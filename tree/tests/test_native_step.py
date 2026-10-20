"""Exercise the real C loop against a compiled stand-in CUDA driver, no GPU.

This validates dispatch/argument/control ABI, not device execution or numerics.
The fake API performs host copies and exposes ordering/argument observations.
"""
import ctypes as C
from pathlib import Path
import shutil
import subprocess

import pytest

from hexcore.native_step import (
    DeviceClear, DeviceCopy, FrameArgument, Kernel, NativeStep, NativeSubmissionError,
)


FAKE_DRIVER = r'''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static unsigned calls, launches, fail_at;
static uintptr_t last_stream;
static unsigned order[8192];
void reset(unsigned fail) { calls = launches = 0; fail_at = fail; }
unsigned call_count(void) { return calls; }
unsigned kernel_count(void) { return launches; }
unsigned command_at(unsigned i) { return order[i]; }
uintptr_t stream_seen(void) { return last_stream; }
static int record(unsigned kind, void *stream) {
    last_stream = (uintptr_t)stream;
    order[calls++] = kind;
    return fail_at && calls == fail_at ? 719 : 0;
}
int cuLaunchKernel(void *function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, void *stream,
    void **args, void **extra) {
    ++launches;
    int status = record((unsigned)(uintptr_t)function, stream);
    if (status) return status;
    if ((uintptr_t)function == 1) {
        uint32_t *out = *(uint32_t **)args[0];
        const uint32_t *input = *(const uint32_t **)args[1];
        for (unsigned i = 0; i < 4; ++i) if (out[i]) return 700;
        out[0] = *input;
        out[1] = (uint32_t)*(uint64_t *)args[2];
        memcpy(out + 2, args[3], sizeof(uint32_t));
        out[3] = gx + 10 * gy + 100 * gz + bx + by + bz + shared;
    } else if ((uintptr_t)function == 2) {
        uint32_t *out = *(uint32_t **)args[0];
        out[3] += *(uint32_t *)args[1];
    }
    return 0;
}
int cuMemcpyDtoDAsync_v2(uint64_t dst, uint64_t src, size_t bytes, void *stream) {
    int status = record(30, stream);
    if (!status) memcpy((void *)(uintptr_t)dst, (void *)(uintptr_t)src, bytes);
    return status;
}
int cuMemsetD8Async(uint64_t dst, unsigned char fill, size_t bytes, void *stream) {
    int status = record(40, stream);
    if (!status) memset((void *)(uintptr_t)dst, fill, bytes);
    return status;
}
'''


class Frame(C.Structure):
    _fields_ = [("input", C.c_uint64), ("step", C.c_uint64), ("clock", C.c_float)]


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for native submission ABI check")
    root = tmp_path_factory.mktemp("native-step")
    source = Path(__file__).parents[1] / "src/hexcore/native/step_submit.c"
    native = root / "step.so"
    fake_source = root / "cuda.c"
    fake_source.write_text(FAKE_DRIVER)
    fake = root / "cuda.so"
    for inp, out in ((source, native), (fake_source, fake)):
        subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-parameter", "-fPIC", "-shared", str(inp), "-o", str(out)],
                       check=True, capture_output=True)
    cuda = C.CDLL(str(fake))
    cuda.reset.argtypes = [C.c_uint]
    cuda.command_at.argtypes = [C.c_uint]
    cuda.stream_seen.restype = C.c_size_t
    return native, fake, cuda


def make_step(libraries, commands, owners=()):
    native, fake, _ = libraries
    return NativeStep(native, commands, frame_type=Frame, owners=owners, cuda_library=fake)


def test_real_native_order_dynamic_arguments_and_sealed_constants(libraries):
    _, _, cuda = libraries
    cuda.reset(0)
    source = C.c_uint32(0xDEADBEEF)
    scratch, output = (C.c_uint32 * 4)(), (C.c_uint32 * 4)()
    addition = C.c_uint32(7)
    step = make_step(libraries, [
        DeviceClear(C.addressof(scratch), 0, C.sizeof(scratch)),
        Kernel(1, (2, 3), (8,), (C.c_uint64(C.addressof(scratch)),
               FrameArgument("input"), FrameArgument("step"), FrameArgument("clock")), 16),
        DeviceCopy(C.addressof(output), C.addressof(scratch), C.sizeof(scratch)),
        Kernel(2, (1,), (1,), (C.c_uint64(C.addressof(output)), addition)),
    ], owners=(scratch, output, source))
    # Modifying the original scalar must not mutate the sealed arguments.
    addition.value = 99
    frame = Frame(C.addressof(source), 17, 0)
    clock_bits = 0x80000000  # negative zero must survive the host ABI exactly
    C.memmove(C.addressof(frame) + Frame.clock.offset, C.byref(C.c_uint32(clock_bits)), 4)
    step.submit(frame, stream=123)
    assert list(output) == [source.value, 17, clock_bits, 2 + 30 + 100 + 8 + 1 + 1 + 16 + 7]
    assert [cuda.command_at(i) for i in range(4)] == [40, 1, 30, 2]
    assert cuda.stream_seen() == 123
    # A later step changes pointer/counter/float arguments without rebuilding.
    second = C.c_uint32(0x12345678)
    frame.input, frame.step, frame.clock = C.addressof(second), 18, 360
    step.submit(frame, stream=456)
    assert list(output)[:3] == [second.value, 18, 0x43B40000]
    assert cuda.call_count() == 8 and cuda.stream_seen() == 456


def test_many_driver_launches_use_one_native_submission(libraries):
    _, _, cuda = libraries
    cuda.reset(0)
    step = make_step(libraries, [Kernel(9, (1,), (128,), ())] * 2854)
    actual = step._submit
    submissions = []
    def count_call(*args):
        submissions.append(1)
        return actual(*args)
    step._submit = count_call
    step.submit(Frame(), stream=0)
    assert len(submissions) == 1
    assert cuda.kernel_count() == step.kernel_count == step.command_count == 2854


def test_first_driver_error_stops_and_poisons_step(libraries):
    _, _, cuda = libraries
    cuda.reset(2)
    step = make_step(libraries, [Kernel(9, (1,), (1,), ())] * 3)
    with pytest.raises(NativeSubmissionError, match="719 at command 1") as caught:
        step.submit(Frame(), stream=0)
    assert caught.value.issued == 1
    assert cuda.call_count() == 2
    with pytest.raises(RuntimeError, match="poisoned"):
        step.submit(Frame(), stream=0)
    assert cuda.call_count() == 2


def test_bad_descriptors_and_frames_fail_before_dispatch(libraries):
    _, _, cuda = libraries
    cuda.reset(0)
    with pytest.raises(ValueError, match="invalid native step"):
        make_step(libraries, [Kernel(0, (1,), (1,), ())])
    with pytest.raises(ValueError, match="unsigned"):
        make_step(libraries, [DeviceClear(123, 256, 4)])
    with pytest.raises(ValueError, match="invalid native step"):
        make_step(libraries, [])
    step = make_step(libraries, [Kernel(9, (1,), (1,), ())])
    with pytest.raises(TypeError, match="frame differs"):
        step.submit(C.c_uint64(), stream=0)
    assert cuda.call_count() == 0
