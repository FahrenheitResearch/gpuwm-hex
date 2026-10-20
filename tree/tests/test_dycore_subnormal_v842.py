"""Native subnormal corpus and a sentinel that fails under forced FTZ."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from test_dycore_options_v842 import (
    native_d1, device_d1, test_complete_nonles_module_bits,
    test_native_cam_and_rayleigh_bits,
)


@pytest.mark.gpu
def test_subnormal_native_upper_and_compiler(native_d1):
    import cupy as cp
    from cupy.cuda import compiler
    from hexcore.cuda_upper_damping_v842 import direct_cache, CUDA_OPTIONS, apply_rayleigh_u_cuda
    folder, data = native_d1
    assert (folder / "subnormal.txt").exists()
    words = np.frombuffer((folder / "subnormal-upper.bin").read_bytes(), dtype="<f4").reshape(5,8)
    source = r'''extern "C" __global__ void gradual(const float* input,
        const float* factor, float* output) {
        int i=threadIdx.x; if(i<8)output[i]=__fmul_rn(input[i],factor[i]);
    }'''
    inputs, factors = cp.asarray(words[0]), cp.asarray(words[1])
    output = cp.empty_like(inputs)
    direct_cache().raw_kernel("gradual", source,
        module_key="hexcore.dycore_subnormal_test")((1,), (8,), (inputs, factors, output))
    actual = cp.asnumpy(output)
    np.testing.assert_array_equal(actual.view(np.uint32), words[2].view(np.uint32))
    # Counterfactual compile proves this sentinel detects the reported hazard.
    control_options = tuple("--ftz=true" if option == "--ftz=false" else option for option in CUDA_OPTIONS)
    binary, _ = compiler.compile_using_nvrtc(source, options=control_options,
        arch=cp.cuda.Device().compute_capability)
    module = cp.cuda.function.Module(); module.load(binary)
    module.get_function("gradual")((1,), (8,), (inputs, factors, output))
    flushed = cp.asnumpy(output)
    assert np.count_nonzero(flushed.view(np.uint32) != words[2].view(np.uint32)) == 8
    tendency = cp.zeros((8,1), cp.float32)
    apply_rayleigh_u_cuda(tendency, factors.reshape(8,1), inputs.reshape(8,1),
        levels=8, days=0.03125, solved=1)
    ray = cp.asnumpy(tendency).ravel()
    np.testing.assert_array_equal(ray.view(np.uint32), words[4].view(np.uint32))
    path = os.environ.get("HEXCORE_D1_SUBNORMAL_RECEIPT")
    if path:
        Path(path).write_text(json.dumps({"status":"PASS", "native_process_count":1,
            "hex_process_count":1, "mpi":False, "words":16, "different_words":0,
            "excluded_words":[], "ftz_control_different_words":8,
            "compiler":"direct NVRTC", "cuda_options":list(CUDA_OPTIONS),
            "ftz_control_options":list(control_options),
            "ftz_control_binary_sha256":hashlib.sha256(binary).hexdigest(),
            "native_binary_sha256":hashlib.sha256((folder/"subnormal-deck").read_bytes()).hexdigest(),
            "input_sha256":hashlib.sha256((folder/"subnormal-upper.bin").read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
            "native_rayleigh_sha256":hashlib.sha256(words[4].tobytes()).hexdigest(),
            "cuda_rayleigh_sha256":hashlib.sha256(ray.tobytes()).hexdigest(),
            "compile_inventory":direct_cache().inventory},indent=2)+"\n")
