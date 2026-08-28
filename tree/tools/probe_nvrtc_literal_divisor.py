#!/usr/bin/env python3
"""Where NVRTC turns ``x / <float literal>`` into ``x * (1/<literal>)``.

The rewrite is a compiler decision, so the probe asks the compiler as well as
the card, and the two instruments answer different questions:

**On the card** -- seven spellings of one quotient, run on whatever device is
present, compared bit for bit against the host's correctly-rounded
``numpy.float32`` quotient.  This is the arm that can be trusted to notice a
one-ulp difference at all: the explicit ``x * (1.0f/12.0f)`` arm must diverge,
or the probe is not measuring anything, and the runtime-divisor arm must be
exact, or it is measuring something else.

**In the compiler** -- the same spellings compiled to PTX for every target
NVRTC supports, recording which instruction each one becomes.  A card of the
affected architecture is not needed to establish that, and asking only the
local card would answer for one target and read as an answer for all of them.

``--expect`` fails unless both instruments agree with the declared
expectations, so a probe that has quietly stopped discriminating is a failure
rather than a green run.

This supersedes the single-target, single-value probe that first isolated the
mechanism while the regional kernels were being proved against the CPU
authority.  It keeps every arm that probe had and adds three things it could
not answer: the population rate rather than one quotient, the sweep across
every target NVRTC supports -- which is what turned "this stack" into
"compute_100 and above" -- and the translation-unit-constant remedy.  Its
default divisor is the shared kernels' 12, and ``--divisor 5`` reproduces the
original reading.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from hexcore.cuda_fp32 import CUDA_FTZ_HELPERS  # noqa: E402

#: What the runtime requests; CuPy's RawModule route appends ``-ftz=true``
#: and, on CUDA 12+, ``--device-as-default-execution-space``.
REQUESTED_OPTIONS = ("--std=c++17", "--fmad=false")
EFFECTIVE_OPTIONS = (
    *REQUESTED_OPTIONS, "-ftz=true", "--device-as-default-execution-space",
)

#: label, body, role, and the declared expectation:
#:   "divides"    -- must compile to div.rn.f32 on every supported target
#:   "boundary"   -- must divide on the low targets and multiply on the
#:                   high ones, which is what locates the boundary
#:   "multiplies" -- the rewrite written out by hand; must multiply
#:                   everywhere, and is the control proving the value
#:                   instrument can see a one-ulp difference at all
_ARMS: tuple[tuple[str, str, str, str], ...] = (
    (
        "mpas_div, source literal",
        "out[0] = mpas_div(numerator, {literal});",
        "the shape the shared kernels shipped before this lane",
        "boundary",
    ),
    (
        "mpas_div, translation-unit constant",
        "out[0] = mpas_div(numerator, probe_denominator);",
        "the remedy this lane applies to the shared kernels",
        "divides",
    ),
    (
        "mpas_div, runtime argument",
        "out[0] = mpas_div(numerator, divisor);",
        "the remedy the regional kernels apply",
        "divides",
    ),
    (
        "plain operator, source literal",
        "out[0] = numerator / ({literal});",
        "the same hazard without the helper",
        "boundary",
    ),
    (
        "plain operator, runtime argument",
        "out[0] = numerator / divisor;",
        "control",
        "divides",
    ),
    (
        "__fdiv_rn, source literal",
        "out[0] = __fdiv_rn(numerator, {literal});",
        "the IEEE-division intrinsic, which resists the rewrite but bypasses "
        "the FTZ subnormal guard mpas_div carries",
        "divides",
    ),
    (
        "__fdiv_rn, runtime argument",
        "out[0] = __fdiv_rn(numerator, divisor);",
        "control",
        "divides",
    ),
    (
        "multiply by the reciprocal literal",
        "out[0] = numerator * (1.0f / ({literal}));",
        "the rewrite written out by hand -- the probe's own control",
        "multiplies",
    ),
)

_KERNEL = """
__constant__ float probe_denominator = {literal};

extern "C" __global__ void probe(
    const float numerator, const float divisor, float *out)
{{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    {body}
}}
"""

_DIV = re.compile(r"^\s*div\.rn(?:\.ftz)?\.f32\b")
_MUL_IMM = re.compile(r"^\s*mul\.rn(?:\.ftz)?\.f32\s+%f\d+,\s*%f\d+,\s*0f([0-9A-Fa-f]{8});")


def _source(body_template: str, literal: str) -> str:
    return CUDA_FTZ_HELPERS + _KERNEL.format(
        literal=literal, body=body_template.format(literal=literal)
    )


def _ptx(source: str, arch: str) -> str:
    from cupy.cuda import nvrtc

    program = nvrtc.createProgram(source, "probe.cu", [], [])
    try:
        nvrtc.compileProgram(
            program, [*EFFECTIVE_OPTIONS, f"--gpu-architecture=compute_{arch}"]
        )
        image = nvrtc.getPTX(program)
    finally:
        nvrtc.destroyProgram(program)
    return bytes(image).decode() if isinstance(image, (bytes, bytearray)) else image


def _instruction(source: str, arch: str, reciprocal_hex: str) -> str:
    """How the divide became code at ``arch``."""

    for line in _ptx(source, arch).splitlines():
        if _DIV.match(line):
            return "div.rn.f32"
        match = _MUL_IMM.match(line)
        if match and match.group(1).upper() == reciprocal_hex:
            return "mul.rn.f32 by the reciprocal"
    return "neither"


def _device_record() -> dict[str, Any]:
    import cupy
    from cupy.cuda import nvrtc, runtime

    properties = runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    record: dict[str, Any] = {
        "name": name,
        "sm": f"sm_{properties['major']}{properties['minor']}",
        "multiprocessors": int(properties["multiProcessorCount"]),
        "total_global_mem_gib": round(properties["totalGlobalMem"] / (1 << 30), 3),
        "cupy_version": cupy.__version__,
        "numpy_version": np.__version__,
        "nvrtc_get_version": list(nvrtc.getVersion()),
        "cuda_runtime_version": int(runtime.runtimeGetVersion()),
        "cuda_driver_version": int(runtime.driverGetVersion()),
        "host": platform.node(),
        "platform": platform.platform(),
    }
    try:
        from gpuwm.certify.compile_platform import compile_platform_fingerprint

        record["gpuwm_compile_platform_fingerprint"] = dict(
            compile_platform_fingerprint()
        )
    except Exception as error:  # pragma: no cover - reported, never inferred
        record["gpuwm_compile_platform_fingerprint"] = {
            "status": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
        }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--divisor", type=float, default=12.0,
        help="the shared kernels' third-order stencil denominator",
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expect", action="store_true")
    args = parser.parse_args(argv)

    import cupy as cp
    from cupy.cuda import nvrtc

    divisor32 = np.float32(args.divisor)
    literal = f"{float(divisor32)!r}f"
    reciprocal_hex = struct.pack(
        ">f", np.float32(1.0) / divisor32
    ).hex().upper()
    architectures = [str(value) for value in nvrtc.getSupportedArchs()]

    # -- the card ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    numerators = (rng.standard_normal(args.samples) * 1.0e3).astype(np.float32)
    host = (numerators / divisor32).astype(np.float32)
    host_bits = host.view(np.uint32)
    device_numerators = cp.asarray(numerators)

    arms: list[dict[str, Any]] = []
    for label, body, role, expectation in _ARMS:
        source = _source(body, literal)
        sweep_source = CUDA_FTZ_HELPERS + f"""
__constant__ float probe_denominator = {literal};

extern "C" __global__ void sweep(
    const int n, const float divisor, const float *numerators, float *out)
{{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const float numerator = numerators[i];
    {body.format(literal=literal).replace("out[0]", "out[i]")}
}}
"""
        kernel = cp.RawKernel(
            sweep_source, "sweep", options=REQUESTED_OPTIONS, backend="nvrtc"
        )
        values = cp.zeros(args.samples, dtype=cp.float32)
        threads = 256
        kernel(
            ((args.samples + threads - 1) // threads,), (threads,),
            (np.int32(args.samples), divisor32, device_numerators, values),
        )
        cp.cuda.runtime.deviceSynchronize()
        bits = values.get().view(np.uint32)
        mismatches = int(np.count_nonzero(bits != host_bits))
        arms.append({
            "arm": label,
            "role": role,
            "expression": body.format(literal=literal),
            "expectation": expectation,
            "on_this_card": {
                "mismatches_vs_correctly_rounded_host": mismatches,
                "rate": mismatches / args.samples,
            },
            "compiled_to": {
                f"compute_{arch}": _instruction(source, arch, reciprocal_hex)
                for arch in architectures
            },
        })

    # -- the instrument's own controls ------------------------------------
    by_label = {arm["arm"]: arm for arm in arms}
    hand_written = by_label["multiply by the reciprocal literal"]
    runtime_divisor = by_label["mpas_div, runtime argument"]
    can_see_one_ulp = (
        hand_written["on_this_card"]["mismatches_vs_correctly_rounded_host"] > 0
    )
    exact_form_is_exact = (
        runtime_divisor["on_this_card"]["mismatches_vs_correctly_rounded_host"] == 0
    )

    # The boundary is read off the measurement, not asserted from a table.
    literal_arm = by_label["mpas_div, source literal"]
    rewritten_targets = sorted(
        int(target.removeprefix("compute_"))
        for target, instruction in literal_arm["compiled_to"].items()
        if instruction == "mul.rn.f32 by the reciprocal"
    )
    divided_targets = sorted(
        int(target.removeprefix("compute_"))
        for target, instruction in literal_arm["compiled_to"].items()
        if instruction == "div.rn.f32"
    )
    disagreements: list[str] = []
    expected_sets = {
        "divides": {"div.rn.f32"},
        "multiplies": {"mul.rn.f32 by the reciprocal"},
        "boundary": {"div.rn.f32", "mul.rn.f32 by the reciprocal"},
    }
    for arm in arms:
        instructions = set(arm["compiled_to"].values())
        expected = expected_sets[arm["expectation"]]
        if instructions != expected:
            disagreements.append(
                f"{arm['arm']} declares {arm['expectation']!r} but compiled "
                f"to {sorted(instructions)}"
            )
    if divided_targets and rewritten_targets:
        if max(divided_targets) >= min(rewritten_targets):
            disagreements.append(
                "the rewritten and divided target sets interleave; the "
                "boundary is not a threshold"
            )
    else:
        disagreements.append(
            "the literal divisor is not compiled both ways across the "
            "supported targets, so no boundary is measured"
        )
    if not can_see_one_ulp:
        disagreements.append(
            "the hand-written reciprocal multiply did not diverge on this "
            "card, so the value instrument cannot see a one-ulp difference"
        )
    if not exact_form_is_exact:
        disagreements.append(
            "the runtime-divisor arm diverged on this card, so the value "
            "instrument is measuring something other than the rewrite"
        )

    document = {
        "schema": "mpas-port.nvrtc-reciprocal-probe/v2",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "device": _device_record(),
        "compile_options": {
            "requested": list(REQUESTED_OPTIONS),
            "effective": list(EFFECTIVE_OPTIONS),
            "note": (
                "CuPy's RawModule route appends -ftz=true after caller options "
                "and --device-as-default-execution-space on CUDA 12+; the PTX "
                "arm compiles with both explicitly"
            ),
        },
        "divisor": {
            "value": float(divisor32),
            "literal_in_source": literal,
            "reciprocal_bits": f"0x{reciprocal_hex}",
            "reciprocal_is_exact": bool(
                np.float64(1.0) / np.float64(divisor32)
                == np.float64(np.float32(1.0) / divisor32)
            ),
        },
        "population": {
            "samples": int(args.samples),
            "distribution": f"float32 N(0,1)*1e3, seed {args.seed}",
        },
        "supported_targets": architectures,
        "arms": arms,
        "boundary": {
            "targets_that_divide": divided_targets,
            "targets_that_multiply_by_the_reciprocal": rewritten_targets,
            "lowest_rewritten_target": (
                min(rewritten_targets) if rewritten_targets else None
            ),
        },
        "instrument_validation": {
            "value_arm_can_see_one_ulp": can_see_one_ulp,
            "value_arm_is_exact_for_the_exact_form": exact_form_is_exact,
            "disagreements": disagreements,
        },
    }
    document["verdict"] = (
        "a source-literal divisor is compiled to a reciprocal multiply for "
        f"every target from compute_{min(rewritten_targets)} up and to a real "
        "division below it; a host-writable divisor and __fdiv_rn divide on "
        "every target"
        if not disagreements
        else "the probe did not reproduce its declared expectations"
    )

    text = json.dumps(document, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.expect and disagreements:
        print(f"PROBE FAILED: {disagreements}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
