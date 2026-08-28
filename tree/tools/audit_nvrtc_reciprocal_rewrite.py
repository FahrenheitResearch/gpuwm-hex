#!/usr/bin/env python3
"""Census every CUDA translation unit NVRTC rewrites into a reciprocal multiply.

A grep for ``12.0f`` finds what a human already suspected.  This audit asks the
compiler instead: each translation unit is compiled to PTX twice with identical
options -- once for a target below the rewrite boundary and once for a target
above it -- and the two images are compared instruction for instruction.  A site
appears in the census only because the compiler actually emitted
``mul.rn[.ftz].f32`` against the reciprocal of a constant where the lower target
emitted ``div.rn[.ftz].f32`` against that constant.

That makes the audit arbitrary in the sense the house requires: a new
translation unit, a new literal, or a new NVRTC build needs no table entry
here.  It also runs anywhere NVRTC exists -- no card of the affected
architecture is required, because the rewrite happens in the compiler.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# The options the shipped route requests, plus the terminal flag CuPy's
# RawModule appends and the execution-space flag CuPy adds on CUDA 12+.
SHIPPED_OPTIONS = (
    "--std=c++17",
    "--fmad=false",
    "-ftz=true",
    "--device-as-default-execution-space",
)
BELOW_BOUNDARY_ARCH = "86"
ABOVE_BOUNDARY_ARCH = "120"

# A PTX virtual register, whatever this NVRTC calls one.
#
# THE BREAKAGE THIS PREVENTS, measured 2026-08-27 on the proving RTX 5090 (#373).
# These patterns used to spell the register `%f\d+`, which is what NVRTC
# 13.0 emits for a float32 operand.  **NVRTC 13.3 targeting `compute_120`
# names the same registers `%r`**, so on that toolchain the census matched
# nothing and went blind in BOTH directions at once:
#
#   * `test_no_translation_unit_is_rewritten_into_a_reciprocal_multiply`
#     reported GREEN because `rewrite_sites` found zero rewritten sites --
#     a silent green, the worst failure this program has a name for;
#   * `test_the_affected_kernels_still_divide_on_every_target[120]` reported
#     RED for divisions that were, measured, right there in the PTX
#     (`div.rn.ftz.f32 %r333, %r24, %r2` -- two of them, exactly as pinned).
#
# Only the teeth test caught it. The opcode still carries `.f32`, so
# widening the REGISTER spelling widens what this can SEE and changes
# nothing about what it REQUIRES: an immediate operand is `0f........`,
# which does not begin with `%`, so a rewritten multiply can never be
# mistaken for a surviving division.
_REG = r"%\w+"
_DIV_CONST = re.compile(
    rf"^\s*div\.rn(?:\.ftz)?\.f32\s+({_REG}),\s*({_REG}),\s*0f([0-9A-Fa-f]{{8}});"
)
_MUL_CONST = re.compile(
    rf"^\s*mul\.rn(?:\.ftz)?\.f32\s+({_REG}),\s*({_REG}),\s*0f([0-9A-Fa-f]{{8}});"
)
_FUNC = re.compile(r"^\s*(?:\.\w+\s+)*\.entry\s+([A-Za-z_][A-Za-z0-9_]*)")


def _f32_from_hex(text: str) -> float:
    return float(struct.unpack(">f", bytes.fromhex(text))[0])


def _hex_from_f32(value: float) -> str:
    return struct.pack(">f", np.float32(value)).hex().upper()


def compile_ptx(source: str, arch: str) -> str:
    """PTX for ``source`` at ``arch`` under the shipped option set."""

    from cupy.cuda import nvrtc

    program = nvrtc.createProgram(source, "unit.cu", [], [])
    try:
        nvrtc.compileProgram(
            program,
            [*SHIPPED_OPTIONS, f"--gpu-architecture=compute_{arch}"],
        )
        image = nvrtc.getPTX(program)
    finally:
        nvrtc.destroyProgram(program)
    if isinstance(image, (bytes, bytearray)):
        return bytes(image).decode("utf-8", errors="replace")
    return str(image)


def _current_entry(lines: list[str], index: int) -> str:
    for back in range(index, -1, -1):
        match = _FUNC.match(lines[back])
        if match:
            return match.group(1)
    return "<file scope>"


def rewrite_sites(source: str) -> dict[str, Any]:
    """Sites where the higher target multiplies where the lower one divides."""

    low = compile_ptx(source, BELOW_BOUNDARY_ARCH)
    high = compile_ptx(source, ABOVE_BOUNDARY_ARCH)
    low_lines = low.splitlines()
    high_lines = high.splitlines()

    divisors_below: dict[str, int] = {}
    for line in low_lines:
        match = _DIV_CONST.match(line)
        if match:
            key = match.group(3).upper()
            divisors_below[key] = divisors_below.get(key, 0) + 1

    sites: list[dict[str, Any]] = []
    for index, line in enumerate(high_lines):
        match = _MUL_CONST.match(line)
        if not match:
            continue
        immediate = match.group(3).upper()
        multiplier = _f32_from_hex(immediate)
        if multiplier == 0.0:
            continue
        implied = np.float32(1.0) / np.float32(multiplier)
        implied_hex = _hex_from_f32(implied)
        if implied_hex not in divisors_below:
            continue
        # Only count it when the reciprocal round-trips: mul by 1/d where the
        # lower target divides by exactly d.
        if _hex_from_f32(np.float32(1.0) / implied) != immediate:
            continue
        sites.append({
            "entry": _current_entry(high_lines, index),
            "divisor": float(implied),
            "divisor_bits": f"0x{implied_hex}",
            "reciprocal_bits": f"0x{immediate}",
            "reciprocal_is_exact": bool(
                np.float64(1.0) / np.float64(implied)
                == np.float64(np.float32(multiplier))
            ),
            "ptx_below_boundary": (
                f"div.rn.ftz.f32 ..., 0f{implied_hex}"
            ),
            "ptx_above_boundary": high_lines[index].strip(),
        })

    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "ptx_sha256": {
            f"compute_{BELOW_BOUNDARY_ARCH}": hashlib.sha256(
                low.encode("utf-8")
            ).hexdigest(),
            f"compute_{ABOVE_BOUNDARY_ARCH}": hashlib.sha256(
                high.encode("utf-8")
            ).hexdigest(),
        },
        "div_by_constant_below_boundary": sum(divisors_below.values()),
        "rewritten_sites": sites,
        "count": len(sites),
    }


def translation_units() -> Iterator[tuple[str, str, str]]:
    """Yield ``(module_key, python_anchor, source)`` for every unit."""

    import importlib

    def attribute(module_name: str, attribute_name: str) -> Any:
        module = importlib.import_module(f"hexcore.{module_name}")
        return getattr(module, attribute_name)

    simple = (
        ("hexcore.cuda_acoustic", "cuda_acoustic", "_CUDA_SOURCE"),
        ("hexcore.cuda_acoustic_v841", "cuda_acoustic_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_horizontal", "cuda_horizontal", "_CUDA_SOURCE"),
        (
            "hexcore.cuda_horizontal_v841",
            "cuda_horizontal_v841",
            "_CUDA_SOURCE",
        ),
        ("hexcore.cuda_transport", "cuda_transport", "_CUDA_SOURCE"),
        ("hexcore.cuda_transport_v841", "cuda_transport_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_dynamics_v841", "cuda_dynamics_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_driver", "cuda_driver", "_CUDA_SOURCE"),
        (
            "hexcore.cuda_driver.physics_v841",
            "cuda_driver",
            "CUDA_V841_PHYSICS_DRIVER_SOURCE",
        ),
        ("hexcore.cuda_gwdo_v841", "cuda_gwdo_v841", "_CUDA_SOURCE"),
        ("hexcore.cuda_physics_v841", "cuda_physics_v841", "_CUDA_SOURCE"),
        (
            "hexcore.cuda_physics_prep_v841",
            "cuda_physics_prep_v841",
            "_CUDA_SOURCE",
        ),
        ("hexcore.cuda_regional_v841", "cuda_regional_v841", "CUDA_REGIONAL_SOURCE"),
    )
    for module_key, module_name, attribute_name in simple:
        yield (
            module_key,
            f"src/hexcore/{module_name}.py::{attribute_name}",
            attribute(module_name, attribute_name),
        )

    # Built lazily rather than held as a module constant.
    from hexcore import cuda_acoustic_lts

    yield (
        "hexcore.cuda_acoustic_lts",
        "src/hexcore/cuda_acoustic_lts.py::_build_source()",
        cuda_acoustic_lts._build_source(),
    )

    from hexcore.cuda_backend import recovery

    yield (
        "hexcore.cuda_backend.recovery",
        "src/hexcore/cuda_backend/recovery.py::RECOVERY_CUDA_SOURCE",
        recovery.RECOVERY_CUDA_SOURCE,
    )


def source_bearing_files() -> set[str]:
    """Every module under ``src`` that carries CUDA source text.

    The census enumerates translation units by name.  This walk is the control
    on that list: a new CUDA-bearing module that nobody added to
    :func:`translation_units` shows up here as uncovered rather than being
    quietly left out of the audit.
    """

    # A literal entry-point definition, not the escaped form a regex uses to
    # scan someone else's source -- ``cuda_ftz`` audits the other modules'
    # translation units and defines none of its own.
    definition = re.compile(r'extern +"C" +__global__ +void +')
    found: set[str] = set()
    for path in sorted(_REPO_SRC.rglob("*.py")):
        if path.name == "cuda_fp32.py":
            continue  # the shared prelude; it defines no kernel of its own
        text = path.read_text(encoding="utf-8", errors="replace")
        if definition.search(text):
            found.add(path.relative_to(_REPO_SRC).as_posix())
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--expect-clean", action="store_true",
        help="exit non-zero when any translation unit still has a rewrite site",
    )
    args = parser.parse_args(argv)

    from cupy.cuda import nvrtc, runtime

    units: dict[str, Any] = {}
    total = 0
    covered: set[str] = set()
    for module_key, anchor, source in translation_units():
        record = rewrite_sites(source)
        record["python_anchor"] = anchor
        units[module_key] = record
        total += record["count"]
        covered.add(anchor.split("::", 1)[0].removeprefix("src/"))

    uncovered = sorted(source_bearing_files() - covered)

    document = {
        "schema": "mpas-port.nvrtc-reciprocal-census/v1",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "method": (
            "each translation unit compiled to PTX twice with identical "
            f"options, for compute_{BELOW_BOUNDARY_ARCH} and "
            f"compute_{ABOVE_BOUNDARY_ARCH}; a site is counted only where the "
            "higher target emits mul.rn.f32 by the reciprocal of a constant "
            "the lower target divides by"
        ),
        "compiler": {
            "nvrtc_get_version": list(nvrtc.getVersion()),
            "cuda_runtime_version": int(runtime.runtimeGetVersion()),
            "cuda_driver_version": int(runtime.driverGetVersion()),
            "options": list(SHIPPED_OPTIONS),
            "host": platform.node(),
        },
        "total_rewritten_sites": total,
        "cuda_bearing_modules_not_censused": uncovered,
        "translation_units": units,
    }
    text = json.dumps(document, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.expect_clean and (total or uncovered):
        print(
            f"AUDIT FAILED: {total} rewrite sites remain; "
            f"uncensused CUDA-bearing modules {uncovered}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
