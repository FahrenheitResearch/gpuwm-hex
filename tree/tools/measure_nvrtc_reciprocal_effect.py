#!/usr/bin/env python3
"""Per-site effect of the NVRTC reciprocal rewrite, on real regional bytes.

The rewrite happens in the compiler and only for targets at or above the
boundary the census measures, so a card below that boundary cannot show it by
running the shipped kernel.  This tool reproduces the higher target's
arithmetic on any card instead, at instruction resolution:

``shipped``
    the ordinary NVRTC source-to-cubin route, exactly as the runtime compiles
    it.
``assembled``
    the same translation unit taken to PTX and assembled by ``ptxas``.  This
    is the control on the emulation pipeline -- it must agree with ``shipped``
    bit for bit, or the pipeline is changing arithmetic on its own and no
    conclusion drawn through it is worth anything.
``rewritten``
    the same PTX with every census-identified ``div.rn.f32`` by a constant
    replaced by ``mul.rn.f32`` by that constant's reciprocal -- the exact
    substitution the higher target's compiler makes -- then assembled the same
    way.

Each is compared against the CPU authority the port already carries.  The
difference between ``shipped`` and ``rewritten`` is the defect, measured on the
bytes it actually runs on rather than on a synthetic population.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_TREE = Path(__file__).resolve().parents[1]
for _path in (_TREE / "src", _TREE / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from audit_nvrtc_reciprocal_rewrite import (  # noqa: E402
    BELOW_BOUNDARY_ARCH,
    compile_ptx,
    rewrite_sites,
)

_DIV_CONST = re.compile(
    r"^(\s*)div\.rn(\.ftz)?\.f32(\s+)(%f\d+),(\s*)(%f\d+),\s*0f([0-9A-Fa-f]{8});"
)


def _find_ptxas() -> str:
    found = shutil.which("ptxas")
    if found:
        return found
    roots = [
        Path(os.environ.get("CUDA_PATH", "")),
        *sorted(
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*"),
            reverse=True,
        ),
        Path("/usr/local/cuda"),
    ]
    for root in roots:
        for name in ("bin/ptxas.exe", "bin/ptxas"):
            candidate = root / name
            if candidate.is_file():
                return str(candidate)
    raise SystemExit("ptxas is required to assemble the emulation modules")


PTXAS = _find_ptxas()


def _assemble(ptx: str, arch: str) -> str:
    directory = Path(tempfile.mkdtemp(prefix="nvrtc-recip-"))
    ptx_path = directory / "unit.ptx"
    cubin_path = directory / "unit.cubin"
    ptx_path.write_text(ptx, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [PTXAS, "-arch", f"sm_{arch}", "-o", str(cubin_path), str(ptx_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ptxas refused the emulation module: {result.stderr}")
    return str(cubin_path)


def _rewrite_ptx(ptx: str, divisor_bits: set[str]) -> tuple[str, int]:
    """Apply the higher target's substitution to the lower target's PTX."""

    patched: list[str] = []
    applied = 0
    for line in ptx.splitlines():
        match = _DIV_CONST.match(line)
        if match and match.group(7).upper() in divisor_bits:
            divisor = struct.unpack(">f", bytes.fromhex(match.group(7)))[0]
            reciprocal = struct.pack(
                ">f", np.float32(1.0) / np.float32(divisor)
            ).hex().upper()
            ftz = match.group(2) or ""
            patched.append(
                f"{match.group(1)}mul.rn{ftz}.f32{match.group(3)}"
                f"{match.group(4)},{match.group(5)}{match.group(6)}, 0f{reciprocal};"
            )
            applied += 1
        else:
            patched.append(line)
    return "\n".join(patched) + "\n", applied


class Arms:
    """The three compiled arms of one translation unit."""

    def __init__(self, source: str, arch: str, cupy_module: Any) -> None:
        self.census = rewrite_sites(source)
        divisor_bits = {
            site["divisor_bits"].removeprefix("0x").upper()
            for site in self.census["rewritten_sites"]
        }
        ptx = compile_ptx(source, arch)
        rewritten_ptx, applied = _rewrite_ptx(ptx, divisor_bits)
        if applied != len(self.census["rewritten_sites"]):
            raise SystemExit(
                f"emulation patched {applied} instructions but the census "
                f"named {len(self.census['rewritten_sites'])}"
            )
        self.substitutions = applied
        self.shipped = cupy_module.RawModule(
            code=source,
            options=("--std=c++17", "--fmad=false"),
            backend="nvrtc",
            enable_cooperative_groups=False,
        )
        self.assembled = cupy_module.RawModule(path=_assemble(ptx, arch))
        self.rewritten = cupy_module.RawModule(
            path=_assemble(rewritten_ptx, arch)
        )

    def kernels(self, name: str) -> dict[str, Any]:
        return {
            "shipped": self.shipped.get_function(name),
            "assembled": self.assembled.get_function(name),
            "rewritten": self.rewritten.get_function(name),
        }


def _launch(kernel: Any, total: int, args: tuple[Any, ...], cupy_module: Any) -> None:
    threads = 256
    blocks = (total + threads - 1) // threads
    kernel((blocks,), (threads,), args)
    cupy_module.cuda.runtime.deviceSynchronize()


def _compare(name: str, authority: np.ndarray, arms: dict[str, np.ndarray],
             scope: str) -> dict[str, Any]:
    reference = np.ascontiguousarray(authority, np.float32).ravel().view(np.uint32)
    record: dict[str, Any] = {
        "payload": name,
        "scope": scope,
        "values": int(reference.size),
        "authority_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "arms": {},
    }
    authority_values = np.ascontiguousarray(authority, np.float32).ravel()
    for arm, values in arms.items():
        flat = np.ascontiguousarray(values, np.float32).ravel()
        bits = flat.view(np.uint32)
        differ = bits != reference
        count = int(np.count_nonzero(differ))
        # Same-sign float32 bit patterns are ordered, so the unsigned bit
        # distance is the ulp distance; the fluxes here never change sign
        # under a one-ulp divisor perturbation.
        ulps = np.abs(bits.astype(np.int64) - reference.astype(np.int64))
        scale = np.abs(authority_values.astype(np.float64))
        nonzero = scale > 0.0
        relative = np.zeros_like(scale)
        relative[nonzero] = np.abs(
            flat.astype(np.float64)[nonzero] - authority_values.astype(np.float64)[nonzero]
        ) / scale[nonzero]
        record["arms"][arm] = {
            "mismatches_vs_cpu_authority": count,
            "rate": count / max(reference.size, 1),
            "max_ulp": int(ulps.max()) if reference.size else 0,
            "mismatches_beyond_one_ulp": int(np.count_nonzero(ulps > 1)),
            "max_relative_difference": float(relative.max()) if relative.size else 0.0,
            "sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
        }
    return record


def _host_flux3(module: Any, q0, q1, q2, q3, velocity, coefficient):
    return np.asarray(
        module.flux3(q0, q1, q2, q3, velocity, coefficient), np.float32
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path,
        default=Path(os.environ.get("GPUWM_HEX_REGIONAL_REFERENCE_DIR", "")),
        help="the regional reference mirror the contract deck uses",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.reference or not args.reference.is_dir():
        raise SystemExit(
            "--reference (or GPUWM_HEX_REGIONAL_REFERENCE_DIR) must name the "
            "regional reference mirror"
        )

    import cupy as cp
    from cupy.cuda import nvrtc, runtime

    import run_cuda_regional_contract as contract
    from hexcore import cuda_driver, cuda_transport, dynamics
    from hexcore import transport as transport_module

    bundle = contract.Bundle(args.reference, cp)
    nlev, ncells, nedges = bundle.nlev, bundle.ncells, bundle.nedges
    fzm = np.asarray(bundle.vertical.fzm, np.float32)
    fzp = np.asarray(bundle.vertical.fzp, np.float32)

    payloads: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # transport_vertical_flux -- the kernel carrying the standing claim.
    # ------------------------------------------------------------------ #
    transport_arms = Arms(cuda_transport._CUDA_SOURCE, BELOW_BOUNDARY_ARCH, cp)
    stage = np.ascontiguousarray(bundle.scalars_stage, np.float32)
    ww = np.ascontiguousarray(
        np.asarray(bundle.saved.vertical_velocity, np.float32)
    )
    coefficient = np.float32(0.25)
    host_transport = np.asarray(
        transport_module._atmosphere_vertical_flux(
            stage[0], ww, fzm, fzp, 3, float(coefficient)
        ),
        np.float32,
    )
    d_stage, d_ww = cp.asarray(stage), cp.asarray(ww)
    d_fzm, d_fzp = cp.asarray(fzm), cp.asarray(fzp)
    results: dict[str, np.ndarray] = {}
    for arm, kernel in transport_arms.kernels("transport_vertical_flux").items():
        out = cp.zeros((1, nlev + 1, ncells), dtype=cp.float32)
        _launch(kernel, ncells, (
            np.int32(1), np.int32(nlev), np.int32(ncells), coefficient,
            d_stage, d_ww, d_fzm, d_fzp, out,
        ), cp)
        results[arm] = out.get()[0]
    payloads.append({
        **_compare(
            "transport_vertical_flux", host_transport, results,
            "every interface of every cell",
        ),
        "translation_unit": "hexcore.cuda_transport",
        "python_anchor": "src/hexcore/cuda_transport.py::_CUDA_SOURCE",
        "cpu_authority": "hexcore.transport._atmosphere_vertical_flux",
        "census_sites": transport_arms.census["count"],
    })

    # ------------------------------------------------------------------ #
    # The three driver flux kernels.  Their interior branch is the frozen
    # statement function ``dynamics.flux3`` on the same operands, so that is
    # what the comparison uses -- the boundary levels carry no literal
    # divisor and are outside this lane's question.
    # ------------------------------------------------------------------ #
    driver_arms = Arms(cuda_driver._CUDA_SOURCE, BELOW_BOUNDARY_ARCH, cp)
    interior = slice(2, nlev - 1)

    u = np.ascontiguousarray(np.asarray(bundle.state.rho_u, np.float32))
    rw = np.ascontiguousarray(np.asarray(bundle.state.rho_w, np.float32))
    theta = np.ascontiguousarray(np.asarray(bundle.state.rho_theta, np.float32))
    theta_saved = np.ascontiguousarray(
        np.asarray(bundle.saved.rho_theta_perturbation, np.float32)
    ) if hasattr(bundle.saved, "rho_theta_perturbation") else theta
    w = np.ascontiguousarray(
        np.asarray(bundle.saved.vertical_velocity, np.float32)
    )
    cells_on_edge = np.ascontiguousarray(
        np.asarray(bundle.mesh.arrays["cellsOnEdge"], np.int32)
    )
    safe_coe = np.where(cells_on_edge < 0, 0, cells_on_edge)

    d_u, d_rw, d_theta = cp.asarray(u), cp.asarray(rw), cp.asarray(theta)
    d_theta_saved = cp.asarray(theta_saved)
    d_w = cp.asarray(w)
    d_coe = cp.asarray(np.ascontiguousarray(safe_coe))

    levels = np.arange(nlev + 1)[interior]

    # -- vertical_u_flux_f32
    velocity_u = np.float32(0.5) * (
        rw[interior][:, safe_coe[:, 0]] + rw[interior][:, safe_coe[:, 1]]
    )
    host_u = _host_flux3(
        dynamics,
        u[levels - 2], u[levels - 1], u[levels], u[levels + 1],
        velocity_u, 1.0,
    )
    results = {}
    for arm, kernel in driver_arms.kernels("vertical_u_flux_f32").items():
        out = cp.zeros((nlev + 1, nedges), dtype=cp.float32)
        _launch(kernel, nedges, (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            d_u, d_rw, d_coe, d_fzm, d_fzp, out,
        ), cp)
        results[arm] = out.get()[interior]
    payloads.append({
        **_compare(
            "vertical_u_flux_f32", host_u, results,
            "interior levels 2..nlev-2, every edge",
        ),
        "translation_unit": "hexcore.cuda_driver",
        "python_anchor": "src/hexcore/cuda_driver.py::_CUDA_SOURCE",
        "cpu_authority": "hexcore.dynamics.flux3",
        "census_sites": 2,
    })

    # -- theta_vertical_flux_f32
    theta_coefficient = np.float32(0.25)
    host_theta = _host_flux3(
        dynamics,
        theta[levels - 2], theta[levels - 1], theta[levels], theta[levels + 1],
        rw[levels], float(theta_coefficient),
    )
    results = {}
    for arm, kernel in driver_arms.kernels("theta_vertical_flux_f32").items():
        out = cp.zeros((nlev + 1, ncells), dtype=cp.float32)
        _launch(kernel, ncells, (
            np.int32(nlev), np.int32(ncells), theta_coefficient,
            d_theta, d_theta_saved, d_rw, d_rw, d_fzm, d_fzp, out,
        ), cp)
        results[arm] = out.get()[interior]
    payloads.append({
        **_compare(
            "theta_vertical_flux_f32", host_theta, results,
            "interior levels 2..nlev-2, every cell",
        ),
        "translation_unit": "hexcore.cuda_driver",
        "python_anchor": "src/hexcore/cuda_driver.py::_CUDA_SOURCE",
        "cpu_authority": "hexcore.dynamics.flux3",
        "census_sites": 2,
    })

    # -- w_vertical_flux_f32
    velocity_w = np.float32(0.5) * (rw[levels] + rw[levels - 1])
    host_w = _host_flux3(
        dynamics,
        w[levels - 2], w[levels - 1], w[levels], w[levels + 1],
        velocity_w, 1.0,
    )
    results = {}
    for arm, kernel in driver_arms.kernels("w_vertical_flux_f32").items():
        out = cp.zeros((nlev + 1, ncells), dtype=cp.float32)
        _launch(kernel, ncells, (
            np.int32(nlev), np.int32(ncells), d_w, d_rw, out,
        ), cp)
        results[arm] = out.get()[interior]
    payloads.append({
        **_compare(
            "w_vertical_flux_f32", host_w, results,
            "interior levels 2..nlev-2, every cell",
        ),
        "translation_unit": "hexcore.cuda_driver",
        "python_anchor": "src/hexcore/cuda_driver.py::_CUDA_SOURCE",
        "cpu_authority": "hexcore.dynamics.flux3",
        "census_sites": 2,
    })

    properties = runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    document = {
        "schema": "mpas-port.nvrtc-reciprocal-effect/v1",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "device": {
            "name": device_name,
            "sm": f"sm_{properties['major']}{properties['minor']}",
        },
        "compiler": {
            "nvrtc_get_version": list(nvrtc.getVersion()),
            "cuda_driver_version": int(runtime.driverGetVersion()),
            "ptxas": PTXAS,
        },
        "reference": {
            "directory": str(args.reference),
            "n_cells": ncells,
            "n_edges": nedges,
            "n_vert_levels": nlev,
        },
        "emulation": {
            "method": (
                "the lower target's PTX with every census-identified "
                "div.rn.f32-by-constant replaced by mul.rn.f32 by that "
                "constant's reciprocal, assembled by ptxas for this card"
            ),
            "substitutions": {
                "hexcore.cuda_transport": transport_arms.substitutions,
                "hexcore.cuda_driver": driver_arms.substitutions,
            },
        },
        "payloads": payloads,
    }
    control_ok = all(
        payload["arms"]["assembled"]["sha256"]
        == payload["arms"]["shipped"]["sha256"]
        for payload in payloads
    )
    document["pipeline_control"] = {
        "assembled_matches_shipped_everywhere": control_ok,
        "why": (
            "the PTX-to-cubin route must be arithmetic-neutral, or the "
            "rewritten arm's numbers measure the pipeline instead of the "
            "rewrite"
        ),
    }
    text = json.dumps(document, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
