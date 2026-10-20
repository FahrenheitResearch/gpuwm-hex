"""Cell-local acoustic fusion seeded from the unchanged MPAS CUDA kernels.

The edge update stays separate: it reads the previous cell perturbations.
After it completes, prepare, flux sums and the vertical solve access mutable
values only in their own cell. Neighbor slots and all arithmetic are retained.
"""
from __future__ import annotations

import os
import re

from .cuda_acoustic_v841 import _CUDA_SOURCE as SHARED_SOURCE
from .cuda_regional_v841 import CUDA_REGIONAL_SOURCE
from .cuda_fp32 import CUDA_FTZ_HELPERS

MODULE_KEY = "hexcore.cuda_acoustic_fusion_v841"
KERNEL_NAME = "acoustic_cell_fused_v841"


def fusion_enabled() -> bool:
    return os.environ.get("GPUWM_HEX_ACOUSTIC_FUSION", "0") == "1"


def _extract(source: str, name: str) -> tuple[list[str], str]:
    marker = f'extern "C" __global__ void {name}('
    start = source.index(marker) + len(marker)
    end = source.index(")\n{", start)
    declarations = [part.strip() for part in source[start:end].split(",")]
    first = end + 2
    depth = 1
    last = first + 1
    while depth:
        depth += (source[last] == "{") - (source[last] == "}")
        last += 1
    return declarations, source[first + 1:last - 1]


def _name(declaration: str) -> str:
    return re.search(r"(\w+)$", declaration).group(1)


def build_source() -> tuple[str, tuple[str, ...]]:
    names = ("acoustic_prepare_v841", "acoustic_rs_ts_regional_v841",
             "acoustic_column_solve_regional_v841")
    helpers = []
    calls = []
    parameters: dict[str, str] = {}
    for name, source in zip(names, (SHARED_SOURCE, CUDA_REGIONAL_SOURCE,
                                   CUDA_REGIONAL_SOURCE)):
        declarations, body = _extract(source, name)
        for declaration in declarations:
            argument = _name(declaration)
            old = parameters.get(argument)
            if old is None or ("*" in declaration and not declaration.startswith("const ")):
                parameters[argument] = declaration
            elif old.replace("const ", "") != declaration.replace("const ", ""):
                raise ValueError(f"inconsistent fused argument {argument}")
        if name == names[1]:
            # Retain the exact flux body and ascending slot accumulation.
            body = body.replace("REGIONAL_ELEMENT_LOOP(nlev * ncells)",
                                "for (int k = 0; k < nlev; ++k)")
            body = body.replace("        const int cell = element % ncells;\n", "")
            body = body.replace("        const int k = element / ncells;\n", "")
            body = "\n    const int cell = blockDim.x * blockIdx.x + threadIdx.x;\n" \
                   "    if (cell >= ncells) return;\n" + body
        helper = name + "_cell"
        helpers.append("__device__ __forceinline__ void " + helper + "(\n    " +
                       ",\n    ".join(declarations) + ")\n{" + body + "}\n")
        calls.append(helper + "(" + ", ".join(map(_name, declarations)) + ");")
    macros = "\n#define C2(k,c,nc) ((k)*(nc)+(c))\n" \
             "#define RC2(k,c,nc) ((k)*(nc)+(c))\n" \
             "#define RE2(k,e,ne) ((k)*(ne)+(e))\n" \
             "#define RCES(c,s,me) ((c)*(me)+(s))\n"
    source = CUDA_FTZ_HELPERS + macros + "\n".join(helpers)
    source += 'extern "C" __global__ void ' + KERNEL_NAME + "(\n    " + \
              ",\n    ".join(parameters.values()) + ")\n{\n    " + \
              "\n    ".join(calls) + "\n}\n"
    return source, tuple(parameters)


CUDA_FUSED_SOURCE, ARGUMENT_NAMES = build_source()


def launch(runtime, values: dict, ncells: int) -> None:
    kernel = runtime.kernels.cache.raw_kernel(
        KERNEL_NAME, CUDA_FUSED_SOURCE, module_key=MODULE_KEY)
    kernel(((ncells + 127) // 128,), (128,),
           tuple(values[name] for name in ARGUMENT_NAMES))


def advance_cell(runtime, mesh, out, forcing, coefficients, context, *,
                 nlev, ncells, nedges, max_edges, dts, small_step, fzm, fzp, rdzw):
    import numpy as np
    import cupy as cp
    values = {name: np.int32(value) for name, value in (
        ("nlev", nlev), ("ncells", ncells), ("nedges", nedges),
        ("max_edges", max_edges), ("small_step", small_step))}
    values.update(dts=np.float32(dts), fzm=fzm, fzp=fzp, rdzw=rdzw,
                  spec_zone_mask_cell=runtime.spec_zone_mask_cell)
    for name in ("rw_p", "rtheta_pp", "rtheta_pp_old", "rho_pp", "ww_avg", "ru_p"):
        values[name] = getattr(out, name)
    for name in ("theta_m", "tend_rho", "tend_rt", "zz", "rho_zz", "dss",
                 "w", "rw", "rw_save", "tend_rw"):
        values[name] = getattr(forcing, name)
    for name in ("cofrz", "coftz", "cofwr", "cofwz", "cofwt", "a_tri",
                 "alpha_tri", "gamma_tri"):
        values[name] = getattr(coefficients, name)
    for name in ("inv_area_cell", "ewm", "etp", "etm", "ewp"):
        values[name] = getattr(context, name)
    for name in ("n_edges_on_cell", "edges_on_cell", "cells_on_edge", "edge_sign_on_cell", "dv_edge"):
        values[name] = getattr(mesh, name)
    values["rs"] = cp.zeros((nlev, ncells), dtype=cp.float32)
    values["ts"] = cp.zeros_like(values["rs"])
    launch(runtime, values, ncells)
