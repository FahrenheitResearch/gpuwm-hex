"""CUDA split-explicit acoustic kernels for the dry MPAS-A path.

The scalar authority is :mod:`hexcore.acoustic`, transcribed from frozen
MPAS-A v8.2.3 ``mpas_atm_time_integration.F:1818-2523``.  Arrays retain the
logical CPU order ``(level, entity)`` and are float32 CuPy arrays.  A CUDA
thread owns an entire vertical column where source ordering or the Thomas
recurrence requires it; no host staging occurs inside these routines.

NVRTC is compiled with ``--fmad=false``.  The initial gate deliberately avoids
fast math so differences from the single-precision Fortran ruler remain
diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

extern "C" __global__ void acoustic_coefficients(
    const int nlev, const int ncells,
    const float dts, const float epssm,
    const float gravity, const float rgas, const float cp,
    const float *zz, const float *cqw, const float *pressure,
    const float *theta, const float *rho_base,
    const float *rtheta_base, const float *pressure_base,
    const float *rtheta_p, const float *qtot,
    const float *rdzw, const float *fzm, const float *fzp,
    const float *rdzu,
    float *cofwr, float *cofwz, float *coftz, float *cofwt,
    float *cofrz, float *a_tri, float *b_tri, float *c_tri,
    float *alpha_tri, float *gamma_tri)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float dtseps = 0.5f * dts * (1.0f + epssm);
    const float rcv = rgas / (cp - rgas);
    const float csquared = cp * rcv;

    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        cofwr[i] = 0.0f;
        cofwz[i] = 0.0f;
        cofwt[i] = 0.5f * dtseps * rcv * zz[i] * gravity * rho_base[i]
            / (1.0f + qtot[i]) * pressure[i]
            / ((rtheta_base[i] + rtheta_p[i]) * pressure_base[i]);
        a_tri[i] = 0.0f;
        b_tri[i] = 1.0f;
        c_tri[i] = 0.0f;
        alpha_tri[i] = 0.0f;
        gamma_tri[i] = 0.0f;
        coftz[C2(k, cell, ncells)] = 0.0f;
    }
    coftz[C2(nlev, cell, ncells)] = 0.0f;

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float interp_zz = fzm[k] * zz[i] + fzp[k] * zz[im];
        cofwr[i] = 0.5f * dtseps * gravity * interp_zz;
        const float interp_p = fzm[k] * pressure[i] + fzp[k] * pressure[im];
        cofwz[i] = dtseps * csquared * interp_zz * rdzu[k]
            * cqw[i] * interp_p;
        coftz[i] = dtseps * (fzm[k] * theta[i] + fzp[k] * theta[im]);
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const int ip = C2(k + 1, cell, ncells);
        a_tri[i] = -cofwz[i] * coftz[im] * rdzw[k - 1] * zz[im]
            + cofwr[i] * cofrz[k - 1]
            - cofwt[im] * coftz[im] * rdzw[k - 1];
        b_tri[i] = 1.0f
            + cofwz[i] * coftz[i]
                * (rdzw[k] * zz[i] + rdzw[k - 1] * zz[im])
            - coftz[i] * (cofwt[i] * rdzw[k]
                - cofwt[im] * rdzw[k - 1])
            + cofwr[i] * (cofrz[k] - cofrz[k - 1]);
        c_tri[i] = -cofwz[i] * coftz[ip] * rdzw[k] * zz[i]
            - cofwr[i] * cofrz[k]
            + cofwt[i] * coftz[ip] * rdzw[k];
        const float denominator = b_tri[i] - a_tri[i] * gamma_tri[im];
        alpha_tri[i] = 1.0f / denominator;
        gamma_tri[i] = c_tri[i] * alpha_tri[i];
    }
}

extern "C" __global__ void acoustic_cofrz(
    const int nlev, const float dts, const float epssm,
    const float *rdzw, float *cofrz)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < nlev) cofrz[k] = mpas_mul(
        mpas_mul(mpas_mul(0.5f, dts), mpas_add(1.0f, epssm)), rdzw[k]);
}

extern "C" __global__ void tendency_w_to_omega(
    const int nlev, const int ncells, const int nedges,
    const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *edge_sign_on_cell,
    const float *u_tendency, const float *fzm, const float *fzp,
    const float *zz, const float *zb, const float *zb3,
    float *omega_tendency)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int slot = 0; slot < count; ++slot) {
        const int edge = edges_on_cell[CES(cell, slot, max_edges)];
        if (edge < 0 || edge >= nedges) continue;
        const float sign = edge_sign_on_cell[CES(cell, slot, max_edges)];
        for (int k = 1; k < nlev; ++k) {
            const float ut = u_tendency[E2(k, edge, nedges)];
            const float flux = mpas_mul(sign, mpas_add(
                mpas_mul(fzm[k], ut),
                mpas_mul(fzp[k],
                    u_tendency[E2(k - 1, edge, nedges)])));
            const int metric = ZBS(k, cell, slot, ncells, max_edges);
            omega_tendency[C2(k, cell, ncells)] = mpas_sub(
                omega_tendency[C2(k, cell, ncells)],
                mpas_mul(mpas_add(zb[metric], mpas_mul(
                    mpas_copysign(1.0f, ut), zb3[metric])), flux));
        }
    }
    for (int k = 1; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        omega_tendency[index] = mpas_mul(omega_tendency[index], mpas_add(
            mpas_mul(fzm[k], zz[index]),
            mpas_mul(fzp[k], zz[C2(k - 1, cell, ncells)])));
    }
}

extern "C" __global__ void acoustic_ru(
    const int nlev, const int nedges, const int ncells,
    const int small_step, const float dts,
    const float gravity, const float rgas, const float cp,
    const int *cells_on_edge, const float *dc_edge,
    const float *zz, const float *exner, const float *cqu,
    const float *zxu, const float *tend_ru,
    const float *rho_pp, const float *rtheta_pp,
    float *ru_p, float *ru_avg)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, tend_ru[index]);
            ru_p[index] = value;
            ru_avg[index] = value;
        } else {
            const float half_zz = mpas_mul(0.5f, mpas_add(
                zz[C2(k, c1, ncells)], zz[C2(k, c0, ncells)]));
            float pgrad = mpas_div(mpas_div(mpas_sub(
                rtheta_pp[C2(k, c1, ncells)],
                rtheta_pp[C2(k, c0, ncells)]), dc_edge[edge]), half_zz);
            const float csquared = cp * rgas / (cp - rgas);
            pgrad = mpas_mul(mpas_mul(mpas_mul(mpas_mul(
                cqu[index], 0.5f), csquared), mpas_add(
                    exner[C2(k, c0, ncells)],
                    exner[C2(k, c1, ncells)])), pgrad);
            pgrad = mpas_add(pgrad, mpas_mul(mpas_mul(mpas_mul(
                0.5f, gravity), zxu[index]), mpas_add(
                    rho_pp[C2(k, c0, ncells)],
                    rho_pp[C2(k, c1, ncells)])));
            ru_p[index] = mpas_add(ru_p[index], mpas_mul(
                dts, mpas_sub(tend_ru[index], pgrad)));
            ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
        }
    }
}

extern "C" __global__ void acoustic_prepare(
    const int nlev, const int ncells, const int small_step,
    float *rw_p, float *rtheta_pp, float *rtheta_pp_old,
    float *rho_pp, float *ww_avg)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (small_step == 1) {
            rtheta_pp_old[index] = 0.0f;
            rtheta_pp[index] = 0.0f;
            rho_pp[index] = 0.0f;
        } else {
            rtheta_pp_old[index] = rtheta_pp[index];
        }
    }
    if (small_step == 1) {
        for (int k = 0; k <= nlev; ++k) {
            const int index = C2(k, cell, ncells);
            rw_p[index] = 0.0f;
            ww_avg[index] = 0.0f;
        }
    }
}

extern "C" __global__ void acoustic_rs_ts(
    const int nlev, const int ncells, const int nedges,
    const int max_edges, const float dts, const float epssm,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const int *cells_on_edge, const float *edge_sign_on_cell,
    const float *dv_edge, const float *area_cell,
    const float *theta_m, const float *rdzw,
    const float *cofrz, const float *coftz,
    const float *ru_p, const float *rw_p,
    const float *rho_pp, const float *rtheta_pp,
    const float *tend_rho, const float *tend_rt,
    float *rs, float *ts)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        float r = 0.0f;
        float t = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            const float flux = mpas_div(mpas_mul(mpas_mul(mpas_mul(
                edge_sign_on_cell[CES(cell, slot, max_edges)], dts),
                dv_edge[edge]), ru_p[E2(k, edge, nedges)]), area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                theta_m[C2(k, c1, ncells)],
                theta_m[C2(k, c0, ncells)])));
        }
        const float resm = (1.0f - epssm) / (1.0f + epssm);
        r = mpas_add(r, mpas_sub(mpas_add(rho_pp[index],
            mpas_mul(dts, tend_rho[index])), mpas_mul(
                mpas_mul(cofrz[k], resm), mpas_sub(
                    rw_p[C2(k + 1, cell, ncells)],
                    rw_p[C2(k, cell, ncells)]))));
        t = mpas_add(t, mpas_sub(mpas_add(rtheta_pp[index],
            mpas_mul(dts, tend_rt[index])), mpas_mul(
                mpas_mul(resm, rdzw[k]), mpas_sub(
                    mpas_mul(coftz[C2(k + 1, cell, ncells)],
                        rw_p[C2(k + 1, cell, ncells)]),
                    mpas_mul(coftz[C2(k, cell, ncells)],
                        rw_p[C2(k, cell, ncells)])))));
        rs[index] = r;
        ts[index] = t;
    }
}

extern "C" __global__ void acoustic_column_solve(
    const int nlev, const int ncells,
    const float dts, const float epssm,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w,
    const float *rw, const float *rw_save,
    const float *tend_rw, const float *rs, const float *ts,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float resm = (1.0f - epssm) / (1.0f + epssm);
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(
            mpas_mul(0.5f, mpas_sub(1.0f, epssm)), rw_p[i]));
        rw_p[i] = mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(
            rw_p[i], mpas_mul(dts, tend_rw[i])), mpas_mul(cofwz[i],
                mpas_add(mpas_sub(mpas_mul(zz[i], ts[i]),
                    mpas_mul(zz[im], ts[im])), mpas_mul(resm,
                        mpas_sub(mpas_mul(zz[i], rtheta_pp[i]),
                            mpas_mul(zz[im], rtheta_pp[im])))))),
            mpas_mul(cofwr[i], mpas_add(mpas_add(rs[i], rs[im]),
                mpas_mul(resm, mpas_add(rho_pp[i], rho_pp[im]))))),
            mpas_mul(cofwt[i], mpas_add(ts[i],
                mpas_mul(resm, rtheta_pp[i])))),
            mpas_mul(cofwt[im], mpas_add(ts[im],
                mpas_mul(resm, rtheta_pp[im]))));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        rw_p[i] = mpas_mul(mpas_sub(rw_p[i],
            mpas_mul(a_tri[i], rw_p[im])), alpha_tri[i]);
    }
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = C2(k, cell, ncells);
        rw_p[i] = mpas_sub(rw_p[i], mpas_mul(
            gamma_tri[i], rw_p[C2(k + 1, cell, ncells)]));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rw_save[i], rw[i]);
        const float density_interface = mpas_mul(
            mpas_add(mpas_mul(fzm[k], zz[i]), mpas_mul(fzp[k], zz[im])),
            mpas_add(mpas_mul(fzm[k], rho_zz[i]),
                mpas_mul(fzp[k], rho_zz[im])));
        rw_p[i] = mpas_sub(mpas_div(mpas_sub(mpas_add(
            rw_p[i], delta_saved), mpas_mul(mpas_mul(mpas_mul(
                dts, dss[i]), density_interface), w[i])),
            mpas_add(1.0f, mpas_mul(dts, dss[i]))), delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(mpas_mul(
            0.5f, mpas_add(1.0f, epssm)), rw_p[i]));
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(cofrz[k], mpas_sub(
            rw_p[C2(k + 1, cell, ncells)], rw_p[i])));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(rdzw[k], mpas_sub(
            mpas_mul(coftz[C2(k + 1, cell, ncells)],
                rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(coftz[i], rw_p[i]))));
    }
}
"""

_CACHE: Any | None = None
_KERNELS: dict[tuple[Any, str], Any] = {}


def _cupy() -> Any:
    from .cuda_backend import require_cuda

    require_cuda(min_compute=(12, 0))
    import cupy as cp

    return cp


def _kernel(name: str, kernel_cache: Any | None = None) -> Any:
    global _CACHE
    from .cuda_backend import KernelCache, require_cuda

    selected = kernel_cache
    if selected is None:
        if _CACHE is None:
            capability = require_cuda(min_compute=(12, 0))
            _CACHE = KernelCache(capability=capability)
        selected = _CACHE
    # Keep the cache object in the key.  Numeric id reuse after a cache is
    # destroyed can otherwise return a kernel owned by an earlier cache and
    # silently omit this TU from the new compile manifest.
    key = (selected, name)
    if key not in _KERNELS:
        _KERNELS[key] = selected.raw_kernel(
            name,
            _CUDA_SOURCE,
            module_key="hexcore.cuda_acoustic",
        )
    return _KERNELS[key]


def _launch_1d(kernel: Any, count: int, arguments: tuple[Any, ...]) -> None:
    threads = 128
    kernel(((count + threads - 1) // threads,), (threads,), arguments)


def _mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    camel = {
        "n_edges_on_cell": "nEdgesOnCell",
        "edges_on_cell": "edgesOnCell",
        "cells_on_edge": "cellsOnEdge",
        "edge_sign_on_cell": "edgeSignOnCell",
        "dv_edge": "dvEdge",
        "dc_edge": "dcEdge",
        "area_cell": "areaCell",
    }.get(name)
    if camel is not None and hasattr(mesh, camel):
        return getattr(mesh, camel)
    arrays = getattr(mesh, "arrays", None)
    if arrays is not None:
        if name in arrays:
            return arrays[name]
        if camel is not None and camel in arrays:
            return arrays[camel]
    raise AttributeError(f"device mesh has no field {name!r}")


def _float32(value: Any, name: str) -> Any:
    from .cuda_backend import require_resident_array

    return require_resident_array(name, value, dtype=np.float32)


def _int32(value: Any, name: str) -> Any:
    from .cuda_backend import require_resident_array

    return require_resident_array(name, value, dtype=np.int32)


@dataclass(frozen=True, slots=True)
class CudaVerticalImplicitCoefficients:
    cofwr: Any
    cofwz: Any
    coftz: Any
    cofwt: Any
    cofrz: Any
    a_tri: Any
    b_tri: Any
    c_tri: Any
    alpha_tri: Any
    gamma_tri: Any


@dataclass(slots=True)
class CudaAcousticState:
    ru_p: Any
    rw_p: Any
    rtheta_pp: Any
    rtheta_pp_old: Any
    rho_pp: Any
    ru_avg: Any
    ww_avg: Any

    @classmethod
    def zeros(cls, nlev: int, ncells: int, nedges: int) -> "CudaAcousticState":
        cp = _cupy()
        return cls(
            ru_p=cp.zeros((nlev, nedges), dtype=cp.float32),
            rw_p=cp.zeros((nlev + 1, ncells), dtype=cp.float32),
            rtheta_pp=cp.zeros((nlev, ncells), dtype=cp.float32),
            rtheta_pp_old=cp.zeros((nlev, ncells), dtype=cp.float32),
            rho_pp=cp.zeros((nlev, ncells), dtype=cp.float32),
            ru_avg=cp.zeros((nlev, nedges), dtype=cp.float32),
            ww_avg=cp.zeros((nlev + 1, ncells), dtype=cp.float32),
        )

    def copy(self) -> "CudaAcousticState":
        cp = _cupy()
        return CudaAcousticState(
            **{
                name: cp.array(getattr(self, name), copy=True)
                for name in self.__slots__
            }
        )


@dataclass(frozen=True, slots=True)
class CudaAcousticForcing:
    rho_zz: Any
    theta_m: Any
    zz: Any
    exner: Any
    cqu: Any
    zxu: Any
    dss: Any
    tend_ru: Any
    tend_rho: Any
    tend_rt: Any
    tend_rw: Any
    w: Any
    rw: Any
    rw_save: Any


def compute_vertical_implicit_coefficients_cuda(
    *,
    dts: float,
    epssm: float,
    zz: Any,
    cqw: Any,
    pressure: Any,
    theta: Any,
    rho_base: Any,
    rho_theta_base: Any,
    pressure_base: Any,
    rho_theta_perturbation: Any,
    qtot: Any,
    rdzw: Any,
    fzm: Any,
    fzp: Any,
    rdzu: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    kernel_cache: Any | None = None,
) -> CudaVerticalImplicitCoefficients:
    """Build the vertically implicit coefficients entirely on the device."""

    cp_module = _cupy()
    zz = _float32(zz, "zz")
    if zz.ndim != 2:
        raise ValueError("zz must have shape (nVertLevels,nCells)")
    nlev, ncells = map(int, zz.shape)
    cell_shape = (nlev, ncells)
    inputs: dict[str, Any] = {}
    for name, value in {
        "cqw": cqw,
        "pressure": pressure,
        "theta": theta,
        "rho_base": rho_base,
        "rho_theta_base": rho_theta_base,
        "pressure_base": pressure_base,
        "rho_theta_perturbation": rho_theta_perturbation,
        "qtot": qtot,
    }.items():
        array = _float32(value, name)
        if array.shape != cell_shape:
            raise ValueError(f"{name} shape {array.shape} != {cell_shape}")
        inputs[name] = array
    vertical: dict[str, Any] = {}
    for name, value in {"rdzw": rdzw, "fzm": fzm, "fzp": fzp, "rdzu": rdzu}.items():
        array = _float32(value, name)
        if array.shape != (nlev,):
            raise ValueError(f"{name} shape {array.shape} != ({nlev},)")
        vertical[name] = array

    def zero() -> Any:
        return cp_module.empty(cell_shape, dtype=cp_module.float32)

    coftz = cp_module.empty((nlev + 1, ncells), dtype=cp_module.float32)
    cofrz = cp_module.empty((nlev,), dtype=cp_module.float32)
    result = CudaVerticalImplicitCoefficients(
        cofwr=zero(),
        cofwz=zero(),
        coftz=coftz,
        cofwt=zero(),
        cofrz=cofrz,
        a_tri=zero(),
        b_tri=zero(),
        c_tri=zero(),
        alpha_tri=zero(),
        gamma_tri=zero(),
    )
    _launch_1d(
        _kernel("acoustic_cofrz", kernel_cache),
        nlev,
        (np.int32(nlev), np.float32(dts), np.float32(epssm), vertical["rdzw"], cofrz),
    )
    _launch_1d(
        _kernel("acoustic_coefficients", kernel_cache),
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.float32(dts),
            np.float32(epssm),
            np.float32(gravity),
            np.float32(rgas),
            np.float32(cp),
            zz,
            inputs["cqw"],
            inputs["pressure"],
            inputs["theta"],
            inputs["rho_base"],
            inputs["rho_theta_base"],
            inputs["pressure_base"],
            inputs["rho_theta_perturbation"],
            inputs["qtot"],
            vertical["rdzw"],
            vertical["fzm"],
            vertical["fzp"],
            vertical["rdzu"],
            result.cofwr,
            result.cofwz,
            result.coftz,
            result.cofwt,
            result.cofrz,
            result.a_tri,
            result.b_tri,
            result.c_tri,
            result.alpha_tri,
            result.gamma_tri,
        ),
    )
    return result


def convert_w_tendency_to_omega_cuda(
    mesh: Any,
    w_tendency: Any,
    u_tendency: Any,
    *,
    fzm: Any,
    fzp: Any,
    zz: Any,
    zb_cell: Any,
    zb3_cell: Any,
    kernel_cache: Any | None = None,
) -> Any:
    """Apply the terrain w-tendency coupling on-device."""

    cp = _cupy()
    out = cp.array(_float32(w_tendency, "w_tendency"), copy=True)
    u = _float32(u_tendency, "u_tendency")
    nlev, nedges = map(int, u.shape)
    if out.ndim != 2 or out.shape[0] != nlev + 1:
        raise ValueError("w_tendency must have shape (nVertLevels+1,nCells)")
    ncells = int(out.shape[1])
    edges = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
    counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
    signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    max_edges = int(edges.shape[1])
    expected_metric = (nlev + 1, ncells, max_edges)
    zb = _float32(zb_cell, "zb_cell")
    zb3 = _float32(zb3_cell, "zb3_cell")
    if zb.shape != expected_metric or zb3.shape != expected_metric:
        raise ValueError(f"terrain metrics must have shape {expected_metric}")
    _launch_1d(
        _kernel("tendency_w_to_omega", kernel_cache),
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            counts,
            edges,
            signs,
            u,
            _float32(fzm, "fzm"),
            _float32(fzp, "fzp"),
            _float32(zz, "zz"),
            zb,
            zb3,
            out,
        ),
    )
    return out


def advance_acoustic_step_cuda(
    mesh: Any,
    state: CudaAcousticState,
    forcing: CudaAcousticForcing,
    coefficients: CudaVerticalImplicitCoefficients,
    *,
    dts: float,
    small_step: int,
    epssm: float,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    in_place: bool = True,
    kernel_cache: Any | None = None,
) -> CudaAcousticState:
    """Advance one closed/global acoustic substep without host transfers."""

    if small_step < 1:
        raise ValueError("small_step must be at least one")
    out = state if in_place else state.copy()
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    if out.rho_pp.shape != (nlev, ncells) or out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError("acoustic state shapes are inconsistent")
    cells_on_edge = _int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")
    edges_on_cell = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
    counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
    signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    dv_edge = _float32(_mesh_value(mesh, "dv_edge"), "dv_edge")
    dc_edge = _float32(_mesh_value(mesh, "dc_edge"), "dc_edge")
    area_cell = _float32(_mesh_value(mesh, "area_cell"), "area_cell")
    max_edges = int(edges_on_cell.shape[1])
    fzm = _float32(fzm, "fzm")
    fzp = _float32(fzp, "fzp")
    rdzw = _float32(rdzw, "rdzw")

    _launch_1d(
        _kernel("acoustic_ru", kernel_cache),
        nedges,
        (
            np.int32(nlev),
            np.int32(nedges),
            np.int32(ncells),
            np.int32(small_step),
            np.float32(dts),
            np.float32(gravity),
            np.float32(rgas),
            np.float32(cp),
            cells_on_edge,
            dc_edge,
            _float32(forcing.zz, "forcing.zz"),
            _float32(forcing.exner, "forcing.exner"),
            _float32(forcing.cqu, "forcing.cqu"),
            _float32(forcing.zxu, "forcing.zxu"),
            _float32(forcing.tend_ru, "forcing.tend_ru"),
            out.rho_pp,
            out.rtheta_pp,
            out.ru_p,
            out.ru_avg,
        ),
    )
    _launch_1d(
        _kernel("acoustic_prepare", kernel_cache),
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(small_step),
            out.rw_p,
            out.rtheta_pp,
            out.rtheta_pp_old,
            out.rho_pp,
            out.ww_avg,
        ),
    )
    cp = _cupy()
    rs = cp.empty((nlev, ncells), dtype=cp.float32)
    ts = cp.empty_like(rs)
    _launch_1d(
        _kernel("acoustic_rs_ts", kernel_cache),
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            np.float32(dts),
            np.float32(epssm),
            counts,
            edges_on_cell,
            cells_on_edge,
            signs,
            dv_edge,
            area_cell,
            _float32(forcing.theta_m, "forcing.theta_m"),
            rdzw,
            coefficients.cofrz,
            coefficients.coftz,
            out.ru_p,
            out.rw_p,
            out.rho_pp,
            out.rtheta_pp,
            _float32(forcing.tend_rho, "forcing.tend_rho"),
            _float32(forcing.tend_rt, "forcing.tend_rt"),
            rs,
            ts,
        ),
    )
    _launch_1d(
        _kernel("acoustic_column_solve", kernel_cache),
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.float32(dts),
            np.float32(epssm),
            _float32(forcing.zz, "forcing.zz"),
            _float32(forcing.rho_zz, "forcing.rho_zz"),
            fzm,
            fzp,
            rdzw,
            _float32(forcing.dss, "forcing.dss"),
            _float32(forcing.w, "forcing.w"),
            _float32(forcing.rw, "forcing.rw"),
            _float32(forcing.rw_save, "forcing.rw_save"),
            _float32(forcing.tend_rw, "forcing.tend_rw"),
            rs,
            ts,
            coefficients.cofwr,
            coefficients.cofwz,
            coefficients.coftz,
            coefficients.cofwt,
            coefficients.cofrz,
            coefficients.a_tri,
            coefficients.alpha_tri,
            coefficients.gamma_tri,
            out.rw_p,
            out.rho_pp,
            out.rtheta_pp,
            out.ww_avg,
        ),
    )
    return out


__all__ = [
    "CudaAcousticForcing",
    "CudaAcousticState",
    "CudaVerticalImplicitCoefficients",
    "advance_acoustic_step_cuda",
    "compute_vertical_implicit_coefficients_cuda",
    "convert_w_tendency_to_omega_cuda",
]
