"""Additive CUDA mirror of the MPAS-A v8.4.1 acoustic equations.

Kernel names and the NVRTC module key are release-specific so the frozen
v8.2.3 translation unit remains numerically selectable and independently
fingerprinted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cuda_acoustic import (
    CudaAcousticForcing,
    CudaAcousticState,
    CudaVerticalImplicitCoefficients,
    _float32,
    _int32,
    _launch_1d,
    _mesh_value,
)
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_backend.containers import require_resident_array
from .cuda_v841 import CudaV841Context


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

extern "C" __global__ void acoustic_cofrz_v841(
    const int nlev, const float *rdzw, float *cofrz)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < nlev) cofrz[k] = rdzw[k];
}

extern "C" __global__ void acoustic_coefficients_v841(
    const int nlev, const int ncells, const float dts,
    const float gravity, const float rgas, const float cp,
    const float *zz, const float *cqw, const float *exner,
    const float *theta, const float *rho_base,
    const float *rtheta_base, const float *exner_base,
    const float *rtheta_p, const float *qtot,
    const float *rdzw, const float *fzm, const float *fzp,
    const float *rdzu, const float *etp, const float *ewp,
    float *cofwr, float *cofwz, float *coftz, float *cofwt,
    const float *cofrz, float *a_tri, float *b_tri, float *c_tri,
    float *alpha_tri, float *gamma_tri, int *singular)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float one = 1.0f;
    const float half = 0.5f;
    const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
    const float csquared = mpas_mul(cp, rcv);
    const float dts2 = mpas_mul(dts, dts);

    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        cofwr[i] = 0.0f;
        cofwz[i] = 0.0f;
        coftz[i] = 0.0f;
        float value = mpas_mul(half, rcv);
        value = mpas_mul(value, zz[i]);
        value = mpas_mul(value, gravity);
        value = mpas_mul(value, rho_base[i]);
        value = mpas_div(value, mpas_add(one, qtot[i]));
        value = mpas_mul(value, exner[i]);
        value = mpas_div(value, mpas_mul(
            mpas_add(rtheta_base[i], rtheta_p[i]), exner_base[i]));
        cofwt[i] = value;
        a_tri[i] = 0.0f;
        b_tri[i] = 0.0f;
        c_tri[i] = 0.0f;
        alpha_tri[i] = 0.0f;
        gamma_tri[i] = 0.0f;
    }
    coftz[C2(nlev, cell, ncells)] = 0.0f;
    if (nlev > 0) b_tri[C2(0, cell, ncells)] = one;

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float interp_zz = mpas_add(mpas_mul(fzm[k], zz[i]),
            mpas_mul(fzp[k], zz[im]));
        cofwr[i] = mpas_mul(mpas_mul(half, gravity), interp_zz);
        float zcoef = mpas_mul(csquared, interp_zz);
        zcoef = mpas_mul(zcoef, rdzu[k]);
        zcoef = mpas_mul(zcoef, cqw[i]);
        zcoef = mpas_mul(zcoef, mpas_add(mpas_mul(fzm[k], exner[i]),
            mpas_mul(fzp[k], exner[im])));
        cofwz[i] = zcoef;
        coftz[i] = mpas_add(mpas_mul(fzm[k], theta[i]),
            mpas_mul(fzp[k], theta[im]));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const int ip = C2(k + 1, cell, ncells);

        float avalue = mpas_sub(0.0f, mpas_mul(mpas_mul(mpas_mul(
            cofwz[i], coftz[im]), rdzw[k - 1]), zz[im]));
        avalue = mpas_add(avalue, mpas_mul(cofwr[i], cofrz[k - 1]));
        avalue = mpas_sub(avalue, mpas_mul(mpas_mul(
            cofwt[im], coftz[im]), rdzw[k - 1]));
        avalue = mpas_mul(avalue, etp[k - 1]);
        a_tri[i] = mpas_mul(avalue, ewp[k - 1]);

        const float upper_metric = mpas_mul(mpas_mul(
            etp[k], rdzw[k]), zz[i]);
        const float lower_metric = mpas_mul(mpas_mul(
            etp[k - 1], rdzw[k - 1]), zz[im]);
        float bvalue = mpas_mul(mpas_mul(cofwz[i], coftz[i]),
            mpas_add(upper_metric, lower_metric));
        const float upper_heat = mpas_mul(mpas_mul(
            etp[k], cofwt[i]), rdzw[k]);
        const float lower_heat = mpas_mul(mpas_mul(
            etp[k - 1], cofwt[im]), rdzw[k - 1]);
        bvalue = mpas_sub(bvalue, mpas_mul(coftz[i],
            mpas_sub(upper_heat, lower_heat)));
        const float upper_mass = mpas_mul(etp[k], cofrz[k]);
        const float lower_mass = mpas_mul(etp[k - 1], cofrz[k - 1]);
        bvalue = mpas_add(bvalue, mpas_mul(cofwr[i],
            mpas_sub(upper_mass, lower_mass)));
        b_tri[i] = mpas_mul(bvalue, ewp[k]);

        float cvalue = mpas_sub(0.0f, mpas_mul(mpas_mul(mpas_mul(
            cofwz[i], coftz[ip]), rdzw[k]), zz[i]));
        cvalue = mpas_sub(cvalue, mpas_mul(cofwr[i], cofrz[k]));
        cvalue = mpas_add(cvalue, mpas_mul(mpas_mul(
            cofwt[i], coftz[ip]), rdzw[k]));
        cvalue = mpas_mul(cvalue, etp[k]);
        c_tri[i] = mpas_mul(cvalue, ewp[k + 1]);
    }
    if (nlev > 0) c_tri[C2(nlev - 1, cell, ncells)] = 0.0f;

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float denominator = mpas_add(one, mpas_mul(dts2,
            mpas_sub(b_tri[i], mpas_mul(a_tri[i], gamma_tri[im]))));
        if (!(isfinite(denominator) && denominator != 0.0f))
            atomicExch(singular, 1);
        alpha_tri[i] = mpas_div(one, denominator);
        gamma_tri[i] = mpas_mul(mpas_mul(dts2, c_tri[i]), alpha_tri[i]);
    }
}

extern "C" __global__ void acoustic_ru_v841(
    const int nlev, const int nedges, const int ncells,
    const int small_step, const float dts,
    const float gravity, const float rgas, const float cp,
    const int *cells_on_edge, const float *inv_dc_edge,
    const float *zz, const float *exner, const float *cqu,
    const float *zxu, const float *tend_ru,
    const float *rho_pp, const float *rtheta_pp,
    float *ru_p, float *ru_avg)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    const float half = 0.5f;
    const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
    const float csquared = mpas_mul(cp, rcv);
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, tend_ru[index]);
            ru_p[index] = value;
            ru_avg[index] = value;
        } else {
            const float normal = mpas_mul(mpas_sub(
                rtheta_pp[C2(k, c1, ncells)],
                rtheta_pp[C2(k, c0, ncells)]), inv_dc_edge[edge]);
            const float normalized = mpas_div(normal, mpas_mul(half, mpas_add(
                zz[C2(k, c1, ncells)], zz[C2(k, c0, ncells)])));
            float pgrad = mpas_mul(cqu[index], half);
            pgrad = mpas_mul(pgrad, csquared);
            pgrad = mpas_mul(pgrad, mpas_add(exner[C2(k, c0, ncells)],
                exner[C2(k, c1, ncells)]));
            pgrad = mpas_mul(pgrad, normalized);
            const float terrain = mpas_mul(mpas_mul(mpas_mul(
                half, zxu[index]), gravity), mpas_add(
                    rho_pp[C2(k, c0, ncells)],
                    rho_pp[C2(k, c1, ncells)]));
            pgrad = mpas_add(pgrad, terrain);
            ru_p[index] = mpas_add(ru_p[index], mpas_mul(
                dts, mpas_sub(tend_ru[index], pgrad)));
            ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
        }
    }
}

extern "C" __global__ void acoustic_prepare_v841(
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

extern "C" __global__ void acoustic_rs_ts_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const float dts, const int *n_edges_on_cell, const int *edges_on_cell,
    const int *cells_on_edge, const float *edge_sign_on_cell,
    const float *dv_edge, const float *inv_area_cell,
    const float *theta_m, const float *rdzw,
    const float *cofrz, const float *coftz, const float *ewm,
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
            float flux = mpas_mul(
                edge_sign_on_cell[CES(cell, slot, max_edges)], dts);
            flux = mpas_mul(flux, dv_edge[edge]);
            flux = mpas_mul(flux, ru_p[E2(k, edge, nedges)]);
            flux = mpas_mul(flux, inv_area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                theta_m[C2(k, c1, ncells)],
                theta_m[C2(k, c0, ncells)])));
        }
        r = mpas_add(mpas_add(rho_pp[index], mpas_mul(dts, tend_rho[index])),
            r);
        r = mpas_sub(r, mpas_mul(mpas_mul(dts, cofrz[k]), mpas_sub(
            mpas_mul(ewm[k + 1], rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(ewm[k], rw_p[index]))));
        t = mpas_add(mpas_add(rtheta_pp[index], mpas_mul(dts, tend_rt[index])),
            t);
        t = mpas_sub(t, mpas_mul(mpas_mul(dts, rdzw[k]), mpas_sub(
            mpas_mul(mpas_mul(ewm[k + 1], coftz[C2(k + 1, cell, ncells)]),
                rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(mpas_mul(ewm[k], coftz[index]), rw_p[index]))));
        rs[index] = r;
        ts[index] = t;
    }
}

extern "C" __global__ void acoustic_column_solve_v841(
    const int nlev, const int ncells, const float dts,
    const float *zz, const float *rho_zz,
    const float *fzm, const float *fzp, const float *rdzw,
    const float *dss, const float *w, const float *rw, const float *rw_save,
    const float *tend_rw, const float *rs, const float *ts,
    const float *cofwr, const float *cofwz, const float *coftz,
    const float *cofwt, const float *cofrz,
    const float *a_tri, const float *alpha_tri, const float *gamma_tri,
    const float *etp, const float *etm, const float *ewp, const float *ewm,
    float *rw_p, float *rho_pp, float *rtheta_pp, float *ww_avg)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float dts2 = mpas_mul(dts, dts);
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewm[k], rw_p[i]));

        const float theta_implicit = mpas_sub(
            mpas_mul(mpas_mul(etp[k], zz[i]), ts[i]),
            mpas_mul(mpas_mul(etp[k - 1], zz[im]), ts[im]));
        const float theta_explicit = mpas_sub(
            mpas_mul(mpas_mul(etm[k], zz[i]), rtheta_pp[i]),
            mpas_mul(mpas_mul(etm[k - 1], zz[im]), rtheta_pp[im]));
        const float density_implicit = mpas_add(
            mpas_mul(etp[k], rs[i]), mpas_mul(etp[k - 1], rs[im]));
        const float density_explicit = mpas_add(
            mpas_mul(etm[k], rho_pp[i]),
            mpas_mul(etm[k - 1], rho_pp[im]));
        const float heat_upper = mpas_add(
            mpas_mul(etp[k], ts[i]), mpas_mul(etm[k], rtheta_pp[i]));
        const float heat_lower = mpas_add(
            mpas_mul(etp[k - 1], ts[im]),
            mpas_mul(etm[k - 1], rtheta_pp[im]));

        float inner = mpas_sub(tend_rw[i], mpas_mul(cofwz[i],
            mpas_add(theta_implicit, theta_explicit)));
        inner = mpas_sub(inner, mpas_mul(cofwr[i],
            mpas_add(density_implicit, density_explicit)));
        inner = mpas_add(inner, mpas_mul(cofwt[i], heat_upper));
        inner = mpas_add(inner, mpas_mul(cofwt[im], heat_lower));
        rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, inner));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        rw_p[i] = mpas_mul(mpas_sub(rw_p[i], mpas_mul(mpas_mul(
            dts2, a_tri[i]), rw_p[im])), alpha_tri[i]);
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
        float value = mpas_add(rw_p[i], delta_saved);
        value = mpas_sub(value, mpas_mul(mpas_mul(mpas_mul(
            dts, dss[i]), density_interface), w[i]));
        value = mpas_div(value, mpas_add(1.0f, mpas_mul(dts, dss[i])));
        rw_p[i] = mpas_sub(value, delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(mpas_mul(dts, cofrz[k]),
            mpas_sub(mpas_mul(ewp[k + 1], rw_p[C2(k + 1, cell, ncells)]),
                mpas_mul(ewp[k], rw_p[i]))));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(mpas_mul(dts, rdzw[k]),
            mpas_sub(mpas_mul(mpas_mul(ewp[k + 1],
                    coftz[C2(k + 1, cell, ncells)]),
                rw_p[C2(k + 1, cell, ncells)]),
                mpas_mul(mpas_mul(ewp[k], coftz[i]), rw_p[i]))));
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
            _CACHE = KernelCache(capability=require_cuda(min_compute=(12, 0)))
        selected = _CACHE
    key = (selected, name)
    if key not in _KERNELS:
        _KERNELS[key] = selected.raw_kernel(
            name,
            _CUDA_SOURCE,
            module_key="mpas_port.cuda_acoustic_v841",
        )
    return _KERNELS[key]


def compute_vertical_implicit_coefficients_cuda_v841(
    *,
    dts: float,
    context: CudaV841Context,
    zz: Any,
    cqw: Any,
    exner: Any,
    theta: Any,
    rho_base: Any,
    rho_theta_base: Any,
    exner_base: Any,
    rho_theta_perturbation: Any,
    qtot: Any,
    rdzw: Any,
    fzm: Any,
    fzp: Any,
    rdzu: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    validation_flag: Any | None = None,
    kernel_cache: Any | None = None,
) -> CudaVerticalImplicitCoefficients:
    """Build released unscaled v8.4.1 coefficients on the device."""

    if not np.isfinite(dts) or dts <= 0.0:
        raise ValueError("dts must be finite and positive")
    if (
        not all(np.isfinite(value) for value in (gravity, rgas, cp))
        or gravity <= 0.0
        or rgas <= 0.0
        or cp <= rgas
    ):
        raise ValueError("gravity, rgas, and cp must define a finite dry atmosphere")
    cp_module = _cupy()
    zz = _float32(zz, "zz")
    if zz.ndim != 2:
        raise ValueError("zz must have shape (nVertLevels,nCells)")
    nlev, ncells = map(int, zz.shape)
    cell_shape = (nlev, ncells)
    for name, shape in {
        "etp": (nlev,),
        "etm": (nlev,),
        "ewp": (nlev + 1,),
        "ewm": (nlev + 1,),
    }.items():
        value = _float32(getattr(context, name), f"context.{name}")
        if value.shape != shape:
            raise ValueError(f"context.{name} shape {value.shape} != {shape}")
    inputs: dict[str, Any] = {}
    for name, value in {
        "cqw": cqw,
        "exner": exner,
        "theta": theta,
        "rho_base": rho_base,
        "rho_theta_base": rho_theta_base,
        "exner_base": exner_base,
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

    def cell_array() -> Any:
        return cp_module.empty(cell_shape, dtype=cp_module.float32)

    result = CudaVerticalImplicitCoefficients(
        cofwr=cell_array(),
        cofwz=cell_array(),
        coftz=cp_module.empty((nlev + 1, ncells), dtype=cp_module.float32),
        cofwt=cell_array(),
        cofrz=cp_module.empty((nlev,), dtype=cp_module.float32),
        a_tri=cell_array(),
        b_tri=cell_array(),
        c_tri=cell_array(),
        alpha_tri=cell_array(),
        gamma_tri=cell_array(),
    )
    owns_flag = validation_flag is None
    singular = (
        cp_module.zeros((1,), dtype=cp_module.int32)
        if owns_flag
        else require_resident_array(
            "validation_flag", validation_flag, dtype=np.int32, shape=(1,)
        )
    )
    _launch_1d(
        _kernel("acoustic_cofrz_v841", kernel_cache),
        nlev,
        (np.int32(nlev), vertical["rdzw"], result.cofrz),
    )
    _launch_1d(
        _kernel("acoustic_coefficients_v841", kernel_cache),
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.float32(dts),
            np.float32(gravity), np.float32(rgas), np.float32(cp),
            zz, inputs["cqw"], inputs["exner"], inputs["theta"],
            inputs["rho_base"], inputs["rho_theta_base"], inputs["exner_base"],
            inputs["rho_theta_perturbation"], inputs["qtot"],
            vertical["rdzw"], vertical["fzm"], vertical["fzp"], vertical["rdzu"],
            context.etp, context.ewp,
            result.cofwr, result.cofwz, result.coftz, result.cofwt,
            result.cofrz, result.a_tri, result.b_tri, result.c_tri,
            result.alpha_tri, result.gamma_tri, singular,
        ),
    )
    if owns_flag and int(cp_module.asnumpy(singular)[0]) != 0:
        raise FloatingPointError("singular v8.4.1 vertical acoustic system")
    return result


def advance_acoustic_step_cuda_v841(
    mesh: Any,
    state: CudaAcousticState,
    forcing: CudaAcousticForcing,
    coefficients: CudaVerticalImplicitCoefficients,
    *,
    context: CudaV841Context,
    dts: float,
    small_step: int,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    in_place: bool = True,
    kernel_cache: Any | None = None,
) -> CudaAcousticState:
    """Advance one closed/global released v8.4.1 acoustic substep."""

    if small_step < 1:
        raise ValueError("small_step must be at least one")
    if not np.isfinite(dts) or dts <= 0.0:
        raise ValueError("dts must be finite and positive")
    if (
        not all(np.isfinite(value) for value in (gravity, rgas, cp))
        or gravity <= 0.0
        or rgas <= 0.0
        or cp <= rgas
    ):
        raise ValueError("gravity, rgas, and cp must define a finite dry atmosphere")
    out = state if in_place else state.copy()
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    if out.rho_pp.shape != (nlev, ncells) or out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError("acoustic state shapes are inconsistent")
    mesh.validate()
    if int(mesh.n_cells) != ncells or int(mesh.n_edges) != nedges:
        raise ValueError("acoustic state dimensions differ from device mesh")
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(getattr(mesh, "n_vertices")),
    )
    state_shapes = {
        "ru_p": (nlev, nedges),
        "ru_avg": (nlev, nedges),
        "rw_p": (nlev + 1, ncells),
        "ww_avg": (nlev + 1, ncells),
        "rho_pp": (nlev, ncells),
        "rtheta_pp": (nlev, ncells),
        "rtheta_pp_old": (nlev, ncells),
    }
    for name, shape in state_shapes.items():
        value = _float32(getattr(out, name), f"state.{name}")
        if value.shape != shape:
            raise ValueError(f"state.{name} shape {value.shape} != {shape}")
    forcing_shapes = {
        "rho_zz": (nlev, ncells),
        "theta_m": (nlev, ncells),
        "zz": (nlev, ncells),
        "exner": (nlev, ncells),
        "cqu": (nlev, nedges),
        "zxu": (nlev, nedges),
        "dss": (nlev, ncells),
        "tend_ru": (nlev, nedges),
        "tend_rho": (nlev, ncells),
        "tend_rt": (nlev, ncells),
        "tend_rw": (nlev + 1, ncells),
        "w": (nlev + 1, ncells),
        "rw": (nlev + 1, ncells),
        "rw_save": (nlev + 1, ncells),
    }
    for name, shape in forcing_shapes.items():
        value = _float32(getattr(forcing, name), f"forcing.{name}")
        if value.shape != shape:
            raise ValueError(f"forcing.{name} shape {value.shape} != {shape}")
    coefficient_shapes = {
        "cofwr": (nlev, ncells),
        "cofwz": (nlev, ncells),
        "coftz": (nlev + 1, ncells),
        "cofwt": (nlev, ncells),
        "cofrz": (nlev,),
        "a_tri": (nlev, ncells),
        "b_tri": (nlev, ncells),
        "c_tri": (nlev, ncells),
        "alpha_tri": (nlev, ncells),
        "gamma_tri": (nlev, ncells),
    }
    for name, shape in coefficient_shapes.items():
        value = _float32(getattr(coefficients, name), f"coefficients.{name}")
        if value.shape != shape:
            raise ValueError(f"coefficients.{name} shape {value.shape} != {shape}")
    cells_on_edge = _int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")
    edges_on_cell = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
    counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
    signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    dv_edge = _float32(_mesh_value(mesh, "dv_edge"), "dv_edge")
    max_edges = int(edges_on_cell.shape[1])
    fzm = _float32(fzm, "fzm")
    fzp = _float32(fzp, "fzp")
    rdzw = _float32(rdzw, "rdzw")
    for name, value in (("fzm", fzm), ("fzp", fzp), ("rdzw", rdzw)):
        if value.shape != (nlev,):
            raise ValueError(f"{name} shape {value.shape} != ({nlev},)")

    _launch_1d(
        _kernel("acoustic_ru_v841", kernel_cache),
        nedges,
        (
            np.int32(nlev), np.int32(nedges), np.int32(ncells),
            np.int32(small_step), np.float32(dts), np.float32(gravity),
            np.float32(rgas), np.float32(cp), cells_on_edge, context.inv_dc_edge,
            _float32(forcing.zz, "forcing.zz"),
            _float32(forcing.exner, "forcing.exner"),
            _float32(forcing.cqu, "forcing.cqu"),
            _float32(forcing.zxu, "forcing.zxu"),
            _float32(forcing.tend_ru, "forcing.tend_ru"),
            out.rho_pp, out.rtheta_pp, out.ru_p, out.ru_avg,
        ),
    )
    _launch_1d(
        _kernel("acoustic_prepare_v841", kernel_cache),
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.int32(small_step),
            out.rw_p, out.rtheta_pp, out.rtheta_pp_old, out.rho_pp, out.ww_avg,
        ),
    )
    cp_module = _cupy()
    rs = cp_module.empty((nlev, ncells), dtype=cp_module.float32)
    ts = cp_module.empty_like(rs)
    _launch_1d(
        _kernel("acoustic_rs_ts_v841", kernel_cache),
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            np.int32(max_edges), np.float32(dts), counts, edges_on_cell,
            cells_on_edge, signs, dv_edge, context.inv_area_cell,
            _float32(forcing.theta_m, "forcing.theta_m"), rdzw,
            coefficients.cofrz, coefficients.coftz, context.ewm,
            out.ru_p, out.rw_p, out.rho_pp, out.rtheta_pp,
            _float32(forcing.tend_rho, "forcing.tend_rho"),
            _float32(forcing.tend_rt, "forcing.tend_rt"), rs, ts,
        ),
    )
    _launch_1d(
        _kernel("acoustic_column_solve_v841", kernel_cache),
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.float32(dts),
            _float32(forcing.zz, "forcing.zz"),
            _float32(forcing.rho_zz, "forcing.rho_zz"), fzm, fzp, rdzw,
            _float32(forcing.dss, "forcing.dss"),
            _float32(forcing.w, "forcing.w"),
            _float32(forcing.rw, "forcing.rw"),
            _float32(forcing.rw_save, "forcing.rw_save"),
            _float32(forcing.tend_rw, "forcing.tend_rw"), rs, ts,
            coefficients.cofwr, coefficients.cofwz, coefficients.coftz,
            coefficients.cofwt, coefficients.cofrz, coefficients.a_tri,
            coefficients.alpha_tri, coefficients.gamma_tri,
            context.etp, context.etm, context.ewp, context.ewm,
            out.rw_p, out.rho_pp, out.rtheta_pp, out.ww_avg,
        ),
    )
    return out


__all__ = [
    "advance_acoustic_step_cuda_v841",
    "compute_vertical_implicit_coefficients_cuda_v841",
]
