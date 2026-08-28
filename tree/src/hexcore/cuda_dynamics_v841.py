"""Additive v8.4.1 CUDA dynamics kernels and split-flux reductions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cuda_backend.containers import require_resident_array
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_v841 import CudaV841Context


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

extern "C" __global__ void vector_momentum_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *inv_dc_edge, const float *angle_edge, const float *f_edge,
    const float *u_init, const float *v_init, float *result)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        float q = 0.0f;
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            const float work_pv = mpas_mul(0.5f, mpas_add(
                pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
            q = mpas_add(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], u[E2(k, neighbor, nedges)]), work_pv));
        }
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            const float reference_u = mpas_sub(
                mpas_mul(u_init[k], cosf(angle_edge[neighbor])),
                mpas_mul(v_init[k], sinf(angle_edge[neighbor])));
            q = mpas_sub(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], reference_u), f_edge[edge]));
        }
        const float gradient = mpas_mul(mpas_sub(
            kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]),
            inv_dc_edge[edge]);
        result[index] = mpas_sub(mpas_mul(rho_edge[index],
            mpas_sub(q, gradient)), mpas_mul(mpas_mul(u[index], 0.5f),
                mpas_add(mass_divergence[C2(k, c0, ncells)],
                    mass_divergence[C2(k, c1, ncells)])));
    }
}

extern "C" __global__ void theta_finish_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux, float *result)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            value = mpas_sub(value, mpas_mul(
                acoustic_sign[CES(cell, slot, max_edges)],
                edge_flux[E2(k, edge, nedges)]));
        }
        value = mpas_mul(value, inv_area_cell[cell]);
        value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
        result[index] = value;
    }
}

extern "C" __global__ void w_finish_v841_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzu,
    const float *edge_flux, const float *vertical_flux, float *result)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k <= nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            float value = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                value = mpas_sub(value, mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)],
                    edge_flux[E2(k, edge, nedges)]));
            }
            value = mpas_mul(value, inv_area_cell[cell]);
            value = mpas_sub(value, mpas_mul(rdzu[k], mpas_sub(
                vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
            result[index] = value;
        }
    }
}

extern "C" __global__ void enforce_rw_endpoints_v841_f32(
    const int nlev, const int ncells, float *rw)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    rw[C2(0, cell, ncells)] = 0.0f;
    rw[C2(nlev, cell, ncells)] = 0.0f;
}

extern "C" __global__ void validate_positive_state_v841_f32(
    const int nvalues, const float *rho, const float *rho_theta, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    const float density = rho[index];
    const float theta_mass = rho_theta[index];
    if (!(isfinite(density) && density > 0.0f
            && isfinite(theta_mass) && theta_mass > 0.0f)) {
        atomicExch(invalid, 1);
    }
}

extern "C" __global__ void validate_recovered_v841_f32(
    const int nlev, const int ncells, const int nedges,
    const float *rho, const float *rho_theta, const float *rho_u,
    const float *rho_w, const float *theta, const float *exner,
    const float *rho_p, const float *rtheta_p, const float *pressure_p,
    const float *normal_velocity, const float *vertical_velocity, int *invalid)
{
    const int owner = blockDim.x * blockIdx.x + threadIdx.x;
    if (owner < ncells) {
        for (int k = 0; k < nlev; ++k) {
            const int index = C2(k, owner, ncells);
            if (!(isfinite(rho[index]) && rho[index] > 0.0f
                    && isfinite(rho_theta[index]) && rho_theta[index] > 0.0f
                    && isfinite(theta[index]) && theta[index] > 0.0f
                    && isfinite(exner[index]) && exner[index] > 0.0f
                    && isfinite(rho_p[index]) && isfinite(rtheta_p[index])
                    && isfinite(pressure_p[index]))) {
                atomicExch(invalid, 1);
            }
        }
        for (int k = 0; k <= nlev; ++k) {
            const int index = C2(k, owner, ncells);
            if (!(isfinite(rho_w[index]) && isfinite(vertical_velocity[index])))
                atomicExch(invalid, 1);
        }
    }
    if (owner < nedges) {
        for (int k = 0; k < nlev; ++k) {
            const int index = E2(k, owner, nedges);
            if (!(isfinite(rho_u[index]) && isfinite(normal_velocity[index])))
                atomicExch(invalid, 1);
        }
    }
}

extern "C" __global__ void validate_finite_array_v841_f32(
    const int nvalues, const float *values, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < nvalues && !isfinite(values[index])) atomicExch(invalid, 1);
}

extern "C" __global__ void split_flux_first_v841_f32(
    const int nlevels, const int nowners, const float *current, float *accumulator)
{
    const int owner = blockDim.x * blockIdx.x + threadIdx.x;
    if (owner >= nowners) return;
    for (int k = 0; k < nlevels; ++k) {
        const int index = k * nowners + owner;
        accumulator[index] = current[index];
    }
}

extern "C" __global__ void split_flux_add_v841_f32(
    const int nlevels, const int nowners, const float *current, float *accumulator)
{
    const int owner = blockDim.x * blockIdx.x + threadIdx.x;
    if (owner >= nowners) return;
    for (int k = 0; k < nlevels; ++k) {
        const int index = k * nowners + owner;
        accumulator[index] = mpas_add(current[index], accumulator[index]);
    }
}

extern "C" __global__ void split_flux_finish_v841_f32(
    const int nlevels, const int nowners, const float reciprocal,
    const float *accumulator, float *average)
{
    const int owner = blockDim.x * blockIdx.x + threadIdx.x;
    if (owner >= nowners) return;
    for (int k = 0; k < nlevels; ++k) {
        const int index = k * nowners + owner;
        average[index] = mpas_mul(accumulator[index], reciprocal);
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
            module_key="hexcore.cuda_dynamics_v841",
        )
    return _KERNELS[key]


def _launch(name: str, count: int, args: tuple[Any, ...], cache: Any | None) -> None:
    if count < 1:
        return
    threads = 128
    _kernel(name, cache)(
        ((int(count) + threads - 1) // threads,),
        (threads,),
        args,
    )


def _shape(value: Any, name: str) -> tuple[int, int]:
    array = require_resident_array(name, value, dtype=np.float32)
    if array.ndim != 2 or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous (level,owner) float32")
    return int(array.shape[0]), int(array.shape[1])


def vector_momentum_tendency_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    normal_velocity: Any,
    rho_edge: Any,
    pv_edge: Any,
    kinetic_energy: Any,
    horizontal_divergence: Any,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev, nedges = _shape(normal_velocity, "normal_velocity")
    ncells = int(mesh.n_cells)
    if nedges != int(mesh.n_edges):
        raise ValueError("normal_velocity edge extent differs from mesh")
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    for name, value, shape in (
        ("rho_edge", rho_edge, (nlev, nedges)),
        ("pv_edge", pv_edge, (nlev, nedges)),
        ("kinetic_energy", kinetic_energy, (nlev, ncells)),
        ("horizontal_divergence", horizontal_divergence, (nlev, ncells)),
    ):
        require_resident_array(name, value, dtype=np.float32, shape=shape)
    out = cp.empty_like(normal_velocity)
    _launch(
        "vector_momentum_v841_f32",
        nedges,
        (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            np.int32(mesh.max_edges2), normal_velocity, rho_edge, pv_edge,
            kinetic_energy, horizontal_divergence, mesh.cells_on_edge,
            mesh.edges_on_edge, mesh.n_edges_on_edge, mesh.weights_on_edge,
            context.inv_dc_edge, mesh.angle_edge, mesh.f_edge,
            context.u_init, context.v_init, out,
        ),
        kernel_cache,
    )
    return out


def theta_finish_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    edge_flux: Any,
    vertical_flux: Any,
    rdzw: Any,
    n_vert_levels: int,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev = int(n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    require_resident_array(
        "edge_flux", edge_flux, dtype=np.float32, shape=(nlev, nedges)
    )
    require_resident_array(
        "vertical_flux", vertical_flux,
        dtype=np.float32, shape=(nlev + 1, ncells),
    )
    require_resident_array("rdzw", rdzw, dtype=np.float32, shape=(nlev,))
    out = cp.empty((nlev, ncells), dtype=cp.float32)
    _launch(
        "theta_finish_v841_f32",
        ncells,
        (
            np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(mesh.max_edges),
            mesh.n_edges_on_cell, mesh.edges_on_cell, mesh.edge_sign_on_cell,
            context.inv_area_cell, rdzw, edge_flux, vertical_flux, out,
        ),
        kernel_cache,
    )
    return out


def w_finish_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    *,
    edge_flux: Any,
    vertical_flux: Any,
    rdzu: Any,
    n_vert_levels: int,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev = int(n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    require_resident_array(
        "edge_flux", edge_flux, dtype=np.float32, shape=(nlev + 1, nedges)
    )
    require_resident_array(
        "vertical_flux", vertical_flux,
        dtype=np.float32, shape=(nlev + 1, ncells),
    )
    require_resident_array("rdzu", rdzu, dtype=np.float32, shape=(nlev,))
    out = cp.empty((nlev + 1, ncells), dtype=cp.float32)
    _launch(
        "w_finish_v841_f32",
        ncells,
        (
            np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(mesh.max_edges),
            mesh.n_edges_on_cell, mesh.edges_on_cell, mesh.edge_sign_on_cell,
            context.inv_area_cell, rdzu, edge_flux, vertical_flux, out,
        ),
        kernel_cache,
    )
    return out


def enforce_rw_endpoints_cuda_v841(
    rho_w: Any,
    *,
    kernel_cache: Any | None = None,
) -> Any:
    nlevels, ncells = _shape(rho_w, "rho_w")
    if nlevels < 2:
        raise ValueError("rho_w must contain bottom and top interfaces")
    _launch(
        "enforce_rw_endpoints_v841_f32",
        ncells,
        (np.int32(nlevels - 1), np.int32(ncells), rho_w),
        kernel_cache,
    )
    return rho_w


def validate_positive_state_cuda_v841(
    rho: Any,
    rho_theta: Any,
    *,
    invalid_flag: Any | None = None,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev, ncells = _shape(rho, "rho")
    require_resident_array(
        "rho_theta", rho_theta, dtype=np.float32, shape=(nlev, ncells)
    )
    owns_flag = invalid_flag is None
    invalid = (
        cp.zeros((1,), dtype=cp.int32)
        if owns_flag
        else require_resident_array(
            "invalid_flag", invalid_flag, dtype=np.int32, shape=(1,)
        )
    )
    _launch(
        "validate_positive_state_v841_f32",
        nlev * ncells,
        (
            np.int32(nlev * ncells), rho, rho_theta, invalid,
        ),
        kernel_cache,
    )
    if owns_flag and int(cp.asnumpy(invalid)[0]) != 0:
        raise FloatingPointError(
            "v8.4.1 CUDA rho and rho_theta must remain finite and positive"
        )
    return invalid


def validate_recovered_state_cuda_v841(
    state: Any,
    saved: Any,
    *,
    invalid_flag: Any,
    kernel_cache: Any | None = None,
) -> Any:
    nlev, ncells = _shape(state.rho, "state.rho")
    nedges = int(state.rho_u.shape[1])
    requirements = {
        "state.rho_theta": (state.rho_theta, (nlev, ncells)),
        "state.rho_u": (state.rho_u, (nlev, nedges)),
        "state.rho_w": (state.rho_w, (nlev + 1, ncells)),
        "saved.theta_m": (saved.theta_m, (nlev, ncells)),
        "saved.exner": (saved.exner, (nlev, ncells)),
        "saved.density_perturbation": (
            saved.density_perturbation, (nlev, ncells)
        ),
        "saved.rho_theta_perturbation": (
            saved.rho_theta_perturbation, (nlev, ncells)
        ),
        "saved.pressure_perturbation": (
            saved.pressure_perturbation, (nlev, ncells)
        ),
        "saved.normal_velocity": (saved.normal_velocity, (nlev, nedges)),
        "saved.vertical_velocity": (
            saved.vertical_velocity, (nlev + 1, ncells)
        ),
    }
    for name, (value, shape) in requirements.items():
        require_resident_array(name, value, dtype=np.float32, shape=shape)
    flag = require_resident_array(
        "invalid_flag", invalid_flag, dtype=np.int32, shape=(1,)
    )
    _launch(
        "validate_recovered_v841_f32",
        max(ncells, nedges),
        (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            state.rho, state.rho_theta, state.rho_u, state.rho_w,
            saved.theta_m, saved.exner, saved.density_perturbation,
            saved.rho_theta_perturbation, saved.pressure_perturbation,
            saved.normal_velocity, saved.vertical_velocity, flag,
        ),
        kernel_cache,
    )
    return flag


def accumulate_finite_array_cuda_v841(
    values: Any,
    *,
    invalid_flag: Any,
    kernel_cache: Any | None = None,
) -> Any:
    array = require_resident_array("values", values, dtype=np.float32)
    flag = require_resident_array(
        "invalid_flag", invalid_flag, dtype=np.int32, shape=(1,)
    )
    _launch(
        "validate_finite_array_v841_f32",
        int(array.size),
        (np.int32(array.size), array, flag),
        kernel_cache,
    )
    return flag


def accumulate_split_flux_cuda_v841(
    current: Any,
    accumulator: Any | None,
    *,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlevels, nowners = _shape(current, "current_split_flux")
    if accumulator is None:
        result = cp.empty_like(current)
        kernel = "split_flux_first_v841_f32"
    else:
        accumulator_levels, accumulator_owners = _shape(
            accumulator, "split_flux_accumulator"
        )
        if (accumulator_levels, accumulator_owners) != (nlevels, nowners):
            raise ValueError("split flux accumulator shape mismatch")
        result = accumulator
        kernel = "split_flux_add_v841_f32"
    _launch(
        kernel,
        nowners,
        (np.int32(nlevels), np.int32(nowners), current, result),
        kernel_cache,
    )
    return result


def finish_split_flux_cuda_v841(
    accumulator: Any,
    split_count: int,
    *,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    if not isinstance(split_count, (int, np.integer)) or isinstance(
        split_count, (bool, np.bool_)
    ) or int(split_count) < 1:
        raise ValueError("split_count must be a positive integer")
    nlevels, nowners = _shape(accumulator, "split_flux_accumulator")
    result = cp.empty_like(accumulator)
    reciprocal = np.float32(1.0) / np.float32(split_count)
    _launch(
        "split_flux_finish_v841_f32",
        nowners,
        (
            np.int32(nlevels), np.int32(nowners), reciprocal,
            accumulator, result,
        ),
        kernel_cache,
    )
    return result


__all__ = [
    "accumulate_split_flux_cuda_v841",
    "accumulate_finite_array_cuda_v841",
    "enforce_rw_endpoints_cuda_v841",
    "finish_split_flux_cuda_v841",
    "theta_finish_cuda_v841",
    "validate_positive_state_cuda_v841",
    "validate_recovered_state_cuda_v841",
    "vector_momentum_tendency_cuda_v841",
    "w_finish_cuda_v841",
]
