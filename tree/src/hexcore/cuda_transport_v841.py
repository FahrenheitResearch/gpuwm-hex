"""MPAS-A v8.4.1 CUDA atmosphere scalar transport.

The v8.2.3 translation unit remains untouched.  This additive module retains
the released stored-inverse and reduction associations needed by the native
split-three outer scalar RK.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import cuda_transport as _v823
from .cuda_transport import CudaAdvectionCoefficients, CudaScalarTransportResult
from .cuda_v841 import CudaV841Context
from .errors import ConfigurationRefusal


_CUDA_SOURCE = _v823._CUDA_SOURCE + r"""
extern "C" __global__ void transport_interpolate_target_v841(
    const int nvalues, const float weight,
    const float *rho_old, const float *rho_new, float *target)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    target[index] = mpas_add(
        mpas_mul(mpas_sub(1.0f, weight), rho_old[index]),
        mpas_mul(weight, rho_new[index]));
}

extern "C" __global__ void validate_density_v841(
    const int nvalues, const float *density, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nvalues) return;
    const float value = density[index];
    if (!(isfinite(value) && value > 0.0f)) atomicExch(invalid, 1);
}

extern "C" __global__ void validate_transport_indices_v841(
    const int ncells, const int nedges, const int max_edges,
    const int width, const int *n_edges_on_cell, const int *edges_on_cell,
    const int *cells_on_cell, const int *cells_on_edge,
    const int *n_adv, const int *adv_cells, int *invalid)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < ncells) {
        const int count = n_edges_on_cell[index];
        if (count < 0 || count > max_edges) atomicExch(invalid, 1);
        const int safe_count = count < 0 ? 0 : (count > max_edges ? max_edges : count);
        for (int slot = 0; slot < safe_count; ++slot) {
            const int edge = edges_on_cell[CES(index, slot, max_edges)];
            const int cell = cells_on_cell[CES(index, slot, max_edges)];
            if (edge < 0 || edge >= nedges || cell < 0 || cell >= ncells)
                atomicExch(invalid, 1);
        }
    }
    if (index < nedges) {
        const int c0 = cells_on_edge[2 * index];
        const int c1 = cells_on_edge[2 * index + 1];
        if (c0 < 0 || c0 >= ncells || c1 < 0 || c1 >= ncells)
            atomicExch(invalid, 1);
        const int count = n_adv[index];
        if (count < 0 || count > width) atomicExch(invalid, 1);
        const int safe_count = count < 0 ? 0 : (count > width ? width : count);
        for (int slot = 0; slot < safe_count; ++slot) {
            const int cell = adv_cells[ADV(index, slot, width)];
            if (cell < 0 || cell >= ncells) atomicExch(invalid, 1);
        }
    }
}

extern "C" __global__ void transport_standard_finish_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *velocity, const float *edge_values,
    const float *vertical_flux, const float *rdzw,
    const float *old, const float *rho_old, const float *target_density,
    const float *source, float *output)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float horizontal = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                horizontal = mpas_add(horizontal, mpas_mul(mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]),
                    edge_values[QE3(tracer, k, edge, nlev, nedges)]));
            }
            const float tendency = mpas_add(
                mpas_mul(horizontal, inv_area_cell[cell]), source[index]);
            const float vertical = mpas_mul(rdzw[k], mpas_sub(
                vertical_flux[QI3(tracer, k + 1, cell, nlev, ncells)],
                vertical_flux[QI3(tracer, k, cell, nlev, ncells)]));
            const float numerator = mpas_add(
                mpas_mul(old[index], rho_old[C2(k, cell, ncells)]),
                mpas_mul(dt, mpas_sub(tendency, vertical)));
            const float rho_new_inv = mpas_div(
                1.0f, target_density[C2(k, cell, ncells)]);
            output[index] = mpas_mul(numerator, rho_new_inv);
        }
    }
}

extern "C" __global__ void transport_target_density_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int advance_density, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *dv_edge,
    const float *inv_area_cell, const float *velocity,
    const float *vertical_velocity, const float *rdzw,
    const float *rho_old, const float *rho_new, float *target_density)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (!advance_density) {
            target_density[index] = rho_new[index];
        } else {
            float horizontal = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                float contribution = mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]);
                contribution = mpas_mul(contribution, dv_edge[edge]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                horizontal = mpas_add(horizontal, contribution);
            }
            target_density[index] = mpas_add(rho_old[index], mpas_mul(dt,
                mpas_sub(horizontal, mpas_mul(rdzw[k], mpas_sub(
                    vertical_velocity[C2(k + 1, cell, ncells)],
                    vertical_velocity[index])))));
        }
    }
}

extern "C" __global__ void transport_edge_values_mono_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *edge_values)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int count = n_adv[edge];
    if (count == 10) {
        for (int k = 0; k < nlev; ++k) {
            const float vel = velocity[E2(k, edge, nedges)];
            const float sign = mpas_copysign(1.0f, vel);
            for (int tracer = 0; tracer < ntracers; ++tracer) {
                const int first_cell = adv_cells[ADV(edge, 0, width)];
                float weight = mpas_add(adv[ADV(edge, 0, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, 0, width)]));
                float value = mpas_mul(weight,
                    stage[Q3(tracer, k, first_cell, nlev, ncells)]);
                for (int slot = 1; slot < 10; ++slot) {
                    const int cell = adv_cells[ADV(edge, slot, width)];
                    weight = mpas_add(adv[ADV(edge, slot, width)],
                        mpas_mul(mpas_mul(coefficient, sign),
                            adv3[ADV(edge, slot, width)]));
                    value = mpas_add(value, mpas_mul(weight,
                        stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
                edge_values[QE3(tracer, k, edge, nlev, nedges)] =
                    mpas_mul(vel, value);
            }
        }
    } else {
        for (int tracer = 0; tracer < ntracers; ++tracer) {
            for (int k = 0; k < nlev; ++k) {
                edge_values[QE3(tracer, k, edge, nlev, nedges)] = 0.0f;
            }
        }
        for (int slot = 0; slot < count; ++slot) {
            const int cell = adv_cells[ADV(edge, slot, width)];
            for (int k = 0; k < nlev; ++k) {
                const float vel = velocity[E2(k, edge, nedges)];
                const float sign = mpas_copysign(1.0f, vel);
                float scalar_weight = mpas_add(adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)]));
                scalar_weight = mpas_mul(vel, scalar_weight);
                for (int tracer = 0; tracer < ntracers; ++tracer) {
                    const int index = QE3(tracer, k, edge, nlev, nedges);
                    edge_values[index] = mpas_add(edge_values[index],
                        mpas_mul(scalar_weight,
                            stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
            }
        }
    }
}

extern "C" __global__ void fct_edge_residual_v841(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const float dt, const float *source_old, const float *velocity,
    const float *dv_edge, const int *cells_on_edge,
    const float *high_flux, float *upwind_flux, float *horizontal_residual)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = QE3(tracer, k, edge, nlev, nedges);
            const float vel = velocity[E2(k, edge, nedges)];
            const float upwind = mpas_mul(mpas_mul(dv_edge[edge], dt),
                mpas_add(mpas_mul(plus(vel), source_old[Q3(
                    tracer, k, c0, nlev, ncells)]),
                    mpas_mul(minus(vel), source_old[Q3(
                    tracer, k, c1, nlev, ncells)])));
            upwind_flux[index] = upwind;
            horizontal_residual[index] = mpas_sub(
                mpas_mul(dt, high_flux[index]), upwind);
        }
    }
}

extern "C" __global__ void fct_horizontal_low_order_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell,
    const float *upwind_flux, const float *horizontal_residual,
    float *mass, float *scale_in, float *scale_out)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float inverse_area = inv_area_cell[cell];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float cell_mass = mass[index];
            float incoming = scale_in[index];
            float outgoing = scale_out[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                const int qe = QE3(tracer, k, edge, nlev, nedges);
                cell_mass = mpas_add(cell_mass, mpas_mul(
                    mpas_mul(sign, upwind_flux[qe]), inverse_area));
                const float signed_residual = mpas_mul(
                    -sign, horizontal_residual[qe]);
                outgoing = mpas_sub(outgoing,
                    mpas_mul(plus(signed_residual), inverse_area));
                incoming = mpas_sub(incoming,
                    mpas_mul(minus(signed_residual), inverse_area));
            }
            mass[index] = cell_mass;
            scale_in[index] = incoming;
            scale_out[index] = outgoing;
        }
    }
}

extern "C" __global__ void fct_finish_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *horizontal_residual, const float *vertical_residual,
    const float *target_density, const float *mass, float *output)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float inverse_area = inv_area_cell[cell];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float value = mass[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                value = mpas_add(value, mpas_mul(mpas_mul(sign,
                    horizontal_residual[QE3(
                        tracer, k, edge, nlev, nedges)]), inverse_area));
            }
            value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
                vertical_residual[QI3(
                    tracer, k + 1, cell, nlev, ncells)],
                vertical_residual[QI3(
                    tracer, k, cell, nlev, ncells)])));
            output[index] = mpas_max(0.0f,
                mpas_div(value, target_density[C2(k, cell, ncells)]));
        }
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
            module_key="hexcore.cuda_transport_v841",
        )
    return _KERNELS[key]


def _launch(
    name: str,
    count: int,
    arguments: tuple[Any, ...],
    kernel_cache: Any | None,
) -> None:
    threads = 128
    _kernel(name, kernel_cache)(
        ((int(count) + threads - 1) // threads,), (threads,), arguments
    )


def _common_inputs(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
) -> tuple[Any, ...]:
    values = _v823._inputs(
        mesh,
        scalar_old,
        scalar_stage,
        rho_zz_old,
        rho_zz_new,
        uh_avg,
        ww_avg,
        fzm,
        fzp,
        rdzw,
    )
    nlev, ncells, nedges = values[11], values[12], values[13]
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    for name, value in (("fzm", values[7]), ("fzp", values[8]), ("rdzw", values[9])):
        if tuple(value.shape) != (nlev,):
            raise ValueError(f"{name} must have shape ({nlev},)")
    return values


def _check_density(
    density: Any,
    *,
    name: str,
    count: int,
    invalid_flag: Any | None,
    kernel_cache: Any | None,
) -> Any:
    cp = _cupy()
    owns_flag = invalid_flag is None
    invalid = (
        cp.zeros((1,), dtype=cp.int32)
        if owns_flag
        else _v823._i32(invalid_flag)
    )
    if tuple(invalid.shape) != (1,):
        raise ValueError("validation_flag must have shape (1,)")
    _launch(
        "validate_density_v841",
        count,
        (np.int32(count), density, invalid),
        kernel_cache,
    )
    if owns_flag and int(cp.asnumpy(invalid)[0]) != 0:
        raise ValueError(f"{name} must remain finite and strictly positive")
    return invalid


def _validate_geometry_and_coefficients(
    mesh: Any,
    coefficients: CudaAdvectionCoefficients,
    *,
    ncells: int,
    nedges: int,
    indices_prevalidated: bool,
    kernel_cache: Any | None,
) -> tuple[Any, ...]:
    cp = _cupy()
    coefficients.validate()
    coefficient_shape = tuple(coefficients.adv_coefs.shape)
    if len(coefficient_shape) != 2 or coefficient_shape[0] != nedges:
        raise ValueError("advection coefficient edge extent differs from mesh")
    width = int(coefficient_shape[1])
    geometry = _v823._geometry(mesh)
    counts, edges, neighbors, cells_on_edge, signs, dv, area, max_edges = geometry
    expected = {
        "n_edges_on_cell": (counts, (ncells,)),
        "edges_on_cell": (edges, (ncells, max_edges)),
        "cells_on_cell": (neighbors, (ncells, max_edges)),
        "cells_on_edge": (cells_on_edge, (nedges, 2)),
        "edge_sign_on_cell": (signs, (ncells, max_edges)),
        "dv_edge": (dv, (nedges,)),
        "area_cell": (area, (ncells,)),
    }
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if not indices_prevalidated:
        invalid = cp.zeros((1,), dtype=cp.int32)
        _launch(
            "validate_transport_indices_v841",
            max(ncells, nedges),
            (
                np.int32(ncells), np.int32(nedges), np.int32(max_edges),
                np.int32(width), counts, edges, neighbors, cells_on_edge,
                coefficients.n_adv_cells_for_edge,
                coefficients.adv_cells_for_edge, invalid,
            ),
            kernel_cache,
        )
        if int(cp.asnumpy(invalid)[0]) != 0:
            raise ValueError(
                "active transport connectivity or stencil index is out of range"
            )
    return geometry


def _standard_high_fluxes(
    stage: Any,
    velocity: Any,
    vertical_velocity: Any,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    coefficient: float,
    ntracers: int,
    nlev: int,
    ncells: int,
    nedges: int,
    kernel_cache: Any | None,
) -> tuple[Any, Any]:
    cp = _cupy()
    edge = cp.empty((ntracers, nlev, nedges), dtype=cp.float32)
    vertical = cp.empty((ntracers, nlev + 1, ncells), dtype=cp.float32)
    width = int(coefficients.adv_coefs.shape[1])
    _launch(
        "transport_edge_values",
        nedges,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(width), np.float32(coefficient),
            stage, velocity, coefficients.adv_coefs,
            coefficients.adv_coefs_3rd,
            coefficients.n_adv_cells_for_edge,
            coefficients.adv_cells_for_edge, edge,
        ),
        kernel_cache,
    )
    _launch(
        "transport_vertical_flux",
        ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.float32(coefficient), stage, vertical_velocity, fzm, fzp,
            vertical,
        ),
        kernel_cache,
    )
    return edge, vertical


def _source(
    old: Any,
    scalar_tendency: Any | None,
    cp: Any,
    *,
    squeezed: bool,
) -> Any:
    if scalar_tendency is None:
        return cp.zeros_like(old)
    source = _v823._f32(scalar_tendency, "scalar_tendency")
    expected_ndim = 2 if squeezed else 3
    if source.ndim != expected_ndim:
        raise ValueError("scalar_tendency carrier rank differs from scalar_old")
    if squeezed:
        source = source[None, ...]
    if source.shape != old.shape:
        raise ValueError("scalar_tendency shape differs from scalar_old")
    return source


def advance_scalars_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_time_integration_order: int,
    config_scalar_adv_order: int,
    config_scalar_vadv_order: int,
    config_coef_3rd_order: float,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    validation_flag: Any | None = None,
    indices_prevalidated: bool = False,
    rho_old_prevalidated: bool = False,
    kernel_cache: Any | None = None,
    halo_exchange: Any | None = None,
) -> CudaScalarTransportResult:
    # ``halo_exchange`` is accepted for signature parity with the monotonic
    # lane: RK stages 1/2 need no mid-function round -- the executor's D round
    # at stage entry covers every ring<=2 read in this function.
    del halo_exchange
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    _v823._validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order,
        config_coef_3rd_order,
    )
    if rk_step not in (1, 2, 3) or config_time_integration_order != 3:
        raise ConfigurationRefusal(
            "config_time_integration_order", config_time_integration_order,
            "the v8.4.1 CUDA scalar path is RK3", "config_time_integration_order=3",
        )
    values = _common_inputs(
        mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, fzm, fzp, rdzw,
    )
    (
        old, stage, squeezed, rho0, rho1, velocity, vertical_velocity,
        fnm, fnp, rdnw, ntracers, nlev, ncells, nedges,
    ) = values
    cp = _cupy()
    source = _source(old, scalar_tendency, cp, squeezed=squeezed)
    counts, edges, _neighbors, _cells, signs, _dv, _area, max_edges = (
        _validate_geometry_and_coefficients(
            mesh, coefficients, ncells=ncells, nedges=nedges,
            indices_prevalidated=indices_prevalidated,
            kernel_cache=kernel_cache,
        )
    )
    if not rho_old_prevalidated:
        _check_density(
            rho0, name="rho_zz_old", count=nlev * ncells,
            invalid_flag=None, kernel_cache=kernel_cache,
        )
    edge_values, vertical_flux = _standard_high_fluxes(
        stage, velocity, vertical_velocity, coefficients, fnm, fnp,
        config_coef_3rd_order, ntracers, nlev, ncells, nedges, kernel_cache,
    )
    weight = np.float32(
        1.0 / 3.0 if advance_density and rk_step == 1
        else 0.5 if advance_density and rk_step == 2
        else 1.0
    )
    target = cp.empty_like(rho0)
    _launch(
        "transport_interpolate_target_v841",
        nlev * ncells,
        (
            np.int32(nlev * ncells), weight, rho0, rho1, target,
        ),
        kernel_cache,
    )
    _check_density(
        target, name="target_density", count=nlev * ncells,
        invalid_flag=validation_flag, kernel_cache=kernel_cache,
    )
    output = cp.empty_like(old)
    _launch(
        "transport_standard_finish_v841",
        ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(max_edges), np.float32(dt), counts,
            edges, signs, context.inv_area_cell, velocity, edge_values,
            vertical_flux, rdnw, old, rho0, target, source, output,
        ),
        kernel_cache,
    )
    return CudaScalarTransportResult(output[0] if squeezed else output, target)


def advance_scalars_monotonic_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    config_scalar_adv_order: int,
    config_scalar_vadv_order: int,
    config_coef_3rd_order: float,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    n_halos: int = 3,
    validation_flag: Any | None = None,
    indices_prevalidated: bool = False,
    rho_old_prevalidated: bool = False,
    kernel_cache: Any | None = None,
    halo_exchange: Any | None = None,
) -> CudaScalarTransportResult:
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    _v823._validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order,
        config_coef_3rd_order,
    )
    if n_halos < 3:
        raise ConfigurationRefusal("nHalos", n_halos, "FCT requires three halos", "nHalos=3")
    values = _common_inputs(
        mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, fzm, fzp, rdzw,
    )
    (
        old, stage, squeezed, rho0, rho1, velocity, vertical_velocity,
        fnm, fnp, rdnw, ntracers, nlev, ncells, nedges,
    ) = values
    cp = _cupy()
    source = _source(old, scalar_tendency, cp, squeezed=squeezed)
    counts, edges, neighbors, cells_on_edge, signs, dv, _area, max_edges = (
        _validate_geometry_and_coefficients(
            mesh, coefficients, ncells=ncells, nedges=nedges,
            indices_prevalidated=indices_prevalidated,
            kernel_cache=kernel_cache,
        )
    )
    if not rho_old_prevalidated:
        _check_density(
            rho0, name="rho_zz_old", count=nlev * ncells,
            invalid_flag=None, kernel_cache=kernel_cache,
        )
    target = cp.empty_like(rho0)
    _launch(
        "transport_target_density_v841",
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            np.int32(max_edges), np.int32(bool(advance_density)), np.float32(dt),
            counts, edges, signs, dv, context.inv_area_cell, velocity,
            vertical_velocity, rdnw, rho0, rho1, target,
        ),
        kernel_cache,
    )
    _check_density(
        target, name="target_density", count=nlev * ncells,
        invalid_flag=validation_flag, kernel_cache=kernel_cache,
    )
    width = int(coefficients.adv_coefs.shape[1])
    high_edge = cp.empty((ntracers, nlev, nedges), dtype=cp.float32)
    _launch(
        "transport_edge_values_mono_v841",
        nedges,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(width),
            np.float32(config_coef_3rd_order), stage, velocity,
            coefficients.adv_coefs, coefficients.adv_coefs_3rd,
            coefficients.n_adv_cells_for_edge, coefficients.adv_cells_for_edge,
            high_edge,
        ),
        kernel_cache,
    )
    high_vertical = cp.empty(
        (ntracers, nlev + 1, ncells), dtype=cp.float32
    )
    _launch(
        "transport_vertical_flux",
        ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.float32(config_coef_3rd_order), stage, vertical_velocity,
            fnm, fnp, high_vertical,
        ),
        kernel_cache,
    )
    source_old = cp.empty_like(old)
    minimum = cp.empty_like(old)
    maximum = cp.empty_like(old)
    _launch(
        "fct_minmax_source", ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(max_edges), np.float32(dt), counts, neighbors, old,
            source, rho0, source_old, minimum, maximum,
        ), kernel_cache,
    )
    mass = cp.empty_like(old)
    vertical_residual = cp.empty_like(high_vertical)
    scale_in = cp.empty_like(old)
    scale_out = cp.empty_like(old)
    _launch(
        "fct_vertical_low_order", ntracers * ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.float32(dt), source_old, rho0, vertical_velocity, rdnw,
            high_vertical, mass, vertical_residual, scale_in, scale_out,
        ), kernel_cache,
    )
    upwind = cp.empty_like(high_edge)
    horizontal_residual = cp.empty_like(high_edge)
    _launch(
        "fct_edge_residual_v841", nedges,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.float32(dt), source_old, velocity, dv,
            cells_on_edge, high_edge, upwind, horizontal_residual,
        ), kernel_cache,
    )
    _launch(
        "fct_horizontal_low_order_v841", ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(max_edges), counts, edges, signs,
            context.inv_area_cell, upwind, horizontal_residual, mass,
            scale_in, scale_out,
        ), kernel_cache,
    )
    _launch(
        "fct_scale", ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells), minimum,
            maximum, target, mass, scale_in, scale_out,
        ), kernel_cache,
    )
    if halo_exchange is not None:
        # Round E (two-rank executor): fct_limit_horizontal reads BOTH endpoint
        # cells' scale factors at owned edges, so ring-1/2 scale factors must be
        # owner truth first (a boundary cell's own factors read ring-2-edge
        # residuals whose input cones leave the K=2 halo).
        halo_exchange.round_fct_scale(scale_in, scale_out)
    _launch(
        "fct_limit_horizontal", nedges,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), cells_on_edge, scale_in, scale_out,
            horizontal_residual,
        ), kernel_cache,
    )
    _launch(
        "fct_limit_vertical", ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells), scale_in,
            scale_out, vertical_residual,
        ), kernel_cache,
    )
    output = cp.empty_like(old)
    _launch(
        "fct_finish_v841", ncells,
        (
            np.int32(ntracers), np.int32(nlev), np.int32(ncells),
            np.int32(nedges), np.int32(max_edges), counts, edges, signs,
            context.inv_area_cell, rdnw, horizontal_residual,
            vertical_residual, target, mass, output,
        ), kernel_cache,
    )
    return CudaScalarTransportResult(output[0] if squeezed else output, target)


def advance_scalar_transport_cuda_v841(
    mesh: Any,
    context: CudaV841Context,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: CudaAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_scalar_advection: bool,
    config_monotonic: bool,
    config_positive_definite: bool,
    config_split_dynamics_transport: bool,
    config_time_integration_order: int,
    kernel_cache: Any | None = None,
    **kwargs: Any,
) -> CudaScalarTransportResult:
    if rk_step not in (1, 2, 3):
        raise ConfigurationRefusal(
            "rk_step", rk_step, "the v8.4.1 CUDA scalar RK has three stages",
            "rk_step=1, 2, or 3",
        )
    if config_time_integration_order != 3:
        raise ConfigurationRefusal(
            "config_time_integration_order", config_time_integration_order,
            "the v8.4.1 CUDA scalar lane mirrors RK3",
            "config_time_integration_order=3",
        )
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    cp = _cupy()
    stage = _v823._f32(scalar_stage, "scalar_stage")
    rho_new = _v823._f32(rho_zz_new, "rho_zz_new")
    if not config_scalar_advection:
        return CudaScalarTransportResult(
            cp.array(stage, copy=True), cp.array(rho_new, copy=True)
        )
    common = dict(
        coefficients=coefficients,
        fzm=fzm,
        fzp=fzp,
        rdzw=rdzw,
        advance_density=config_split_dynamics_transport,
        kernel_cache=kernel_cache,
        **kwargs,
    )
    if rk_step >= 3 and (config_monotonic or config_positive_definite):
        return advance_scalars_monotonic_cuda_v841(
            mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
            uh_avg, ww_avg, dt, **common,
        )
    return advance_scalars_cuda_v841(
        mesh, context, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, dt, rk_step=rk_step,
        config_time_integration_order=config_time_integration_order,
        **common,
    )


__all__ = [
    "advance_scalar_transport_cuda_v841",
    "advance_scalars_cuda_v841",
    "advance_scalars_monotonic_cuda_v841",
]
