"""Device-resident scalar transport and monotonic FCT for MPAS-A.

This is the CUDA transcription of the atmosphere-specific routines in frozen
MPAS-A v8.2.3 ``mpas_atm_time_integration.F:3049-4211``.  It implements the
admitted third-order horizontal/vertical path, including the stage-three
Zalesak limiter and split-transport density update.  All arrays are float32,
level-major CuPy arrays; no routine copies a field to the host.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .cuda_backend.containers import TransferStats, require_resident_array
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .errors import ConfigurationRefusal


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define Q3(q,k,c,nl,nc) ((((q)*(nl) + (k))*(nc)) + (c))
#define QE3(q,k,e,nl,ne) ((((q)*(nl) + (k))*(ne)) + (e))
#define QI3(q,k,c,nl,nc) ((((q)*((nl)+1) + (k))*(nc)) + (c))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,width) ((e)*(width) + (s))

__device__ __forceinline__ float plus(float x) { return mpas_max(0.0f, x); }
__device__ __forceinline__ float minus(float x) { return mpas_min(0.0f, x); }

extern "C" __global__ void transport_edge_values(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells,
    float *edge_values)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            float value = 0.0f;
            const float velocity_sign = mpas_copysign(
                1.0f, velocity[E2(k, edge, nedges)]);
            bool ftz_sensitive = false;
            for (int slot = 0; slot < n_adv[edge]; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                const float base_weight = adv[ADV(edge, slot, width)];
                const float third_weight = adv3[ADV(edge, slot, width)];
                const float stage_value = stage[Q3(
                    tracer, k, cell, nlev, ncells)];
                const float correction = coefficient * velocity_sign
                    * third_weight;
                const float weight = base_weight + correction;
                const float contribution = weight * stage_value;
                const float next = value + contribution;
#if MPAS_FTZ_FALLBACK_ENABLED
                const unsigned int coefficient_mag =
                    mpas_f32_magnitude_bits(coefficient);
                const unsigned int third_mag =
                    mpas_f32_magnitude_bits(third_weight);
                const unsigned int correction_mag =
                    mpas_f32_magnitude_bits(correction);
                const unsigned int base_mag =
                    mpas_f32_magnitude_bits(base_weight);
                const unsigned int weight_mag =
                    mpas_f32_magnitude_bits(weight);
                const unsigned int stage_mag =
                    mpas_f32_magnitude_bits(stage_value);
                const unsigned int contribution_mag =
                    mpas_f32_magnitude_bits(contribution);
                const unsigned int value_mag =
                    mpas_f32_magnitude_bits(value);
                const unsigned int next_mag = mpas_f32_magnitude_bits(next);
                ftz_sensitive = ftz_sensitive
                    || (base_mag != 0u && (base_mag >> 23) == 0u)
                    || (third_mag != 0u && (third_mag >> 23) == 0u)
                    || (stage_mag != 0u && (stage_mag >> 23) == 0u)
                    || (correction_mag == 0u && coefficient_mag != 0u
                        && third_mag != 0u)
                    || (weight_mag == 0u && base_mag != correction_mag
                        && base_mag != 0u && correction_mag != 0u)
                    || (contribution_mag == 0u && weight_mag != 0u
                        && stage_mag != 0u)
                    || (next_mag == 0u && value_mag != contribution_mag
                        && value_mag != 0u && contribution_mag != 0u);
#endif
                value = next;
            }
            if (ftz_sensitive) {
                value = 0.0f;
                for (int slot = 0; slot < n_adv[edge]; ++slot) {
                    const int cell = adv_cells[ADV(edge, slot, width)];
                    const float weight = mpas_add(
                        adv[ADV(edge, slot, width)],
                        mpas_mul(mpas_mul(coefficient, velocity_sign),
                            adv3[ADV(edge, slot, width)]));
                    value = mpas_add(value, mpas_mul(weight,
                        stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
            }
            edge_values[QE3(tracer, k, edge, nlev, nedges)] = value;
        }
    }
}

extern "C" __global__ void transport_vertical_flux(
    const int ntracers, const int nlev, const int ncells,
    const float coefficient, const float *stage, const float *velocity,
    const float *fzm, const float *fzp, float *vertical_flux)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int interface = 0; interface <= nlev; ++interface) {
            float result = 0.0f;
            if (interface > 0 && interface < nlev) {
                const float vel = velocity[C2(interface, cell, ncells)];
                if (interface == 1 || interface == nlev - 1) {
                    result = mpas_mul(vel, mpas_add(
                        mpas_mul(fzm[interface], stage[Q3(
                            tracer, interface, cell, nlev, ncells)]),
                        mpas_mul(fzp[interface], stage[Q3(
                            tracer, interface - 1, cell, nlev, ncells)])));
                } else {
                    const float qim2 = stage[Q3(
                        tracer, interface - 2, cell, nlev, ncells)];
                    const float qim1 = stage[Q3(
                        tracer, interface - 1, cell, nlev, ncells)];
                    const float qi = stage[Q3(
                        tracer, interface, cell, nlev, ncells)];
                    const float qip1 = stage[Q3(
                        tracer, interface + 1, cell, nlev, ncells)];
                    const float base = mpas_div(mpas_mul(vel, mpas_sub(
                        mpas_mul(7.0f, mpas_add(qi, qim1)),
                        mpas_add(qip1, qim2))), 12.0f);
                    const float correction = mpas_div(mpas_mul(
                        mpas_mul(coefficient, mpas_abs(vel)),
                        mpas_sub(mpas_sub(qip1, qim2),
                            mpas_mul(3.0f, mpas_sub(qi, qim1)))), 12.0f);
                    result = mpas_add(base, correction);
                }
            }
            vertical_flux[QI3(tracer, interface, cell, nlev, ncells)] = result;
        }
    }
}

extern "C" __global__ void transport_target_density(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int advance_density, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *dv_edge, const float *area_cell,
    const float *velocity, const float *vertical_velocity,
    const float *rdzw, const float *rho_old, const float *rho_new,
    float *target_density)
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
                horizontal = mpas_add(horizontal, mpas_div(mpas_mul(
                    mpas_mul(sign, velocity[E2(k, edge, nedges)]),
                    dv_edge[edge]), area_cell[cell]));
            }
            target_density[index] = mpas_add(rho_old[index], mpas_mul(dt,
                mpas_sub(horizontal, mpas_mul(rdzw[k], mpas_sub(
                    vertical_velocity[C2(k + 1, cell, ncells)],
                    vertical_velocity[index])))));
        }
    }
}

extern "C" __global__ void transport_standard_finish(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges, const float dt,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *area_cell,
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
            float tendency = source[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                tendency = mpas_add(tendency, mpas_div(mpas_mul(mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]),
                    edge_values[QE3(tracer, k, edge, nlev, nedges)]),
                    area_cell[cell]));
            }
            const float vertical = mpas_mul(rdzw[k], mpas_sub(
                vertical_flux[QI3(tracer, k + 1, cell, nlev, ncells)],
                vertical_flux[QI3(tracer, k, cell, nlev, ncells)]));
            output[index] = mpas_div(mpas_add(
                mpas_mul(old[index], rho_old[C2(k, cell, ncells)]),
                mpas_mul(dt, mpas_sub(tendency, vertical))),
                target_density[C2(k, cell, ncells)]);
        }
    }
}

extern "C" __global__ void fct_minmax_source(
    const int ntracers, const int nlev, const int ncells,
    const int max_edges, const float dt,
    const int *n_edges_on_cell, const int *cells_on_cell,
    const float *old, const float *source, const float *rho_old,
    float *source_old, float *minimum, float *maximum)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            const float current = mpas_add(old[index], mpas_mul(dt,
                mpas_div(source[index], rho_old[C2(k, cell, ncells)])));
            source_old[index] = current;
            float low = current;
            float high = current;
            if (k > 0) {
                const float value = mpas_add(old[Q3(
                    tracer, k - 1, cell, nlev, ncells)], mpas_mul(dt,
                    mpas_div(source[Q3(
                        tracer, k - 1, cell, nlev, ncells)],
                        rho_old[C2(k - 1, cell, ncells)])));
                low = mpas_min(low, value);
                high = mpas_max(high, value);
            }
            if (k + 1 < nlev) {
                const float value = mpas_add(old[Q3(
                    tracer, k + 1, cell, nlev, ncells)], mpas_mul(dt,
                    mpas_div(source[Q3(
                        tracer, k + 1, cell, nlev, ncells)],
                        rho_old[C2(k + 1, cell, ncells)])));
                low = mpas_min(low, value);
                high = mpas_max(high, value);
            }
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int neighbor = cells_on_cell[CES(cell, slot, max_edges)];
                const float value = mpas_add(old[Q3(
                    tracer, k, neighbor, nlev, ncells)], mpas_mul(dt,
                    mpas_div(source[Q3(tracer, k, neighbor, nlev, ncells)],
                        rho_old[C2(k, neighbor, ncells)])));
                low = mpas_min(low, value);
                high = mpas_max(high, value);
            }
            minimum[index] = low;
            maximum[index] = high;
        }
    }
}

extern "C" __global__ void fct_vertical_low_order(
    const int ntracers, const int nlev, const int ncells,
    const float dt, const float *source_old, const float *rho_old,
    const float *vertical_velocity, const float *rdzw,
    const float *high_vertical,
    float *mass, float *vertical_residual, float *scale_in, float *scale_out)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = ntracers * ncells;
    if (index >= total) return;
    const int cell = index % ncells;
    const int tracer = index / ncells;
    for (int k = 0; k < nlev; ++k) {
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_mul(source_old[q], rho_old[C2(k, cell, ncells)]);
    }
    for (int interface = 0; interface <= nlev; ++interface) {
        const int qi = QI3(tracer, interface, cell, nlev, ncells);
        vertical_residual[qi] = mpas_mul(dt, high_vertical[qi]);
    }
    for (int interface = 1; interface < nlev; ++interface) {
        const float velocity = vertical_velocity[C2(interface, cell, ncells)];
        const float flux = mpas_mul(dt, mpas_add(
            mpas_mul(plus(velocity), source_old[Q3(
                tracer, interface - 1, cell, nlev, ncells)]),
            mpas_mul(minus(velocity), source_old[Q3(
                tracer, interface, cell, nlev, ncells)])));
        const int lower_q = Q3(
            tracer, interface - 1, cell, nlev, ncells);
        const int upper_q = Q3(tracer, interface, cell, nlev, ncells);
        mass[lower_q] = mpas_sub(
            mass[lower_q], mpas_mul(flux, rdzw[interface - 1]));
        mass[upper_q] = mpas_add(
            mass[upper_q], mpas_mul(flux, rdzw[interface]));
        const int residual_q = QI3(
            tracer, interface, cell, nlev, ncells);
        vertical_residual[residual_q] = mpas_sub(
            vertical_residual[residual_q], flux);
    }
    for (int k = 0; k < nlev; ++k) {
        const float upper = vertical_residual[QI3(tracer, k + 1, cell, nlev, ncells)];
        const float lower = vertical_residual[QI3(tracer, k, cell, nlev, ncells)];
        const int q = Q3(tracer, k, cell, nlev, ncells);
        scale_in[q] = mpas_mul(-rdzw[k],
            mpas_sub(minus(upper), plus(lower)));
        scale_out[q] = mpas_mul(-rdzw[k],
            mpas_sub(plus(upper), minus(lower)));
    }
}

extern "C" __global__ void fct_edge_residual(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const float dt, const float *source_old, const float *velocity,
    const float *dv_edge, const int *cells_on_edge,
    const float *edge_values, float *upwind_flux, float *horizontal_residual)
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
                mpas_mul(mpas_mul(dt, vel), edge_values[index]), upwind);
        }
    }
}

extern "C" __global__ void fct_horizontal_low_order(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *area_cell,
    const float *upwind_flux, const float *horizontal_residual,
    float *mass, float *scale_in, float *scale_out)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float inverse_area = 1.0f / area_cell[cell];
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

extern "C" __global__ void fct_scale(
    const int ntracers, const int nlev, const int ncells,
    const float *minimum, const float *maximum, const float *target_density,
    const float *mass, float *scale_in, float *scale_out)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            const float density = target_density[C2(k, cell, ncells)];
            const float eps = 1.0e-20f;
            const float incoming = mpas_div(mpas_sub(
                mpas_mul(maximum[index], density), mass[index]),
                mpas_add(scale_in[index], eps));
            const float outgoing = mpas_div(mpas_sub(
                mpas_mul(minimum[index], density), mass[index]),
                mpas_sub(scale_out[index], eps));
            scale_in[index] = mpas_min(1.0f, mpas_max(0.0f, incoming));
            scale_out[index] = mpas_min(1.0f, mpas_max(0.0f, outgoing));
        }
    }
}

extern "C" __global__ void fct_limit_horizontal(
    const int ntracers, const int nlev, const int ncells, const int nedges,
    const int *cells_on_edge, const float *scale_in, const float *scale_out,
    float *residual)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = QE3(tracer, k, edge, nlev, nedges);
            const float flux = residual[index];
            residual[index] = mpas_add(mpas_mul(plus(flux), mpas_min(
                    scale_out[Q3(tracer, k, c0, nlev, ncells)],
                    scale_in[Q3(tracer, k, c1, nlev, ncells)])),
                mpas_mul(minus(flux), mpas_min(
                    scale_in[Q3(tracer, k, c0, nlev, ncells)],
                    scale_out[Q3(tracer, k, c1, nlev, ncells)])));
        }
    }
}

extern "C" __global__ void fct_limit_vertical(
    const int ntracers, const int nlev, const int ncells,
    const float *scale_in, const float *scale_out, float *residual)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int interface = 1; interface < nlev; ++interface) {
            const int qi = QI3(tracer, interface, cell, nlev, ncells);
            const float flux = residual[qi];
            residual[qi] = mpas_add(mpas_mul(plus(flux), mpas_min(
                    scale_out[Q3(
                        tracer, interface - 1, cell, nlev, ncells)],
                    scale_in[Q3(tracer, interface, cell, nlev, ncells)])),
                mpas_mul(minus(flux), mpas_min(
                    scale_out[Q3(tracer, interface, cell, nlev, ncells)],
                    scale_in[Q3(
                        tracer, interface - 1, cell, nlev, ncells)])));
        }
    }
}

extern "C" __global__ void fct_finish(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *area_cell, const float *rdzw,
    const float *horizontal_residual, const float *vertical_residual,
    const float *target_density, const float *mass, float *output)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float inverse_area = 1.0f / area_cell[cell];
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
            _CACHE = KernelCache(
                capability=require_cuda(min_compute=(12, 0)),
            )
        selected = _CACHE
    # Object identity must remain live in the key; a recycled numeric id can
    # bind a fresh certification cache to an old kernel/manifest owner.
    key = (selected, name)
    if key not in _KERNELS:
        _KERNELS[key] = selected.raw_kernel(
            name,
            _CUDA_SOURCE,
            module_key="mpas_port.cuda_transport",
        )
    return _KERNELS[key]


def _launch(
    name: str,
    count: int,
    arguments: tuple[Any, ...],
    kernel_cache: Any | None = None,
) -> None:
    threads = 128
    _kernel(name, kernel_cache)(
        ((count + threads - 1) // threads,), (threads,), arguments
    )


def _mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    camel = {
        "n_edges_on_cell": "nEdgesOnCell",
        "edges_on_cell": "edgesOnCell",
        "cells_on_cell": "cellsOnCell",
        "cells_on_edge": "cellsOnEdge",
        "edge_sign_on_cell": "edgeSignOnCell",
        "dv_edge": "dvEdge",
        "area_cell": "areaCell",
    }.get(name)
    if camel is not None and hasattr(mesh, camel):
        return getattr(mesh, camel)
    arrays = getattr(mesh, "arrays", None)
    if arrays is not None:
        for key in (name, camel):
            if key is not None and key in arrays:
                return arrays[key]
    raise AttributeError(f"device mesh has no field {name!r}")


def _f32(value: Any, name: str) -> Any:
    from .cuda_backend import require_resident_array

    return require_resident_array(name, value, dtype=np.float32)


def _i32(value: Any) -> Any:
    from .cuda_backend import require_resident_array

    return require_resident_array("connectivity", value, dtype=np.int32)


@dataclass(frozen=True, slots=True)
class CudaAdvectionCoefficients:
    adv_coefs: Any
    adv_coefs_3rd: Any
    n_adv_cells_for_edge: Any
    adv_cells_for_edge: Any
    h2d: TransferStats
    horizontal_order: int = 3

    @classmethod
    def from_host(cls, coefficients: Any) -> "CudaAdvectionCoefficients":
        cp = _cupy()
        if int(coefficients.horizontal_order) != 3:
            raise ConfigurationRefusal(
                "config_scalar_adv_order",
                coefficients.horizontal_order,
                "the first CUDA transport kernel admits the frozen order-three path",
                "config_scalar_adv_order=3",
            )
        host_float = {
            "adv_coefs": np.ascontiguousarray(
                np.asarray(coefficients.adv_coefs, dtype=np.float32)
            ),
            "adv_coefs_3rd": np.ascontiguousarray(
                np.asarray(coefficients.adv_coefs_3rd, dtype=np.float32)
            ),
        }
        host_index = {
            "n_adv_cells_for_edge": np.ascontiguousarray(
                np.asarray(coefficients.n_adv_cells_for_edge, dtype=np.int32)
            ),
            "adv_cells_for_edge": np.ascontiguousarray(
                np.asarray(coefficients.adv_cells_for_edge, dtype=np.int32)
            ),
        }
        if host_float["adv_coefs"].ndim != 2:
            raise ValueError("adv_coefs must have shape (nEdges, stencilWidth)")
        nedges, width = host_float["adv_coefs"].shape
        expected = (nedges, width)
        if host_float["adv_coefs_3rd"].shape != expected:
            raise ValueError("adv_coefs_3rd shape differs from adv_coefs")
        if host_index["n_adv_cells_for_edge"].shape != (nedges,):
            raise ValueError("n_adv_cells_for_edge must have shape (nEdges,)")
        if host_index["adv_cells_for_edge"].shape != expected:
            raise ValueError("adv_cells_for_edge shape differs from adv_coefs")
        if not all(np.all(np.isfinite(value)) for value in host_float.values()):
            raise ValueError("advection coefficients contain non-finite values")
        started = time.perf_counter()
        device = {
            **{name: cp.asarray(value) for name, value in host_float.items()},
            **{name: cp.asarray(value) for name, value in host_index.items()},
        }
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        result = cls(
            **device,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        shape = tuple(self.adv_coefs.shape)
        if len(shape) != 2:
            raise ValueError("adv_coefs must have shape (nEdges, stencilWidth)")
        require_resident_array(
            "advection.adv_coefs", self.adv_coefs, dtype=np.float32, shape=shape
        )
        require_resident_array(
            "advection.adv_coefs_3rd",
            self.adv_coefs_3rd,
            dtype=np.float32,
            shape=shape,
        )
        require_resident_array(
            "advection.n_adv_cells_for_edge",
            self.n_adv_cells_for_edge,
            dtype=np.int32,
            shape=(shape[0],),
        )
        require_resident_array(
            "advection.adv_cells_for_edge",
            self.adv_cells_for_edge,
            dtype=np.int32,
            shape=shape,
        )


@dataclass(frozen=True, slots=True)
class CudaScalarTransportResult:
    scalars: Any
    density: Any


def _validate_orders(horizontal: int, vertical: int, coefficient: float) -> None:
    if horizontal != 3:
        raise ConfigurationRefusal(
            "config_scalar_adv_order",
            horizontal,
            "the CUDA scalar path currently transcribes order three",
            "config_scalar_adv_order=3",
        )
    if vertical != 3:
        raise ConfigurationRefusal(
            "config_scalar_vadv_order",
            vertical,
            "the CUDA scalar path currently transcribes order three",
            "config_scalar_vadv_order=3",
        )
    if not np.isfinite(coefficient) or coefficient < 0.0 or coefficient > 1.0:
        raise ConfigurationRefusal(
            "config_coef_3rd_order",
            coefficient,
            "the frozen coefficient range is zero through one",
            "config_coef_3rd_order=0.25",
        )


def _inputs(
    mesh: Any,
    scalar_old: Any,
    scalar_stage: Any,
    rho_old: Any,
    rho_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
) -> tuple[Any, ...]:
    old = _f32(scalar_old, "scalar_old")
    stage = _f32(scalar_stage, "scalar_stage")
    squeezed = old.ndim == 2
    if squeezed:
        old = old[None, ...]
        stage = stage[None, ...]
    if old.ndim != 3 or old.shape != stage.shape:
        raise ValueError("scalar_old/stage must share (nTracers,nLevels,nCells)")
    ntracers, nlev, ncells = map(int, old.shape)
    horizontal = _f32(uh_avg, "uh_avg")
    vertical = _f32(ww_avg, "ww_avg")
    if horizontal.ndim != 2 or horizontal.shape[0] != nlev:
        raise ValueError("uh_avg must have shape (nLevels,nEdges)")
    nedges = int(horizontal.shape[1])
    if vertical.shape != (nlev + 1, ncells):
        raise ValueError("ww_avg must have shape (nLevels+1,nCells)")
    rho0 = _f32(rho_old, "rho_zz_old")
    rho1 = _f32(rho_new, "rho_zz_new")
    if rho0.shape != (nlev, ncells) or rho1.shape != rho0.shape:
        raise ValueError("rho_zz arrays disagree with scalar shape")
    return (
        old,
        stage,
        squeezed,
        rho0,
        rho1,
        horizontal,
        vertical,
        _f32(fzm, "fzm"),
        _f32(fzp, "fzp"),
        _f32(rdzw, "rdzw"),
        ntracers,
        nlev,
        ncells,
        nedges,
    )


def _geometry(mesh: Any) -> tuple[Any, ...]:
    counts = _i32(_mesh_value(mesh, "n_edges_on_cell"))
    edges = _i32(_mesh_value(mesh, "edges_on_cell"))
    neighbors = _i32(_mesh_value(mesh, "cells_on_cell"))
    cells_on_edge = _i32(_mesh_value(mesh, "cells_on_edge"))
    signs = _f32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    return (
        counts,
        edges,
        neighbors,
        cells_on_edge,
        signs,
        _f32(_mesh_value(mesh, "dv_edge"), "dv_edge"),
        _f32(_mesh_value(mesh, "area_cell"), "area_cell"),
        int(edges.shape[1]),
    )


def _high_fluxes(
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
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(width),
            np.float32(coefficient),
            stage,
            velocity,
            coefficients.adv_coefs,
            coefficients.adv_coefs_3rd,
            coefficients.n_adv_cells_for_edge,
            coefficients.adv_cells_for_edge,
            edge,
        ),
        kernel_cache,
    )
    _launch(
        "transport_vertical_flux",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.float32(coefficient),
            stage,
            vertical_velocity,
            fzm,
            fzp,
            vertical,
        ),
        kernel_cache,
    )
    return edge, vertical


def advance_scalars_cuda(
    mesh: Any,
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
    rk_step: int = 3,
    config_time_integration_order: int = 3,
    config_scalar_adv_order: int = 3,
    config_scalar_vadv_order: int = 3,
    config_coef_3rd_order: float = 0.25,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    kernel_cache: Any | None = None,
) -> CudaScalarTransportResult:
    """Unrestricted third-order atmosphere scalar RK update."""

    _validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order, config_coef_3rd_order
    )
    if rk_step not in (1, 2, 3):
        raise ValueError("rk_step must be 1, 2, or 3")
    if config_time_integration_order != 3:
        raise ConfigurationRefusal(
            "config_time_integration_order",
            config_time_integration_order,
            "the first CUDA whole-step path is RK3",
            "config_time_integration_order=3",
        )
    (
        old,
        stage,
        squeezed,
        rho0,
        rho1,
        velocity,
        vertical_velocity,
        fnm,
        fnp,
        rdnw,
        ntracers,
        nlev,
        ncells,
        nedges,
    ) = _inputs(
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
    cp = _cupy()
    if scalar_tendency is None:
        source = cp.zeros_like(old)
    else:
        source = _f32(scalar_tendency, "scalar_tendency")
        if source.ndim == 2:
            source = source[None, ...]
        if source.shape != old.shape:
            raise ValueError("scalar_tendency shape differs from scalar_old")
    counts, edges, _, cells_on_edge, signs, dv, area, max_edges = _geometry(mesh)
    edge_values, vertical_flux = _high_fluxes(
        stage,
        velocity,
        vertical_velocity,
        coefficients,
        fnm,
        fnp,
        config_coef_3rd_order,
        ntracers,
        nlev,
        ncells,
        nedges,
        kernel_cache,
    )
    if advance_density:
        weight = np.float32(1.0 / 3.0 if rk_step == 1 else 0.5 if rk_step == 2 else 1.0)
        target = (np.float32(1.0) - weight) * rho0 + weight * rho1
    else:
        target = cp.array(rho1, copy=True)
    output = cp.empty_like(old)
    _launch(
        "transport_standard_finish",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            np.float32(dt),
            counts,
            edges,
            signs,
            area,
            velocity,
            edge_values,
            vertical_flux,
            rdnw,
            old,
            rho0,
            target,
            source,
            output,
        ),
        kernel_cache,
    )
    return CudaScalarTransportResult(output[0] if squeezed else output, target)


def advance_scalars_monotonic_cuda(
    mesh: Any,
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
    config_scalar_adv_order: int = 3,
    config_scalar_vadv_order: int = 3,
    config_coef_3rd_order: float = 0.25,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    n_halos: int = 3,
    kernel_cache: Any | None = None,
) -> CudaScalarTransportResult:
    """Stage-three Zalesak FCT update, fully resident on the device."""

    _validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order, config_coef_3rd_order
    )
    if n_halos < 3:
        raise ConfigurationRefusal(
            "nHalos", n_halos, "FCT requires three halo rows", "nHalos=3"
        )
    (
        old,
        stage,
        squeezed,
        rho0,
        rho1,
        velocity,
        vertical_velocity,
        fnm,
        fnp,
        rdnw,
        ntracers,
        nlev,
        ncells,
        nedges,
    ) = _inputs(
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
    cp = _cupy()
    source = (
        cp.zeros_like(old)
        if scalar_tendency is None
        else _f32(scalar_tendency, "scalar_tendency")
    )
    if source.ndim == 2:
        source = source[None, ...]
    if source.shape != old.shape:
        raise ValueError("scalar_tendency shape differs from scalar_old")
    counts, edges, neighbors, cells_on_edge, signs, dv, area, max_edges = _geometry(
        mesh
    )
    target = cp.empty_like(rho0)
    _launch(
        "transport_target_density",
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            np.int32(bool(advance_density)),
            np.float32(dt),
            counts,
            edges,
            signs,
            dv,
            area,
            velocity,
            vertical_velocity,
            rdnw,
            rho0,
            rho1,
            target,
        ),
        kernel_cache,
    )
    edge_values, high_vertical = _high_fluxes(
        stage,
        velocity,
        vertical_velocity,
        coefficients,
        fnm,
        fnp,
        config_coef_3rd_order,
        ntracers,
        nlev,
        ncells,
        nedges,
        kernel_cache,
    )
    source_old = cp.empty_like(old)
    minimum = cp.empty_like(old)
    maximum = cp.empty_like(old)
    _launch(
        "fct_minmax_source",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(max_edges),
            np.float32(dt),
            counts,
            neighbors,
            old,
            source,
            rho0,
            source_old,
            minimum,
            maximum,
        ),
        kernel_cache,
    )
    mass = cp.empty_like(old)
    vertical_residual = cp.empty_like(high_vertical)
    scale_in = cp.empty_like(old)
    scale_out = cp.empty_like(old)
    _launch(
        "fct_vertical_low_order",
        ntracers * ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.float32(dt),
            source_old,
            rho0,
            vertical_velocity,
            rdnw,
            high_vertical,
            mass,
            vertical_residual,
            scale_in,
            scale_out,
        ),
        kernel_cache,
    )
    upwind = cp.empty_like(edge_values)
    horizontal_residual = cp.empty_like(edge_values)
    _launch(
        "fct_edge_residual",
        nedges,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.float32(dt),
            source_old,
            velocity,
            dv,
            cells_on_edge,
            edge_values,
            upwind,
            horizontal_residual,
        ),
        kernel_cache,
    )
    _launch(
        "fct_horizontal_low_order",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            counts,
            edges,
            signs,
            area,
            upwind,
            horizontal_residual,
            mass,
            scale_in,
            scale_out,
        ),
        kernel_cache,
    )
    _launch(
        "fct_scale",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            minimum,
            maximum,
            target,
            mass,
            scale_in,
            scale_out,
        ),
        kernel_cache,
    )
    _launch(
        "fct_limit_horizontal",
        nedges,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            cells_on_edge,
            scale_in,
            scale_out,
            horizontal_residual,
        ),
        kernel_cache,
    )
    _launch(
        "fct_limit_vertical",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            scale_in,
            scale_out,
            vertical_residual,
        ),
        kernel_cache,
    )
    output = cp.empty_like(old)
    _launch(
        "fct_finish",
        ncells,
        (
            np.int32(ntracers),
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            counts,
            edges,
            signs,
            area,
            rdnw,
            horizontal_residual,
            vertical_residual,
            target,
            mass,
            output,
        ),
        kernel_cache,
    )
    return CudaScalarTransportResult(output[0] if squeezed else output, target)


def advance_scalar_transport_cuda(
    mesh: Any,
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
    rk_step: int = 3,
    config_scalar_advection: bool = True,
    config_monotonic: bool = True,
    config_positive_definite: bool = False,
    config_split_dynamics_transport: bool = False,
    config_time_integration_order: int = 3,
    kernel_cache: Any | None = None,
    **kwargs: Any,
) -> CudaScalarTransportResult:
    """Frozen dispatcher for the admitted CUDA scalar path."""

    cp = _cupy()
    stage = _f32(scalar_stage, "scalar_stage")
    rho_new = _f32(rho_zz_new, "rho_zz_new")
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
        return advance_scalars_monotonic_cuda(
            mesh,
            scalar_old,
            scalar_stage,
            rho_zz_old,
            rho_zz_new,
            uh_avg,
            ww_avg,
            dt,
            **common,
        )
    return advance_scalars_cuda(
        mesh,
        scalar_old,
        scalar_stage,
        rho_zz_old,
        rho_zz_new,
        uh_avg,
        ww_avg,
        dt,
        rk_step=rk_step,
        config_time_integration_order=config_time_integration_order,
        **common,
    )


__all__ = [
    "CudaAdvectionCoefficients",
    "CudaScalarTransportResult",
    "advance_scalar_transport_cuda",
    "advance_scalars_cuda",
    "advance_scalars_monotonic_cuda",
]
