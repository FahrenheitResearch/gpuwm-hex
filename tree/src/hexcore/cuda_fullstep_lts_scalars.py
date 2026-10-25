"""Scalar transport across rate interfaces for active full-step LTS (one device).

The classes step fine first (``cuda_fullstep_lts_active``).  Every class runs
the UNCHANGED native scalar transport -- two predictor stages and the
flux-corrected third stage -- on its own sub-mesh at its own whole step.  What
this module adds is the interface law, as edits of the third stage's two edge
arrays (low-order flux and antidiffusive residual) at rate-interface edges
BEFORE the native low-order, scale, limit and finish kernels read them, so
both cells of an interface edge are moved by one number:

* MASS.  A class's transport reads the step-mean mass flux ``uhAvg``.  On an
  edge owned by the finer class that value is replaced by the mass the finer
  class actually moved through the edge over the interval (the FP64 register
  of the dynamics stage).  The class's target density is then the refluxed
  density to rounding, and a constant mixing ratio stays constant.

* PACKETS.  The finer class owns the edge.  In each of its steps the limited
  scalar flux through the edge (low-order plus limited residual, the two
  numbers its own cells are updated with) is added to an FP64 register.  When
  the coarser class steps over the interval, its low-order flux on that edge
  IS the register and its residual there is zero: the packet enters the
  coarse cell's provisional low-order mass before the cell's limiter scale
  factors are formed, so the native limiter treats it as already spent.

* THE RESERVE.  Fine first means a coarse column's donor budget is drawn by
  the finer class before the column's own step exists.  Each coarse column on
  an interface therefore carries a reserve: the scalar mass it held at its
  last accepted state.  The finer class may not move more than that out of
  the column through the interface between two of the column's accepted
  states (``lts_reserve_cap`` scales its interface fluxes when it would), and
  the reserve is debited with every accepted packet.  When the column steps,
  its own low-order outflow, horizontal and vertical, is reduced by what the
  packets took beyond the column's own upwind export (``lts_donor_guard``).
  Neither side can commit mass the other already spent, so the provisional
  mass of an interface column is never negative for a reason the interface
  made.

  Both safeguards are measured against the MASS-CONSISTENT export, the
  column's start mixing ratio times the dry mass that left it, and never
  against the content alone.  A level that exports more dry mass in one step
  than it holds (a thin level over steep terrain at a long class step does)
  exports its tracer with that mass in the native scheme too; scaling the
  tracer and not the mass there would break a constant mixing ratio.
  MEASURED 2026-10-11, 48 km mesh with a 3 km patch over the central United
  States, before this rule: the constant tracer ranged 0.896 to 1.093 after
  208 base steps while the winds were still calm.

The native kernels are reached without editing a pinned source: the driver
hands its own kernel cache to the transport module, which caches the resolved
kernel under ``(cache, name)``.  For a class driver's cache that entry is
replaced by a callable that runs a hook, then the same kernel.  Other drivers
have other caches and are not touched.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS

# planes of the scalar mailbox per level of finer owner: accepted scalars at
# the start, the middle and the end of the consumer's step
SCALAR_PLANES = 3
START, MID, END = 0, 1, 2
# the cap leaves the last few ulps of a reserve unspent so FP32 state
# arithmetic on the other side cannot overdraw it
_RESERVE_MARGIN = 1.0 - 2.0 ** -20

_SOURCE = CUDA_FTZ_HELPERS + r"""
#define LTS_LOOP(total) \
    for (long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x; \
         element < (long long)(total); element += (long long)gridDim.x * blockDim.x)

/* Halo scalars from the mailbox.  spec[4*owner] = action (0 skip, 1 write
   unless carried, 2 write always), plane a, plane b; weight[2*owner] = wa, wb
   (wb == 0: plane a's bytes). */
extern "C" __global__ void lts_fill_scalar(
    const int ntr, const int nlev, const int n, const int nloc, const int nmail, const int first,
    const int *mail, const unsigned char *cls, const int *spec, const float *weight,
    const float *G, float *local)
{
    const long long rows = (long long)ntr * nlev;
    LTS_LOOP(rows * n) {
        const int i = (int)(element % n);
        const long long row = element / n;
        const int owner = cls[i] & 31;
        const int action = spec[4 * owner];
        if (action == 0 || ((cls[i] & 64) && action != 2)) continue;
        const float wb = weight[2 * owner + 1];
        float value = G[((long long)spec[4 * owner + 1] * rows + row) * nmail + mail[i]];
        if (wb != 0.0f)
            value = weight[2 * owner] * value
                + wb * G[((long long)spec[4 * owner + 2] * rows + row) * nmail + mail[i]];
        local[row * nloc + first + i] = value;
    }
}

extern "C" __global__ void lts_publish_scalar(
    const int ntr, const int nlev, const int n, const int nloc, const int nmail,
    const int *loc, const int *mail, const int into, float *G, const float *local)
{
    const long long rows = (long long)ntr * nlev;
    LTS_LOOP(rows * n) {
        const int i = (int)(element % n);
        const long long row = element / n;
        G[((long long)into * rows + row) * nmail + mail[i]] = local[row * nloc + loc[i]];
    }
}

extern "C" __global__ void lts_scalar_plane_copy(
    const long long rows, const int n, const int nmail, const int *mail,
    const int from, const int into, float *G)
{
    LTS_LOOP(rows * n) {
        const long long row = element / n;
        const int i = (int)(element % n);
        G[((long long)into * rows + row) * nmail + mail[i]] =
            G[((long long)from * rows + row) * nmail + mail[i]];
    }
}

/* The step-mean mass flux on an edge the finer class owns: what that class
   moved through it over this class's step. */
extern "C" __global__ void lts_accept_mass_flux(
    const int nlev, const int n, const int nedges, const int nslots,
    const int *edge, const int *slot, const double inverse_seconds,
    const float *dv_edge, const double *reg_mass, float *flux)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const long long k = element / n;
        flux[k * nedges + edge[i]] = (float)(
            reg_mass[k * nslots + slot[i]] * inverse_seconds / (double)dv_edge[edge[i]]);
    }
}

/* Coarse side: the accepted packet is the edge's low-order flux; nothing of
   it is left for this class's limiter to scale. */
extern "C" __global__ void lts_packet_apply(
    const int ntr, const int nlev, const int n, const int nedges, const int nslots,
    const int *edge, const int *slot, const double *reg_scalar,
    float *upwind, float *residual)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int i = (int)(element % n);
        const long long row = element / n;
        const long long qe = row * nedges + edge[i];
        upwind[qe] = (float)reg_scalar[row * nslots + slot[i]];
        residual[qe] = 0.0f;
    }
}

/* Coarse side, one thread per (tracer, interface column).  The packets are
   spent; the column's own low-order outflow -- through its edges that are
   not packets, and up and down -- is what is left to adjust.  Counting no
   inflow (inflows are other donors' to scale), the level is overdrawn by
       need   = packets out + own outflow - content,
   and it would have been overdrawn by
       native = q_start * (dry mass out through the packet edges) + own outflow - content
   had the packets carried the column's own mixing ratio, which is what the
   native upwind flux does.  Only the difference is the interface's doing,
   and the own outflow is reduced by exactly that much, each flux by one
   factor together with its residual.  Each flux has one donor, so one thread
   writes it.
   stats: [0] levels scaled, [1] levels whose own outflow could not cover it.
   amounts[3*tracer + 2]: the outflow withheld, in packet units. */
extern "C" __global__ void lts_donor_guard(
    const int ntr, const int nlev, const int n, const int width, const int ncells,
    const int nedges, const int nslots, const int max_edges, const float dt,
    const int *cell, const int *slot, const double *coef, const double *reg_scalar,
    const double *reg_mass,
    const int *n_edges_on_cell, const int *edges_on_cell, const float *acoustic_sign,
    const float *inv_area_cell, const unsigned char *packet_edge,
    const float *source_old, const float *rho_old, const float *ww, const float *rdzw,
    float *upwind, float *residual, float *mass, float *vertical_residual,
    unsigned long long *stats, double *amounts)
{
    LTS_LOOP((long long)ntr * n) {
        const int row = (int)(element % n);
        const long long tracer = element / n;
        const int c = cell[row];
        const double inverse_area = (double)inv_area_cell[c];
        const int count = n_edges_on_cell[c];
        for (int k = 0; k < nlev; ++k) {
            const long long tk = tracer * nlev + k;
            const long long q = tk * ncells + c;
            const double value = (double)source_old[q];
            const double content = value * (double)rho_old[(long long)k * ncells + c];
            double exported = 0.0, air_out = 0.0;
            for (int w = 0; w < width; ++w) {
                const double outward = -coef[row * width + w];
                const int s = slot[row * width + w];
                exported += outward * reg_scalar[tk * nslots + s];
                air_out += fmax(outward * reg_mass[(long long)k * nslots + s], 0.0);
            }
            const double up = (k + 1 < nlev) ? fmax((double)ww[(long long)(k + 1) * ncells + c], 0.0) : 0.0;
            const double down = (k > 0) ? fmin((double)ww[(long long)k * ncells + c], 0.0) : 0.0;
            double outflow = fmax((double)rdzw[k] * (double)dt * (up - down) * value, 0.0);
            for (int s = 0; s < count; ++s) {
                const int edge = edges_on_cell[c * max_edges + s];
                if (packet_edge[edge]) continue;
                const double inward = -(double)acoustic_sign[c * max_edges + s]
                    * (double)upwind[tk * nedges + edge] * inverse_area;
                if (inward < 0.0) outflow -= inward;
            }
            const double need = exported + outflow - content;
            if (need <= 0.0) continue;
            const double native = fmax(fmax(value, 0.0) * air_out + outflow - content, 0.0);
            const double reduce = need - native;
            /* where the native scheme is itself past the content, rounding
               between a scalar packet and a mass packet is not an overdraw */
            if (reduce <= (native > 0.0 ? 4.0e-6 * (fabs(exported) + outflow + fabs(content)) : 0.0))
                continue;
            if (reduce > outflow) atomicAdd(&stats[1], 1ULL);
            if (outflow <= 0.0) continue;
            atomicAdd(&stats[0], 1ULL);
            const double keep = reduce < outflow ? (outflow - reduce) / outflow : 0.0;
            atomicAdd(&amounts[3 * tracer + 2], (1.0 - keep) * outflow / inverse_area);
            const float scale = (float)keep;
            for (int s = 0; s < count; ++s) {
                const int edge = edges_on_cell[c * max_edges + s];
                if (packet_edge[edge]) continue;
                const long long qe = tk * nedges + edge;
                if (-(double)acoustic_sign[c * max_edges + s] * (double)upwind[qe] < 0.0) {
                    upwind[qe] *= scale;
                    residual[qe] *= scale;
                }
            }
            if (up > 0.0 && value > 0.0) {
                const double removed = (1.0 - keep) * (double)dt * up * value;
                mass[q] = (float)((double)mass[q] + removed * (double)rdzw[k]);
                mass[q + ncells] = (float)((double)mass[q + ncells] - removed * (double)rdzw[k + 1]);
                vertical_residual[(tracer * (nlev + 1) + k + 1) * ncells + c] *= scale;
            }
            if (down < 0.0 && value > 0.0) {
                const double removed = (1.0 - keep) * (double)dt * down * value;   /* negative */
                mass[q - ncells] = (float)((double)mass[q - ncells] + removed * (double)rdzw[k - 1]);
                mass[q] = (float)((double)mass[q] - removed * (double)rdzw[k]);
                vertical_residual[(tracer * (nlev + 1) + k) * ncells + c] *= scale;
            }
        }
    }
}

/* Fine side, one thread per (tracer, level, coarse interface column): what
   this step may move out of the column through the interface is capped by
   the column's reserve.  When more dry mass has left the column through the
   interface than it held (air_out > air_start), the allowance grows with it:
   the column's start mixing ratio times that mass, the native upwind export.
   The low-order outflow is scaled first; then the outward residuals share
   what is left (the limiter can only shrink them).
   Also keeps the low-order flux of every interface edge for the capture.
   stats: [2] low-order capped, [3] residual capped.
   amounts[3*tracer + 1]: the outflow withheld, in packet units. */
extern "C" __global__ void lts_reserve_cap(
    const int ntr, const int nlev, const int n, const int width, const int nedges,
    const int nslots, const int *slot, const double *coef, const int *edge_of_slot,
    const int enforce, const double margin, const double *reserve,
    const double *content_start, const double *air_start, const double *reg_mass,
    float *upwind, float *residual, float *hold, unsigned long long *stats, double *amounts)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        const long long k = tk % nlev;
        double low = 0.0, high = 0.0, air_out = 0.0;
        for (int w = 0; w < width; ++w) {
            const double c = coef[row * width + w];
            if (c == 0.0) continue;
            const double outward = c < 0.0 ? 1.0 : -1.0;
            const int s = slot[row * width + w];
            const long long qe = tk * nedges + edge_of_slot[s];
            low += fmax(outward * (double)upwind[qe], 0.0);
            high += fmax(outward * (double)residual[qe], 0.0);
            air_out += fmax(outward * reg_mass[k * nslots + s], 0.0);
        }
        double available = 1.0e300;
        if (enforce) {
            const double left = fmax(reserve[tk * n + row], 0.0);
            const double air = air_start[k * n + row];
            if (air_out > air && air > 0.0)
                available = (left + fmax(content_start[tk * n + row], 0.0) * (air_out / air - 1.0))
                    * (1.0 + 4.0e-6);
            else
                available = left * margin;
        }
        double keep_low = 1.0, keep_high = 1.0;
        if (low > available) {
            keep_low = available / low;
            keep_high = 0.0;
            atomicAdd(&stats[2], 1ULL);
        } else if (low + high > available) {
            keep_high = (available - low) / high;
            atomicAdd(&stats[3], 1ULL);
        }
        if (keep_high < 1.0)
            atomicAdd(&amounts[3 * (tk / nlev) + 1],
                      (1.0 - keep_low) * low + (1.0 - keep_high) * high);
        for (int w = 0; w < width; ++w) {
            const double c = coef[row * width + w];
            if (c == 0.0) continue;
            const double outward = c < 0.0 ? 1.0 : -1.0;
            const int s = slot[row * width + w];
            const long long qe = tk * nedges + edge_of_slot[s];
            if (keep_low < 1.0 && outward * (double)upwind[qe] > 0.0)
                upwind[qe] = (float)((double)upwind[qe] * keep_low);
            if (keep_high < 1.0 && outward * (double)residual[qe] > 0.0)
                residual[qe] = (float)((double)residual[qe] * keep_high);
            hold[tk * nslots + s] = upwind[qe];
        }
    }
}

/* Fine side, after the limiter: the accepted packet of every interface edge
   joins the register and leaves the coarse column's reserve.
   amounts[3*tracer]: the sum of |packet|, the throughput the withheld amounts
   are compared with. */
extern "C" __global__ void lts_packet_capture(
    const int ntr, const int nlev, const int n, const int width, const int nedges,
    const int nslots, const int *slot, const double *coef, const int *edge_of_slot,
    const float *hold, const float *residual, double *reg_scalar, double *reserve,
    double *amounts)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        double net = 0.0, moved = 0.0;
        for (int w = 0; w < width; ++w) {
            const double c = coef[row * width + w];
            if (c == 0.0) continue;
            const int s = slot[row * width + w];
            const double packet = (double)hold[tk * nslots + s]
                + (double)residual[tk * nedges + edge_of_slot[s]];
            reg_scalar[tk * nslots + s] += packet;
            net += (c < 0.0 ? packet : -packet);
            moved += fabs(packet);
        }
        reserve[tk * n + row] -= net;
        if (moved != 0.0) atomicAdd(&amounts[3 * (tk / nlev)], moved);
    }
}

/* A coarse interface column's reserve at its accepted state: area * rho * q,
   the units of an edge packet; and the dry mass area * rho it is a mixing
   ratio of. */
extern "C" __global__ void lts_reserve_set(
    const int ntr, const int nlev, const int n, const int ncells, const int *cell,
    const double *area, const float *rho, const float *scalars, double *reserve,
    double *content_start, double *air_start)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        const long long k = tk % nlev;
        const double air = area[row] * (double)rho[k * ncells + cell[row]];
        const double content = air * (double)scalars[tk * ncells + cell[row]];
        reserve[tk * n + row] = content;
        content_start[tk * n + row] = content;
        if (tk < nlev) air_start[k * n + row] = air;
    }
}

extern "C" __global__ void lts_zero_scalar(
    const int ntr, const int nlev, const int n, const int nslots, const int *slot, double *reg)
{
    LTS_LOOP((long long)ntr * nlev * n)
        reg[(element / n) * (long long)nslots + slot[element % n]] = 0.0;
}

/* What the native finish clamps: the third stage ends with max(0, value /
   target).  This repeats fct_finish_v841's sum for owned cells and adds up
   the mass a negative value would have carried, per tracer, so a change in a
   tracer total can be split into clamped mass and rounding. */
extern "C" __global__ void lts_clamp_audit(
    const int ntr, const int nlev, const int nowned, const int ncells, const int nedges,
    const int max_edges, const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *horizontal_residual, const float *vertical_residual, const float *mass,
    const double *weight, double *created, unsigned long long *count)
{
    LTS_LOOP((long long)ntr * nlev * nowned) {
        const int cell = (int)(element % nowned);
        const long long tk = element / nowned;
        const int k = (int)(tk % nlev);
        const int tracer = (int)(tk / nlev);
        const float inverse_area = inv_area_cell[cell];
        float value = mass[tk * ncells + cell];
        for (int s = 0; s < n_edges_on_cell[cell]; ++s) {
            const int edge = edges_on_cell[cell * max_edges + s];
            const float sign = -acoustic_sign[cell * max_edges + s];
            value = mpas_add(value, mpas_mul(mpas_mul(sign,
                horizontal_residual[tk * nedges + edge]), inverse_area));
        }
        const long long qi = ((long long)tracer * (nlev + 1) + k) * ncells + cell;
        value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
            vertical_residual[qi + ncells], vertical_residual[qi])));
        if (value < 0.0f) {
            atomicAdd(&created[tracer], -(double)value * weight[(long long)k * nowned + cell]);
            atomicAdd(&count[tracer], 1ULL);
        }
    }
}
"""

_KERNEL_NAMES = (
    "lts_fill_scalar", "lts_publish_scalar", "lts_scalar_plane_copy", "lts_accept_mass_flux",
    "lts_packet_apply", "lts_donor_guard", "lts_reserve_cap", "lts_packet_capture",
    "lts_reserve_set", "lts_zero_scalar", "lts_clamp_audit")

_LOW_ORDER = ("fct_horizontal_low_order_v841", "fct_horizontal_low_order_scale_elem_v841")
_FINISH = ("fct_finish_v841", "fct_finish_elem_v841")
STAT_NAMES = ("donor_guard_levels", "packets_overdrew_levels", "reserve_capped_low_order",
              "reserve_capped_residual")
AMOUNT_NAMES = ("packet_throughput", "withheld_by_reserve_cap", "withheld_by_donor_guard")


def tap_kernel(module: Any, cache: Any, name: str, *, before: Any = None, after: Any = None,
               dimensions: tuple[tuple[str, int], ...] = ()) -> None:
    """Run ``before(args)`` / ``after(args)`` around one native kernel of one driver's cache."""

    table = getattr(module, "_KERNELS", None)
    resolve = getattr(module, "_kernel", None)
    if not isinstance(table, dict) or resolve is None:
        raise RuntimeError(
            f"{module.__name__} no longer resolves kernels through _kernel/_KERNELS; the "
            "interface hooks would not run and scalars would cross a rate interface with "
            "two different fluxes")
    kernel = resolve(name, cache, dimensions) if dimensions else resolve(name, cache)
    if getattr(kernel, "lts_tapped", False):
        raise RuntimeError(f"{name} is already tapped for this kernel cache")

    def tapped(grid: Any, block: Any, arguments: tuple[Any, ...], **keywords: Any) -> Any:
        if before is not None:
            before(arguments)
        result = kernel(grid, block, arguments, **keywords)
        if after is not None:
            after(arguments)
        return result

    tapped.lts_tapped = True
    key = (cache, name, dimensions) if dimensions else (cache, name)
    table[key] = tapped


def tap_regional_launch(kernels: Any, name: str, *, before: Any = None, after: Any = None,
                        drop_tail: int = 0) -> None:
    """Run ``before(args)`` / ``after(args)`` around one kernel of a limited-area runtime.

    A limited-area driver launches its specified-zone-aware kernels through its
    own ``CudaRegionalKernels.launch`` (``cuda_regional_v841``), never through
    the global modules' ``_kernel`` table, so a ``tap_kernel`` on the global
    kernel never fires there.  ``drop_tail`` trailing arguments (the regional
    kernels' refusal flag) are cut from the view the hooks see, so a hook
    written for the global kernel's argument list reads the same positions.
    """

    launch = kernels.launch
    tapped_names = getattr(kernels, "lts_tapped_names", frozenset())
    if name in tapped_names:
        raise RuntimeError(f"{name} is already tapped for this regional runtime")

    def tapped(kernel_name: str, count: int, args: tuple[Any, ...]) -> None:
        if kernel_name != name or int(count) < 1:
            return launch(kernel_name, count, args)
        view = args[:len(args) - drop_tail] if drop_tail else args
        if before is not None:
            before(view)
        launch(kernel_name, count, args)
        if after is not None:
            after(view)

    kernels.launch = tapped
    kernels.lts_tapped_names = frozenset(tapped_names | {name})


def tap_acoustic_momentum(driver: Any, *, before: Any, after: Any,
                          dimensions: tuple[tuple[str, int], ...]) -> None:
    """Tap the acoustic normal-momentum update this driver actually launches.

    Global and VR drivers launch ``acoustic_ru_v841``; a limited-area driver
    (``driver.regional_v841`` set) launches ``acoustic_ru_regional_v841``
    instead (the specified-zone pgrad mask), whose arguments are the global
    kernel's with ``spec_zone_mask_edge`` inserted before ``zz`` and the
    ``invalid`` flag appended.  The hooks read only ``nedges`` (1),
    ``small_step`` (3) and the trailing ``ru_p``, ``ru_avg``, so dropping the
    flag gives them the same view.  Breakage without the regional arm: the
    matched interface flux was never imposed on a limited-area mesh with two
    rate classes and the first class step stopped with "class 0 imposed the
    child's interface flux 0 times" (hex 0.4.4, 2026-10-24).
    """

    regional = getattr(driver, "regional_v841", None)
    if regional is None:
        from . import cuda_acoustic_v841 as acoustic_module

        tap_kernel(acoustic_module, driver.cache, "acoustic_ru_v841", before=before, after=after,
                   dimensions=dimensions)
        return
    tap_regional_launch(regional.kernels, "acoustic_ru_regional_v841", before=before, after=after,
                        drop_tail=1)


class ClampAudit:
    """Sum the mass the native third stage clamps, for any driver (fixed arm included)."""

    def __init__(self, driver: Any, weight: Any, n_owned: int, kernel_cache: Any) -> None:
        import cupy as cp
        from . import cuda_transport_v841 as transport

        self.cp = cp
        self.n_owned = int(n_owned)
        self.weight = weight
        self.ntr = int(driver.atmosphere.state.scalars.shape[0])
        self.created = cp.zeros(self.ntr, dtype=cp.float64)
        self.count = cp.zeros(self.ntr, dtype=cp.uint64)
        self._kernel = kernel_cache.raw_kernel(
            "lts_clamp_audit", _SOURCE, module_key="hexcore.cuda_fullstep_lts_scalars")
        self.tapped = False
        if type(self) is ClampAudit:
            for name in _FINISH:
                tap_kernel(transport, driver.cache, name, before=self.observe)
            self.tapped = True

    def observe(self, arguments: tuple[Any, ...]) -> None:
        i32 = np.int32
        ntr, nlev, ncells, nedges, max_edges = (int(v) for v in arguments[:5])
        total = ntr * nlev * self.n_owned
        self._kernel(((total + 255) // 256,), (256,), (
            i32(ntr), i32(nlev), i32(self.n_owned), i32(ncells), i32(nedges), i32(max_edges),
            arguments[5], arguments[6], arguments[7], arguments[8], arguments[9],
            arguments[10], arguments[11], arguments[13], self.weight, self.created, self.count))

    def read(self) -> dict[str, list[float]]:
        cp = self.cp
        return {"clamped_mass": [float(v) for v in cp.asnumpy(self.created)],
                "clamped_values": [int(v) for v in cp.asnumpy(self.count)]}


class _ClassScalars:
    """One class's hooks and device tables."""

    def __init__(self, owner: "ScalarInterfaces", klass: Any, area_owned: np.ndarray) -> None:
        cp = owner.cp
        self.owner = owner
        self.klass = klass
        tables = klass.tables
        nslots = owner.nslots
        # fine side: this class owns edges shared with the next coarser class
        edge_of_slot = np.full(nslots, -1, dtype=np.int32)
        edge_of_slot[np.asarray(tables.fine_slot, dtype=np.int64)] = tables.fine_edge_local
        self.edge_of_slot = cp.asarray(edge_of_slot)
        self.has_fine_side = bool(np.asarray(tables.fine_slot).size)
        # coarse side: edges the next finer class owns, and this class's cells on them
        self.has_coarse_side = bool(np.asarray(tables.coarse_slot).size)
        packet_edge = np.zeros(max(klass.edge_stride, 1), dtype=np.uint8)
        packet_edge[np.asarray(tables.coarse_edge_local, dtype=np.int64)] = 1
        self.packet_edge = cp.asarray(packet_edge)
        rows = int(np.asarray(tables.reflux_cell_local).size)
        self.rows = rows
        self.reserve = cp.zeros((owner.ntr, owner.nlev, max(rows, 1)), dtype=cp.float64)
        self.content_start = cp.zeros_like(self.reserve)
        self.air_start = cp.zeros((owner.nlev, max(rows, 1)), dtype=cp.float64)
        self.row_area = cp.asarray(np.ascontiguousarray(
            np.asarray(area_owned, dtype=np.float64)[np.asarray(tables.reflux_cell_local, dtype=np.int64)]))
        self.stats = cp.zeros(len(STAT_NAMES), dtype=cp.uint64)
        # per tracer: |packet| through this class's fine-side edges, and what
        # the cap (as the finer class) and the guard (as the coarser) withheld
        self.amounts = cp.zeros(3 * owner.ntr, dtype=cp.float64)
        self.expect_mass_flux = False
        self.step_seconds = 0.0
        self.vertical: tuple[Any, ...] | None = None
        self.audit: ClampAudit | None = None

    # -- hooks -----------------------------------------------------------------
    def on_split_flux(self, arguments: tuple[Any, ...]) -> None:
        klass, owner = self.klass, self.owner
        result = arguments[4]
        if (not self.expect_mass_flux or int(arguments[1]) != klass.edge_stride
                or int(result.shape[0]) != owner.nlev):
            return
        self.expect_mass_flux = False
        count = int(klass.coarse_edge.size)
        if count:
            i32 = np.int32
            owner.run("lts_accept_mass_flux", owner.nlev * count, (
                i32(owner.nlev), i32(count), i32(klass.edge_stride), i32(owner.nslots),
                klass.coarse_edge, klass.coarse_slot, np.float64(1.0 / self.step_seconds),
                klass.driver.atmosphere.mesh.dv_edge, owner.active.reg_mass_fine, result))

    def on_vertical(self, arguments: tuple[Any, ...]) -> None:
        # dt, source_old, rho_old, ww, rdzw, vertical_residual
        self.vertical = (arguments[3], arguments[4], arguments[5], arguments[6], arguments[7],
                         arguments[10])

    def on_low_order(self, arguments: tuple[Any, ...]) -> None:
        klass, owner = self.klass, self.owner
        i32 = np.int32
        ntr, nlev, ncells, nedges, max_edges = (int(v) for v in arguments[:5])
        if ntr != owner.ntr or nlev != owner.nlev or ncells != klass.cell_stride:
            raise RuntimeError("the flux-corrected stage ran on a shape the interface tables do not describe")
        counts, edges, signs, inverse_area, upwind, residual, mass = arguments[5:12]
        if self.vertical is None:
            raise RuntimeError("the vertical low-order launch was not seen before the horizontal one; "
                               "the donor guard would read another step's fluxes")
        dt, source_old, rho_old, ww, rdzw, vertical_residual = self.vertical
        self.vertical = None
        if self.has_coarse_side and owner.packets:
            count = int(klass.coarse_edge.size)
            owner.run("lts_packet_apply", ntr * nlev * count, (
                i32(ntr), i32(nlev), i32(count), i32(nedges), i32(owner.nslots),
                klass.coarse_edge, klass.coarse_slot, owner.reg_scalar, upwind, residual))
            if owner.guard:
                owner.run("lts_donor_guard", ntr * self.rows, (
                    i32(ntr), i32(nlev), i32(self.rows), i32(klass.reflux_width), i32(ncells),
                    i32(nedges), i32(owner.nslots), i32(max_edges), dt,
                    klass.reflux_cell, klass.reflux_slot, klass.reflux_coef, owner.reg_scalar,
                    owner.active.reg_mass_fine,
                    counts, edges, signs, inverse_area, self.packet_edge,
                    source_old, rho_old, ww, rdzw, upwind, residual, mass, vertical_residual,
                    self.stats, self.amounts))
        if self.has_fine_side:
            parent = owner.classes[klass.index - 1]
            pk = parent.klass
            owner.run("lts_reserve_cap", ntr * nlev * parent.rows, (
                i32(ntr), i32(nlev), i32(parent.rows), i32(pk.reflux_width), i32(nedges),
                i32(owner.nslots), pk.reflux_slot, pk.reflux_coef, self.edge_of_slot,
                i32(1 if owner.cap else 0), np.float64(_RESERVE_MARGIN),
                parent.reserve, parent.content_start, parent.air_start, owner.active.reg_mass_fine,
                upwind, residual, owner.hold, self.stats, self.amounts))

    def on_finish(self, arguments: tuple[Any, ...]) -> None:
        klass, owner = self.klass, self.owner
        i32 = np.int32
        ntr, nlev, _ncells, nedges = (int(v) for v in arguments[:4])
        if self.has_fine_side:
            parent = owner.classes[klass.index - 1]
            pk = parent.klass
            owner.run("lts_packet_capture", ntr * nlev * parent.rows, (
                i32(ntr), i32(nlev), i32(parent.rows), i32(pk.reflux_width), i32(nedges),
                i32(owner.nslots), pk.reflux_slot, pk.reflux_coef, self.edge_of_slot,
                owner.hold, arguments[10], owner.reg_scalar, parent.reserve, self.amounts))
        if self.audit is not None:
            self.audit.observe(arguments)


class ScalarInterfaces:
    """Scalar mailbox, packet register and reserves of one ``FullStepLtsActive``."""

    def __init__(self, active: Any, kernel_cache: Any, area_cell: Sequence[np.ndarray], *,
                 audit: bool = False) -> None:
        from . import cuda_dynamics_v841 as dynamics
        from . import cuda_transport_v841 as transport

        cp = active.cp
        self.cp = cp
        self.active = active
        plan = active.plan
        self.nlev = int(active.nlev)
        self.ntr = int(active.classes[0].driver.atmosphere.state.scalars.shape[0])
        self.nslots = int(active.reg_mass_fine.shape[1])
        self.packets = True    # False: diagnostic arm, the coarse side applies its own flux (does not conserve)
        self.guard = True      # False: diagnostic arm, the coarse column may be overdrawn
        self.cap = True        # False: diagnostic arm, the finer class may overdraw a reserve
        self._kernels = {name: kernel_cache.raw_kernel(
            name, _SOURCE, module_key="hexcore.cuda_fullstep_lts_scalars") for name in _KERNEL_NAMES}
        self.depth = int(plan.level_depth)
        self.truth_plane = SCALAR_PLANES * self.depth
        self.n_mail = max(int(plan.n_mail_cells), 1)
        self.G = cp.zeros((self.truth_plane + 1, self.ntr, self.nlev, self.n_mail), dtype=cp.float32)
        self.reg_scalar = cp.zeros((self.ntr, self.nlev, self.nslots), dtype=cp.float64)
        self.hold = cp.zeros((self.ntr, self.nlev, self.nslots), dtype=cp.float32)
        self.classes: list[_ClassScalars] = []
        for klass, area in zip(active.classes, area_cell):
            mine = _ClassScalars(self, klass, np.asarray(area)[:klass.n_owned_cells])
            self.classes.append(mine)
            cache = klass.driver.cache
            tap_kernel(dynamics, cache, "split_flux_finish_v841_f32", after=mine.on_split_flux)
            # identity-extend 4688aaa renamed the v841 vertical low-order kernel (native
            # upwind order); tapping the old name left on_vertical unseen and every LTS
            # step with scalars stopped "the vertical low-order launch was not seen".
            tap_kernel(transport, cache, "fct_vertical_low_order_v841", before=mine.on_vertical)
            for name in _LOW_ORDER:
                tap_kernel(transport, cache, name, before=mine.on_low_order)
            for name in _FINISH:
                tap_kernel(transport, cache, name, before=mine.on_finish)
            if audit:
                mine.audit = _UntappedAudit(klass.driver, klass.mass_weight, klass.n_owned_cells, kernel_cache)
        for mine in self.classes:
            scalars = mine.klass.driver.atmosphere.state.scalars
            self.publish(mine.klass, scalars, self.truth_plane)
            for level in range(self.depth):
                self.publish(mine.klass, scalars, level * SCALAR_PLANES + START)
            self.reserve_set(mine)

    def run(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total > 0:
            self._kernels[name](((int(total) + 255) // 256,), (256,), args)

    # -- mailbox -----------------------------------------------------------------
    def fill(self, klass: Any, scalars: Any, spec: Any, weight: Any) -> None:
        i32 = np.int32
        cells = klass.n_local_cells - klass.n_owned_cells
        self.run("lts_fill_scalar", self.ntr * self.nlev * cells, (
            i32(self.ntr), i32(self.nlev), i32(cells), i32(klass.cell_stride), i32(self.n_mail),
            i32(klass.n_owned_cells), klass.halo_cell_mail, klass.halo_cell_class, spec, weight,
            self.G, scalars))

    def publish(self, klass: Any, scalars: Any, plane: int) -> None:
        i32 = np.int32
        cells = int(klass.pub_cell_local.size)
        self.run("lts_publish_scalar", self.ntr * self.nlev * cells, (
            i32(self.ntr), i32(self.nlev), i32(cells), i32(klass.cell_stride), i32(self.n_mail),
            klass.pub_cell_local, klass.pub_cell_mail, i32(plane), self.G, scalars))

    def publish_step_end(self, klass: Any, tick: int) -> None:
        """A class's scalars are accepted at its step ends only: the consumer's middle or end."""

        from .fullstep_lts_classes import trajectory_plane

        active = self.active
        scalars = klass.driver.atmosphere.state.scalars
        planes = active.planes_per_step      # 3 N; N even, so the middle is a plane
        for consumer in range(max(0, klass.index - self.depth), klass.index):
            plane = trajectory_plane(active.class_tick[consumer], active.effective[consumer], tick, planes)
            if plane in (planes // 2, planes):
                self.publish(klass, scalars, (klass.index - consumer - 1) * SCALAR_PLANES
                             + (MID if plane == planes // 2 else END))

    def rotate(self, consumer: Any) -> None:
        """The consumer finished its step: its finer owners' end scalars are its next start scalars."""

        rows = self.ntr * self.nlev
        for distance in range(1, self.depth + 1):
            provider = consumer.index + distance
            if provider >= len(self.classes):
                break
            base = (distance - 1) * SCALAR_PLANES
            mail = self.classes[provider].klass.pub_cell_mail
            count = int(mail.size)
            self.run("lts_scalar_plane_copy", rows * count, (
                np.int64(rows), np.int32(count), np.int32(self.n_mail), mail,
                np.int32(base + END), np.int32(base + START), self.G))

    # -- registers and reserves -----------------------------------------------------
    def begin_step(self, klass: Any, seconds: float) -> None:
        mine = self.classes[klass.index]
        mine.expect_mass_flux = True
        mine.step_seconds = float(seconds)
        mine.vertical = None

    def end_step(self, klass: Any) -> None:
        if self.classes[klass.index].expect_mass_flux:
            raise RuntimeError("the step-mean mass flux was never formed; scalars were moved by a "
                               "flux the interface register does not hold")

    def reserve_set(self, mine: _ClassScalars) -> None:
        if not mine.rows:
            return
        klass = mine.klass
        state = klass.driver.atmosphere.state
        i32 = np.int32
        self.run("lts_reserve_set", self.ntr * self.nlev * mine.rows, (
            i32(self.ntr), i32(self.nlev), i32(mine.rows), i32(klass.cell_stride),
            klass.reflux_cell, mine.row_area, state.rho, state.scalars, mine.reserve,
            mine.content_start, mine.air_start))

    def sync(self, klass: Any) -> None:
        """After the class's reflux: clear its consumed packets, publish, rotate, renew its reserve."""

        mine = self.classes[klass.index]
        count = int(klass.coarse_slot.size)
        self.run("lts_zero_scalar", self.ntr * self.nlev * count, (
            np.int32(self.ntr), np.int32(self.nlev), np.int32(count), np.int32(self.nslots),
            klass.coarse_slot, self.reg_scalar))
        self.publish(klass, klass.driver.atmosphere.state.scalars, self.truth_plane)
        self.rotate(klass)
        self.reserve_set(mine)

    # -- diagnostics ------------------------------------------------------------------
    def register_empty(self) -> bool:
        return not bool(self.cp.any(self.reg_scalar))

    def totals(self) -> dict[str, Any]:
        """FP64 owned-union tracer masses (area * dzw * rho * q) and the extreme mixing ratios."""

        cp = self.cp
        mass = np.zeros(self.ntr)
        low = np.full(self.ntr, np.inf)
        high = np.full(self.ntr, -np.inf)
        for mine in self.classes:
            klass = mine.klass
            state = klass.driver.atmosphere.state
            n = klass.n_owned_cells
            air = state.rho[:, :n].astype(cp.float64) * klass.mass_weight
            for tracer in range(self.ntr):
                q = state.scalars[tracer, :, :n]
                mass[tracer] += float(cp.sum(q.astype(cp.float64) * air))
                low[tracer] = min(low[tracer], float(cp.min(q)))
                high[tracer] = max(high[tracer], float(cp.max(q)))
            del air
        return {"tracer_mass": mass.tolist(), "tracer_min": low.tolist(), "tracer_max": high.tolist()}

    def stats(self) -> dict[str, Any]:
        cp = self.cp
        total = np.zeros(len(STAT_NAMES), dtype=np.int64)
        per_class = []
        for mine in self.classes:
            values = cp.asnumpy(mine.stats).astype(np.int64)
            per_class.append([int(v) for v in values])
            total += values
        result: dict[str, Any] = {name: int(value) for name, value in zip(STAT_NAMES, total)}
        result["per_class"] = per_class
        amounts = np.zeros((self.ntr, 3))
        for mine in self.classes:
            amounts += cp.asnumpy(mine.amounts).reshape(self.ntr, 3)
        for column, name in enumerate(AMOUNT_NAMES):
            result[name] = amounts[:, column].tolist()
        result["withheld_fraction_of_throughput"] = [
            float((row[1] + row[2]) / row[0]) if row[0] else 0.0 for row in amounts]
        if any(mine.audit is not None for mine in self.classes):
            created = np.zeros(self.ntr)
            count = np.zeros(self.ntr, dtype=np.int64)
            rows = []
            for mine in self.classes:
                audit = mine.audit.read()
                rows.append(audit)
                created += np.asarray(audit["clamped_mass"])
                count += np.asarray(audit["clamped_values"], dtype=np.int64)
            result["clamped_mass"] = created.tolist()
            result["clamped_values"] = count.tolist()
            result["clamped_per_class"] = rows
        return result

    def gather(self) -> np.ndarray:
        """Owned-union scalars in global cell order."""

        cp = self.cp
        first = self.classes[0].klass.tables.layout
        target = np.empty((self.ntr, self.nlev, first._global_cells), dtype=np.float32)
        for mine in self.classes:
            klass = mine.klass
            n = klass.n_owned_cells
            l2g = klass.tables.layout.cell_l2g[:n].astype(np.int64)
            target[:, :, l2g] = cp.asnumpy(klass.driver.atmosphere.state.scalars[:, :, :n])
        return target


class _UntappedAudit(ClampAudit):
    """The audit as a step of a class's own finish hook (the tap is the class's)."""
