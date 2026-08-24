"""Local-timestep acoustic kernels, derived by surgery from the pinned text.

OPT-IN.  Nothing in this module runs unless ``config_local_timestep`` is on.

Why derive instead of copy
--------------------------
``src/mpas_port/cuda_driver.py``, ``cuda_horizontal.py`` and ``cuda_horizontal_v841.py``
are exact-byte pinned by ``tools/run_cuda_v841_full_physics_x4.py``'s
``EXECUTION_SOURCE_PINS``, so the rate-class launches cannot be added in place.
A hand-copied kernel would drift from its original the first time the original
changed, and the drift would be silent.  Instead every kernel here is produced
by applying a small number of **asserted** textual edits to the original source
string, read out of the module that owns it at import time.  If an anchor no
longer matches byte for byte, ``_replace_once`` raises and the LTS path refuses
to compile rather than executing stale arithmetic.

What the edits are
------------------
1. The thread preamble becomes an index-list gather:
   ``i = blockDim.x*blockIdx.x + threadIdx.x; if (i >= n) return;``
   becomes
   ``s = ...; if (s >= n_active) return; const int i = active[s];``
   Two parameters are appended to the signature.  **Nothing else changes**, so a
   launch over ``arange(n)`` executes the original arithmetic on the original
   cells, and the ``__global__`` body is byte-comparable to its ancestor once
   the preamble is substituted back.

2. ``recover_edges``/``recover_interfaces`` take the sub-step count as an array
   instead of a scalar, because under LTS each edge and each column executes a
   different number of acoustic sub-steps and the ``ru_avg``/``ww_avg`` time
   average must be divided by the count that entity actually ran.

3. Two kernels are new and have no ancestor: ``lts_reflux_accumulate`` and
   ``lts_reflux_settle``.  They are the conservation mechanism (see below).

Conservation at a class interface
---------------------------------
``acoustic_rs_ts`` moves ``dts * dvEdge * ru_p`` across every edge, added to one
cell and subtracted from the other, so mass telescopes.  When the two cells run
at different rates the coarse cell integrates that edge with its own longer
``dts`` and its own, fewer, samples of ``ru_p``, while the fine cell integrates
the same edge at the fine rate.  The two no longer cancel, and mass is created
or destroyed at the boundary.  That is what kills local time stepping in
practice.

The remedy here is Berger-Colella refluxing, applied to the acoustic mass flux:

* the coarse cell keeps predicting with its own ``dts`` (unchanged arithmetic --
  its main kernel is not modified at all);
* an interface edge accumulates the residual
  ``resid = SUM_fine (dts_fine * dvEdge) * ru_p - SUM_coarse (dts_coarse * dvEdge) * ru_p``
  over the RK stage;
* at the end of the stage ``lts_reflux_settle`` applies ``resid`` to the
  **coarse** cell only.

The coarse cell's stage total then equals the fine-rate integral, which is the
same value the fine cell removed, with the opposite sign.  Nothing is left
unmatched.  ``edgeSignOnCell`` is exactly +-1 (``mpas_atm_core.F:1186-1202``), and
multiplying a binary32 by +-1.0 only flips the sign bit, so the accumulated term
``(dts*dvEdge)*ru_p`` is bit-for-bit the magnitude the fine side used.

The result is conservative in the flux -- no term is dropped or double counted --
but **not bit-exact**: the coarse side applies one rounded sum where the fine
side applied a sequence of rounded increments, and the two cells divide by their
own ``invAreaCell``, which is what the option-off path already does.  So the
statement is APPROXIMATE at binary32 rounding, and the run has to measure the
drift rather than assert it.

Because ``flux_u`` handed to scalar transport is the same fine-rate average, the
coarse cell's mass change and the transported mass flux stay consistent, which
is what keeps the scalar limiter from seeing a source at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_acoustic import _float32, _int32, _mesh_value
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .errors import ConfigurationRefusal


class LocalTimestepSourceDrift(RuntimeError):
    """A pinned kernel's text moved out from under the derived LTS kernel."""


def _replace_once(source: str, old: str, new: str, *, what: str) -> str:
    """Substitute exactly one occurrence, refusing zero or many.

    A silent no-op replacement would leave the LTS kernel running the original,
    whole-domain launch shape while the driver believed it was running a class
    subset -- an A/B arm that reports a delta of exactly zero.  Refusing here is
    the only way that stays visible.
    """

    count = source.count(old)
    if count != 1:
        raise LocalTimestepSourceDrift(
            f"local-timestep derivation anchor for {what!r} matched {count} times, "
            "expected exactly one; the pinned kernel text changed and the derived "
            "kernel would no longer be the same arithmetic"
        )
    return source.replace(old, new, 1)


def _extract_kernel(source: str, name: str) -> str:
    """Return one ``extern "C" __global__`` definition, braces balanced."""

    marker = f'extern "C" __global__ void {name}('
    start = source.find(marker)
    if start < 0:
        raise LocalTimestepSourceDrift(f"kernel {name!r} is absent from its source module")
    if source.find(marker, start + 1) >= 0:
        raise LocalTimestepSourceDrift(f"kernel {name!r} is defined more than once")
    body = source.find("{", source.find(")", start))
    depth = 0
    for offset in range(body, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise LocalTimestepSourceDrift(f"kernel {name!r} has unbalanced braces")


def _gather(kernel: str, *, name: str, index: str, bound: str, new_name: str) -> str:
    """Rename a kernel and turn its thread preamble into an index gather."""

    open_paren = kernel.find("(")
    depth = 0
    signature_end = -1
    for offset in range(open_paren, len(kernel)):
        if kernel[offset] == "(":
            depth += 1
        elif kernel[offset] == ")":
            depth -= 1
            if depth == 0:
                signature_end = offset
                break
    if signature_end < 0:
        raise LocalTimestepSourceDrift(f"kernel {name!r} has no recognisable signature end")
    kernel = (
        kernel[:signature_end]
        + ",\n    const int *lts_active, const int lts_n_active"
        + kernel[signature_end:]
    )
    kernel = _replace_once(
        kernel,
        f'extern "C" __global__ void {name}(',
        f'extern "C" __global__ void {new_name}(',
        what=f"{name} declaration",
    )
    kernel = _replace_once(
        kernel,
        f"const int {index} = blockDim.x * blockIdx.x + threadIdx.x;\n"
        f"    if ({index} >= {bound}) return;",
        "const int lts_slot = blockDim.x * blockIdx.x + threadIdx.x;\n"
        "    if (lts_slot >= lts_n_active) return;\n"
        f"    const int {index} = lts_active[lts_slot];",
        what=f"{name} thread preamble",
    )
    return kernel


_REFLUX_SOURCE = r"""
/* Refluxing at a rate-class interface.  ``sign_scale`` is +1 when the fine
   side contributes its sub-step flux and -1 when the coarse side's own
   prediction is removed, so ``resid`` ends the stage holding
   fine_integral - coarse_prediction for every interface edge. */
extern "C" __global__ void lts_reflux_accumulate(
    const int nlev, const int nedges, const int n_iface,
    const float dts, const float sign_scale,
    const int *iface_edges, const float *dv_edge,
    const float *ru_p, float *resid)
{
    const int slot = blockDim.x * blockIdx.x + threadIdx.x;
    if (slot >= n_iface) return;
    const int edge = iface_edges[slot];
    const float scaled = mpas_mul(sign_scale, mpas_mul(dts, dv_edge[edge]));
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        resid[index] = mpas_add(resid[index],
            mpas_mul(scaled, ru_p[index]));
    }
}

/* Hand the accumulated residual to the coarse cell.  ``coarse_sign`` is that
   cell's ``edgeSignOnCell`` entry for the edge, exactly +-1.  The flux
   composition mirrors acoustic_rs_ts term for term so the coarse cell's stage
   total becomes the fine-rate integral.

   ONE THREAD PER COARSE CELL, not per interface edge.  A coarse cell can carry
   more than one interface edge -- 404 of 529 do on the published x4.163842
   mesh, up to three each -- and a per-edge kernel has to atomicAdd into the
   shared cell.  Binary32 addition is not associative, so the scheduler's order
   decides the last bits and two identical runs diverge; measured over 120
   steps that seed reached 59-99% of the domain.  That defeats the port's own
   dual-run byte comparison, which is the corruption screen on cards with no
   ECC.  Summing a cell's own slots in index order and storing once makes the
   result defined.  Each interface edge belongs to exactly one coarse cell, so
   clearing ``resid`` inside this loop still touches every edge exactly once. */
extern "C" __global__ void lts_reflux_settle(
    const int nlev, const int nedges, const int ncells, const int n_coarse,
    const int *coarse_cells, const int *slot_offsets,
    const int *slot_edges, const float *slot_signs,
    const float *inv_area_cell,
    const float *theta_m, const int *cells_on_edge,
    float *resid, float *rho_pp, float *rtheta_pp)
{
    const int slot = blockDim.x * blockIdx.x + threadIdx.x;
    if (slot >= n_coarse) return;
    const int cell = coarse_cells[slot];
    const int begin = slot_offsets[slot];
    const int end = slot_offsets[slot + 1];
    const float inv_area = inv_area_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        const int cindex = C2(k, cell, ncells);
        float drho = 0.0f;
        float drtheta = 0.0f;
        for (int s = begin; s < end; ++s) {
            const int edge = slot_edges[s];
            const int eindex = E2(k, edge, nedges);
            float flux = mpas_mul(slot_signs[s], resid[eindex]);
            flux = mpas_mul(flux, inv_area);
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            drho = mpas_add(drho, mpas_sub(0.0f, flux));
            drtheta = mpas_add(drtheta, mpas_sub(0.0f,
                mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                    theta_m[C2(k, c1, ncells)],
                    theta_m[C2(k, c0, ncells)]))));
            resid[eindex] = 0.0f;
        }
        rho_pp[cindex] = mpas_add(rho_pp[cindex], drho);
        rtheta_pp[cindex] = mpas_add(rtheta_pp[cindex], drtheta);
    }
}

/* Time-average recovery with a per-entity sub-step count.  Under LTS an edge
   that ran at rate r executed acoustic_steps/r sub-steps, so ru_avg must be
   divided by that count and not by the stage's nominal count. */
extern "C" __global__ void lts_recover_edges(
    const int nlev, const int nedges, const int *edge_steps,
    const float *ru_saved, const float *ru_p, const float *ru_avg_p,
    float *ru, float *flux_u)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const float steps = (float)edge_steps[edge];
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        ru[index] = mpas_add(ru_saved[index], ru_p[index]);
        flux_u[index] = mpas_add(ru_saved[index],
            mpas_div(ru_avg_p[index], steps));
    }
}

extern "C" __global__ void lts_recover_interfaces(
    const int nlev, const int ncells, const int *cell_steps,
    const float *rw_saved, const float *rw_p, const float *ww_avg_p,
    float *rw, float *flux_w)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const float steps = (float)cell_steps[cell];
    for (int k = 0; k <= nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            rw[index] = 0.0f;
        } else {
            rw[index] = mpas_add(rw_saved[index], rw_p[index]);
        }
        flux_w[index] = mpas_add(rw_saved[index],
            mpas_div(ww_avg_p[index], steps));
    }
}
"""


def _build_source() -> str:
    from . import cuda_acoustic_v841, cuda_horizontal

    acoustic = cuda_acoustic_v841._CUDA_SOURCE
    horizontal = cuda_horizontal._CUDA_SOURCE

    pieces = [
        CUDA_FTZ_HELPERS,
        "#define C2(k,c,nc) ((k)*(nc) + (c))\n"
        "#define E2(k,e,ne) ((k)*(ne) + (e))\n"
        "#define CES(c,s,me) ((c)*(me) + (s))\n"
        "__device__ __forceinline__ int lidx(int k, int i, int n)"
        " { return k * n + i; }\n",
    ]
    # cofrz is a per-level kernel with no cell or edge dimension, so it is
    # reused verbatim rather than gathered.
    pieces.append(_extract_kernel(acoustic, "acoustic_cofrz_v841"))
    for name, index, bound in (
        ("acoustic_coefficients_v841", "cell", "ncells"),
        ("acoustic_ru_v841", "edge", "nedges"),
        ("acoustic_prepare_v841", "cell", "ncells"),
        ("acoustic_rs_ts_v841", "cell", "ncells"),
        ("acoustic_column_solve_v841", "cell", "ncells"),
    ):
        kernel = _extract_kernel(acoustic, name)
        pieces.append(
            _gather(
                kernel,
                name=name,
                index=index,
                bound=bound,
                new_name=name.replace("_v841", "_lts"),
            )
        )
    damping = _extract_kernel(horizontal, "divergence_damping_f32")
    pieces.append(
        _gather(
            damping,
            name="divergence_damping_f32",
            index="edge",
            bound="nedges",
            new_name="divergence_damping_lts",
        )
    )
    pieces.append(_REFLUX_SOURCE)
    return "\n\n".join(pieces)


_SOURCE: str | None = None
_KERNELS: dict[tuple[Any, str], Any] = {}
_CACHE: Any | None = None


def local_timestep_cuda_source() -> str:
    """The derived translation unit, built once and cached."""

    global _SOURCE
    if _SOURCE is None:
        _SOURCE = _build_source()
    return _SOURCE


def kernel(name: str, kernel_cache: Any | None = None) -> Any:
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
            local_timestep_cuda_source(),
            module_key="mpas_port.cuda_acoustic_lts",
        )
    return _KERNELS[key]


_THREADS = 128


def launch(name: str, count: int, args: tuple[Any, ...], *, kernel_cache: Any = None) -> None:
    if int(count) <= 0:
        return
    blocks = (int(count) + _THREADS - 1) // _THREADS
    kernel(name, kernel_cache)((blocks,), (_THREADS,), args)


@dataclass(frozen=True, slots=True)
class DeviceRateClassing:
    """Device-resident launch lists and the interface bookkeeping."""

    rates: tuple[int, ...]
    cell_lists: tuple[Any, ...]
    edge_lists: tuple[Any, ...]
    all_cells: Any
    all_edges: Any
    iface_edges: Any
    iface_coarse_cell: Any
    iface_coarse_sign: Any
    iface_coarse_rate: Any
    n_iface: int
    # Interface edges regrouped by their coarse cell, so the settle kernel can
    # sum a cell's own contributions in a defined order instead of racing
    # atomics into the same address.
    settle_cells: Any
    settle_offsets: Any
    settle_edges: Any
    settle_signs: Any
    n_settle: int
    resid: Any
    edge_rate_host: np.ndarray
    cell_rate_host: np.ndarray

    @property
    def single_class(self) -> bool:
        return len(self.rates) == 1 and int(self.rates[0]) == 1


def _effective_rate(rate: int, stage_steps: int) -> int:
    """Largest divisor of ``stage_steps`` that does not exceed ``rate``.

    A rate coarser than a stage's whole sub-step count silently becomes that
    stage's count, so the class never takes a step longer than the stage.
    """

    best = 1
    for candidate in range(1, int(stage_steps) + 1):
        if int(stage_steps) % candidate == 0 and candidate <= int(rate):
            best = candidate
    return best


def build_device_classing(
    classing: Any,
    mesh: Any,
    *,
    n_vert_levels: int,
    cp: Any,
) -> DeviceRateClassing:
    """Move a :class:`~mpas_port.lts_v841.LocalTimestepClassing` to the device."""

    cells_on_edge = np.asarray(cp.asnumpy(_int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")))
    edges_on_cell = np.asarray(cp.asnumpy(_int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")))
    signs = np.asarray(cp.asnumpy(_float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")))
    counts = np.asarray(cp.asnumpy(_int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")))

    cell_rate = np.asarray(classing.cell_rate, dtype=np.int32)
    edge_rate = np.asarray(classing.edge_rate, dtype=np.int32)
    n_cells = int(cell_rate.size)
    n_edges = int(edge_rate.size)
    if cells_on_edge.shape != (n_edges, 2):
        raise ValueError("device cellsOnEdge does not match the classing")

    iface = np.asarray(classing.interface_edges, dtype=np.int32)
    if iface.size:
        c0 = cells_on_edge[iface, 0].astype(np.int64)
        c1 = cells_on_edge[iface, 1].astype(np.int64)
        take_c1 = cell_rate[c1] > cell_rate[c0]
        coarse = np.where(take_c1, c1, c0).astype(np.int32)
        # edgeSignOnCell for the coarse cell's slot holding this edge.
        coarse_sign = np.empty(iface.size, dtype=np.float32)
        for position, (edge, cell) in enumerate(zip(iface.tolist(), coarse.tolist())):
            slots = edges_on_cell[cell, : int(counts[cell])]
            match = np.flatnonzero(slots == edge)
            if match.size != 1:
                raise ValueError(
                    "an interface edge is not listed exactly once on its coarse cell"
                )
            coarse_sign[position] = signs[cell, int(match[0])]
        if not np.all(np.abs(np.abs(coarse_sign) - 1.0) < 1e-6):
            raise ValueError("edgeSignOnCell must be exactly +-1 for the reflux to balance")
        coarse_rate = cell_rate[coarse.astype(np.int64)].astype(np.int32)
    else:
        coarse = np.zeros(0, dtype=np.int32)
        coarse_sign = np.zeros(0, dtype=np.float32)
        coarse_rate = np.zeros(0, dtype=np.int32)

    # CSR by coarse cell.  ``np.argsort(kind="stable")`` fixes the within-cell
    # order to ascending interface-edge index, so the settle sums the same
    # sequence every run and on every card.
    if iface.size:
        order = np.argsort(coarse.astype(np.int64), kind="stable")
        sorted_cells = coarse.astype(np.int64)[order]
        settle_cells, starts = np.unique(sorted_cells, return_index=True)
        settle_offsets = np.append(starts, sorted_cells.size).astype(np.int32)
        settle_edges = iface[order].astype(np.int32)
        settle_signs = coarse_sign[order].astype(np.float32)
        settle_cells = settle_cells.astype(np.int32)
        if int(settle_offsets[-1]) != int(iface.size):
            raise ValueError("the reflux settle grouping lost an interface edge")
    else:
        settle_cells = np.zeros(0, dtype=np.int32)
        settle_offsets = np.zeros(1, dtype=np.int32)
        settle_edges = np.zeros(0, dtype=np.int32)
        settle_signs = np.zeros(0, dtype=np.float32)

    def to_device_int(value: np.ndarray) -> Any:
        return cp.asarray(np.ascontiguousarray(value, dtype=np.int32))

    return DeviceRateClassing(
        rates=tuple(int(value) for value in classing.rates),
        cell_lists=tuple(to_device_int(value) for value in classing.cell_lists),
        edge_lists=tuple(to_device_int(value) for value in classing.edge_lists),
        all_cells=to_device_int(np.arange(n_cells, dtype=np.int32)),
        all_edges=to_device_int(np.arange(n_edges, dtype=np.int32)),
        iface_edges=to_device_int(iface),
        iface_coarse_cell=to_device_int(coarse),
        iface_coarse_sign=cp.asarray(np.ascontiguousarray(coarse_sign, dtype=np.float32)),
        iface_coarse_rate=to_device_int(coarse_rate),
        n_iface=int(iface.size),
        settle_cells=to_device_int(settle_cells),
        settle_offsets=to_device_int(settle_offsets),
        settle_edges=to_device_int(settle_edges),
        settle_signs=cp.asarray(np.ascontiguousarray(settle_signs, dtype=np.float32)),
        n_settle=int(settle_cells.size),
        resid=(
            cp.zeros((int(n_vert_levels), n_edges), dtype=cp.float32)
            if iface.size
            else None
        ),
        edge_rate_host=edge_rate,
        cell_rate_host=cell_rate,
    )


def stage_rate_plan(
    classing: DeviceRateClassing, stage_acoustic_steps: int
) -> tuple[tuple[int, int, int], ...]:
    """``(class_index, effective_rate, executed_substeps)`` for one RK stage."""

    plan = []
    for index, rate in enumerate(classing.rates):
        effective = _effective_rate(rate, stage_acoustic_steps)
        plan.append((index, effective, int(stage_acoustic_steps) // effective))
    return tuple(plan)


def entity_step_counts(
    classing: DeviceRateClassing, stage_acoustic_steps: int, *, cp: Any
) -> tuple[Any, Any]:
    """Per-cell and per-edge executed sub-step counts for one RK stage."""

    total = int(stage_acoustic_steps)
    cell_effective = np.array(
        [_effective_rate(int(value), total) for value in classing.cell_rate_host],
        dtype=np.int32,
    )
    edge_effective = np.array(
        [_effective_rate(int(value), total) for value in classing.edge_rate_host],
        dtype=np.int32,
    )
    return (
        cp.asarray(np.ascontiguousarray(total // cell_effective, dtype=np.int32)),
        cp.asarray(np.ascontiguousarray(total // edge_effective, dtype=np.int32)),
    )


def refuse_unsupported_ladder(rates: tuple[int, ...], stage_acoustic_steps: tuple[int, ...]) -> None:
    from .lts_v841 import admissible_rates

    allowed = admissible_rates(stage_acoustic_steps)
    bad = [rate for rate in rates if rate not in allowed]
    if bad:
        raise ConfigurationRefusal(
            "config_local_timestep_rates",
            tuple(rates),
            "a rate that does not divide every RK stage's acoustic sub-step "
            "count cannot skip whole sub-steps, so the class would have to "
            "change the stage's sub-step count -- a different scheme, not a "
            "different rate",
            f"rates drawn from {allowed}",
        )


__all__ = [
    "DeviceRateClassing",
    "LocalTimestepSourceDrift",
    "build_device_classing",
    "entity_step_counts",
    "kernel",
    "launch",
    "local_timestep_cuda_source",
    "refuse_unsupported_ladder",
    "stage_rate_plan",
]
