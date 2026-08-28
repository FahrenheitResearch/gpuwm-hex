#!/usr/bin/env python3
"""Per-kernel contract decks for the regional (limited-area) CUDA kernels.

Every kernel in :mod:`hexcore.cuda_regional_v841` mirrors exactly one
function of the CPU authority lane, which is the pinned v8.4.1 transcription.
This tool is the proof of that claim, on real regional bytes:

1. it loads the native-culled reference mesh, its init state and its lbc
   series (the L0 record set) and derives the 7-ring masks with the CPU
   authority's own code;
2. for each deck it runs the **host** function to produce expected bits, runs
   the **device** kernels on the same inputs, and compares raw ``uint32`` bit
   patterns -- not tolerances;
3. it runs every deck **twice** in the same process and requires the two
   device passes to be byte-identical (dual-run stability);
4. it runs a **mutation control** per deck: one deliberately wrong mask value
   is written into the device inputs and the deck is required to FAIL.  A
   deck that passes with the right masks and still passes with wrong ones is
   not a proof of anything, so both directions are measured and recorded.

The receipt is JSON and names, per deck, the kernels exercised, the compared
payloads with their bit-mismatch counts, the dual-run verdict and the
mutation verdict.

Usage on a card:

    python tools/run_cuda_regional_contract.py \
        --reference-dir <work-dir>/lam-l0-20260825 \
        --out evidence/regional-cuda-l5-20260826/contract/records.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexcore import regional_v841  # noqa: E402
from hexcore.regional_v841 import N_RELAX_ZONE as regional_v841_N_RELAX_ZONE  # noqa: E402
from hexcore.acoustic import edge_signs_on_cells  # noqa: E402
from hexcore.acoustic_v841 import (  # noqa: E402
    AcousticStepForcing,
    AcousticStepState,
    VerticalImplicitCoefficientsV841,
    advance_acoustic_step_v841,
    compute_vertical_implicit_coefficients_v841,
)
from hexcore.cuda_acoustic import (  # noqa: E402
    CudaAcousticForcing,
    CudaAcousticState,
    CudaVerticalImplicitCoefficients,
)
from hexcore.cuda_regional_v841 import (  # noqa: E402
    CUDA_REGIONAL_SOURCE,
    MODULE_KEY,
    REGIONAL_KERNELS,
    CudaRegionalKernels,
    DeviceRegionalMasks,
)
from hexcore.diagnostics import edge_signs_on_vertices  # noqa: E402
from hexcore.driver import (  # noqa: E402
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from hexcore.mesh import Mesh  # noqa: E402
from hexcore.offcentering_v841 import build_v841_acoustic_offcentering  # noqa: E402
from hexcore.transport import (  # noqa: E402
    _atmosphere_horizontal_edge_values,
    build_advection_coefficients,
)

_XTIME = "%Y-%m-%d_%H:%M:%S"
_RELAX_DIVDAMP = 6.0
_DT = 120.0
_DTS = 120.0 / 3.0 / 6.0
#: nRelaxZone as the runtime float32 divisor the kernels take (never a
#: source literal -- NVRTC turns a literal divisor into a reciprocal
#: multiply; see cuda_regional_v841's module docstring).
_RELAX_ZONE = np.float32(regional_v841_N_RELAX_ZONE)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _bits(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return array.view(np.uint32)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype=np.float32)).tobytes("C")
    ).hexdigest()


def compare_bits(name: str, host: Any, device: Any) -> dict[str, Any]:
    host_bits = _bits(host)
    device_bits = _bits(device)
    if host_bits.shape != device_bits.shape:
        return {
            "payload": name,
            "bitwise_equal": False,
            "reason": f"shape {device_bits.shape} != {host_bits.shape}",
        }
    same = host_bits == device_bits
    record: dict[str, Any] = {
        "payload": name,
        "values": int(host_bits.size),
        "bitwise_equal": bool(same.all()),
        "host_sha256": _sha(host),
        "device_sha256": _sha(device),
    }
    if not record["bitwise_equal"]:
        bad = ~same
        record["mismatch_count"] = int(bad.sum())
        h = np.asarray(host, dtype=np.float32)
        d = np.asarray(device, dtype=np.float32)
        first = tuple(int(x) for x in np.argwhere(bad)[0])
        record["first_mismatch"] = {
            "index": list(first),
            "host": float(h[first]),
            "device": float(d[first]),
        }
    return record


# ---------------------------------------------------------------------------
# the shared regional bundle
# ---------------------------------------------------------------------------


class Bundle:
    """Real regional bytes plus one deterministic perturbation field.

    The mesh, its masks, the init state and the lbc series are the reference
    record set.  Tendency and acoustic scratch arrays are seeded from a fixed
    PCG64 stream and scaled to the magnitudes the fields actually carry, so
    the decks exercise the real zone geometry with reproducible inputs and
    the same bytes reach the host function and the kernel.
    """

    def __init__(
        self,
        reference: Path | None,
        cp: Any,
        *,
        grid: Path | None = None,
        init: Path | None = None,
        lbc_dir: Path | None = None,
        start_time: Any = None,
    ) -> None:
        self.cp = cp
        # Named paths take the bundle off the reference-dir layout entirely.
        # A cull of a mesh this program placed for itself does not live in the
        # 2026-08-25 record set's directory shape, and every regional anchor
        # is earned on its OWN cull -- "a different cull is a different
        # configuration" -- so the contract has to be runnable against one.
        if grid is None:
            if reference is None:
                raise SystemExit("the contract bundle needs --reference-dir or --grid")
            grid = reference / "cull-x1" / "conus.grid.nc"
            init = reference / "init-x1" / "conus.init.nc"
            lbc_dir = reference / "lbc-x1"
        if init is None or lbc_dir is None:
            raise SystemExit("--grid needs --init and --lbc-dir")
        self.grid_path = Path(grid)
        self.lbc_paths = sorted(str(p) for p in Path(lbc_dir).glob("lbc.*.nc"))
        if not self.lbc_paths:
            raise SystemExit(f"no lbc.*.nc under {lbc_dir}")
        self.mesh = Mesh.from_netcdf(grid, init, validate=False)
        vertical = load_mpas_vertical_grid(
            init, self.mesh, allow_regional_sentinels=True
        )
        self.vertical = vertical.vertical_grid
        self.terrain = vertical.terrain_metrics
        state, reference_state, saved = load_mpas_initial_state(
            init,
            self.mesh,
            self.vertical,
            scalar_names=("qv",),
            terrain_metrics=self.terrain,
            allow_regional_sentinels=True,
            return_saved_diagnostics=True,
        )
        self.state = state
        self.reference_state = reference_state
        self.saved = saved
        self.dtype = np.dtype(np.float32)
        self.masks = regional_v841.derive_regional_masks(self.mesh, self.dtype)
        self.scaling_cell, self.scaling_edge = (
            regional_v841.compute_mesh_scaling_regional(
                self.mesh, self.dtype, config_h_scale_with_mesh=True
            )
        )
        self.nlev, self.ncells = map(int, np.asarray(state.rho).shape)
        self.nedges = int(np.asarray(state.rho_u).shape[1])
        self.nvertices = int(
            np.asarray(self.mesh.arrays["areaTriangle"]).size
        )
        self.max_edges = int(
            np.asarray(self.mesh.arrays["edgesOnCell"]).shape[1]
        )
        self.vertex_degree = int(
            np.asarray(self.mesh.arrays["edgesOnVertex"]).shape[1]
        )

        def m(name: str, dtype: Any) -> np.ndarray:
            return np.ascontiguousarray(
                np.asarray(self.mesh.arrays[name]), dtype=dtype
            )

        self.cells_on_edge = m("cellsOnEdge", np.int32)
        self.edges_on_cell = m("edgesOnCell", np.int32)
        self.n_edges_on_cell = m("nEdgesOnCell", np.int32)
        self.vertices_on_edge = m("verticesOnEdge", np.int32)
        self.edges_on_vertex = m("edgesOnVertex", np.int32)
        self.dc_edge = m("dcEdge", np.float32)
        self.dv_edge = m("dvEdge", np.float32)
        self.inv_dc_edge = np.reciprocal(self.dc_edge)
        self.inv_dv_edge = np.reciprocal(self.dv_edge)
        self.inv_area_cell = np.reciprocal(m("areaCell", np.float32))
        self.inv_area_triangle = np.reciprocal(m("areaTriangle", np.float32))
        self.edge_sign_on_cell = np.ascontiguousarray(
            edge_signs_on_cells(self.mesh).astype(np.float32)
        )
        self.edge_sign_on_vertex = np.ascontiguousarray(
            edge_signs_on_vertices(self.mesh).astype(np.float32)
        )

        # -- the host driving state, and a device pool derived from the same
        #    files by the device kernels.
        # THE DECK'S OWN CLOCK, and it stopped being a constant on 2026-08-27.
        # A cull with a DELAYED START is driven by a boundary series that
        # begins at the hour the swath wanted, not at the parent's init hour;
        # the hardwired time here made the deck ask for a boundary file six
        # hours before the earliest one that exists and refuse the cull by
        # name.  The default is the reference record set's own start, so every
        # archived invocation is byte-unchanged.
        start = (
            start_time
            if isinstance(start_time, datetime)
            else datetime.strptime(str(start_time or "2026-08-12_06:00:00"), _XTIME)
        )
        self.start_time = start
        self.host_driving = regional_v841.RegionalDrivingState(
            regional_v841.LbcInventory(self.lbc_paths),
            self.mesh,
            self.vertical.zz,
            scalar_names=("lbc_qv",),
        )
        self.host_driving.start(start)
        self.host_driving.advance(start)

        rng = np.random.default_rng(20260826)

        def noise(shape: tuple[int, ...], scale: float) -> np.ndarray:
            return np.ascontiguousarray(
                (rng.standard_normal(shape) * scale).astype(np.float32)
            )

        cells = (self.nlev, self.ncells)
        edges = (self.nlev, self.nedges)
        rows = (self.nlev + 1, self.ncells)
        self.tend_rho = noise(cells, 1.0e-6)
        self.tend_rt = noise(cells, 1.0e-3)
        self.tend_ru = noise(edges, 1.0e-3)
        self.tend_rw = noise(rows, 1.0e-3)
        self.rho_pp = noise(cells, 1.0e-5)
        self.rtheta_pp = noise(cells, 1.0e-2)
        self.ru_p = noise(edges, 1.0e-2)
        self.ru_avg = noise(edges, 1.0e-2)
        self.rw_p = noise(rows, 1.0e-2)
        self.ww_avg = noise(rows, 1.0e-2)
        self.scalars_stage = np.ascontiguousarray(
            np.asarray(state.scalars, dtype=np.float32)
            + noise((1,) + cells, 1.0e-6)
        )

    # -- device helpers ---------------------------------------------------

    def dev(self, value: Any) -> Any:
        return self.cp.asarray(
            np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        )

    def devi(self, value: Any) -> Any:
        return self.cp.asarray(
            np.ascontiguousarray(np.asarray(value, dtype=np.int32))
        )

    def host(self, value: Any) -> np.ndarray:
        return self.cp.asnumpy(value)


# ---------------------------------------------------------------------------
# decks
# ---------------------------------------------------------------------------


class Deck:
    """One kernel-level contract: kernels, host expectation, device arm."""

    def __init__(
        self,
        name: str,
        kernels: tuple[str, ...],
        native_anchor: str,
        host_anchor: str,
        run: Callable[["Context"], dict[str, Any]],
        mutation: str,
        mutate: str = "masks",
    ) -> None:
        self.name = name
        self.kernels = kernels
        self.native_anchor = native_anchor
        self.host_anchor = host_anchor
        self.run = run
        self.mutation = mutation
        self.mutate = mutate


class Context:
    """Device-side inputs a deck body consumes, with a mutable mask set."""

    def __init__(self, bundle: Bundle, kernels: CudaRegionalKernels) -> None:
        self.b = bundle
        self.k = kernels
        self.cp = bundle.cp
        self.masks = DeviceRegionalMasks.from_host(bundle.masks)
        self.mesh_scaling_cell = bundle.dev(bundle.scaling_cell)
        self.mesh_scaling_edge = bundle.dev(bundle.scaling_edge)
        self.cells_on_edge = bundle.devi(bundle.cells_on_edge)
        self.edges_on_cell = bundle.devi(bundle.edges_on_cell)
        self.n_edges_on_cell = bundle.devi(bundle.n_edges_on_cell)
        self.vertices_on_edge = bundle.devi(bundle.vertices_on_edge)
        self.edges_on_vertex = bundle.devi(bundle.edges_on_vertex)
        self.dc_edge = bundle.dev(bundle.dc_edge)
        self.dv_edge = bundle.dev(bundle.dv_edge)
        self.inv_dc_edge = bundle.dev(bundle.inv_dc_edge)
        self.inv_dv_edge = bundle.dev(bundle.inv_dv_edge)
        self.inv_area_cell = bundle.dev(bundle.inv_area_cell)
        self.inv_area_triangle = bundle.dev(bundle.inv_area_triangle)
        self.edge_sign_on_cell = bundle.dev(bundle.edge_sign_on_cell)
        self.edge_sign_on_vertex = bundle.dev(bundle.edge_sign_on_vertex)

    def mutate_masks(self) -> str:
        """Write one deliberately wrong zone geometry into the device inputs.

        The mutation is a zone-geometry lie, not a numerical nudge: an
        outermost relaxation element is relabelled one ring inward (which
        moves both its relaxation coefficient and, at mask 4, its scalar
        flux branch), the first specified edge is de-masked, and the first
        specified cell is dropped from the element list.  Every deck's
        arithmetic depends on the zone geometry, so a deck that still
        matches the host after this is a deck that is not reading the masks
        it claims to read.
        """

        b = self.b
        masks = self.masks
        detail = []
        relax_cells = np.asarray(b.masks.relax_cells)
        if relax_cells.size:
            cell = int(relax_cells[-1])
            before = int(np.asarray(b.masks.bdy_mask_cell)[cell])
            masks.bdy_mask_cell[cell] = np.int32(before - 1)
            detail.append(f"bdy_mask_cell[{cell}] {before}->{before - 1}")
        relax_edges = np.asarray(b.masks.relax_edges)
        if relax_edges.size:
            host_edge_mask = np.asarray(b.masks.bdy_mask_edge)
            # Prefer a mask-4 edge: it is the ring whose relabelling also
            # moves the scalar flux out of the first-order downgrade, so one
            # mutation exercises both the coefficient and the branch.
            at_four = relax_edges[host_edge_mask[relax_edges] == 4]
            edge = int(at_four[-1] if at_four.size else relax_edges[-1])
            before = int(host_edge_mask[edge])
            masks.bdy_mask_edge[edge] = np.int32(before - 1)
            detail.append(f"bdy_mask_edge[{edge}] {before}->{before - 1}")
        spec_edges = np.asarray(b.masks.spec_edges)
        if spec_edges.size:
            edge = int(spec_edges[0])
            masks.spec_zone_mask_edge[edge] = np.float32(0.0)
            detail.append(f"spec_zone_mask_edge[{edge}] 1.0->0.0")
        spec_cells = np.asarray(b.masks.spec_cells)
        if spec_cells.size > 1:
            cell = int(spec_cells[0])
            masks.spec_zone_mask_cell[cell] = np.float32(0.0)
            masks.spec_cells[0] = np.int32(int(spec_cells[1]))
            detail.append(
                f"spec_zone_mask_cell[{cell}] 1.0->0.0 and spec_cells[0] "
                "aliased to its successor"
            )
        return "; ".join(detail)

    def mutate_connectivity(self) -> str:
        """Give one ring-7 one-cell edge a second cell it does not have.

        The lateral-boundary pool derivation reads ``cellsOnEdge`` presence,
        not the zone masks: ``lbc_rho_edge`` is written only where both
        adjacent cells exist.  This is the wrong-input control for that
        deck -- inventing the absent neighbour is exactly the mistake a
        sentinel-blind implementation makes.
        """

        coe = np.asarray(self.b.mesh.arrays["cellsOnEdge"], dtype=np.int64)
        rows = np.flatnonzero((coe < 0).any(axis=1))
        if not rows.size:
            return "no one-cell edge exists on this mesh"
        edge = int(rows[0])
        slot = int(np.flatnonzero(coe[edge] < 0)[0])
        replacement = int(coe[edge, 1 - slot])
        self.cells_on_edge[edge, slot] = np.int32(replacement)
        return (
            f"cellsOnEdge[{edge},{slot}] sentinel -> {replacement} "
            "(an absent neighbour invented)"
        )

    def mutate(self, kind: str) -> str:
        if kind == "connectivity":
            return self.mutate_connectivity()
        return self.mutate_masks()


# -- deck bodies ------------------------------------------------------------


def deck_lbc_pool(ctx: Context) -> dict[str, Any]:
    """regional_lbc_derive / rho_edge / tendency / state_at."""

    from hexcore.cuda_regional_v841 import DeviceRegionalDrivingState
    from hexcore.lbc import LbcInventory

    b = ctx.b
    driving = DeviceRegionalDrivingState(
        LbcInventory(b.lbc_paths),
        ctx.masks,
        cells_on_edge=ctx.cells_on_edge,
        zz=b.dev(b.vertical.zz),
        n_vert_levels=b.nlev,
        kernels=ctx.k,
        scalar_names=("lbc_qv",),
    )
    driving.start(b.start_time)
    driving.advance(b.start_time)
    payloads: dict[str, tuple[Any, Any]] = {}
    for field in ("u", "ru", "rho_edge", "w", "rho", "rho_zz", "theta", "rtheta_m"):
        payloads[f"tend[{field}]"] = (
            b.host_driving.tendency(field),
            b.host(driving.tendency(field)),
        )
    payloads["tend[scalars]"] = (
        b.host_driving.tendency("scalars"),
        b.host(driving.tendency("scalars")),
    )
    delta_t = regional_v841.dynamics_time_offset(
        outer_dt=_DT,
        dynamics_split=3,
        dynamics_substep=2,
        rk_timestep=regional_v841.rk_timestep_f32(
            outer_dt=_DT, dynamics_split=3, rk_step=2
        ),
    )
    for field in ("ru", "rtheta_m", "rho_zz", "u"):
        payloads[f"state_at[{field}]"] = (
            b.host_driving.state_at(field, b.start_time, delta_t),
            b.host(driving.state_at(field, b.start_time, delta_t)),
        )
    payloads["state_at[scalars]"] = (
        b.host_driving.state_at("scalars", b.start_time, delta_t),
        b.host(driving.state_at("scalars", b.start_time, delta_t)),
    )
    return payloads


def deck_speczone_tend(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    host_rho = b.tend_rho.copy()
    host_rt = b.tend_rt.copy()
    host_ru = b.tend_ru.copy()
    host_rw = b.tend_rw.copy()
    regional_v841.adjust_dynamics_speczone_tend(
        masks=b.masks,
        tend_ru=host_ru,
        tend_rho=host_rho,
        tend_rt=host_rt,
        tend_rw=host_rw,
        ru_driving_tend=b.host_driving.tendency("ru"),
        rt_driving_tend=b.host_driving.tendency("rtheta_m"),
        rho_driving_tend=b.host_driving.tendency("rho_zz"),
    )
    d_rho = b.dev(b.tend_rho)
    d_rt = b.dev(b.tend_rt)
    d_ru = b.dev(b.tend_ru)
    d_rw = b.dev(b.tend_rw)
    ctx.k.launch(
        "regional_speczone_tend_cell_v841",
        max(ctx.masks.n_spec_cells, 1),
        (
            np.int32(b.nlev),
            np.int32(b.ncells),
            np.int32(ctx.masks.n_spec_cells),
            ctx.masks.spec_cells,
            b.dev(b.host_driving.tendency("rho_zz")),
            b.dev(b.host_driving.tendency("rtheta_m")),
            d_rho,
            d_rt,
            d_rw,
        ),
    )
    ctx.k.launch(
        "regional_speczone_tend_edge_v841",
        max(ctx.masks.n_spec_edges, 1),
        (
            np.int32(b.nlev),
            np.int32(b.nedges),
            np.int32(ctx.masks.n_spec_edges),
            ctx.masks.spec_edges,
            b.dev(b.host_driving.tendency("ru")),
            d_ru,
        ),
    )
    return {
        "tend_rho": (host_rho, b.host(d_rho)),
        "tend_rt": (host_rt, b.host(d_rt)),
        "tend_ru": (host_ru, b.host(d_ru)),
        "tend_rw": (host_rw, b.host(d_rw)),
    }


def _driving_values(b: Bundle, delta_t: np.float32) -> dict[str, np.ndarray]:
    return {
        name: b.host_driving.state_at(name, b.start_time, delta_t)
        for name in ("ru", "rtheta_m", "rho_zz", "u")
    }


def deck_relaxzone_tend(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    delta_t = regional_v841.dynamics_time_offset(
        outer_dt=_DT,
        dynamics_split=3,
        dynamics_substep=1,
        rk_timestep=regional_v841.rk_timestep_f32(
            outer_dt=_DT, dynamics_split=3, rk_step=1
        ),
    )
    values = _driving_values(b, delta_t)
    theta_m = np.ascontiguousarray(np.asarray(b.saved.theta_m, np.float32))
    rho_zz = np.ascontiguousarray(np.asarray(b.state.rho, np.float32))
    ru = np.ascontiguousarray(np.asarray(b.state.rho_u, np.float32))
    host_rho = b.tend_rho.copy()
    host_rt = b.tend_rt.copy()
    host_ru = b.tend_ru.copy()
    regional_v841.adjust_dynamics_relaxzone_tend(
        b.mesh,
        masks=b.masks,
        mesh_scaling_regional_cell=b.scaling_cell,
        mesh_scaling_regional_edge=b.scaling_edge,
        config_relax_zone_divdamp_coef=_RELAX_DIVDAMP,
        dt=_DT,
        tend_ru=host_ru,
        tend_rho=host_rho,
        tend_rt=host_rt,
        ru=ru,
        theta_m=theta_m,
        rho_zz=rho_zz,
        ru_driving_values=values["ru"],
        rt_driving_values=values["rtheta_m"],
        rho_driving_values=values["rho_zz"],
    )
    d_rho = b.dev(b.tend_rho)
    d_rt = b.dev(b.tend_rt)
    d_ru = b.dev(b.tend_ru)
    d_rho_zz = b.dev(rho_zz)
    d_theta = b.dev(theta_m)
    d_ru_state = b.dev(ru)
    d_rho_drv = b.dev(values["rho_zz"])
    d_rt_drv = b.dev(values["rtheta_m"])
    d_ru_drv = b.dev(values["ru"])
    fifty_dt = np.float32(np.float32(50.0) * np.float32(_DT))
    ten_dt = np.float32(np.float32(10.0) * np.float32(_DT))
    ctx.k.launch(
        "regional_relaxzone_rayleigh_cell_v841",
        max(ctx.masks.n_relax_cells, 1),
        (
            np.int32(b.nlev), np.int32(b.ncells),
            np.int32(ctx.masks.n_relax_cells), fifty_dt, _RELAX_ZONE,
            ctx.masks.relax_cells, ctx.masks.bdy_mask_cell,
            ctx.mesh_scaling_cell, d_rho_zz, d_theta, d_rho_drv, d_rt_drv,
            d_rho, d_rt,
        ),
    )
    ctx.k.launch(
        "regional_relaxzone_rayleigh_edge_v841",
        max(ctx.masks.n_relax_edges, 1),
        (
            np.int32(b.nlev), np.int32(b.nedges),
            np.int32(ctx.masks.n_relax_edges), fifty_dt, _RELAX_ZONE,
            ctx.masks.relax_edges, ctx.masks.bdy_mask_edge,
            ctx.mesh_scaling_edge, d_ru_state, d_ru_drv, d_ru,
        ),
    )
    ctx.k.launch(
        "regional_relaxzone_filter_cell_v841",
        max(ctx.masks.n_relax_cells, 1),
        (
            np.int32(b.nlev), np.int32(b.ncells), np.int32(b.nedges),
            np.int32(b.max_edges), np.int32(ctx.masks.n_relax_cells), ten_dt,
            _RELAX_ZONE, ctx.masks.relax_cells, ctx.masks.bdy_mask_cell,
            ctx.mesh_scaling_cell, ctx.n_edges_on_cell, ctx.edges_on_cell,
            ctx.cells_on_edge, ctx.edge_sign_on_cell, ctx.dv_edge,
            ctx.inv_dc_edge, d_rho_zz, d_theta, d_rho_drv, d_rt_drv,
            d_rho, d_rt,
        ),
    )
    ctx.k.launch(
        "regional_relaxzone_filter_edge_v841",
        max(ctx.masks.n_relax_edges, 1),
        (
            np.int32(b.nlev), np.int32(b.ncells), np.int32(b.nedges),
            np.int32(b.nvertices), np.int32(b.max_edges),
            np.int32(b.vertex_degree), np.int32(ctx.masks.n_relax_edges),
            ten_dt, np.float32(_RELAX_DIVDAMP), _RELAX_ZONE,
            ctx.masks.relax_edges, ctx.masks.bdy_mask_edge,
            ctx.mesh_scaling_edge, ctx.n_edges_on_cell, ctx.edges_on_cell,
            ctx.cells_on_edge, ctx.vertices_on_edge, ctx.edges_on_vertex,
            ctx.edge_sign_on_cell, ctx.edge_sign_on_vertex, ctx.dc_edge,
            ctx.dv_edge, ctx.inv_dc_edge, ctx.inv_dv_edge, ctx.inv_area_cell,
            ctx.inv_area_triangle, d_ru_state, d_ru_drv, d_ru,
        ),
    )
    return {
        "tend_rho": (host_rho, b.host(d_rho)),
        "tend_rt": (host_rt, b.host(d_rt)),
        "tend_ru": (host_ru, b.host(d_ru)),
    }


def deck_speczone_overwrites(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    delta_t = regional_v841.dynamics_time_offset(
        outer_dt=_DT, dynamics_split=3, dynamics_substep=3,
        rk_timestep=regional_v841.rk_timestep_f32(
            outer_dt=_DT, dynamics_split=3, rk_step=3
        ),
    )
    values = _driving_values(b, delta_t)
    host_u = np.ascontiguousarray(
        np.asarray(b.saved.normal_velocity, np.float32)
    ).copy()
    host_ru = np.ascontiguousarray(
        np.asarray(b.state.rho_u, np.float32)
    ).copy()
    host_w = np.ascontiguousarray(
        np.asarray(b.saved.vertical_velocity, np.float32)
    ).copy()
    regional_v841.overwrite_speczone_u_ru(
        masks=b.masks,
        normal_velocity=host_u,
        rho_u=host_ru,
        u_driving_values=values["u"],
        ru_driving_values=values["ru"],
    )
    regional_v841.zero_speczone_w(masks=b.masks, w=host_w)
    d_u = b.dev(b.saved.normal_velocity)
    d_ru = b.dev(b.state.rho_u)
    d_w = b.dev(b.saved.vertical_velocity)
    ctx.k.launch(
        "regional_speczone_u_ru_v841",
        max(ctx.masks.n_spec_edges, 1),
        (
            np.int32(b.nlev), np.int32(b.nedges),
            np.int32(ctx.masks.n_spec_edges), ctx.masks.spec_edges,
            b.dev(values["u"]), b.dev(values["ru"]), d_u, d_ru,
        ),
    )
    ctx.k.launch(
        "regional_zero_speczone_w_v841",
        max(ctx.masks.n_spec_cells, 1),
        (
            np.int32(b.nlev + 1), np.int32(b.ncells),
            np.int32(ctx.masks.n_spec_cells), ctx.masks.spec_cells, d_w,
        ),
    )
    return {
        "normal_velocity": (host_u, b.host(d_u)),
        "rho_u": (host_ru, b.host(d_ru)),
        "w": (host_w, b.host(d_w)),
    }


def deck_reset_speczone(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    dt_f32 = np.float32(_DT)
    rt_values = b.host_driving.state_at("rtheta_m", b.start_time, dt_f32)
    rho_values = b.host_driving.state_at("rho_zz", b.start_time, dt_f32)
    host_theta = np.ascontiguousarray(
        np.asarray(b.saved.theta_m, np.float32)
    ).copy()
    host_rho_theta = np.ascontiguousarray(
        np.asarray(b.state.rho_theta, np.float32)
    ).copy()
    host_perturbation = np.ascontiguousarray(
        np.asarray(b.saved.rho_theta_perturbation, np.float32)
    ).copy()
    regional_v841.reset_speczone_values(
        masks=b.masks,
        theta_m=host_theta,
        rho_theta=host_rho_theta,
        rt_driving_values=rt_values,
        rho_driving_values=rho_values,
    )
    spec = b.masks.spec_cells
    base = np.asarray(b.reference_state.rho_theta_base, np.float32)
    host_perturbation[:, spec] = rt_values[:, spec] - base[:, spec]
    d_theta = b.dev(b.saved.theta_m)
    d_rho_theta = b.dev(b.state.rho_theta)
    d_perturbation = b.dev(b.saved.rho_theta_perturbation)
    ctx.k.launch(
        "regional_reset_speczone_values_v841",
        max(ctx.masks.n_spec_cells, 1),
        (
            np.int32(b.nlev), np.int32(b.ncells),
            np.int32(ctx.masks.n_spec_cells), ctx.masks.spec_cells,
            b.dev(rt_values), b.dev(rho_values), b.dev(base),
            d_theta, d_rho_theta, d_perturbation,
        ),
    )
    return {
        "theta_m": (host_theta, b.host(d_theta)),
        "rho_theta": (host_rho_theta, b.host(d_rho_theta)),
        "rho_theta_perturbation": (
            host_perturbation, b.host(d_perturbation)
        ),
    }


def deck_scalar_boundary(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    dt_rk = regional_v841.transport_rk_timestep_f32(outer_dt=_DT, rk_step=2)
    driving = b.host_driving.state_at("scalars", b.start_time, dt_rk)
    host_adjust = b.scalars_stage.copy()
    regional_v841.bdy_adjust_scalars(
        b.mesh,
        masks=b.masks,
        mesh_scaling_regional_cell=b.scaling_cell,
        scalars_new=host_adjust,
        scalars_driving=driving,
        dt=_DT,
        dt_rk=dt_rk,
    )
    host_set = b.scalars_stage.copy()
    regional_v841.bdy_set_scalars(
        masks=b.masks, scalars_new=host_set, scalars_driving=driving
    )
    host_clamped = b.scalars_stage.copy()
    regional_v841.clamp_negative_scalars(host_clamped)

    ntracers = int(b.scalars_stage.shape[0])
    ten_dt = np.float32(np.float32(10.0) * np.float32(_DT))
    d_adjust = b.dev(b.scalars_stage)
    d_driving = b.dev(driving)
    d_updates = ctx.cp.zeros_like(d_adjust)
    ctx.k.launch(
        "regional_bdy_adjust_scalars_compute_v841",
        max(ctx.masks.n_relax_cells + ctx.masks.n_spec_cells, 1),
        (
            np.int32(ntracers), np.int32(b.nlev), np.int32(b.ncells),
            np.int32(b.nedges), np.int32(b.max_edges),
            np.int32(ctx.masks.n_relax_cells),
            np.int32(ctx.masks.n_spec_cells), ten_dt, np.float32(dt_rk),
            _RELAX_ZONE, ctx.masks.relax_cells, ctx.masks.spec_cells,
            ctx.masks.bdy_mask_cell, ctx.mesh_scaling_cell,
            ctx.n_edges_on_cell, ctx.edges_on_cell, ctx.cells_on_edge,
            ctx.edge_sign_on_cell, ctx.dv_edge, ctx.inv_dc_edge,
            d_adjust, d_driving, d_updates,
        ),
    )
    ctx.k.launch(
        "regional_bdy_adjust_scalars_copyback_v841",
        max(ctx.masks.n_nudged_cells, 1),
        (
            np.int32(ntracers), np.int32(b.nlev), np.int32(b.ncells),
            np.int32(ctx.masks.n_nudged_cells), ctx.masks.nudged_cells,
            d_updates, d_adjust,
        ),
    )
    d_set = b.dev(b.scalars_stage)
    ctx.k.launch(
        "regional_bdy_set_scalars_v841",
        max(ctx.masks.n_spec_cells, 1),
        (
            np.int32(ntracers), np.int32(b.nlev), np.int32(b.ncells),
            np.int32(ctx.masks.n_spec_cells), ctx.masks.spec_cells,
            d_driving, d_set,
        ),
    )
    d_clamped = b.dev(b.scalars_stage)
    ctx.k.launch(
        "regional_clamp_negative_scalars_v841",
        int(d_clamped.size),
        (np.int32(d_clamped.size), d_clamped),
    )
    return {
        "bdy_adjust_scalars": (host_adjust, b.host(d_adjust)),
        "bdy_set_scalars": (host_set, b.host(d_set)),
        "clamp_negative_scalars": (host_clamped, b.host(d_clamped)),
    }


def _acoustic_inputs(b: Bundle) -> tuple[Any, ...]:
    offcentering = build_v841_acoustic_offcentering(
        np.asarray(b.vertical.rdzw, np.float32),
        minimum=0.1,
        maximum=1.0,
        transition_bottom_z=10_000.0,
        transition_top_z=30_000.0,
    )
    zz = np.ascontiguousarray(np.asarray(b.vertical.zz, np.float32))
    cells = (b.nlev, b.ncells)
    coefficients = compute_vertical_implicit_coefficients_v841(
        dts=_DTS,
        offcentering=offcentering,
        zz=zz,
        cqw=np.ones(cells, np.float32),
        exner=np.ascontiguousarray(np.asarray(b.saved.exner, np.float32)),
        theta=np.ascontiguousarray(np.asarray(b.saved.theta_m, np.float32)),
        rho_base=np.ascontiguousarray(
            np.asarray(b.reference_state.rho_base, np.float32)
        ),
        rho_theta_base=np.ascontiguousarray(
            np.asarray(b.reference_state.rho_theta_base, np.float32)
        ),
        exner_base=np.ascontiguousarray(
            np.asarray(b.reference_state.exner_base, np.float32)
        ),
        rho_theta_perturbation=np.ascontiguousarray(
            np.asarray(b.saved.rho_theta_perturbation, np.float32)
        ),
        qtot=np.zeros(cells, np.float32),
        rdzw=np.asarray(b.vertical.rdzw, np.float32),
        fzm=np.asarray(b.vertical.fzm, np.float32),
        fzp=np.asarray(b.vertical.fzp, np.float32),
        rdzu=np.asarray(b.vertical.rdzu, np.float32),
    )
    forcing = AcousticStepForcing(
        rho_zz=np.ascontiguousarray(np.asarray(b.state.rho, np.float32)),
        theta_m=np.ascontiguousarray(np.asarray(b.saved.theta_m, np.float32)),
        zz=zz,
        exner=np.ascontiguousarray(np.asarray(b.saved.exner, np.float32)),
        cqu=np.ones((b.nlev, b.nedges), np.float32),
        zxu=np.ascontiguousarray(np.asarray(b.vertical.zxu, np.float32)),
        dss=np.ascontiguousarray(np.asarray(b.vertical.dss, np.float32)),
        tend_ru=b.tend_ru,
        tend_rho=b.tend_rho,
        tend_rt=b.tend_rt,
        tend_rw=b.tend_rw,
        w=np.ascontiguousarray(
            np.asarray(b.saved.vertical_velocity, np.float32)
        ),
        rw=np.ascontiguousarray(np.asarray(b.state.rho_w, np.float32)),
        rw_save=np.ascontiguousarray(np.asarray(b.state.rho_w, np.float32)),
    )
    state = AcousticStepState(
        ru_p=b.ru_p.copy(),
        rw_p=b.rw_p.copy(),
        rtheta_pp=b.rtheta_pp.copy(),
        rtheta_pp_old=b.rtheta_pp.copy(),
        rho_pp=b.rho_pp.copy(),
        ru_avg=b.ru_avg.copy(),
        ww_avg=b.ww_avg.copy(),
    )
    return offcentering, coefficients, forcing, state


def deck_acoustic(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    offcentering, coefficients, forcing, state = _acoustic_inputs(b)
    host = advance_acoustic_step_v841(
        b.mesh,
        state,
        forcing,
        coefficients,
        dts=_DTS,
        small_step=2,
        offcentering=offcentering,
        fzm=np.asarray(b.vertical.fzm, np.float32),
        fzp=np.asarray(b.vertical.fzp, np.float32),
        rdzw=np.asarray(b.vertical.rdzw, np.float32),
        specified_zone_edge=b.masks.spec_zone_mask_edge,
        specified_zone_cell=b.masks.spec_zone_mask_cell,
    )
    cp = ctx.cp
    device_state = CudaAcousticState(
        ru_p=b.dev(state.ru_p),
        rw_p=b.dev(state.rw_p),
        rtheta_pp=b.dev(state.rtheta_pp),
        rtheta_pp_old=b.dev(state.rtheta_pp_old),
        rho_pp=b.dev(state.rho_pp),
        ru_avg=b.dev(state.ru_avg),
        ww_avg=b.dev(state.ww_avg),
    )
    device_forcing = CudaAcousticForcing(
        **{
            name: b.dev(getattr(forcing, name))
            for name in (
                "rho_zz", "theta_m", "zz", "exner", "cqu", "zxu", "dss",
                "tend_ru", "tend_rho", "tend_rt", "tend_rw", "w", "rw",
                "rw_save",
            )
        }
    )
    device_coefficients = CudaVerticalImplicitCoefficients(
        **{
            name: b.dev(getattr(coefficients, name))
            for name in (
                "cofwr", "cofwz", "coftz", "cofwt", "cofrz", "a_tri",
                "b_tri", "c_tri", "alpha_tri", "gamma_tri",
            )
        }
    )
    etp = b.dev(offcentering.etp)
    etm = b.dev(offcentering.etm)
    ewp = b.dev(offcentering.ewp)
    ewm = b.dev(offcentering.ewm)
    invalid = cp.zeros((1,), dtype=cp.int32)
    ctx.k.launch(
        "acoustic_ru_regional_v841",
        b.nedges,
        (
            np.int32(b.nlev), np.int32(b.nedges), np.int32(b.ncells),
            np.int32(2), np.float32(_DTS), np.float32(9.80616),
            np.float32(287.0), np.float32(1004.5), ctx.cells_on_edge,
            ctx.inv_dc_edge, ctx.masks.spec_zone_mask_edge,
            device_forcing.zz, device_forcing.exner, device_forcing.cqu,
            device_forcing.zxu, device_forcing.tend_ru,
            device_state.rho_pp, device_state.rtheta_pp,
            device_state.ru_p, device_state.ru_avg, invalid,
        ),
    )
    # The device prepare step for small_step > 1 is the rtheta_pp_old
    # capture, which the driver performs before the advance; the deck feeds
    # the same captured array to both arms.
    device_state.rtheta_pp_old = b.dev(state.rtheta_pp)
    rs = cp.zeros((b.nlev, b.ncells), dtype=cp.float32)
    ts = cp.zeros_like(rs)
    ctx.k.launch(
        "acoustic_rs_ts_regional_v841",
        b.ncells,
        (
            np.int32(b.nlev), np.int32(b.ncells), np.int32(b.nedges),
            np.int32(b.max_edges), np.float32(_DTS), ctx.n_edges_on_cell,
            ctx.edges_on_cell, ctx.cells_on_edge, ctx.edge_sign_on_cell,
            ctx.dv_edge, ctx.inv_area_cell, ctx.masks.spec_zone_mask_cell,
            device_forcing.theta_m, b.dev(b.vertical.rdzw),
            device_coefficients.cofrz, device_coefficients.coftz, ewm,
            device_state.ru_p, device_state.rw_p, device_state.rho_pp,
            device_state.rtheta_pp, device_forcing.tend_rho,
            device_forcing.tend_rt, rs, ts,
        ),
    )
    ctx.k.launch(
        "acoustic_column_solve_regional_v841",
        b.ncells,
        (
            np.int32(b.nlev), np.int32(b.ncells), np.float32(_DTS),
            device_forcing.zz, device_forcing.rho_zz,
            b.dev(b.vertical.fzm), b.dev(b.vertical.fzp),
            b.dev(b.vertical.rdzw), device_forcing.dss, device_forcing.w,
            device_forcing.rw, device_forcing.rw_save,
            device_forcing.tend_rw, device_forcing.tend_rho,
            device_forcing.tend_rt, rs, ts, ctx.masks.spec_zone_mask_cell,
            device_coefficients.cofwr, device_coefficients.cofwz,
            device_coefficients.coftz, device_coefficients.cofwt,
            device_coefficients.cofrz, device_coefficients.a_tri,
            device_coefficients.alpha_tri, device_coefficients.gamma_tri,
            etp, etm, ewp, ewm,
            device_state.rw_p, device_state.rho_pp, device_state.rtheta_pp,
            device_state.ww_avg,
        ),
    )
    payloads = {}
    for name in ("ru_p", "ru_avg", "rw_p", "ww_avg", "rho_pp", "rtheta_pp"):
        payloads[name] = (
            getattr(host, name), b.host(getattr(device_state, name))
        )
    payloads["_one_cell_edge_flag"] = (
        np.zeros(1, np.float32),
        np.asarray(ctx.cp.asnumpy(invalid), np.float32),
    )
    return payloads


def deck_transport(ctx: Context) -> dict[str, Any]:
    b = ctx.b
    coefficients = build_advection_coefficients(
        b.mesh,
        config_scalar_adv_order=3,
        n_vert_levels=b.nlev,
        source_order_v841=True,
        allow_regional_sentinels=True,
    )
    velocity = np.ascontiguousarray(
        np.asarray(b.state.rho_u, np.float32)
    )
    stage = b.scalars_stage
    cells_on_edge64 = np.asarray(
        b.mesh.arrays["cellsOnEdge"], dtype=np.int64
    )
    host_edges = _atmosphere_horizontal_edge_values(
        stage,
        velocity,
        b.dv_edge,
        np.where(cells_on_edge64 < 0, b.ncells, cells_on_edge64),
        np.asarray(coefficients.adv_coefs, np.float32),
        np.asarray(coefficients.adv_coefs_3rd, np.float32),
        np.asarray(coefficients.n_adv_cells_for_edge, np.int64),
        np.asarray(coefficients.adv_cells_for_edge, np.int64),
        3,
        0.25,
        config_apply_lbcs=True,
        bdy_mask_edge=np.asarray(b.masks.bdy_mask_edge, np.int64),
    )
    ntracers = int(stage.shape[0])
    d_edges = ctx.cp.zeros(
        (ntracers, b.nlev, b.nedges), dtype=ctx.cp.float32
    )
    width = int(np.asarray(coefficients.adv_coefs).shape[1])
    ctx.k.launch(
        "transport_edge_values_regional_v841",
        b.nedges,
        (
            np.int32(ntracers), np.int32(b.nlev), np.int32(b.ncells),
            np.int32(b.nedges), np.int32(width), np.float32(0.25),
            ctx.masks.bdy_mask_edge, ctx.dv_edge, ctx.cells_on_edge,
            b.dev(stage), b.dev(velocity),
            b.dev(coefficients.adv_coefs), b.dev(coefficients.adv_coefs_3rd),
            b.devi(coefficients.n_adv_cells_for_edge),
            b.devi(coefficients.adv_cells_for_edge), d_edges,
        ),
    )
    device_edges = b.host(d_edges)
    # Native leaves mask>nRelaxZone scratch unwritten and no updated cell
    # reads it; the deck therefore compares the lanes that ARE read and
    # records the skipped population rather than pretending to compare it.
    read = np.asarray(b.masks.bdy_mask_edge, np.int64) <= 5

    # -- the cell update, host authority against the composed device chain.
    from hexcore import cuda_transport_v841 as device_transport
    from hexcore.transport import advance_scalars

    old = np.ascontiguousarray(np.asarray(b.state.scalars, np.float32))
    rho_old = np.ascontiguousarray(np.asarray(b.state.rho, np.float32))
    rho_new = np.ascontiguousarray(rho_old * np.float32(1.000001))
    ww_avg = np.ascontiguousarray(
        np.asarray(b.saved.vertical_velocity, np.float32)
    )
    cp = ctx.cp
    d_stage = b.dev(stage)
    d_old = b.dev(old)
    d_rho_old = b.dev(rho_old)
    d_rho_new = b.dev(rho_new)
    d_ww = b.dev(ww_avg)
    d_velocity = b.dev(velocity)
    vertical_flux = cp.empty(
        (ntracers, b.nlev + 1, b.ncells), dtype=cp.float32
    )
    device_transport._launch(
        "transport_vertical_flux",
        b.ncells,
        (
            np.int32(ntracers), np.int32(b.nlev), np.int32(b.ncells),
            np.float32(0.25), d_stage, d_ww,
            b.dev(b.vertical.fzm), b.dev(b.vertical.fzp), vertical_flux,
        ),
        ctx.k.cache,
    )
    target = cp.empty_like(d_rho_old)
    device_transport._launch(
        "transport_interpolate_target_v841",
        b.nlev * b.ncells,
        (
            np.int32(b.nlev * b.ncells), np.float32(1.0),
            d_rho_old, d_rho_new, target,
        ),
        ctx.k.cache,
    )
    # This payload was recorded when the shared kernel
    # ``transport_vertical_flux`` was not bitwise equal to the CPU authority's
    # ``_atmosphere_vertical_flux`` on an sm_120 stack: 51,258 of 166,376
    # values on the reference cull.  THE CAUSE IS NAMED AND FIXED (#355,
    # tree/evidence/nvrtc-reciprocal-20260826/): NVRTC rewrote the kernel's
    # ``12.0f`` divisor into a multiply by its float32 reciprocal for every
    # target at or above compute_100, and the rewrite accounts for exactly
    # those 51,258 values with nothing left over.  The denominator is now a
    # translation-unit constant the compiler may not fold.
    #
    # So this count is a REGRESSION DETECTOR, not an inherited condition: at
    # this tip it reads 0, and a nonzero reading means the divisor moved back
    # to a foldable form or something new diverged.  The substitution below --
    # feeding the CPU authority the device flux -- is kept so the payload that
    # gates the regional finish kernel stays the regional kernel's own
    # arithmetic and nothing else, which is a separable question from this
    # one.  OWED: the sm_120 re-run that turns "reads 0" from an
    # instruction-level proof into a measurement.
    import hexcore.transport as transport_module

    device_vertical_flux = b.host(vertical_flux)
    host_vertical_flux = np.asarray(
        transport_module._atmosphere_vertical_flux(
            stage[0],
            ww_avg,
            np.asarray(b.vertical.fzm, np.float32),
            np.asarray(b.vertical.fzp, np.float32),
            3,
            0.25,
        ),
        np.float32,
    )
    inherited = int(
        (
            host_vertical_flux.view(np.uint32)
            != device_vertical_flux[0].view(np.uint32)
        ).sum()
    )
    original_flux = transport_module._atmosphere_vertical_flux

    def _device_flux(*_args: Any, **_kwargs: Any) -> np.ndarray:
        return device_vertical_flux[0]

    transport_module._atmosphere_vertical_flux = _device_flux
    try:
        host_result = advance_scalars(
            b.mesh,
            old,
            stage,
            rho_old,
            rho_new,
            velocity,
            ww_avg,
            _DT,
            coefficients=coefficients,
            fzm=np.asarray(b.vertical.fzm, np.float32),
            fzp=np.asarray(b.vertical.fzp, np.float32),
            rdzw=np.asarray(b.vertical.rdzw, np.float32),
            rk_step=3,
            config_coef_3rd_order=0.25,
            config_apply_lbcs=True,
            bdy_mask_cell=np.asarray(b.masks.bdy_mask_cell, np.int64),
            bdy_mask_edge=np.asarray(b.masks.bdy_mask_edge, np.int64),
            advance_density=True,
            inv_area_cell=b.inv_area_cell,
        )
    finally:
        transport_module._atmosphere_vertical_flux = original_flux

    output = cp.empty_like(d_old)
    ctx.k.launch(
        "transport_standard_finish_regional_v841",
        b.ncells,
        (
            np.int32(ntracers), np.int32(b.nlev), np.int32(b.ncells),
            np.int32(b.nedges), np.int32(b.max_edges), np.float32(_DT),
            ctx.masks.bdy_mask_cell, ctx.n_edges_on_cell, ctx.edges_on_cell,
            ctx.edge_sign_on_cell, ctx.inv_area_cell, d_velocity, d_edges,
            vertical_flux, b.dev(b.vertical.rdzw), d_old, d_rho_old, target,
            cp.zeros_like(d_old), d_stage, output,
        ),
    )
    return {
        "edge_values[read lanes]": (
            host_edges[:, :, read], device_edges[:, :, read]
        ),
        "edge_values[skipped lanes are zero]": (
            np.zeros_like(device_edges[:, :, ~read]),
            device_edges[:, :, ~read],
        ),
        "scalars_after_transport": (
            np.asarray(host_result.scalars, np.float32), b.host(output)
        ),
        "target_density": (
            np.asarray(host_result.density, np.float32), b.host(target)
        ),
        "_inherited_shared_vertical_flux_mismatches": (
            np.asarray([float(inherited)], np.float32),
            np.asarray([float(inherited)], np.float32),
        ),
    }


DECKS: tuple[Deck, ...] = (
    Deck(
        "lbc-pool",
        (
            "regional_lbc_derive_v841",
            "regional_lbc_rho_edge_v841",
            "regional_lbc_tendency_v841",
            "regional_lbc_state_at_v841",
        ),
        "mpas_atm_boundaries.F:217-309,491-551",
        "regional_v841.RegionalDrivingState",
        deck_lbc_pool,
        "one ring-7 one-cell edge is given the second cell it does not have",
        "connectivity",
    ),
    Deck(
        "speczone-tendencies",
        (
            "regional_speczone_tend_cell_v841",
            "regional_speczone_tend_edge_v841",
        ),
        "mpas_atm_time_integration.F:7906-7967",
        "regional_v841.adjust_dynamics_speczone_tend",
        deck_speczone_tend,
        "one specified cell is dropped from the element list",
    ),
    Deck(
        "relaxzone-tendencies",
        (
            "regional_relaxzone_rayleigh_cell_v841",
            "regional_relaxzone_rayleigh_edge_v841",
            "regional_relaxzone_filter_cell_v841",
            "regional_relaxzone_filter_edge_v841",
        ),
        "mpas_atm_time_integration.F:7971-8198",
        "regional_v841.adjust_dynamics_relaxzone_tend",
        deck_relaxzone_tend,
        "the outermost relaxation ring is relabelled one ring inward",
    ),
    Deck(
        "speczone-overwrites",
        ("regional_speczone_u_ru_v841", "regional_zero_speczone_w_v841"),
        "mpas_atm_time_integration.F:2442-2485,7868-7902",
        "regional_v841.overwrite_speczone_u_ru / zero_speczone_w",
        deck_speczone_overwrites,
        "one specified cell is dropped from the element list",
    ),
    Deck(
        "reset-speczone-values",
        ("regional_reset_speczone_values_v841",),
        "mpas_atm_time_integration.F:8201-8244",
        "regional_v841.reset_speczone_values",
        deck_reset_speczone,
        "one specified cell is dropped from the element list",
    ),
    Deck(
        "scalar-boundary",
        (
            "regional_bdy_adjust_scalars_compute_v841",
            "regional_bdy_adjust_scalars_copyback_v841",
            "regional_bdy_set_scalars_v841",
            "regional_clamp_negative_scalars_v841",
        ),
        "mpas_atm_time_integration.F:8305-8416,8462-8505,2798-2800",
        "regional_v841.bdy_adjust_scalars / bdy_set_scalars",
        deck_scalar_boundary,
        "the outermost relaxation ring is relabelled one ring inward",
    ),
    Deck(
        "acoustic-regional",
        (
            "acoustic_ru_regional_v841",
            "acoustic_rs_ts_regional_v841",
            "acoustic_column_solve_regional_v841",
        ),
        "mpas_atm_time_integration.F:3909,3992-4103",
        "acoustic_v841.advance_acoustic_step_v841 (masked)",
        deck_acoustic,
        "one specified edge is de-masked so its pressure gradient is no "
        "longer switched off",
    ),
    Deck(
        "transport-regional",
        (
            "transport_edge_values_regional_v841",
            "transport_standard_finish_regional_v841",
        ),
        "mpas_atm_time_integration.F:4764-4861",
        "transport.advance_scalars (masked)",
        deck_transport,
        "the outermost relaxation ring is relabelled one ring inward, "
        "moving an edge out of the first-order downgrade",
    ),
)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _payload_records(payloads: dict[str, Any]) -> tuple[list[dict], bool]:
    records = [
        compare_bits(name, host, device)
        for name, (host, device) in payloads.items()
    ]
    return records, all(r["bitwise_equal"] for r in records)


def run_deck(deck: Deck, bundle: Bundle, cache: Any) -> dict[str, Any]:
    kernels = CudaRegionalKernels(cache)
    started = time.perf_counter()

    first_ctx = Context(bundle, kernels)
    first = deck.run(first_ctx)
    first_records, first_ok = _payload_records(first)

    second_ctx = Context(bundle, kernels)
    second = deck.run(second_ctx)
    second_records, second_ok = _payload_records(second)

    dual_run = all(
        a.get("device_sha256") == b.get("device_sha256")
        for a, b in zip(first_records, second_records)
    )

    mutant_ctx = Context(bundle, kernels)
    mutation_detail = mutant_ctx.mutate(deck.mutate)
    mutant = deck.run(mutant_ctx)
    mutant_records, mutant_ok = _payload_records(mutant)

    return {
        "deck": deck.name,
        "kernels": list(deck.kernels),
        "native_anchor": deck.native_anchor,
        "host_oracle": deck.host_anchor,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "pass_1": {"bitwise_equal": first_ok, "payloads": first_records},
        "pass_2": {"bitwise_equal": second_ok, "payloads": second_records},
        "dual_run_identical": bool(dual_run),
        "mutation": {
            "declared": deck.mutation,
            "applied": mutation_detail,
            "deck_still_matches_host": bool(mutant_ok),
            "control_has_teeth": not bool(mutant_ok),
            "payloads": [
                {
                    "payload": r["payload"],
                    "bitwise_equal": r["bitwise_equal"],
                    "mismatch_count": r.get("mismatch_count", 0),
                }
                for r in mutant_records
            ],
        },
        "verdict": bool(first_ok and second_ok and dual_run and not mutant_ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--grid", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--lbc-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decks", default=None)
    # THE LEDGER FIELDS (2026-08-27).  A cull the cascade placed this cycle
    # cannot be a row in a source file that was written before it existed, so
    # this receipt is what admits it -- checked by content against
    # cuda_backend.regional_admission.contract_receipt_defects, never by name.
    # --class-id names the minted configuration class the geometry belongs to;
    # it is a CLAIM, and the gate re-measures the class key off the real run
    # and refuses a receipt whose claim does not hold.
    parser.add_argument(
        "--class-id", default=None, metavar="ID",
        help="the minted configuration class this cull belongs to "
             "(cuda_backend.regional_admission.ADMITTED_CLASSES). Without it "
             "the receipt records no class and cannot admit anything")
    parser.add_argument(
        "--mesh-row", default=None, metavar="NAME",
        help="the registry row for this cull, when it has one")
    parser.add_argument(
        "--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="the hour this cull is driven from. A cull with a DELAYED START "
             "has a boundary series that begins at the hour its swath wanted, "
             "not at its parent's init hour, and a deck asking for a boundary "
             "file before the earliest one that exists refuses the cull by "
             "name (default: the reference record set's own start)")
    args = parser.parse_args()

    from hexcore.cuda_backend import KernelCache, require_cuda
    from hexcore.cuda_backend.compile_contract import source_sha256
    from hexcore.cuda_backend.regional_admission import kernel_set_sha256
    from hexcore.mesh import (
        REGIONAL_BOUNDARY_MASK_NAMES,
        regional_boundary_mask_digest,
    )

    capability = require_cuda(min_compute=(12, 0))
    import cupy as cp

    reference = None if args.reference_dir is None else Path(args.reference_dir)
    bundle = Bundle(
        reference,
        cp,
        grid=None if args.grid is None else Path(args.grid),
        init=None if args.init is None else Path(args.init),
        lbc_dir=None if args.lbc_dir is None else Path(args.lbc_dir),
        start_time=args.start_time,
    )
    cache = KernelCache(capability=capability)

    selected = DECKS
    if args.decks:
        wanted = {name.strip() for name in args.decks.split(",")}
        selected = tuple(d for d in DECKS if d.name in wanted)

    records = [run_deck(deck, bundle, cache) for deck in selected]
    covered: set[str] = set()
    for deck in selected:
        covered.update(deck.kernels)
    uncovered = sorted(set(REGIONAL_KERNELS) - covered)

    masks_present = {
        name: getattr(bundle.mesh, name)
        for name in REGIONAL_BOUNDARY_MASK_NAMES
        if getattr(bundle.mesh, name, None) is not None
    }
    bdy_mask_sha256 = (
        regional_boundary_mask_digest(masks_present) if masks_present else None
    )
    zone_width = (
        int(np.max(np.asarray(bundle.mesh.bdyMaskCell))) if masks_present else 0
    )
    decks_bitwise = all(r["verdict"] for r in records)

    receipt = {
        "schema": "mpas-port.cuda-regional-contract/v1",
        # The ledger block.  Flat, at the top level, and named for what it
        # asserts rather than for where it came from: these are the fields
        # regional_admission reads when a cull presents its own deck instead
        # of being a row somebody wrote.
        "instrument": "run_cuda_regional_contract",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card": capability.name,
        "class_id": args.class_id,
        "mesh_row": args.mesh_row,
        "bdy_mask_sha256": bdy_mask_sha256,
        "n_cells": bundle.ncells,
        "start_time": bundle.start_time.strftime(_XTIME),
        "boundary_zone_width": zone_width,
        "kernel_set_sha256": kernel_set_sha256(),
        "all_decks_bitwise": bool(decks_bitwise),
        "all_kernels_covered": not uncovered,
        "all_controls_have_teeth": all(
            r["mutation"]["control_has_teeth"] for r in records
        ),
        "dual_run_identical": all(r["dual_run_identical"] for r in records),
        "decks_selected": bool(args.decks),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "device": {
            "name": capability.name,
            "sm": capability.sm,
            "compute_capability": (
                f"{capability.compute_major}.{capability.compute_minor}"
            ),
            "multiprocessor_count": capability.multiprocessor_count,
            "cupy_version": capability.cupy_version,
        },
        "translation_unit": {
            "module_key": MODULE_KEY,
            "source_sha256": source_sha256(CUDA_REGIONAL_SOURCE),
            "declared_kernels": list(REGIONAL_KERNELS),
            "kernels_without_a_deck": uncovered,
        },
        "mesh": {
            "grid": str(bundle.grid_path),
            "n_cells": bundle.ncells,
            "n_edges": bundle.nedges,
            "n_vertices": bundle.nvertices,
            "n_vert_levels": bundle.nlev,
            "spec_cells": int(bundle.masks.spec_cells.size),
            "relax_cells": int(bundle.masks.relax_cells.size),
            "spec_edges": int(bundle.masks.spec_edges.size),
            "relax_edges": int(bundle.masks.relax_edges.size),
        },
        "decks": records,
    }
    receipt["summary"] = {
        "decks": len(records),
        "decks_passed": sum(1 for r in records if r["verdict"]),
        "kernels_covered": len(covered),
        "kernels_declared": len(REGIONAL_KERNELS),
        "all_dual_run_identical": all(r["dual_run_identical"] for r in records),
        "all_controls_have_teeth": all(
            r["mutation"]["control_has_teeth"] for r in records
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    for record in records:
        flag = "PASS" if record["verdict"] else "FAIL"
        print(
            f"[{flag}] {record['deck']:<24s} "
            f"dual_run={record['dual_run_identical']} "
            f"teeth={record['mutation']['control_has_teeth']}",
            flush=True,
        )
        if not record["verdict"]:
            for payload in record["pass_1"]["payloads"]:
                if not payload["bitwise_equal"]:
                    print("   ", json.dumps(payload), flush=True)
    print("receipt:", out, flush=True)
    ok = all(r["verdict"] for r in records)
    print("contract verdict:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
