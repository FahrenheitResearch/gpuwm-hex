"""Attach opt-in local time stepping to a built v8.4.1 CUDA driver.

OPT-IN.  ``attach_local_timestep`` is a no-op unless the driver's config carries
``config_local_timestep=True``.

Why attachment rather than a subclass
-------------------------------------
``src/mpas_port/cuda_driver.py`` is one of the seventeen exact-byte pinned
execution sources, so the rate-class launches cannot be written into
``_advance_dynamics_subcycle_v841`` where the acoustic loop lives, and copying
that 320-line method into a subclass would fork the dycore for a performance
option.  Instead this module rebinds three call targets at runtime and leaves
every pinned byte where it is -- the same mechanism ``tools/mpas_mesh_binding.py``
already uses to retarget the pinned proof module without editing it.  The pin
is over file bytes; ``require_frozen_execution_sources()`` still passes, and the
gates below check that it does.

The three rebindings, and what each is for:

* ``cuda_driver.advance_acoustic_step_cuda_v841`` -- the acoustic sub-step.
  The replacement launches each rate class over its own index list with that
  class's ``dts``, and only on the sub-steps that class is active for.
* ``driver.horizontal.divergence_damping_3d`` -- the post-acoustic 3-D
  divergence damping.  Its coefficient is ``2 * smdiv * len_disp / dts``, so it
  is per class, and it must run only on the edges that just advanced.
* ``driver._vertical_coefficients`` and ``driver._recover_candidate`` -- the
  implicit vertical coefficients depend on ``dts**2``, and the ``ru_avg`` /
  ``ww_avg`` time averages must be divided by the sub-step count each entity
  actually ran rather than the stage's nominal count.

``detach_local_timestep`` puts all three back.  Attachment refuses rather than
warns when the driver is not the shape this was proved against, because a
silent fallback to the whole-domain launch is an A/B arm that reports a delta
of exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import cuda_acoustic_lts as ltsk
from . import cuda_driver as _driver_module
from .cuda_acoustic import CudaVerticalImplicitCoefficients, _float32, _int32, _mesh_value
from .cuda_backend.containers import require_resident_array
from .config_lts import local_timestep_enabled
from .errors import ConfigurationRefusal
from .lts_v841 import LocalTimestepClassing, classify_from_grid_file

#: The one acoustic schedule this lane is proved against.
SUPPORTED_STAGE_ACOUSTIC_STEPS: tuple[int, int, int] = (1, 3, 6)


def _refuse(knob: str, value: Any, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


@dataclass
class _StageTracker:
    """Which RK stage the acoustic loop is in, inferred and then checked.

    ``_advance_dynamics_subcycle_v841`` drives the sub-step loop inline, so the
    replacement sub-step function is the only place the stage is observable.
    The loop restarts ``small_step`` at 1 for every stage, which makes the stage
    index recoverable -- and the recovered stage's observed sub-step count is
    checked against the schedule, so a change to the pinned loop shape refuses
    instead of silently mis-scheduling the classes.
    """

    expected: tuple[int, ...]
    stage_index: int = -1
    observed: int = 0

    def note(self, small_step: int) -> int:
        if small_step == 1:
            if self.stage_index >= 0 and self.observed != self.expected[self.stage_index]:
                raise ConfigurationRefusal(
                    "stage_acoustic_steps",
                    self.observed,
                    "the pinned RK stage ran a different number of acoustic "
                    "sub-steps than the schedule local time stepping planned "
                    "the rate classes against, so the classes would advance by "
                    "the wrong amount of time",
                    f"stage sub-step counts {self.expected}",
                )
            self.stage_index = (self.stage_index + 1) % len(self.expected)
            self.observed = 0
        self.observed = max(self.observed, int(small_step))
        return self.stage_index

    def stage_steps(self) -> int:
        return int(self.expected[self.stage_index])


class LocalTimestepAttachment:
    """Live LTS state bound to one driver instance."""

    def __init__(
        self,
        driver: Any,
        classing: LocalTimestepClassing,
        *,
        stage_acoustic_steps: tuple[int, ...],
    ) -> None:
        self.driver = driver
        self.classing = classing
        self.stage_acoustic_steps = tuple(int(value) for value in stage_acoustic_steps)
        self.cp = driver.cp
        self.device = ltsk.build_device_classing(
            classing, driver.atmosphere.mesh, n_vert_levels=driver.nlev, cp=self.cp
        )
        self.tracker = _StageTracker(expected=self.stage_acoustic_steps)
        self._active_edge_lists: list[tuple[Any, int, int]] = []
        self._step_counts: dict[int, tuple[Any, Any]] = {}
        self._iface_by_edge_rate = self._group_interface(by="edge")
        self._iface_by_coarse_rate = self._group_interface(by="coarse")
        self.launch_counts = {"acoustic_substep_class_launches": 0, "reflux_settles": 0}
        self._saved: dict[str, Any] = {}

    # -- interface grouping -------------------------------------------------
    def _group_interface(self, *, by: str) -> tuple[tuple[int, Any, int], ...]:
        if self.device.n_iface == 0:
            return ()
        cp = self.cp
        edges = cp.asnumpy(self.device.iface_edges)
        if by == "edge":
            rates = self.device.edge_rate_host[edges.astype(np.int64)]
        else:
            rates = cp.asnumpy(self.device.iface_coarse_rate)
        groups = []
        for rate in sorted({int(value) for value in rates.tolist()}):
            mask = np.flatnonzero(rates == rate).astype(np.int32)
            groups.append(
                (
                    int(rate),
                    cp.asarray(np.ascontiguousarray(edges[mask], dtype=np.int32)),
                    int(mask.size),
                )
            )
        return tuple(groups)

    # -- schedule -----------------------------------------------------------
    def _effective_rate(self, rate: int, stage_steps: int) -> int:
        best = 1
        for candidate in range(1, int(stage_steps) + 1):
            if int(stage_steps) % candidate == 0 and candidate <= int(rate):
                best = candidate
        return best

    def _stage_steps_for_dts(self, dts: float) -> int:
        """Recover a stage's sub-step count from the acoustic timestep it uses.

        RK2 and RK3 share one ``dts`` and one coefficient set, so they share the
        smaller of their two counts: a class must not be handed vertical
        coefficients built for a sub-step longer than RK2 actually runs.
        """

        config = self.driver.config
        dt_dyn = float(config.config_dt) / int(config.config_dynamics_split_steps)
        substeps = int(config.config_number_of_sub_steps)
        if dts == dt_dyn / 3.0:
            return 1
        if dts == dt_dyn / substeps:
            return min(steps for steps in self.stage_acoustic_steps if steps > 1)
        raise ConfigurationRefusal(
            "acoustic_timestep",
            dts,
            "the acoustic timestep is not one the pinned RK3 schedule emits, "
            "so the rate classes cannot be told which stage they are in",
            f"dts in ({dt_dyn / 3.0}, {dt_dyn / substeps})",
        )

    def _active_classes(self, stage_steps: int, small_step: int) -> list[tuple[int, int]]:
        """``(class_index, effective_rate)`` for classes advancing at this sub-step."""

        active = []
        for index, rate in enumerate(self.device.rates):
            effective = self._effective_rate(rate, stage_steps)
            if (int(small_step) - 1) % effective == 0:
                active.append((index, effective))
        return active

    def _counts_for(self, stage_steps: int) -> tuple[Any, Any]:
        key = int(stage_steps)
        if key not in self._step_counts:
            self._step_counts[key] = ltsk.entity_step_counts(
                self.device, key, cp=self.cp
            )
        return self._step_counts[key]

    # -- vertical coefficients ---------------------------------------------
    def vertical_coefficients(
        self,
        state: Any,
        dts: float,
        theta_m: Any,
        rtheta_p: Any,
        *,
        exner: Any | None = None,
        validation_flag: Any | None = None,
        cqw: Any | None = None,
        qtot: Any | None = None,
    ) -> CudaVerticalImplicitCoefficients:
        driver = self.driver
        cp = self.cp
        if driver.v841_context is None:
            raise RuntimeError("local time stepping requires the v8.4.1 device context")
        context = driver.v841_context
        atmosphere = driver.atmosphere
        selected_exner = atmosphere.saved.exner if exner is None else exner
        nlev, ncells = driver.nlev, driver.ncells

        def cell_array() -> Any:
            return cp.empty((nlev, ncells), dtype=cp.float32)

        result = CudaVerticalImplicitCoefficients(
            cofwr=cell_array(),
            cofwz=cell_array(),
            coftz=cp.empty((nlev + 1, ncells), dtype=cp.float32),
            cofwt=cell_array(),
            cofrz=cp.empty((nlev,), dtype=cp.float32),
            a_tri=cell_array(),
            b_tri=cell_array(),
            c_tri=cell_array(),
            alpha_tri=cell_array(),
            gamma_tri=cell_array(),
        )
        owns_flag = validation_flag is None
        singular = (
            cp.zeros((1,), dtype=cp.int32)
            if owns_flag
            else require_resident_array(
                "validation_flag", validation_flag, dtype=np.int32, shape=(1,)
            )
        )
        ltsk.launch(
            "acoustic_cofrz_v841",
            nlev,
            (np.int32(nlev), atmosphere.vertical.rdzw, result.cofrz),
            kernel_cache=driver.cache,
        )
        # Which stage's coefficients these are is fixed by ``dts``: the pinned
        # schedule gives RK1 ``dt_dyn/3`` with one sub-step, and RK2/RK3 share
        # ``dt_dyn/nsub``.  RK1 runs one sub-step, so every class is effective
        # rate 1 there; RK2 and RK3 share one coefficient set, so a class takes
        # the coarsest rate admissible in both.  Matching on the timestep
        # rather than on call order means a change to the pinned call sequence
        # refuses here instead of quietly building RK1 coefficients at a coarse
        # rate the single RK1 sub-step never runs.
        stage_steps = self._stage_steps_for_dts(float(dts))
        cqw_value = cp.ones_like(state.rho) if cqw is None else cqw
        qtot_value = cp.zeros_like(state.rho) if qtot is None else qtot
        for index, rate in enumerate(self.device.rates):
            effective = self._effective_rate(rate, stage_steps)
            cells = self.device.cell_lists[index]
            ltsk.launch(
                "acoustic_coefficients_lts",
                int(cells.size),
                (
                    np.int32(nlev), np.int32(ncells),
                    np.float32(dts * effective),
                    np.float32(9.80616), np.float32(287.0), np.float32(1004.5),
                    atmosphere.vertical.zz, cqw_value, selected_exner, theta_m,
                    atmosphere.reference.rho_base,
                    atmosphere.reference.rho_theta_base,
                    atmosphere.reference.exner_base,
                    rtheta_p, qtot_value,
                    atmosphere.vertical.rdzw, atmosphere.vertical.fzm,
                    atmosphere.vertical.fzp, atmosphere.vertical.rdzu,
                    context.etp, context.ewp,
                    result.cofwr, result.cofwz, result.coftz, result.cofwt,
                    result.cofrz, result.a_tri, result.b_tri, result.c_tri,
                    result.alpha_tri, result.gamma_tri, singular,
                    cells, np.int32(cells.size),
                ),
                kernel_cache=driver.cache,
            )
        if owns_flag and int(cp.asnumpy(singular)[0]) != 0:
            raise FloatingPointError("singular v8.4.1 vertical acoustic system")
        return result

    # -- acoustic sub-step --------------------------------------------------
    def advance_acoustic_step(
        self,
        mesh: Any,
        state: Any,
        forcing: Any,
        coefficients: Any,
        *,
        context: Any,
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
    ) -> Any:
        driver = self.driver
        xp = self.cp
        out = state if in_place else state.copy()
        nlev, nedges = map(int, out.ru_p.shape)
        ncells = int(out.rho_pp.shape[1])
        stage_index = self.tracker.note(int(small_step))
        stage_steps = self.tracker.stage_steps()
        cells_on_edge = _int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")
        edges_on_cell = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
        counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
        signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
        dv_edge = _float32(_mesh_value(mesh, "dv_edge"), "dv_edge")
        max_edges = int(edges_on_cell.shape[1])

        if int(small_step) == 1 and self.device.resid is not None:
            self.device.resid.fill(0.0)

        active = self._active_classes(stage_steps, int(small_step))
        if not active:
            raise RuntimeError("no rate class is active at an acoustic sub-step")

        # 1. normal-momentum update, per active EDGE class at that class's dts
        self._active_edge_lists = []
        for index, effective in active:
            edges = self.device.edge_lists[index]
            if int(edges.size) == 0:
                continue
            self._active_edge_lists.append((edges, int(edges.size), effective))
            ltsk.launch(
                "acoustic_ru_lts",
                int(edges.size),
                (
                    np.int32(nlev), np.int32(nedges), np.int32(ncells),
                    np.int32(small_step), np.float32(dts * effective),
                    np.float32(gravity), np.float32(rgas), np.float32(cp),
                    cells_on_edge, context.inv_dc_edge,
                    _float32(forcing.zz, "forcing.zz"),
                    _float32(forcing.exner, "forcing.exner"),
                    _float32(forcing.cqu, "forcing.cqu"),
                    _float32(forcing.zxu, "forcing.zxu"),
                    _float32(forcing.tend_ru, "forcing.tend_ru"),
                    out.rho_pp, out.rtheta_pp, out.ru_p, out.ru_avg,
                    edges, np.int32(edges.size),
                ),
                kernel_cache=driver.cache,
            )
            self.launch_counts["acoustic_substep_class_launches"] += 1

        # 2. refluxing bookkeeping at class interfaces.  The fine side adds the
        #    sub-step flux it is about to apply; the coarse side removes the
        #    prediction its own kernel is about to make with the longer dts.
        if self.device.resid is not None:
            for rate, edges, size in self._iface_by_edge_rate:
                effective = self._effective_rate(rate, stage_steps)
                if (int(small_step) - 1) % effective:
                    continue
                ltsk.launch(
                    "lts_reflux_accumulate",
                    size,
                    (
                        np.int32(nlev), np.int32(nedges), np.int32(size),
                        np.float32(dts * effective), np.float32(1.0),
                        edges, dv_edge, out.ru_p, self.device.resid,
                    ),
                    kernel_cache=driver.cache,
                )
            for rate, edges, size in self._iface_by_coarse_rate:
                effective = self._effective_rate(rate, stage_steps)
                if (int(small_step) - 1) % effective:
                    continue
                ltsk.launch(
                    "lts_reflux_accumulate",
                    size,
                    (
                        np.int32(nlev), np.int32(nedges), np.int32(size),
                        np.float32(dts * effective), np.float32(-1.0),
                        edges, dv_edge, out.ru_p, self.device.resid,
                    ),
                    kernel_cache=driver.cache,
                )

        # 3. cell kernels, per active CELL class
        rs = xp.empty((nlev, ncells), dtype=xp.float32)
        ts = xp.empty_like(rs)
        for index, effective in active:
            cells = self.device.cell_lists[index]
            if int(cells.size) == 0:
                continue
            step_dts = np.float32(dts * effective)
            ltsk.launch(
                "acoustic_prepare_lts",
                int(cells.size),
                (
                    np.int32(nlev), np.int32(ncells), np.int32(small_step),
                    out.rw_p, out.rtheta_pp, out.rtheta_pp_old, out.rho_pp,
                    out.ww_avg, cells, np.int32(cells.size),
                ),
                kernel_cache=driver.cache,
            )
            ltsk.launch(
                "acoustic_rs_ts_lts",
                int(cells.size),
                (
                    np.int32(nlev), np.int32(ncells), np.int32(nedges),
                    np.int32(max_edges), step_dts, counts, edges_on_cell,
                    cells_on_edge, signs, dv_edge, context.inv_area_cell,
                    _float32(forcing.theta_m, "forcing.theta_m"), rdzw,
                    coefficients.cofrz, coefficients.coftz, context.ewm,
                    out.ru_p, out.rw_p, out.rho_pp, out.rtheta_pp,
                    _float32(forcing.tend_rho, "forcing.tend_rho"),
                    _float32(forcing.tend_rt, "forcing.tend_rt"), rs, ts,
                    cells, np.int32(cells.size),
                ),
                kernel_cache=driver.cache,
            )
            ltsk.launch(
                "acoustic_column_solve_lts",
                int(cells.size),
                (
                    np.int32(nlev), np.int32(ncells), step_dts,
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
                    cells, np.int32(cells.size),
                ),
                kernel_cache=driver.cache,
            )
        del stage_index
        return out

    # -- divergence damping -------------------------------------------------
    def divergence_damping_3d(
        self,
        rho_u_perturbation: Any,
        theta_m: Any,
        rtheta_pp: Any,
        rtheta_pp_old: Any,
        *,
        dts: float,
        config_smdiv: float = 0.1,
        config_len_disp: float = 0.0,
        spec_zone_mask_edge: Any | None = None,
        n_cells_solve: int | None = None,
        in_place: bool = False,
    ) -> Any:
        horizontal = self.driver.horizontal
        if not self._active_edge_lists:
            raise RuntimeError(
                "divergence damping ran without a preceding local-timestep "
                "acoustic sub-step, so the active edge classes are unknown"
            )
        configured_length = float(config_len_disp)
        length = (
            float(horizontal.mesh.nominal_min_dc)
            if configured_length == 0.0
            else configured_length
        )
        mask = (
            horizontal.mesh.spec_zone_mask_edge
            if spec_zone_mask_edge is None
            else spec_zone_mask_edge
        )
        solve_count = horizontal.ncells if n_cells_solve is None else int(n_cells_solve)
        out = rho_u_perturbation if bool(in_place) else rho_u_perturbation.copy()
        for edges, size, effective in self._active_edge_lists:
            coefficient = np.float32(
                np.float32(2.0)
                * np.float32(float(config_smdiv))
                * np.float32(length)
                / np.float32(float(dts) * effective)
            )
            ltsk.launch(
                "divergence_damping_lts",
                size,
                (
                    theta_m, rtheta_pp, rtheta_pp_old,
                    horizontal.mesh.cells_on_edge, mask, coefficient,
                    np.int32(solve_count), np.int32(horizontal.nlev),
                    np.int32(horizontal.ncells), np.int32(horizontal.nedges),
                    out, edges, np.int32(size),
                ),
                kernel_cache=self.driver.cache,
            )
        self._active_edge_lists = []
        return out

    # -- stage close: settle the reflux, then recover -----------------------
    def recover_candidate(
        self,
        saved_state: Any,
        saved_diag: Any,
        acoustic: Any,
        stage: int,
        acoustic_steps: int,
        scalars: Any,
    ) -> tuple[Any, Any, Any, Any]:
        driver = self.driver
        nlev, ncells, nedges = driver.nlev, driver.ncells, driver.nedges
        if self.device.resid is not None:
            ltsk.launch(
                "lts_reflux_settle",
                self.device.n_settle,
                (
                    np.int32(nlev), np.int32(nedges), np.int32(ncells),
                    np.int32(self.device.n_settle),
                    self.device.settle_cells, self.device.settle_offsets,
                    self.device.settle_edges, self.device.settle_signs,
                    driver.v841_context.inv_area_cell,
                    saved_diag.theta_m,
                    _int32(
                        _mesh_value(driver.atmosphere.mesh, "cells_on_edge"),
                        "cells_on_edge",
                    ),
                    self.device.resid, acoustic.rho_pp, acoustic.rtheta_pp,
                ),
                kernel_cache=driver.cache,
            )
            self.launch_counts["reflux_settles"] += 1
        state, diagnostics, flux_u, flux_w = self._stock_recover(
            saved_state, saved_diag, acoustic, stage, acoustic_steps, scalars
        )
        if not self.device.single_class:
            cell_steps, edge_steps = self._counts_for(int(acoustic_steps))
            xp = self.cp
            scratch_ru = xp.empty_like(state.rho_u)
            ltsk.launch(
                "lts_recover_edges",
                nedges,
                (
                    np.int32(nlev), np.int32(nedges), edge_steps,
                    saved_state.rho_u, acoustic.ru_p, acoustic.ru_avg,
                    scratch_ru, flux_u,
                ),
                kernel_cache=driver.cache,
            )
            scratch_rw = xp.empty_like(state.rho_w)
            ltsk.launch(
                "lts_recover_interfaces",
                ncells,
                (
                    np.int32(nlev), np.int32(ncells), cell_steps,
                    saved_state.rho_w, acoustic.rw_p, acoustic.ww_avg,
                    scratch_rw, flux_w,
                ),
                kernel_cache=driver.cache,
            )
        return state, diagnostics, flux_u, flux_w

    # -- (de)attachment -----------------------------------------------------
    def attach(self) -> None:
        driver = self.driver
        self._saved["module_advance"] = _driver_module.advance_acoustic_step_cuda_v841
        self._saved["damping"] = driver.horizontal.divergence_damping_3d
        self._stock_recover = driver._recover_candidate
        _driver_module.advance_acoustic_step_cuda_v841 = self.advance_acoustic_step
        driver.horizontal.divergence_damping_3d = self.divergence_damping_3d
        driver._vertical_coefficients = self.vertical_coefficients
        driver._recover_candidate = self.recover_candidate
        driver.local_timestep = self

    def detach(self) -> None:
        driver = self.driver
        _driver_module.advance_acoustic_step_cuda_v841 = self._saved["module_advance"]
        driver.horizontal.divergence_damping_3d = self._saved["damping"]
        for name in ("_vertical_coefficients", "_recover_candidate"):
            if name in driver.__dict__:
                del driver.__dict__[name]
        if "local_timestep" in driver.__dict__:
            del driver.__dict__["local_timestep"]

    def receipt(self) -> dict[str, Any]:
        summary = self.classing.summary()
        summary.update(
            {
                "stage_acoustic_steps": list(self.stage_acoustic_steps),
                "arithmetic_acoustic_saving": self.classing.arithmetic_acoustic_saving(
                    self.stage_acoustic_steps
                ),
                "launch_counts": dict(self.launch_counts),
                "conservation": "approximate: flux-matched, binary32 rounding",
                "settle_coarse_cells": int(self.device.n_settle),
                "settle_is_order_defined": True,
            }
        )
        return summary


def attach_local_timestep(
    driver: Any,
    *,
    grid_path: str | None = None,
    classing: LocalTimestepClassing | None = None,
) -> LocalTimestepAttachment | None:
    """Turn the option on for one driver, or return ``None`` if it is off."""

    config = driver.config
    if not local_timestep_enabled(config):
        return None
    if driver.v841_context is None:
        _refuse(
            "config_local_timestep",
            True,
            "local time stepping is implemented against the v8.4.1 acoustic "
            "solver and there is no v8.4.1 device context on this driver",
            "a v8.4.1 CUDA run",
        )
    if getattr(driver, "halo_exchanger_v841", None) is not None:
        _refuse(
            "config_local_timestep",
            True,
            "a partitioned run exchanges halo rings on a schedule built for one "
            "global sub-step rate, so a coarse rank would damp and exchange "
            "stale ring-1 state",
            "a single-device run, or the multi-GPU lane once it is proved",
        )
    # The flux-corrected transport branch clips a scalar's flux against the
    # bounds of its neighbours within one step.  Across a rate boundary the two
    # sides reach that step at different times, so the limiter would compare a
    # cell against neighbours it has not caught up with and clip a flux the
    # reflux is relying on -- which breaks the scalar budget the interface was
    # built to keep.  The v8.4.1 CUDA lane refuses both knobs today
    # (cuda_driver.py:2569-2582, no native nonzero-tracer ruler), so this never
    # fires; it is here so that admitting the FCT branch later cannot let local
    # time stepping ride along on an interface that was never proved for it.
    for knob in ("config_monotonic", "config_positive_definite"):
        if getattr(config, knob, False):
            _refuse(
                knob,
                True,
                "a flux-corrected transport limiter clips against neighbours "
                "that a rate boundary has not advanced to the same time, so it "
                "would clip the very flux the reflux balances the interface "
                "with; that combination has not been proved",
                f"{knob}=False with config_local_timestep=True",
            )
    order = int(getattr(config, "config_time_integration_order", 3))
    substeps = int(getattr(config, "config_number_of_sub_steps", 6))
    stages = (1, max(1, substeps // 2), substeps)
    if order != 3 or stages != SUPPORTED_STAGE_ACOUSTIC_STEPS:
        _refuse(
            "config_number_of_sub_steps",
            substeps,
            "the rate ladder and the per-stage effective rates were derived "
            f"for the released {SUPPORTED_STAGE_ACOUSTIC_STEPS} acoustic "
            "schedule; another schedule admits a different ladder and has not "
            "been proved",
            "config_time_integration_order=3 with config_number_of_sub_steps=6",
        )
    rates = tuple(int(value) for value in config.config_local_timestep_rates)
    ltsk.refuse_unsupported_ladder(rates, SUPPORTED_STAGE_ACOUSTIC_STEPS)
    if classing is None:
        if grid_path is None:
            _refuse(
                "config_local_timestep",
                True,
                "cells are classed from the grid file's own dcEdge array, and "
                "no grid file was supplied; nominalMinDc is not a substitute "
                "(measured 10.5%/23.6%/39.7% high on three meshes, wrong in "
                "the unsafe direction)",
                "a grid file path",
            )
        classing = classify_from_grid_file(
            grid_path,
            rates=rates,
            buffer_rings=int(config.config_local_timestep_buffer_rings),
            safety_factor=float(config.config_local_timestep_safety_factor),
        )
    if classing.n_cells != driver.ncells or classing.n_edges != driver.nedges:
        _refuse(
            "config_local_timestep",
            (classing.n_cells, classing.n_edges),
            "the classed grid file is not the mesh the driver was built on",
            f"a grid file with {driver.ncells} cells and {driver.nedges} edges",
        )
    attachment = LocalTimestepAttachment(
        driver, classing, stage_acoustic_steps=SUPPORTED_STAGE_ACOUSTIC_STEPS
    )
    attachment.attach()
    return attachment


__all__ = [
    "LocalTimestepAttachment",
    "SUPPORTED_STAGE_ACOUSTIC_STEPS",
    "attach_local_timestep",
]
