"""Incremental analysis update (IAU) forcing for the full-step local-time-step executor.

An analysis increment enters the forecast as a constant tendency over a window instead of being
inserted at once: the forecast starts from the BACKGROUND, and for model time t in [0, W) every
class step adds increment / W to the held physics tendencies of the coupled prognostics, exactly
the slot MPAS-A's own IAU uses (mpas_atm_iau.F atm_add_tend_anal_incr adds to tend_ru_physics,
tend_rtheta_physics, tend_rho_physics and the scalar tendencies once per outer step, before the
dynamics substeps). The dynamics then integrate the forcing through every RK stage and acoustic
substep, so the increment is admitted at the rate the flow can adjust to and the fast
gravity/acoustic response a sudden insertion excites is not launched (Bloom et al. 1996).

The increment is formed in hex's own coupled variables with hex's own conversion of an init file
(``hexcore.driver.load_mpas_initial_state``, the transcription of atm_init_coupled_diagnostics the
forecast itself starts from), as analysis minus background:

    d rho_zz,  d (rho_zz theta_m),  d ru (edge),  d (rho_zz q_s)  per model scalar

so after the window the forced state has received the whole increment (to the dynamics' own
response), and a zero increment is no forcing at all. Vertical momentum is not forced: native
MPAS IAU has no w increment either (Registry tend_iau carries u, rho, theta and moisture only).

Over a class step [t0, t0 + dt] the added tendency is increment * overlap([t0, t0+dt], [0, W]) /
(dt * W), so with LTS (classes stepping at different lengths) every class receives the same total
over the window, and a window that is a whole number of the coarsest step (the usual case) puts
each class step either wholly in or wholly out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Mass species that load the column (surface pressure); number concentrations and
#: rime volumes are not mass.
MASS_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
GRAVITY = 9.80616


def _load(path: Path, prepared: Any, scalar_names: Sequence[str]):
    from netCDF4 import Dataset

    from hexcore.driver import load_mpas_initial_state
    from hexcore.drivers.run_cuda_v841_forecast import select_initial_scalar_names

    with Dataset(path, "r") as dataset:
        source = select_initial_scalar_names(dataset.variables, tuple(scalar_names))
    state, _reference = load_mpas_initial_state(
        path, prepared.mesh, prepared.vertical, scalar_names=source,
        terrain_metrics=getattr(prepared, "terrain_metrics", None), cold_start=True)
    return state, tuple(source)


def analysis_increment(*, prepared: Any, scalar_names: Sequence[str], analysis: Path,
                       background: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Coupled-variable increment (analysis - background) in global order, and its receipt."""

    an, an_names = _load(Path(analysis), prepared, scalar_names)
    bg, bg_names = _load(Path(background), prepared, scalar_names)
    rho_a = np.asarray(an.rho, dtype=np.float64)
    rho_b = np.asarray(bg.rho, dtype=np.float64)
    increment = {
        "rho": (rho_a - rho_b).astype(np.float32),
        "rho_theta": (np.asarray(an.rho_theta, np.float64) - np.asarray(bg.rho_theta, np.float64)).astype(np.float32),
        "rho_u": (np.asarray(an.rho_u, np.float64) - np.asarray(bg.rho_u, np.float64)).astype(np.float32),
    }
    scalars = np.zeros((len(scalar_names),) + rho_a.shape, dtype=np.float32)
    species: dict[str, str] = {}
    for index, name in enumerate(scalar_names):
        if name in an_names and name in bg_names:
            qa = np.asarray(an.scalars[an_names.index(name)], np.float64)
            qb = np.asarray(bg.scalars[bg_names.index(name)], np.float64)
            scalars[index] = (rho_a * qa - rho_b * qb).astype(np.float32)
            species[name] = "rho_zz q increment"
        else:
            # A species neither file carries (cold-augmented) has no analysis to admit.
            species[name] = "not in both files: zero"
    increment["scalars"] = scalars
    for name, value in increment.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"IAU increment {name} is not finite; the analysis or the "
                                     "background file is damaged")
    def rms(a):
        return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))
    receipt = {
        "analysis": str(analysis), "background": str(background),
        "rms": {"rho_zz": rms(increment["rho"]), "rho_theta_m": rms(increment["rho_theta"]),
                "ru": rms(increment["rho_u"]),
                **{f"rho_zz_{n}": rms(scalars[i]) for i, n in enumerate(scalar_names)
                   if species[n] != "not in both files: zero"}},
        "max_abs": {"ru": float(np.abs(increment["rho_u"]).max()),
                    "rho_theta_m": float(np.abs(increment["rho_theta"]).max())},
        "species": species,
        "nonzero_columns": int(np.count_nonzero(np.any(increment["rho_theta"] != 0, axis=0))),
    }
    return increment, receipt


class IauForcing:
    """Per-class device copies of the increment and the window arithmetic."""

    def __init__(self, increment: Mapping[str, np.ndarray], layouts: Sequence[Any], *,
                 window_seconds: float, cp: Any, window_start: float = 0.0,
                 devices: Sequence[int] | None = None, cell_widths: Sequence[int] | None = None):
        if not window_seconds > 0:
            raise ValueError("the IAU window must be positive; a zero window is direct insertion, "
                             "which is the run without --iau-analysis")
        self.window = float(window_seconds)
        # Model time (seconds since the run's original start) where the window begins: 0 on a cold start,
        # the carried start when a restart continues the window (hexcore.iau_native.resolve_restart_window).
        self.window_start = float(window_start)
        self.cp = cp
        # An exactly zero increment leaves the tendency word untouched (y + a*0 would turn a -0
        # tendency into +0), so a zero increment is the background run bit for bit.
        self._axpy = cp.ElementwiseKernel("float32 x, float32 a", "float32 y",
                                          "if (x != 0.0f) y = y + a * x;", "hex_iau_axpy")
        self.classes: list[dict[str, Any] | None] = []
        from contextlib import nullcontext

        for index, layout in enumerate(layouts):
            cells = np.asarray(layout.cell_l2g, dtype=np.int64)
            edges = np.asarray(layout.edge_l2g, dtype=np.int64)
            pad = 0 if cell_widths is None else int(cell_widths[index]) - cells.size

            def take(values, axis):
                taken = np.take(values, cells, axis=axis)
                if pad > 0:
                    # A same-host rank of a limited-area mesh carries native's garbage column past its
                    # local cells; it has no analysis, so its increment is zero.
                    shape = list(taken.shape)
                    shape[axis] = pad
                    taken = np.concatenate([taken, np.zeros(shape, taken.dtype)], axis=axis)
                return np.ascontiguousarray(taken)

            # Each class's copies live on the card its class steps on (a same-host rank's device).
            with (nullcontext() if devices is None else cp.cuda.Device(int(devices[index]))):
                self.classes.append({
                    "rho": cp.asarray(take(increment["rho"], 1)),
                    "rho_theta": cp.asarray(take(increment["rho_theta"], 1)),
                    "rho_u": cp.asarray(np.ascontiguousarray(increment["rho_u"][:, edges])),
                    "scalars": cp.asarray(take(increment["scalars"], 2)),
                })
        self.applied_seconds = [0.0] * len(layouts)
        self.calls = [0] * len(layouts)
        self.first_start = [None] * len(layouts)

    def add(self, index: int, held: Any, start: float, dt: float, **_context: Any) -> None:
        """Add this class step's share of the increment to the held physics tendencies.

        ``_context`` (the step's driver, edge fields and geometry) is what the native route
        (hexcore.iau_native) reads; this route's increment is already in coupled variables.
        """

        arrays = self.classes[index]
        if arrays is None:
            return
        end = self.window_start + self.window
        overlap = max(0.0, min(start + dt, end) - max(start, self.window_start))
        if self.first_start[index] is None:
            self.first_start[index] = float(start)
        if overlap <= 0.0:
            if start >= end:
                self.classes[index] = None   # window over: release the device copies
            return
        a = np.float32(overlap / (dt * self.window))
        for name in ("rho", "rho_theta", "rho_u", "scalars"):
            target = getattr(held, name)
            if target.shape != arrays[name].shape:
                raise ValueError(f"IAU {name} increment {arrays[name].shape} does not match the "
                                 f"class tendency {target.shape}")
            self._axpy(arrays[name], a, target)
        self.applied_seconds[index] += overlap
        self.calls[index] += 1

    def receipt(self) -> dict[str, Any]:
        # The part of the window each class's run covers: all of it on a cold start, the rest of it when
        # a restart continues a carried window.
        due = [self.window if first is None else
               max(0.0, self.window_start + self.window - max(float(first), self.window_start))
               for first in self.first_start]
        return {"window_seconds": self.window, "window_start_seconds": self.window_start,
                "class_applied_seconds": self.applied_seconds, "class_due_seconds": due,
                "class_calls": self.calls, "class_first_start_seconds": self.first_start,
                "complete": all(abs(s - d) < 1e-6 for s, d in zip(self.applied_seconds, due))}


@dataclass
class BalanceSeries:
    """Surface-pressure tendency at every macro boundary, from the column mass of owned cells.

    ps - p_top = g * sum_k rho_zz (1 + sum mass species) dzw, so the column mass change between
    two macro boundaries gives dps/dt in each owned cell. The mean absolute value over the domain
    (and over the finest class, where the storms and the radar increments are) is the standard
    noise measure for initialization shock (Lynch and Huang 1992; Lei and Whitaker 2016).
    """

    path: Path
    mass_index: tuple[int, ...]
    dzw: Any
    previous: list[Any] = field(default_factory=list)
    previous_time: float | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(cls, path: Path, scalar_names: Sequence[str], dzw: np.ndarray, cp: Any) -> "BalanceSeries":
        index = tuple(i for i, n in enumerate(scalar_names) if n in MASS_SPECIES)
        return cls(Path(path), index, cp.asarray(np.asarray(dzw, dtype=np.float64))[:, None])

    def sample(self, active: Any, time_seconds: float, cp: Any) -> dict[str, Any] | None:
        masses, rates = [], []
        for klass in active.classes:
            state = klass.driver.atmosphere.state
            n = klass.n_owned_cells
            q = cp.zeros(state.rho[:, :n].shape, dtype=cp.float64)
            for i in self.mass_index:
                q += state.scalars[i, :, :n].astype(cp.float64)
            masses.append(cp.sum(state.rho[:, :n].astype(cp.float64) * (1.0 + q) * self.dzw, axis=0))
            rates.append(int(klass.tables.rate))
        row = None
        if self.previous_time is not None:
            dt = time_seconds - self.previous_time
            values, fine = [], []
            for now, before, rate in zip(masses, self.previous, rates):
                d = cp.asnumpy(cp.abs(now - before)) * GRAVITY / dt * 36.0   # hPa per hour
                values.append(d)
                if rate == 1:
                    fine.append(d)
            allv = np.concatenate(values)
            finev = np.concatenate(fine) if fine else allv
            row = {"time_seconds": time_seconds, "interval_seconds": dt,
                   "mean_abs_dps_dt_hpa_per_h": float(allv.mean()),
                   "fine_mean_abs_dps_dt_hpa_per_h": float(finev.mean()),
                   "fine_p99_abs_dps_dt_hpa_per_h": float(np.percentile(finev, 99)),
                   "fine_max_abs_dps_dt_hpa_per_h": float(finev.max()),
                   "fine_cells": int(finev.size), "cells": int(allv.size)}
            self.rows.append(row)
            with self.path.open("a", encoding="utf-8") as stream:
                import json
                stream.write(json.dumps(row) + "\n")
        self.previous = masses
        self.previous_time = time_seconds
        return row
