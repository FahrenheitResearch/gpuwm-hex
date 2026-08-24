"""Split-explicit RK schedule and large-step state recovery.

Frozen sources:

* RK/acoustic schedule: ``mpas_atm_time_integration.F:638-686``;
* recovery after acoustic substeps:
  ``mpas_atm_time_integration.F:2715-2904``.

The state records below express logical MPAS fields.  They intentionally make
no C/F-contiguity or GPU structure-of-arrays decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .acoustic import edge_signs_on_cells
from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]


def accumulate_split_flux(
    current: FloatArray,
    accumulator: FloatArray | None,
) -> FloatArray:
    """Accumulate one dynamics-subcycle mass flux in frozen source order.

    ``atm_rk_dynamics_substep_finish`` uses ``current + accumulator`` for
    subcycles after the first one (F:6045-6050).  An explicit ufunc output
    keeps every operation in the array's RKIND; in particular, float32 inputs
    are rounded to binary32 after each source-order addition.
    """

    value = np.asarray(current)
    if value.dtype.kind != "f":
        raise TypeError("split mass flux must be floating")
    if accumulator is None:
        return value.copy()
    saved = np.asarray(accumulator)
    if saved.shape != value.shape:
        raise ValueError(
            f"split mass-flux shape changed from {saved.shape} to {value.shape}"
        )
    if saved.dtype != value.dtype:
        raise TypeError(
            f"split mass-flux dtype changed from {saved.dtype} to {value.dtype}"
        )
    result = np.empty_like(value)
    np.add(value, saved, out=result)
    return result


def finish_split_flux(
    accumulator: FloatArray,
    dynamics_splits: int,
) -> FloatArray:
    """Apply frozen ``1 / real(dynamics_split)`` after flux accumulation."""

    value = np.asarray(accumulator)
    if value.dtype.kind != "f":
        raise TypeError("split mass flux must be floating")
    if (
        not isinstance(dynamics_splits, (int, np.integer))
        or isinstance(dynamics_splits, (bool, np.bool_))
        or dynamics_splits < 1
    ):
        raise ConfigurationRefusal(
            "config_dynamics_split_steps",
            dynamics_splits,
            "the source-order flux average requires a positive integer count",
            "config_dynamics_split_steps>=1",
        )
    scalar = value.dtype.type
    inverse = scalar(1.0) / scalar(dynamics_splits)
    result = np.empty_like(value)
    np.multiply(value, inverse, out=result)
    return result


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


@dataclass(frozen=True, slots=True)
class RKStage:
    stage: int
    large_timestep: float
    acoustic_timestep: float
    acoustic_steps: int


@dataclass(frozen=True, slots=True)
class RKSchedule:
    order: int
    full_timestep: float
    dynamics_splits: int
    stages: tuple[RKStage, RKStage, RKStage]

    @classmethod
    def from_mpas(
        cls,
        dt: float,
        *,
        order: int = 3,
        acoustic_substeps: int = 6,
        dynamics_splits: int = 1,
    ) -> "RKSchedule":
        """Transcribe the source schedule, including integer half-step counts."""
        if dt <= 0.0:
            raise ConfigurationRefusal("config_dt", dt, "the timestep must be positive", "config_dt>0")
        if dynamics_splits < 1:
            raise ConfigurationRefusal(
                "config_dynamics_split_steps",
                dynamics_splits,
                "at least one dynamics split is required",
                "config_dynamics_split_steps=1",
            )
        if acoustic_substeps < 1:
            raise ConfigurationRefusal(
                "config_number_of_sub_steps",
                acoustic_substeps,
                "at least one acoustic substep is required",
                "config_number_of_sub_steps=6",
            )
        dt_dyn = dt / dynamics_splits
        if order == 3:
            large = (dt_dyn / 3.0, dt_dyn / 2.0, dt_dyn)
            acoustic_dt = (dt_dyn / 3.0, dt_dyn / acoustic_substeps, dt_dyn / acoustic_substeps)
            counts = (1, max(1, acoustic_substeps // 2), acoustic_substeps)
        elif order == 2:
            large = (dt_dyn / 2.0, dt_dyn / 2.0, dt_dyn)
            acoustic_dt = (dt_dyn / acoustic_substeps,) * 3
            counts = (
                max(1, acoustic_substeps // 2),
                max(1, acoustic_substeps // 2),
                acoustic_substeps,
            )
        else:
            raise ConfigurationRefusal(
                "config_time_integration_order",
                order,
                "frozen MPAS has only second- and third-order schedule branches",
                "config_time_integration_order=3",
            )
        stages = tuple(
            RKStage(index + 1, large[index], acoustic_dt[index], counts[index])
            for index in range(3)
        )
        return cls(order=order, full_timestep=dt, dynamics_splits=dynamics_splits, stages=stages)  # type: ignore[arg-type]


@dataclass(slots=True)
class RecoveryState:
    ww_avg: FloatArray
    rw_save: FloatArray
    w: FloatArray
    rw: FloatArray
    rw_p: FloatArray
    rtheta_p: FloatArray
    rtheta_pp: FloatArray
    rtheta_p_save: FloatArray
    rho_p: FloatArray
    rho_p_save: FloatArray
    rho_pp: FloatArray
    rho_zz: FloatArray
    ru_avg: FloatArray
    ru_save: FloatArray
    ru_p: FloatArray
    u: FloatArray
    ru: FloatArray
    exner: FloatArray
    pressure_p: FloatArray
    theta_m: FloatArray

    def copy(self) -> "RecoveryState":
        return RecoveryState(**{name: np.asarray(getattr(self, name)).copy() for name in self.__slots__})


@dataclass(frozen=True, slots=True)
class RecoveryBackground:
    rho_base: FloatArray
    exner_base: FloatArray
    rtheta_base: FloatArray
    zz: FloatArray


def recover_large_step_variables(
    mesh: object,
    state: RecoveryState,
    background: RecoveryBackground,
    *,
    dt: float,
    acoustic_steps: int,
    rk_step: int,
    rt_diabatic_tendency: FloatArray,
    fzm: FloatArray,
    fzp: FloatArray,
    cf1: float,
    cf2: float,
    cf3: float,
    zb_cell: FloatArray,
    zb3_cell: FloatArray,
    boundary_mask_cell: NDArray[np.integer[Any]] | None = None,
    relaxation_zone: int = 0,
    rgas: float = 287.0,
    cp: float = 1004.5,
    reference_pressure: float = 100_000.0,
) -> RecoveryState:
    """Recover full density, velocity, theta, Exner, and pressure fields.

    This is the closed/global scalar authority.  Regional boundary cells are
    honored when a mask is supplied; halo exchange remains caller-owned.
    """
    if acoustic_steps < 1:
        raise ConfigurationRefusal(
            "number_of_sub_steps", acoustic_steps, "the time average divides by this count", "number_of_sub_steps>=1"
        )
    if rk_step not in (1, 2, 3):
        raise ConfigurationRefusal("rk_step", rk_step, "MPAS RK stages are 1, 2, or 3", "rk_step=1")
    out = state.copy()
    nlev, ncells = out.rho_p.shape
    nedges = out.ru.shape[1]
    if out.rw.shape != (nlev + 1, ncells):
        raise ValueError("rw/w fields must have shape (nVertLevels+1, nCells)")
    if background.rho_base.shape != (nlev, ncells):
        raise ValueError("background fields do not match cell state")
    if boundary_mask_cell is None:
        boundary_mask_cell = np.zeros(ncells, dtype=np.int32)
    inv_ns = out.rho_p.dtype.type(1.0 / acoustic_steps)
    rcv = out.rho_p.dtype.type(rgas / (cp - rgas))

    out.rho_p[:] = out.rho_p_save + out.rho_pp
    out.rho_zz[:] = out.rho_p + background.rho_base
    if np.any(out.rho_zz == 0.0):
        raise FloatingPointError("zero full density during large-step recovery")
    out.w[0] = 0.0
    out.ww_avg[1:nlev] = out.rw_save[1:nlev] + out.ww_avg[1:nlev] * inv_ns
    out.rw[1:nlev] = out.rw_save[1:nlev] + out.rw_p[1:nlev]
    for level in range(1, nlev):
        out.w[level] = out.rw[level] / (
            fzm[level] * background.zz[level] + fzp[level] * background.zz[level - 1]
        )
    out.w[nlev] = 0.0

    if rk_step == 3:
        out.rtheta_p[:] = (
            out.rtheta_p_save
            + out.rtheta_pp
            - out.rtheta_p.dtype.type(dt) * out.rho_zz * rt_diabatic_tendency
        )
    else:
        out.rtheta_p[:] = out.rtheta_p_save + out.rtheta_pp
    full_rtheta = out.rtheta_p + background.rtheta_base
    out.theta_m[:] = full_rtheta / out.rho_zz
    if rk_step == 3:
        exner_argument = background.zz * out.exner.dtype.type(rgas / reference_pressure) * full_rtheta
        if np.any(exner_argument <= 0.0):
            raise FloatingPointError("non-positive Exner power argument")
        out.exner[:] = exner_argument**rcv
        out.pressure_p[:] = background.zz * out.pressure_p.dtype.type(rgas) * (
            out.exner * out.rtheta_p
            + background.rtheta_base * (out.exner - background.exner_base)
        )

    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    out.ru_avg[:] = out.ru_save + out.ru_avg * inv_ns
    out.ru[:] = out.ru_save + out.ru_p
    for edge in range(nedges):
        cell0, cell1 = cells_on_edge[edge]
        if cell0 < 0 or cell1 < 0:
            raise ConfigurationRefusal(
                "regional_boundary_density",
                f"edge {edge}",
                "full velocity recovery needs density on both sides",
                "explicit boundary-cell density",
            )
        out.u[:, edge] = (
            out.u.dtype.type(2.0)
            * out.ru[:, edge]
            / (out.rho_zz[:, cell0] + out.rho_zz[:, cell1])
        )

    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    signs = edge_signs_on_cells(mesh).astype(out.w.dtype, copy=False)
    if zb_cell.shape != (nlev + 1, ncells, edges.shape[1]):
        raise ValueError("zb_cell must have shape (nVertLevels+1, nCells, maxEdges)")
    if zb3_cell.shape != zb_cell.shape:
        raise ValueError("zb3_cell shape differs from zb_cell")
    for cell in range(ncells):
        if boundary_mask_cell[cell] > relaxation_zone:
            continue
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            bottom_flux = cf1 * out.ru[0, edge] + cf2 * out.ru[1, edge] + cf3 * out.ru[2, edge]
            out.w[0, cell] += signs[cell, slot] * (
                zb_cell[0, cell, slot]
                + np.copysign(out.w.dtype.type(1.0), bottom_flux) * zb3_cell[0, cell, slot]
            ) * bottom_flux
            for level in range(1, nlev):
                flux = fzm[level] * out.ru[level, edge] + fzp[level] * out.ru[level - 1, edge]
                out.w[level, cell] += signs[cell, slot] * (
                    zb_cell[level, cell, slot]
                    + np.copysign(out.w.dtype.type(1.0), flux) * zb3_cell[level, cell, slot]
                ) * flux
        out.w[0, cell] /= (
            cf1 * out.rho_zz[0, cell]
            + cf2 * out.rho_zz[1, cell]
            + cf3 * out.rho_zz[2, cell]
        )
        for level in range(1, nlev):
            out.w[level, cell] /= (
                fzm[level] * out.rho_zz[level, cell]
                + fzp[level] * out.rho_zz[level - 1, cell]
            )
    return out
