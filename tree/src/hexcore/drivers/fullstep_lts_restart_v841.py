"""Checkpoint and resume of an active full-step LTS forecast at a macro boundary.

A full-step LTS run has one dycore driver and one column-physics backend per
rate class, each on its own local sub-mesh (owned cells, the class's scratch
zone of coarser cells it advances itself, and halo rings), plus the interface
state the classes hand each other.  A checkpoint written at a macro boundary
-- every class at the same instant, every interface register empty -- holds:

* per class: the driver's prognostic state and saved diagnostics on EVERY local
  column (a scratch column's atmosphere feeds the class's owned cells inside
  the next class step, so it is not reconstructible from the owners), the
  backend's ``restart_state()`` (its land state covers the scratch columns too)
  and the GF advective-forcing carrier the class's last step formed;
* the accepted-state mailboxes (``G_cell``/``G_face``/``G_edge`` and the scalar
  mailbox).  The trajectory planes hold a finer class's committed endpoint from
  BEFORE its own reflux; the next macro's coarser steps read them as their start
  states, and the post-reflux driver state cannot reproduce them;
* the scalar interface's reserves, per-class statistics and its hold array;
* the class clock (``tick``, ``class_tick``, ``start_seconds``) so every time
  the resumed run forms is the same float as the uninterrupted run's.

A plain resume rebuilds the class stacks from these bytes and is bit-identical
to the uninterrupted run (measured by history hashes; see the branch report).
A DA-cycling resume (``--da-analysis``) replaces the dycore state by the
recoupled analysis on every local column of every class and starts the
interface from that state, exactly as a cold start would, while the physics
backends resume from the checkpoint -- the native DA restart's semantics.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

CHECKPOINT_SCHEMA = "gpuwm-hex.fullstep-lts-checkpoint/v1"

_SAVED = ("theta_m", "exner", "density_perturbation", "rho_theta_perturbation",
          "pressure_perturbation", "normal_velocity", "vertical_velocity")
_EDGE_FIELDS = frozenset(("rho_u", "normal_velocity"))
_ACTIVE_ARRAYS = ("G_cell", "G_face", "G_edge")
_REGISTERS = ("reg_mass_fine", "reg_mass_coarse", "reg_theta_fine", "reg_theta_coarse")
_CLASS_SCALAR_ARRAYS = ("reserve", "content_start", "air_start", "stats", "amounts")


def plan_digest(plan: Any) -> str:
    """The class decomposition a checkpoint's local arrays are indexed by."""

    digest = hashlib.sha256()
    for tables in plan.classes:
        layout = tables.layout
        digest.update(np.int64([tables.index, tables.rate, layout.n_owned_cells, layout.n_owned_edges,
                                layout.n_local_cells, layout.n_local_edges]).tobytes())
        digest.update(np.ascontiguousarray(layout.cell_l2g, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(layout.edge_l2g, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _host(cp: Any, array: Any, extent: int | None = None) -> np.ndarray:
    value = cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)
    if extent is not None and value.ndim >= 1 and value.shape[-1] > extent:
        # a limited-area driver carries native's garbage column past the real
        # local entities; the restore re-pads it exactly as a cold start does
        value = value[..., :extent]
    return np.ascontiguousarray(value)


def download_class_state(driver: Any, layout: Any) -> dict[str, Any]:
    """One class driver's state and saved diagnostics on every local column, on the host."""

    cp = driver.cp
    state = driver.atmosphere.state
    saved = driver.atmosphere.saved
    cells, edges = int(layout.n_local_cells), int(layout.n_local_edges)
    out = {"time_seconds": float(state.time_seconds)}
    for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
        out[name] = _host(cp, getattr(state, name), edges if name in _EDGE_FIELDS else cells)
    out["saved"] = {name: _host(cp, getattr(saved, name), edges if name in _EDGE_FIELDS else cells)
                    for name in _SAVED}
    return out


def class_restart_state(entry: Mapping[str, Any]) -> tuple[Any, Any]:
    """``(PrognosticState, DrySavedDiagnostics)`` for a class stack constructor."""

    from hexcore.driver import DrySavedDiagnostics
    from hexcore.state import PrognosticState

    state = PrognosticState(rho=entry["rho"], rho_theta=entry["rho_theta"], rho_u=entry["rho_u"],
                            rho_w=entry["rho_w"], scalars=entry["scalars"],
                            time_seconds=float(entry["time_seconds"]))
    saved = DrySavedDiagnostics(**{name: entry["saved"][name] for name in _SAVED})
    return state, saved


def slice_global_state(state: Any, saved: Any, layout: Any) -> tuple[Any, Any]:
    """A global (analysis) state on one class's local columns, in local order."""

    from hexcore.driver import DrySavedDiagnostics
    from hexcore.state import PrognosticState

    cells = np.asarray(layout.cell_l2g, dtype=np.int64)
    edges = np.asarray(layout.edge_l2g, dtype=np.int64)

    def take(array: Any, edge: bool) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(array)[..., edges if edge else cells])

    local_state = PrognosticState(
        rho=take(state.rho, False), rho_theta=take(state.rho_theta, False), rho_u=take(state.rho_u, True),
        rho_w=take(state.rho_w, False), scalars=take(state.scalars, False),
        time_seconds=float(state.time_seconds))
    local_saved = DrySavedDiagnostics(**{name: take(getattr(saved, name), name in _EDGE_FIELDS)
                                         for name in _SAVED})
    return local_state, local_saved


def download_checkpoint(*, active: Any, stacks: Sequence[Mapping[str, Any]], forcing: Sequence[Any],
                        class_calls: Sequence[Mapping[str, Any]], step: int, label: str, valid_time: str,
                        base_dt: float, top: int, rates: np.ndarray, n_cells: int, n_edges: int,
                        global_state: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The whole resumable state of a full-step LTS run at a macro boundary."""

    cp = active.cp
    cp.cuda.get_current_stream().synchronize()
    if not active.registers_empty():
        raise RuntimeError(
            "an LTS checkpoint was asked for with a non-empty interface register: the macro "
            "boundary is the only instant at which no transfer is half applied")
    if any(int(tick) != int(active.tick) for tick in active.class_tick):
        raise RuntimeError("an LTS checkpoint was asked for while the classes stand at different ticks")
    expected = float(step) * float(base_dt)
    classes = []
    for klass, stack, carrier, calls in zip(active.classes, stacks, forcing, class_calls):
        driver = stack["driver"]
        if float(driver.atmosphere.state.time_seconds) != expected:
            raise RuntimeError(f"class {klass.index} stands at {driver.atmosphere.state.time_seconds} s, "
                               f"not the checkpoint's {expected} s")
        layout = klass.tables.layout
        gf = None
        if carrier is not None:
            gf = {"rthdynten": _host(cp, carrier.rthdynten), "rqvdynten": _host(cp, carrier.rqvdynten),
                  "time_seconds": float(carrier.time_seconds)}
        classes.append({
            "index": int(klass.index), "rate": int(klass.rate),
            "state": download_class_state(driver, layout),
            "backend_state": stack["backend"].restart_state(),
            "gf_dynamics_tendencies": gf,
            "class_calls": {key: (list(value) if isinstance(value, list) else value)
                            for key, value in calls.items()},
        })
    interface: dict[str, Any] = {name: _host(cp, getattr(active, name)) for name in _ACTIVE_ARRAYS}
    scalars = None
    if active.scalars is not None:
        owner = active.scalars
        scalars = {"G": _host(cp, owner.G), "hold": _host(cp, owner.hold),
                   "classes": [{name: _host(cp, getattr(mine, name)) for name in _CLASS_SCALAR_ARRAYS}
                               for mine in owner.classes]}
    return {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "label": label,
        "valid_time": valid_time,
        "dt_seconds": float(base_dt),
        "top_rate": int(top),
        "n_cells": int(n_cells),
        "n_edges": int(n_edges),
        "rates_sha256": hashlib.sha256(np.ascontiguousarray(rates).tobytes()).hexdigest(),
        "plan_sha256": plan_digest(active.plan),
        "clock": {"start_seconds": float(active.start_seconds), "tick": int(active.tick),
                  "class_tick": [int(v) for v in active.class_tick]},
        "classes": classes,
        "interface": interface,
        "scalar_interface": scalars,
        # the owned-union state in global order: what a DA system reads, and
        # the bytes the restart identity line hashes
        "global_state": dict(global_state),
    }


def validate_checkpoint(checkpoint: Mapping[str, Any], *, base_dt: float, top: int, rates: np.ndarray,
                        plan: Any, n_cells: int, schedule: Mapping[str, Any]) -> int:
    """Refuse a checkpoint this run cannot resume; return its step."""

    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "--restart-from is not a full-step LTS checkpoint: an active LTS run resumes one driver "
            "and one physics backend per rate class, which a single-stack checkpoint does not hold")
    if float(checkpoint["dt_seconds"]) != float(base_dt) or int(checkpoint["n_cells"]) != int(n_cells):
        raise ValueError("the LTS checkpoint was written on a different mesh or base timestep")
    digest = hashlib.sha256(np.ascontiguousarray(rates).tobytes()).hexdigest()
    if checkpoint["rates_sha256"] != digest or checkpoint["plan_sha256"] != plan_digest(plan):
        raise ValueError(
            "the LTS checkpoint's rate classes differ from this run's: its local arrays are indexed "
            "by another class decomposition, so every class would resume on the wrong columns "
            "(the rates come from the --init file and the LTS options; resume with the original ones)")
    if int(checkpoint["top_rate"]) != int(top):
        raise ValueError("the LTS checkpoint was written with a different top rate (macro length)")
    step = int(checkpoint["step"])
    if step not in schedule["capture_steps"] or step >= int(schedule["steps"]):
        raise ValueError("the checkpoint step must be a history frame of this run and before its end; "
                         "--start-time/--hours name the ORIGINAL run")
    if step % int(top):
        raise ValueError("the LTS checkpoint is not on a macro boundary")
    if schedule["valid_times"][step] != checkpoint["valid_time"]:
        raise ValueError("the checkpoint valid time differs from this schedule's")
    return step


def restore_interface(active: Any, checkpoint: Mapping[str, Any], *, interface: bool) -> None:
    """Put the class clock back; with ``interface`` also the mailboxes and scalar reserves."""

    cp = active.cp
    clock = checkpoint["clock"]
    active.start_seconds = float(clock["start_seconds"])
    active.tick = int(clock["tick"])
    active.class_tick = [int(v) for v in clock["class_tick"]]
    if len(active.class_tick) != len(active.classes):
        raise ValueError("the checkpoint holds a different number of rate classes")
    if not interface:
        return

    def put(target: Any, value: np.ndarray, what: str) -> None:
        if tuple(target.shape) != tuple(value.shape) or target.dtype != value.dtype:
            raise ValueError(f"checkpoint {what} has shape {value.shape} {value.dtype}, the run "
                             f"built {tuple(target.shape)} {target.dtype}")
        target.set(np.ascontiguousarray(value))

    for name in _ACTIVE_ARRAYS:
        put(getattr(active, name), checkpoint["interface"][name], name)
    for name in _REGISTERS:
        getattr(active, name).fill(0)
    saved = checkpoint["scalar_interface"]
    if (saved is None) != (active.scalars is None):
        raise ValueError("the checkpoint and this run disagree on whether scalars cross the interface")
    if saved is not None:
        owner = active.scalars
        put(owner.G, saved["G"], "scalar mailbox")
        put(owner.hold, saved["hold"], "scalar hold")
        owner.reg_scalar.fill(0)
        for mine, entry in zip(owner.classes, saved["classes"]):
            for name in _CLASS_SCALAR_ARRAYS:
                put(getattr(mine, name), entry[name], f"class {mine.klass.index} scalar {name}")
    cp.cuda.get_current_stream().synchronize()


def restore_forcing(cp: Any, entry: Mapping[str, Any] | None, *, time_seconds: float) -> Any:
    if entry is None:
        return None
    from hexcore.cuda_driver import CudaV841GfDynamicsTendencies

    carrier = CudaV841GfDynamicsTendencies(
        rthdynten=cp.asarray(np.ascontiguousarray(entry["rthdynten"], dtype=np.float32)),
        rqvdynten=cp.asarray(np.ascontiguousarray(entry["rqvdynten"], dtype=np.float32)),
        time_seconds=float(entry["time_seconds"]))
    width = int(np.asarray(entry["rthdynten"]).shape[-1])
    carrier.validate(cp=cp, n_vert_levels=int(np.asarray(entry["rthdynten"]).shape[0]), n_cells=width)
    if float(carrier.time_seconds) != float(time_seconds):
        raise RuntimeError("a restored class GF forcing carrier is not stamped with the restored model time")
    return carrier


def assemble_seam_leaves(stacks: Sequence[Mapping[str, Any]], layouts: Sequence[Any], *,
                         n_global_cells: int, seam_leaf_fields: Any) -> dict[str, np.ndarray]:
    """Each class backend's seam leaves on its OWNED columns, in global column order.

    A leaf's column axis is the one whose length is the class's local column
    count (native's garbage column included on a limited-area mesh); a leaf
    with no such axis carries no column data and must agree across classes.
    """

    merged: dict[str, np.ndarray] = {}
    for stack, layout in zip(stacks, layouts):
        leaves = seam_leaf_fields(stack["backend"].restart_state())
        local, owned = int(layout.n_local_cells), int(layout.n_owned_cells)
        l2g = np.asarray(layout.cell_l2g, dtype=np.int64)[:owned]
        for name, value in leaves.items():
            value = np.asarray(value)
            axes = [axis for axis, size in enumerate(value.shape) if size in (local, local + 1)]
            if not axes:
                if name in merged and not np.array_equal(merged[name], value):
                    raise ValueError(f"seam leaf {name} has no column axis and differs between classes")
                merged.setdefault(name, value)
                continue
            axis = axes[-1]
            if name not in merged:
                shape = list(value.shape)
                shape[axis] = int(n_global_cells)
                merged[name] = np.zeros(shape, dtype=value.dtype)
            target = np.moveaxis(merged[name], axis, -1)
            target[..., l2g] = np.moveaxis(value, axis, -1)[..., :owned]
    return merged


def write_class_background(forecast: Any, *, active: Any, stacks: Sequence[Mapping[str, Any]], host: Mapping[str, Any],
                           snapshot: Mapping[str, Any], label: str, valid_time: str, soil_levels: int,
                           init_fallback: Mapping[str, np.ndarray], output_root: Any) -> dict[str, Any]:
    """The JEDI background file of a full-step LTS frame: the native writer on assembled global views."""

    from types import SimpleNamespace

    layouts = [klass.tables.layout for klass in active.classes]
    leaves = assemble_seam_leaves(stacks, layouts, n_global_cells=int(layouts[0]._global_cells),
                                  seam_leaf_fields=forecast._seam_leaf_fields)
    pressure = active.gather(("pressure_perturbation",))["pressure_perturbation"]
    view = {"driver": SimpleNamespace(cp=active.cp, atmosphere=SimpleNamespace(
        saved=SimpleNamespace(pressure_perturbation=pressure)))}
    return forecast.write_jedi_background(
        output_root=output_root, stack=view, host=host, snapshot=snapshot, label=label, valid_time=valid_time,
        soil_levels=int(soil_levels), init_fallback=init_fallback,
        backend_state={"seam": {"arrays": {f"fields/{name}": value for name, value in leaves.items()}}},
        effective_radii=None)
