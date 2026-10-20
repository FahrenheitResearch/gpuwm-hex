"""Rank-local checkpoints at a drained, committed shared step boundary."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

SCHEMA = "gpuwm-hex.samehost-checkpoint/v1"


def identity(*, host, grid_sha256, layouts_sha256, part_sha256):
    from hexcore.stochastic import StochasticConfig
    stochastic = StochasticConfig.parse(host.get("stochastic"))
    result = {
        "grid_sha256": grid_sha256, "layouts_sha256": layouts_sha256,
        "part_sha256": part_sha256,
        "configuration_sha256": hashlib.sha256(json.dumps(
            asdict(host["config"]), sort_keys=True, default=str).encode()).hexdigest(),
        "scalar_names": list(host["scalar_names"]),
        "physics_backend": host.get("physics_backend"),
        "seam_options": dict(host.get("seam_options", {})),
        "constructor_sha256": hashlib.sha256(json.dumps(
            host.get("constructor_receipt", {}), sort_keys=True, default=str).encode()).hexdigest(),
    }
    if stochastic.enabled:
        result["stochastic"] = stochastic.state_json()
    return result


def validate(checkpoint, *, expected, schedule, rank_count):
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA:
        raise ValueError("same-host restart needs a rank-local checkpoint; a single-card checkpoint lacks rank halo and physics state")
    if checkpoint.get("identity") != expected:
        raise ValueError("same-host checkpoint mesh, partition, configuration or species differ; its local arrays would resume on different columns or physics")
    if len(checkpoint.get("ranks", [])) != rank_count:
        raise ValueError("same-host checkpoint rank count differs from the requested partition")
    step = int(checkpoint["step"])
    if step not in schedule["capture_steps"] or not 0 <= step < int(schedule["steps"]):
        raise ValueError("checkpoint must be a history boundary before the end; --start-time/--hours name the ORIGINAL run")
    if schedule["valid_times"][step] != checkpoint["valid_time"]:
        raise ValueError("same-host checkpoint valid time differs from the requested original schedule")
    expected_seconds = step * float(checkpoint["dt_seconds"])
    for entry in checkpoint["ranks"]:
        if float(entry["state"]["time_seconds"]) != expected_seconds:
            raise ValueError("same-host checkpoint rank state does not share the committed clock")
        carrier = entry["forcing"]
        if carrier is not None and float(carrier["time_seconds"]) != expected_seconds:
            raise ValueError("same-host checkpoint forcing does not share the committed clock")
    return step


def download_rank(stack, layout, forcing):
    from .fullstep_lts_restart_v841 import download_class_state
    cp = stack["driver"].cp
    cp.cuda.get_current_stream().synchronize()
    carrier = None if forcing is None else {
        "rthdynten": np.ascontiguousarray(cp.asnumpy(forcing.rthdynten)),
        "rqvdynten": np.ascontiguousarray(cp.asnumpy(forcing.rqvdynten)),
        "time_seconds": float(forcing.time_seconds),
    }
    return {"state": download_class_state(stack["driver"], layout),
            "backend_state": stack["backend"].restart_state(), "forcing": carrier,
            "stochastic": (stack["driver"].stochastic.process.snapshot()
                           if hasattr(stack["driver"], "stochastic") else None)}


def constructor_kwargs(entry):
    from .fullstep_lts_restart_v841 import class_restart_state
    state, saved = class_restart_state(entry["state"])
    return {"state": state, "saved_diagnostics": saved,
            "backend_restart": entry["backend_state"],
            "stochastic_restart": entry.get("stochastic")}


def restore_forcing(entry, stack):
    from hexcore.cuda_driver import CudaV841GfDynamicsTendencies
    value = entry["forcing"]
    if value is None:
        return None
    cp = stack["driver"].cp
    carrier = CudaV841GfDynamicsTendencies(
        rthdynten=cp.asarray(value["rthdynten"]), rqvdynten=cp.asarray(value["rqvdynten"]),
        time_seconds=float(value["time_seconds"]))
    carrier.validate(cp=cp, n_vert_levels=int(stack["driver"].atmosphere.state.rho.shape[0]),
                     n_cells=int(stack["driver"].atmosphere.state.rho.shape[-1]))
    return carrier


def write(path, *, entries, expected, step, schedule, dt, extra=None):
    payload = {"schema": SCHEMA, "identity": expected, "ranks": entries,
               "step": step, "valid_time": schedule["valid_times"][step], "dt_seconds": dt}
    # ``extra``: records a restart reads besides the rank state (the IAU window, hexcore.iau_native).
    payload.update(extra or {})
    from hexcore.checkpoint_io import write_checkpoint
    path = Path(path)
    write_checkpoint(path, payload)
    return {"path": str(path), "bytes": path.stat().st_size, "step": step}
