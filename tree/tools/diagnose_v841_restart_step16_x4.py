#!/usr/bin/env python3
"""Bounded 15+1 restart-divergence instrument for the v8.4.1 CUDA proof.

Two arms, two separate processes:

* ``--mode baseline``: construct the stack exactly as the frozen proof does,
  run steps 1..15, download the F030 checkpoint (pickled to disk for the
  restored arm), then execute step 16 instrumented, dumping raw device arrays
  at every cutpoint.
* ``--mode restored``: in a genuinely fresh process, construct a fresh driver
  and backend, restore the pickled F030 checkpoint through the exact restart
  path of the frozen proof (``_construct_device_stack`` with
  ``state``/``saved_diagnostics``/``backend_restart``), verify bitwise F030
  rehydration, then execute the same instrumented step 16.

Cutpoints (each: npz of raw arrays + JSON manifest with per-array sha256):

  0. prestep       - restored/pre-step inputs: full MPAS state + saved
                     diagnostics, the complete backend restart payload
                     (held tendencies, radiation/PBL/GF and NoahMP state,
                     precipitation buckets, effective radii, cadence
                     counters), previous-surface-update carriers, clocks,
                     opaque seam export hash, device pointers.
  1. phase1_raw    - raw Arwen phase-1 tendencies du/dv/dtheta/dq*/h_diabatic
                     plus cadence flags (radiation_ran/surface_pbl_ran/
                     cumulus_ran) and call counters.
  2. phase1_held   - conservative coupled tendencies entering the dycore.
  3. post_dycore   - RK candidate atmosphere (pre-WSM6).
  4. post_wsm6     - post-WSM6/recovery candidate state + update carriers.
  5. committed     - committed MPAS atmosphere + full Arwen backend state.

This tool only READS model state between exactly the frozen proof's own
per-step operations; every dump is a pure device-to-host download.  The step
sequence is byte-for-byte the sequence of ``execute_composite_step`` in
``tools/run_cuda_v841_full_physics_x4.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_cuda_v841_full_physics_x4 as runner  # noqa: E402

CUTPOINTS = (
    "prestep",
    "phase1_raw",
    "phase1_held",
    "post_dycore",
    "post_wsm6",
    "committed",
)


def _host(value: Any) -> np.ndarray:
    import cupy as cp

    if isinstance(value, cp.ndarray):
        return np.ascontiguousarray(cp.asnumpy(value))
    return np.ascontiguousarray(np.asarray(value))


def _ptr(value: Any) -> int | None:
    data = getattr(value, "data", None)
    return None if data is None else int(getattr(data, "ptr", 0))


class CutpointDumper:
    def __init__(self, root: Path, arm: str) -> None:
        self.root = root
        self.arm = arm
        self.manifests: dict[str, Any] = {}

    def dump(self, cutpoint: str, arrays: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
        if cutpoint not in CUTPOINTS:
            raise ValueError(cutpoint)
        host_arrays = {name: _host(value) for name, value in sorted(arrays.items())}
        manifest = {
            "cutpoint": cutpoint,
            "arm": self.arm,
            "meta": dict(meta),
            "arrays": {
                name: {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": runner.array_sha256(value),
                }
                for name, value in host_arrays.items()
            },
        }
        npz_path = self.root / f"{cutpoint}.npz"
        np.savez(npz_path, **host_arrays)
        (self.root / f"{cutpoint}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=repr) + "\n",
            encoding="utf-8",
        )
        self.manifests[cutpoint] = manifest
        for name, record in sorted(manifest["arrays"].items()):
            print(f"[{self.arm}] {cutpoint}/{name} sha256={record['sha256']}")
        sys.stdout.flush()


def _backend_restart_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def walk(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                walk(f"{prefix}/{key}" if prefix else str(key), item[key])
        elif isinstance(item, np.ndarray):
            arrays[prefix] = item

    walk("backend", payload)
    return arrays


def _backend_restart_scalars(payload: Mapping[str, Any]) -> dict[str, Any]:
    scalars: dict[str, Any] = {}

    def walk(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                walk(f"{prefix}/{key}" if prefix else str(key), item[key])
        elif isinstance(item, np.ndarray):
            return
        else:
            scalars[prefix] = item if isinstance(item, (str, int, float, bool)) or item is None else repr(item)

    walk("backend", payload)
    return scalars


def dump_prestep(dumper: CutpointDumper, stack: Mapping[str, Any], previous: Mapping[str, Any] | None) -> None:
    driver = stack["driver"]
    backend = stack["backend"]
    state = driver.atmosphere.state
    saved = driver.atmosphere.saved
    restart = backend.restart_state()
    backend_fp = runner.fingerprint_nested_arrays(restart)
    arrays: dict[str, Any] = {
        "state/rho": state.rho,
        "state/rho_theta": state.rho_theta,
        "state/rho_u": state.rho_u,
        "state/rho_w": state.rho_w,
        "state/scalars": state.scalars,
        "saved/theta_m": saved.theta_m,
        "saved/exner": saved.exner,
        "saved/density_perturbation": saved.density_perturbation,
        "saved/rho_theta_perturbation": saved.rho_theta_perturbation,
        "saved/pressure_perturbation": saved.pressure_perturbation,
        "saved/normal_velocity": saved.normal_velocity,
        "saved/vertical_velocity": saved.vertical_velocity,
    }
    arrays.update(_backend_restart_arrays(restart))
    if previous is not None:
        for name, value in sorted(previous.items()):
            arrays[f"previous_surface/{name}"] = value
    carrier = stack.get("gf_dynamics_tendencies")
    if carrier is not None:
        arrays["gf_forcing/rthdynten"] = carrier.rthdynten
        arrays["gf_forcing/rqvdynten"] = carrier.rqvdynten
    meta = {
        "gf_forcing_time_seconds": (
            None if carrier is None else float(carrier.time_seconds)
        ),
        "model_time_seconds": float(state.time_seconds),
        "dt_seconds": runner.DT_SECONDS,
        "backend_restart_scalars": _backend_restart_scalars(restart),
        "backend_export_sha256": backend_fp["sha256"],
        "backend_phase": dict(backend.step_receipt()).get("phase"),
        "previous_surface_present": previous is not None,
        "device_pointers": {
            "state/rho": _ptr(state.rho),
            "state/scalars": _ptr(state.scalars),
            **(
                {}
                if previous is None
                else {
                    f"previous_surface/{name}": _ptr(value)
                    for name, value in sorted(previous.items())
                }
            ),
        },
    }
    dumper.dump("prestep", arrays, meta)


def run_instrumented_step16(
    stack: Mapping[str, Any],
    dumper: CutpointDumper,
    *,
    step_number: int = runner.CHECKPOINT_STEP + 1,
    previous: Mapping[str, Any] | None = None,
    derive_previous: bool = True,
) -> None:
    """Exactly execute_composite_step for one step, with read-only dumps."""

    from hexcore.cuda_physics_v841 import (
        clamp_wsm6_scalars_in_place_v841 as clamp,
        couple_raw_column_physics_v841 as couple,
        recover_post_rk_wsm6_state_v841 as recover,
    )

    driver = stack["driver"]
    backend = stack["backend"]
    scalar_names = runner.SCALAR_NAMES
    if derive_previous:
        # _run_steps derives the previous-surface carriers from the committed
        # diagnostic snapshot at the start of the continuation in BOTH arms.
        previous = runner._previous_surface_updates(stack)
    # The GF advective-forcing carrier the previous step formed (#327): the
    # baseline arm holds it on the stack from _run_steps; a restored arm
    # holds the checkpoint's re-seeded pair.  begin_step below consumes it
    # exactly as execute_composite_step does.
    gf_dynamics_tendencies = stack.get("gf_dynamics_tendencies")
    dump_prestep(dumper, stack, previous)

    start_atmosphere = driver.atmosphere
    start_state = start_atmosphere.state
    start_time = float(start_state.time_seconds)
    if start_time != (step_number - 1) * runner.DT_SECONDS:
        raise RuntimeError(
            f"instrumented step {step_number} must begin at "
            f"{(step_number - 1) * runner.DT_SECONDS}, got {start_time}"
        )

    raw = backend.begin_step(
        atmosphere=driver.atmosphere,
        scalar_names=scalar_names,
        dt=runner.DT_SECONDS,
        dynamics_tendencies=gf_dynamics_tendencies,
    )
    receipt = dict(backend.step_receipt())
    # h_diabatic is not forwarded on the raw carrier (MPAS declines the ARW
    # replay); read the seam's held output buffer directly, read-only.
    seam_h_diabatic = backend._seam._out["h_diabatic"]
    # Post-phase1 persisted seam arrays (held radiation/PBL/cumulus
    # tendencies among them), read-only via the seam's own manifest.
    seam_manifest = {
        f"post_phase1/{key}": value
        for key, value in backend._seam._restart_manifest().items()
    }
    dumper.dump(
        "phase1_raw",
        {
            "raw/du": raw.du,
            "raw/dv": raw.dv,
            "raw/dtheta": raw.dtheta,
            **{f"raw/d{name}": raw.dscalars[name] for name in scalar_names},
            "raw/h_diabatic": seam_h_diabatic,
            **seam_manifest,
        },
        {
            "time_seconds": float(raw.time_seconds),
            "cadence": receipt.get("cadence"),
            "arwen_step_index": receipt.get("arwen_step_index"),
            "device_pointers": {
                "raw/du": _ptr(raw.du),
                "raw/dv": _ptr(raw.dv),
            },
        },
    )
    edge_fields = driver.horizontal.recover_edge_fields(
        start_state.rho, start_state.rho_u
    )
    held = couple(
        raw,
        state=start_state,
        scalar_names=scalar_names,
        geometry=stack["physics_geometry"],
        rho_edge=edge_fields.rho_edge,
        kernel_cache=driver.cache,
    )
    dumper.dump(
        "phase1_held",
        {
            "held/rho": held.rho,
            "held/rho_u": held.rho_u,
            "held/rho_theta": held.rho_theta,
            "held/scalars": held.scalars,
        },
        {"time_seconds": float(held.time_seconds)},
    )
    candidate = driver.step_device_with_physics(held)
    cstate = candidate.atmosphere.state
    csaved = candidate.atmosphere.saved
    dumper.dump(
        "post_dycore",
        {
            "candidate/rho": cstate.rho,
            "candidate/rho_theta": cstate.rho_theta,
            "candidate/rho_u": cstate.rho_u,
            "candidate/rho_w": cstate.rho_w,
            "candidate/scalars": cstate.scalars,
            "candidate_saved/theta_m": csaved.theta_m,
            "candidate_saved/exner": csaved.exner,
            "candidate_saved/density_perturbation": csaved.density_perturbation,
            "candidate_saved/rho_theta_perturbation": csaved.rho_theta_perturbation,
            "candidate_saved/pressure_perturbation": csaved.pressure_perturbation,
            "candidate_saved/normal_velocity": csaved.normal_velocity,
            "candidate_saved/vertical_velocity": csaved.vertical_velocity,
        },
        {"time_seconds": float(cstate.time_seconds)},
    )
    clamp_d2h = clamp(
        cstate.scalars,
        scalar_names=scalar_names,
        kernel_cache=driver.cache,
    )
    update = backend.finish_step(
        atmosphere=candidate.atmosphere,
        scalar_names=scalar_names,
        dt=runner.DT_SECONDS,
    )
    recovery = recover(
        cstate,
        update,
        scalar_names=scalar_names,
        kernel_cache=driver.cache,
        phase2_dt_seconds=runner.DT_SECONDS,
        previous_surface_updates=previous,
    )
    dumper.dump(
        "post_wsm6",
        {
            "post/rho": cstate.rho,
            "post/rho_theta": cstate.rho_theta,
            "post/rho_u": cstate.rho_u,
            "post/rho_w": cstate.rho_w,
            "post/scalars": cstate.scalars,
            "update/theta": update.theta,
            "update/qv": update.qv,
            "update/qc": update.qc,
            "update/qr": update.qr,
            "update/qi": update.qi,
            "update/qs": update.qs,
            "update/qg": update.qg,
            "update/rainnc": update.rainnc,
            "update/rainncv": update.rainncv,
            "update/snownc": update.snownc,
            "update/snowncv": update.snowncv,
            "update/graupelnc": update.graupelnc,
            "update/graupelncv": update.graupelncv,
            "update/sr": update.sr,
            "update/effc": update.effc,
            "update/effi": update.effi,
            "update/effs": update.effs,
        },
        {
            "time_seconds": float(update.time_seconds),
            "clamp_d2h": clamp_d2h.as_dict() if hasattr(clamp_d2h, "as_dict") else repr(clamp_d2h),
        },
    )
    committed = driver.commit_post_wsm6_candidate(candidate, recovery)
    backend.commit_step()
    fstate = committed.atmosphere.state
    fsaved = committed.atmosphere.saved
    restart = backend.restart_state()
    backend_fp = runner.fingerprint_nested_arrays(restart)
    arrays = {
        "final/rho": fstate.rho,
        "final/rho_theta": fstate.rho_theta,
        "final/rho_u": fstate.rho_u,
        "final/rho_w": fstate.rho_w,
        "final/scalars": fstate.scalars,
        "final_saved/theta_m": fsaved.theta_m,
        "final_saved/exner": fsaved.exner,
        "final_saved/density_perturbation": fsaved.density_perturbation,
        "final_saved/rho_theta_perturbation": fsaved.rho_theta_perturbation,
        "final_saved/pressure_perturbation": fsaved.pressure_perturbation,
        "final_saved/normal_velocity": fsaved.normal_velocity,
        "final_saved/vertical_velocity": fsaved.vertical_velocity,
    }
    arrays.update(_backend_restart_arrays(restart))
    dumper.dump(
        "committed",
        arrays,
        {
            "model_time_seconds": float(fstate.time_seconds),
            "backend_export_sha256": backend_fp["sha256"],
            "backend_restart_scalars": _backend_restart_scalars(restart),
        },
    )


def _fresh_dir(path: Path, label: str) -> Path:
    selected = path.expanduser().absolute()
    if selected.exists():
        raise FileExistsError(f"{label} must be absent: {selected}")
    selected.mkdir(parents=True, exist_ok=False)
    return selected


def _authority_paths(assets_root: Path | None) -> dict[str, Path]:
    if assets_root is None:
        return runner.default_authority_paths()
    names = {
        "grid": "x4.163842.grid.nc",
        "static": "x4.163842.static.nc",
        "init": "x4.163842.init.nc",
        "native_f000": "history.2026-08-10_12.00.00.nc",
        "native_f030": "history.2026-08-10_12.30.00.nc",
        "native_f001": "history.2026-08-10_13.00.00.nc",
        "native_validation_receipt": "native-gwdo-authority-receipt.json",
        "native_launch_receipt": "native-gwdo-launch-receipt.json",
        "native_closure": "run-closure.status",
    }
    if set(names) != set(runner.AUTHORITY_PINS):
        raise RuntimeError("authority role inventory changed")
    return {role: assets_root / name for role, name in names.items()}


def _build_common(cache_root: Path, arwen_checkout: Path, assets_root: Path | None) -> dict[str, Any]:
    paths = _authority_paths(assets_root)
    # Source pins verify first: the checkout guard imports the seam manifest
    # from a pinned module, so that module's bytes are proven before its
    # constants are trusted.
    source_receipt = runner.require_frozen_execution_sources()
    runner.verify_arwen_checkout_git(arwen_checkout)
    authority_receipt = runner.verify_authorities(paths)
    host = runner._prepare_host_execution(paths, authority_receipt)
    from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(arwen_checkout)
    from hexcore.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    runner.gpu_memory_admission(cp)
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    return {
        "host": host,
        "cache": cache,
        "source_receipt": source_receipt,
        "authority_receipt": authority_receipt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "restored",
            "combined",
            "baseline-trace",
            "restored-trace",
            "proof-trace",
            "proof-dump-step",
            "restored-dump-step",
        ),
        required=True,
    )
    parser.add_argument(
        "--instrument-step",
        type=int,
        default=runner.CHECKPOINT_STEP + 1,
        help="step number to instrument in the *-dump-step modes",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="baseline: written; restored: read")
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=ROOT / "work" / "arwen19-mpas-column-corrected",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="directory carrying the real (non-symlink) authority files",
    )
    args = parser.parse_args(argv)
    arwen_checkout = args.arwen_checkout.expanduser().absolute()
    started = time.perf_counter()

    cache_root = _fresh_dir(args.cache_root, "cache root")
    dump_root = _fresh_dir(args.dump_root, "dump root")

    assets_root = None if args.assets_root is None else args.assets_root.expanduser().absolute()
    common = _build_common(cache_root, arwen_checkout, assets_root)
    host = common["host"]
    cache = common["cache"]

    if args.mode in ("baseline-trace", "restored-trace", "proof-trace"):
        trace: dict[int, Any] = {}

        def trace_observer(step: int, current_stack: Mapping[str, Any]) -> None:
            boundary = runner.fingerprint_execution_boundary(current_stack)
            trace[step] = {
                "atmosphere_sha256": boundary["atmosphere"]["sha256"],
                "backend_sha256": boundary["backend"]["sha256"],
                "atmosphere": runner._fingerprint_leaf_projection(
                    boundary["atmosphere"]
                ),
                "backend": runner._fingerprint_leaf_projection(
                    boundary["backend"]
                ),
            }
            print(
                f"[{args.mode}] step {step} atmosphere={boundary['atmosphere']['sha256'][:16]} "
                f"backend={boundary['backend']['sha256'][:16]}",
                flush=True,
            )

        if args.mode == "baseline-trace":
            stack = runner._construct_device_stack(
                host=host, cache=cache, arwen_checkout=arwen_checkout
            )
            print("[baseline-trace] running steps 1..15", flush=True)
            runner._run_steps(
                stack=stack,
                start_step=0,
                end_step=runner.CHECKPOINT_STEP,
                capture_steps=set(),
            )
        elif args.mode == "proof-trace":
            # Byte-for-byte the frozen proof's BASELINE arm schedule:
            # captures at steps 0 and 15, then the F030 checkpoint
            # download (pickled for the restored arm), then the second
            # _run_steps leg 15..30 with the F001 capture at step 30.
            checkpoint_path = args.checkpoint.expanduser().absolute()
            if checkpoint_path.exists():
                raise FileExistsError(checkpoint_path)
            stack = runner._construct_device_stack(
                host=host, cache=cache, arwen_checkout=arwen_checkout
            )
            print("[proof-trace] running steps 1..15 with proof captures", flush=True)
            runner._run_steps(
                stack=stack,
                start_step=0,
                end_step=runner.CHECKPOINT_STEP,
                capture_steps={0, runner.CHECKPOINT_STEP},
            )
            checkpoint = runner.download_driver_checkpoint(
                stack["driver"],
                stack["backend"],
                dynamics_tendencies=stack.get("gf_dynamics_tendencies"),
            )
            with checkpoint_path.open("wb") as stream:
                pickle.dump(checkpoint, stream, protocol=4)
            digest = runner.sha256_file(checkpoint_path)
            print(
                f"[proof-trace] checkpoint pickled: {checkpoint_path} sha256={digest}",
                flush=True,
            )
        else:
            checkpoint_path = args.checkpoint.expanduser().absolute()
            with checkpoint_path.open("rb") as stream:
                checkpoint = pickle.load(stream)
            stack = runner._construct_device_stack(
                host=host,
                cache=cache,
                arwen_checkout=arwen_checkout,
                state=checkpoint.state,
                saved_diagnostics=checkpoint.saved_diagnostics,
                backend_restart=checkpoint.backend_state,
                gf_dynamics_tendencies=checkpoint.gf_dynamics_tendencies,
            )
            restored_f030 = runner.fingerprint_execution_boundary(stack)
            runner.require_fingerprint_identity(
                "F030 restored MPAS atmosphere (trace)",
                checkpoint.atmosphere_fingerprint,
                restored_f030["atmosphere"],
            )
            runner.require_fingerprint_identity(
                "F030 restored Arwen backend (trace)",
                checkpoint.backend_fingerprint,
                restored_f030["backend"],
            )
        print(f"[{args.mode}] tracing steps 16..30", flush=True)
        # proof-trace and restored-trace mirror the frozen proof arms,
        # which both capture the F001 snapshot at step 30; the legacy
        # baseline-trace mode keeps its capture-free schedule.
        trace_captures = (
            set() if args.mode == "baseline-trace" else {runner.FULL_STEPS}
        )
        trace_snapshots, _, _ = runner._run_steps(
            stack=stack,
            start_step=runner.CHECKPOINT_STEP,
            end_step=runner.FULL_STEPS,
            capture_steps=trace_captures,
            boundary_observer=trace_observer,
        )
        if runner.FULL_STEPS in trace_snapshots:
            f001 = trace_snapshots[runner.FULL_STEPS]
            projection = {
                name: record["sha256"]
                for name, record in sorted(
                    f001["receipt"]["arrays"].items()
                )
            }
            (dump_root / "f001_snapshot_hashes.json").write_text(
                json.dumps(projection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{args.mode}] F001 snapshot captured; "
                f"{len(projection)} array hashes recorded",
                flush=True,
            )
        (dump_root / "trace.json").write_text(
            json.dumps(
                {str(step): value for step, value in sorted(trace.items())},
                indent=2,
                sort_keys=True,
                default=repr,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "ok", "mode": args.mode, "dump_root": str(dump_root)}))
        return 0

    if args.mode in ("proof-dump-step", "restored-dump-step"):
        target = int(args.instrument_step)
        if not (runner.CHECKPOINT_STEP + 1 <= target <= runner.FULL_STEPS):
            raise ValueError(
                f"--instrument-step must be in "
                f"[{runner.CHECKPOINT_STEP + 1}, {runner.FULL_STEPS}]"
            )
        trace: dict[int, Any] = {}

        def trace_observer(step: int, current_stack: Mapping[str, Any]) -> None:
            boundary = runner.fingerprint_execution_boundary(current_stack)
            trace[step] = {
                "atmosphere_sha256": boundary["atmosphere"]["sha256"],
                "backend_sha256": boundary["backend"]["sha256"],
            }
            print(
                f"[{args.mode}] step {step} atmosphere={boundary['atmosphere']['sha256'][:16]} "
                f"backend={boundary['backend']['sha256'][:16]}",
                flush=True,
            )

        checkpoint_path = args.checkpoint.expanduser().absolute()
        if args.mode == "proof-dump-step":
            # Reproduce the proof baseline arm exactly: captures {0,15},
            # checkpoint download, then traced composite steps up to
            # target-1, then the instrumented target step.
            if checkpoint_path.exists():
                raise FileExistsError(checkpoint_path)
            stack = runner._construct_device_stack(
                host=host, cache=cache, arwen_checkout=arwen_checkout
            )
            print("[proof-dump-step] steps 1..15 with proof captures", flush=True)
            runner._run_steps(
                stack=stack,
                start_step=0,
                end_step=runner.CHECKPOINT_STEP,
                capture_steps={0, runner.CHECKPOINT_STEP},
            )
            checkpoint = runner.download_driver_checkpoint(
                stack["driver"],
                stack["backend"],
                dynamics_tendencies=stack.get("gf_dynamics_tendencies"),
            )
            with checkpoint_path.open("wb") as stream:
                pickle.dump(checkpoint, stream, protocol=4)
            print(
                f"[proof-dump-step] checkpoint pickled: {checkpoint_path} "
                f"sha256={runner.sha256_file(checkpoint_path)}",
                flush=True,
            )
            arm = "proof"
        else:
            with checkpoint_path.open("rb") as stream:
                checkpoint = pickle.load(stream)
            stack = runner._construct_device_stack(
                host=host,
                cache=cache,
                arwen_checkout=arwen_checkout,
                state=checkpoint.state,
                saved_diagnostics=checkpoint.saved_diagnostics,
                backend_restart=checkpoint.backend_state,
                gf_dynamics_tendencies=checkpoint.gf_dynamics_tendencies,
            )
            restored_f030 = runner.fingerprint_execution_boundary(stack)
            runner.require_fingerprint_identity(
                "F030 restored MPAS atmosphere (dump-step)",
                checkpoint.atmosphere_fingerprint,
                restored_f030["atmosphere"],
            )
            runner.require_fingerprint_identity(
                "F030 restored Arwen backend (dump-step)",
                checkpoint.backend_fingerprint,
                restored_f030["backend"],
            )
            print("[restored-dump-step] F030 rehydration bitwise identical", flush=True)
            arm = "restored"
        previous_carrier: Mapping[str, Any] | None = None
        if target > runner.CHECKPOINT_STEP + 1:
            print(
                f"[{args.mode}] composite steps 16..{target - 1} (traced)",
                flush=True,
            )
            _, previous_carrier, _ = runner._run_steps(
                stack=stack,
                start_step=runner.CHECKPOINT_STEP,
                end_step=target - 1,
                capture_steps=set(),
                boundary_observer=trace_observer,
            )
            derive = False
        else:
            derive = True
        dumper = CutpointDumper(dump_root, arm)
        run_instrumented_step16(
            stack,
            dumper,
            step_number=target,
            previous=previous_carrier,
            derive_previous=derive,
        )
        (dump_root / "trace.json").write_text(
            json.dumps(
                {str(step): value for step, value in sorted(trace.items())},
                indent=2,
                sort_keys=True,
                default=repr,
            )
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "mode": args.mode,
            "instrument_step": target,
            "dump_root": str(dump_root),
            "checkpoint": str(checkpoint_path),
            "cutpoints": {
                name: dumper.manifests[name]["arrays"]
                for name in dumper.manifests
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        (dump_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "ok", "mode": args.mode, "dump_root": str(dump_root)}))
        return 0

    if args.mode in ("baseline", "combined"):
        checkpoint_path = args.checkpoint.expanduser().absolute()
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        stack = runner._construct_device_stack(
            host=host, cache=cache, arwen_checkout=arwen_checkout
        )
        print("[baseline] stack constructed; running steps 1..15", flush=True)
        runner._run_steps(
            stack=stack,
            start_step=0,
            end_step=runner.CHECKPOINT_STEP,
            capture_steps={0, runner.CHECKPOINT_STEP},
        )
        print("[baseline] 15 steps complete; downloading F030 checkpoint", flush=True)
        checkpoint = runner.download_driver_checkpoint(
            stack["driver"],
            stack["backend"],
            dynamics_tendencies=stack.get("gf_dynamics_tendencies"),
        )
        with checkpoint_path.open("wb") as stream:
            pickle.dump(checkpoint, stream, protocol=4)
        digest = runner.sha256_file(checkpoint_path)
        print(f"[baseline] checkpoint pickled: {checkpoint_path} sha256={digest}", flush=True)
        if args.mode == "combined":
            base_dir = dump_root / "uninterrupted"
            base_dir.mkdir()
            dumper = CutpointDumper(base_dir, "baseline")
            run_instrumented_step16(stack, dumper)
            # Reproduce the frozen proof's exact in-process restart topology:
            # drop the uninterrupted graph, free the pools, and rebuild the
            # restart stack with the SAME KernelCache in the SAME process.
            import cupy as cp

            del stack
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            restart_stack = runner._construct_device_stack(
                host=host,
                cache=cache,
                arwen_checkout=arwen_checkout,
                state=checkpoint.state,
                saved_diagnostics=checkpoint.saved_diagnostics,
                backend_restart=checkpoint.backend_state,
                gf_dynamics_tendencies=checkpoint.gf_dynamics_tendencies,
            )
            restored_f030 = runner.fingerprint_execution_boundary(restart_stack)
            runner.require_fingerprint_identity(
                "F030 restored MPAS atmosphere (in-process)",
                checkpoint.atmosphere_fingerprint,
                restored_f030["atmosphere"],
            )
            runner.require_fingerprint_identity(
                "F030 restored Arwen backend (in-process)",
                checkpoint.backend_fingerprint,
                restored_f030["backend"],
            )
            print("[combined] in-process F030 rehydration bitwise identical; stepping", flush=True)
            rest_dir = dump_root / "restored-inprocess"
            rest_dir.mkdir()
            dumper = CutpointDumper(rest_dir, "restored-inprocess")
            run_instrumented_step16(restart_stack, dumper)
        else:
            dumper = CutpointDumper(dump_root, "baseline")
            run_instrumented_step16(stack, dumper)
    else:
        checkpoint_path = args.checkpoint.expanduser().absolute()
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        print(f"[restored] checkpoint loaded from {checkpoint_path}", flush=True)
        stack = runner._construct_device_stack(
            host=host,
            cache=cache,
            arwen_checkout=arwen_checkout,
            state=checkpoint.state,
            saved_diagnostics=checkpoint.saved_diagnostics,
            backend_restart=checkpoint.backend_state,
            gf_dynamics_tendencies=checkpoint.gf_dynamics_tendencies,
        )
        restored_f030 = runner.fingerprint_execution_boundary(stack)
        runner.require_fingerprint_identity(
            "F030 restored MPAS atmosphere (fresh process)",
            checkpoint.atmosphere_fingerprint,
            restored_f030["atmosphere"],
        )
        runner.require_fingerprint_identity(
            "F030 restored Arwen backend (fresh process)",
            checkpoint.backend_fingerprint,
            restored_f030["backend"],
        )
        print("[restored] F030 rehydration is bitwise identical; stepping", flush=True)
        dumper = CutpointDumper(dump_root, "restored")
        run_instrumented_step16(stack, dumper)

    summary = {
        "mode": args.mode,
        "dump_root": str(dump_root),
        "checkpoint": str(args.checkpoint),
        "cutpoints": {name: dumper.manifests[name]["arrays"] for name in dumper.manifests},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (dump_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "mode": args.mode, "dump_root": str(dump_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
