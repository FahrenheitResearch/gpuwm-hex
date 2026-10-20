#!/usr/bin/env python3
"""Per-step cost profile of one ``gpuwm-hex forecast``, installed from outside.

Nothing here is on the default path: the door runs unchanged unless this
script is the entry point.  It runs the real door in-process
(``hexcore.cli.main(["forecast", ...])``) with timers installed by
monkeypatching around the named phases of every composite step -- the
physics seam's two phases and what they call inside the pinned engine, the
dynamics half and its RK internals, the post-RK clamp/recover/commit, the
per-step host health gate, and the history capture and write -- counts every
``cupy.asnumpy`` host copy by caller file:line, and opens two disjoint step
windows: one bracketed by ``cudaProfilerStart``/``Stop`` so that
``nsys profile --capture-range=cudaProfilerApi`` records exactly those
steps, and one under ``cProfile`` for the host thread.

Every region records host wall and NVTX ranges. Optional ``--events`` adds
CUDA events on the current stream. The default uses no per-region device
events and no per-step synchronization beyond the model's existing loop.
Capture start and stop synchronize once to keep the bounded trace complete.
Instrumentation changes timing, so measure throughput in a separate plain
forecast and compare full output hashes before using the phase attribution.

The engine (gpuwm) files are pinned by digest and are not edited: its
classes are wrapped in memory after the pinned factory has been loaded and
verified, and the identity checks the pin performs (``_PHASE1_ORCHESTRATION
is PhysicsDriver.compute``, ``_PHASE2_MICROPHYSICS is microphysics.apply``)
are left untouched, which is why the engine's inner ``_run_*`` methods are
wrapped and its ``compute``/``apply`` are not.

Usage on the node with the card::

    python tools/profile_forecast_step.py --report DIR \
        --nsys-window 101:220 --cprofile-window 301:420 \
        -- <every argument gpuwm-hex forecast takes>

``--selftest`` installs every patch against the named ``--repo`` and
``--gpuwm-checkout`` without running the forecast and prints what it found.
"""

from __future__ import annotations

import argparse
import collections
import cProfile
import functools
import hashlib
import io
import json
import math
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DRIVER_SCHEMA = "mpas-port.cuda-v841-engineering-forecast/v1"
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _window(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    first, last = text.split(":")
    window = (int(first), int(last))
    if window[0] < 1 or window[1] < window[0]:
        raise ValueError(f"window {text!r} must be FIRST:LAST with 1 <= FIRST <= LAST")
    return window


def _short(path: str) -> str:
    """A file path with the roots a reader knows cut off."""
    for marker in ("/site-packages/", "/gpuwm-hex-lanes/", "/hex-lanes/"):
        index = path.find(marker)
        if index >= 0:
            return path[index + len(marker):]
    return path


def completion_evidence(
    *, rc: int, completed_steps: int, receipt: Any,
    final_time_seconds: Any, forcing_time_seconds: Any,
) -> dict[str, Any]:
    """Require a published complete run, including the next-step forcing carrier."""
    reasons = []
    requested = executed = None
    if rc != 0:
        reasons.append("CLI returned nonzero")
    if type(completed_steps) is not int or completed_steps <= 0:
        reasons.append("no positive completed composite count")
    if not isinstance(receipt, dict):
        reasons.append("driver receipt is absent or malformed")
    else:
        if receipt.get("schema") != DRIVER_SCHEMA:
            reasons.append("driver receipt schema is unrecognized")
        if receipt.get("status") != "passed":
            reasons.append("driver receipt does not report passed")
        for key in ("sources_unchanged", "authorities_unchanged", "arwen_checkout_unchanged"):
            if receipt.get(key) is not True:
                reasons.append(f"driver receipt does not confirm {key}")
        forecast = receipt.get("forecast")
        if not isinstance(forecast, dict):
            reasons.append("driver forecast receipt is absent or malformed")
        else:
            if "refusal" not in forecast or forecast["refusal"] is not None:
                reasons.append("driver forecast is refused or lacks explicit no-refusal evidence")
            if forecast.get("full_physics_cuda_executed") is not True:
                reasons.append("driver receipt does not confirm full physics execution")
            requested = forecast.get("steps_requested")
            executed = forecast.get("steps_executed")
            counts_valid = type(requested) is int and type(executed) is int and requested > 0 and executed > 0
            if not counts_valid or requested != executed or completed_steps != executed:
                reasons.append("requested, executed and completed composite step counts disagree")
            schedule = forecast.get("schedule")
            walls = forecast.get("walls")
            if not isinstance(schedule, dict) or not isinstance(walls, dict):
                reasons.append("driver schedule or wall receipt is absent or malformed")
            else:
                if type(schedule.get("steps")) is not int or schedule["steps"] != requested:
                    reasons.append("scheduled and requested steps disagree")
                if type(walls.get("steps")) is not int or walls["steps"] != executed:
                    reasons.append("wall receipt and executed steps disagree")
                dt = schedule.get("dt_seconds")
                numeric = lambda value: type(value) in (int, float) and math.isfinite(value)
                if not numeric(dt) or dt <= 0 or not counts_valid:
                    reasons.append("driver timestep or requested endpoint is malformed")
                else:
                    expected_time = requested * dt
                    if not numeric(walls.get("forecast_seconds")) or walls["forecast_seconds"] != expected_time:
                        reasons.append("driver forecast endpoint disagrees with requested steps")
                    if not numeric(final_time_seconds) or final_time_seconds != expected_time:
                        reasons.append("retained atmosphere endpoint disagrees with requested steps")
                    if not numeric(forcing_time_seconds) or forcing_time_seconds != expected_time:
                        reasons.append("final next-step forcing is absent or has a different endpoint")
    return {"eligible": not reasons, "reasons": reasons, "steps_requested": requested, "steps_executed": executed}


class StepProfiler:
    """Region timers, host-copy census and step windows for one forecast."""

    def __init__(
        self,
        report_dir: Path,
        *,
        nsys_window: tuple[int, int] | None,
        cprofile_window: tuple[int, int] | None,
        use_nvtx: bool = True,
        use_events: bool = False,
        baseline: bool = False,
    ) -> None:
        self.report_dir = report_dir
        self.nsys_window = nsys_window
        self.cprofile_window = cprofile_window
        self.use_nvtx = use_nvtx
        self.use_events = use_events
        self.baseline = baseline
        self.final_stack: Any = None
        self.last_result: Any = None
        self.proof: Any = None
        self.completed_steps = 0
        self.driver_receipt_path: Path | None = None
        self.cp: Any = None
        self.nvtx: Any = None
        self.step = 0
        self.steps: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.preloop: dict[str, Any] = self._new_step(0)
        self.stack: list[tuple[str, float, Any]] = []
        self.pending: list[tuple[str, str | None, int, float, float, Any, Any]] = []
        self.asnumpy_by_site: dict[str, list[float]] = collections.defaultdict(
            lambda: [0, 0.0, 0]
        )
        self.patched: list[str] = []
        self.missing: list[str] = []
        self.cprofile: cProfile.Profile | None = None
        self.engine_patched = False
        self.driver_patched = False
        self.t_run_start = time.perf_counter()
        self.nsys_marks: dict[str, float] = {}

    # ------------------------------------------------------------ steps
    @staticmethod
    def _new_step(step: int) -> dict[str, Any]:
        return {
            "step": step,
            "t_open": time.perf_counter(),
            "t_composite_end": None,
            "regions": [],
            "asnumpy_calls": 0,
            "asnumpy_seconds": 0.0,
            "asnumpy_bytes": 0,
            "asnumpy_sites": {},
        }

    def open_step(self) -> None:
        self.step += 1
        step = self.step
        if self.nsys_window and step == self.nsys_window[0]:
            self.cp.cuda.get_current_stream().synchronize()
            self.cp.cuda.profiler.start()
            self.nsys_marks["start"] = time.perf_counter()
        if self.cprofile_window and step == self.cprofile_window[0]:
            self.cprofile = cProfile.Profile()
            self.cprofile.enable()
        self.current = self._new_step(step)
        if self.nvtx is not None:
            self.nvtx.RangePush(f"step {step}")

    def close_step(self, *, final: bool = False) -> None:
        current = self.current
        if current is None:
            # Regions recorded outside any step (device-stack construction,
            # the step-0 capture) belong to the pre-loop bucket.
            if self.pending:
                self._flush_pending(self.preloop)
            return
        self._flush_pending(current)
        current["t_close"] = time.perf_counter()
        if self.nvtx is not None:
            self.nvtx.RangePop()
        step = current["step"]
        if self.nsys_window and step == self.nsys_window[1]:
            self.cp.cuda.get_current_stream().synchronize()
            self.cp.cuda.profiler.stop()
            self.nsys_marks["stop"] = time.perf_counter()
        if self.cprofile is not None and self.cprofile_window and step == self.cprofile_window[1]:
            self.cprofile.disable()
            self._dump_cprofile()
        self.steps.append(current)
        self.current = None

    def _flush_pending(self, target: dict[str, Any]) -> None:
        cp = self.cp
        for label, parent, depth, t0, t1, ev0, ev1 in self.pending:
            try:
                gpu_ms = float(cp.cuda.get_elapsed_time(ev0, ev1)) if ev0 is not None else None
            except Exception:
                gpu_ms = float("nan")
            target["regions"].append(
                {
                    "label": label,
                    "parent": parent,
                    "depth": depth,
                    "t0": t0,
                    "t1": t1,
                    "host_s": t1 - t0,
                    "gpu_ms": gpu_ms,
                }
            )
        self.pending = []

    def _dump_cprofile(self) -> None:
        assert self.cprofile is not None
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.cprofile.dump_stats(str(self.report_dir / "host.prof"))
        stream = io.StringIO()
        stats = pstats.Stats(self.cprofile, stream=stream)
        stats.sort_stats("tottime").print_stats(60)
        stats.sort_stats("cumulative").print_stats(60)
        for pattern in ("asnumpy", r"\.get\b", r"\.item\b", "synchronize", "__float__", "__bool__", "__int__"):
            stream.write(f"\n===== callers of {pattern}\n")
            stats.print_callers(pattern)
        (self.report_dir / "host-cprofile.txt").write_text(stream.getvalue(), encoding="utf-8")
        self.cprofile = None

    # ---------------------------------------------------------- regions
    def begin(self, label: str) -> None:
        if self.nvtx is not None:
            self.nvtx.RangePush(label)
        event = self.cp.cuda.Event() if self.use_events else None
        if event is not None:
            event.record()
        self.stack.append((label, time.perf_counter(), event))

    def end(self) -> None:
        label, t0, ev0 = self.stack.pop()
        ev1 = self.cp.cuda.Event() if self.use_events else None
        if ev1 is not None:
            ev1.record()
        t1 = time.perf_counter()
        if self.nvtx is not None:
            self.nvtx.RangePop()
        parent = self.stack[-1][0] if self.stack else None
        self.pending.append((label, parent, len(self.stack), t0, t1, ev0, ev1))

    def wrap(
        self,
        owner: Any,
        name: str,
        label: str,
        *,
        owner_name: str,
        label_from: Callable[[str, tuple, dict], str] | None = None,
        static: bool = False,
    ) -> None:
        original = getattr(owner, name, None)
        if original is None:
            self.missing.append(f"{owner_name}.{name}")
            return
        if static:
            original = original.__func__ if hasattr(original, "__func__") else original
        prof = self

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            tag = label if label_from is None else label_from(label, args, kwargs)
            prof.begin(tag)
            try:
                return original(*args, **kwargs)
            finally:
                prof.end()

        # The process writer and prepared-cache serializer must obtain the
        # original callable's module globals, including the rebound mesh size.
        wrapped.__prep_cache_transparent__ = True
        setattr(owner, name, staticmethod(wrapped) if static else wrapped)
        self.patched.append(f"{owner_name}.{name} -> {label}")

    # ------------------------------------------------------------ cupy
    def attach(self) -> None:
        import cupy

        self.cp = cupy
        if self.use_nvtx:
            try:
                from cupy.cuda import nvtx

                nvtx.RangePush("profile-attach")
                nvtx.RangePop()
                self.nvtx = nvtx
            except Exception:
                self.nvtx = None
        self._patch_asnumpy()

    def _patch_asnumpy(self) -> None:
        cupy = self.cp
        original = cupy.asnumpy
        prof = self

        @functools.wraps(original)
        def asnumpy(a: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(a, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                frame = sys._getframe(1)
                site = f"{_short(frame.f_code.co_filename)}:{frame.f_lineno}"
                nbytes = int(getattr(a, "nbytes", 0) or 0)
                entry = prof.asnumpy_by_site[site]
                entry[0] += 1
                entry[1] += elapsed
                entry[2] += nbytes
                target = prof.current if prof.current is not None else prof.preloop
                target["asnumpy_calls"] += 1
                target["asnumpy_seconds"] += elapsed
                target["asnumpy_bytes"] += nbytes
                sites = target["asnumpy_sites"]
                sites[site] = sites.get(site, 0) + 1

        cupy.asnumpy = asnumpy
        self.patched.append("cupy.asnumpy -> host-copy census")

    # --------------------------------------------------------- installs
    def install(self) -> None:
        """Hook the two load points; everything else installs behind them."""
        import hexcore.forecast_door as door
        import hexcore.cuda_arwen_physics_v841 as arwen

        prof = self
        original_load = door._load_drivers

        @functools.wraps(original_load)
        def load_drivers(repo: Path) -> Any:
            binding, driver = original_load(repo)
            prof.patch_final_identity(driver)
            if not prof.baseline:
                prof.patch_driver(driver)
                prof.patch_hexcore()
            return binding, driver

        door._load_drivers = load_drivers
        self.patched.append("hexcore.forecast_door._load_drivers -> driver + hexcore patches")

        if self.baseline:
            return

        original_factory = arwen._load_pinned_arwen_factory

        @functools.wraps(original_factory)
        def load_factory(checkout: Any) -> Any:
            result = original_factory(checkout)
            prof.patch_engine()
            return result

        arwen._load_pinned_arwen_factory = load_factory
        self.patched.append("hexcore.cuda_arwen_physics_v841._load_pinned_arwen_factory -> engine patches")

    def patch_final_identity(self, driver: Any) -> None:
        """Retain references only; hash them after the real run has returned."""
        proof = driver.proof
        self.proof = proof
        original_construct = proof._construct_device_stack
        original_step = proof.execute_composite_step
        original_execute = driver.execute_forecast
        prof = self

        @functools.wraps(original_construct)
        def construct(*args: Any, **kwargs: Any) -> Any:
            result = original_construct(*args, **kwargs)
            prof.final_stack = result
            return result

        @functools.wraps(original_step)
        def step(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            prof.last_result = result
            prof.completed_steps += 1
            return result

        @functools.wraps(original_execute)
        def execute(*args: Any, **kwargs: Any) -> Any:
            output_root = kwargs.get("output_root")
            if output_root is not None:
                prof.driver_receipt_path = Path(output_root) / driver.RECEIPT_NAME
            return original_execute(*args, **kwargs)

        proof._construct_device_stack = construct
        proof.execute_composite_step = step
        driver.execute_forecast = execute
        self.patched.append("proof construction + composite result -> retained references for final hashes")

    def write_final_identity(self, rc: int) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": "gpuwm-hex.profile-final-identity/v1",
            "rc": int(rc),
            "boundary_available": self.final_stack is not None,
            "completed_composite_steps": self.completed_steps,
            "complete_run_identity_eligible": False,
            "note": "Hashes are taken after the run returns, outside the engine step timers and trace window.",
        }
        final_time = forcing_time = None
        if self.final_stack is not None:
            try:
                proof = self.proof
                payload["execution_boundary"] = proof.fingerprint_execution_boundary(self.final_stack)
                final_time = payload["execution_boundary"]["atmosphere"].get("model_time_seconds")
                forcing = getattr(self.last_result, "dynamics_tendencies", None)
                if forcing is not None:
                    forcing_time = float(forcing.time_seconds)
                    cp = self.final_stack["driver"].cp
                    payload["dynamics_tendencies"] = proof.fingerprint_nested_arrays({
                        "rthdynten": cp.asnumpy(forcing.rthdynten),
                        "rqvdynten": cp.asnumpy(forcing.rqvdynten),
                        "time_seconds": float(forcing.time_seconds),
                    })
            except Exception as error:
                payload["hash_error"] = f"{type(error).__name__}: {error}"
                payload["complete_run_identity_eligible"] = False
        receipt = None
        if self.driver_receipt_path is not None:
            payload["driver_receipt_path"] = str(self.driver_receipt_path)
            try:
                raw = self.driver_receipt_path.read_bytes()
                receipt = json.loads(raw)
                payload["driver_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
            except (OSError, ValueError) as error:
                payload["driver_receipt_error"] = f"{type(error).__name__}: {error}"
        evidence = completion_evidence(
            rc=int(rc), completed_steps=self.completed_steps, receipt=receipt,
            final_time_seconds=final_time, forcing_time_seconds=forcing_time,
        )
        if "hash_error" in payload:
            evidence["eligible"] = False
            evidence["reasons"].append("final boundary hashing failed")
        payload["completion_evidence"] = evidence
        payload["complete_run_identity_eligible"] = evidence["eligible"]
        path = self.report_dir / "final-identity.json"
        path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return path

    def patch_driver(self, driver: Any) -> None:
        if self.driver_patched:
            return
        self.driver_patched = True
        proof = driver.proof
        prof = self
        original = proof.execute_composite_step

        @functools.wraps(original)
        def composite(*args: Any, **kwargs: Any) -> Any:
            prof.close_step()
            prof.open_step()
            prof.begin("composite")
            try:
                return original(*args, **kwargs)
            finally:
                prof.end()
                assert prof.current is not None
                prof.current["t_composite_end"] = time.perf_counter()

        proof.execute_composite_step = composite
        self.patched.append("driver.proof.execute_composite_step -> step bracket")
        self.wrap(driver, "step_health_gate", "health_gate", owner_name="driver")
        self.wrap(driver, "asdict", "receipt_asdict", owner_name="driver")
        self.wrap(proof, "capture_snapshot", "history_capture", owner_name="proof")
        self.wrap(proof, "physical_snapshot_gate", "history_capture_gate", owner_name="proof")
        self.wrap(proof, "write_snapshot_netcdf", "history_write", owner_name="proof")
        self.wrap(proof, "require_arwen_v2_surface_execution", "receipt_surface_check", owner_name="proof")
        self.wrap(proof, "_construct_device_stack", "startup_device_stack", owner_name="proof")
        self.wrap(proof, "gpu_memory_admission", "startup_memory_admission", owner_name="proof")

    def patch_hexcore(self) -> None:
        import hexcore.cuda_arwen_physics_v841 as arwen
        import hexcore.cuda_driver as cuda_driver
        import hexcore.cuda_physics_v841 as physics
        import hexcore.cuda_regional_forecast_v841 as regional
        import hexcore.cuda_horizontal_v841 as horizontal

        backend = arwen.PersistentTwoPhaseCudaPhysicsBackendV841
        owner = "Backend"
        self.wrap(backend, "begin_step", "phase1_physics", owner_name=owner)
        self.wrap(backend, "finish_step", "phase2_physics", owner_name=owner)
        self.wrap(backend, "commit_step", "backend_commit", owner_name=owner)
        self.wrap(backend, "_restore_boundary", "backend_rollback", owner_name=owner)

        def prep_label(label: str, args: tuple, kwargs: dict) -> str:
            return "prep_phase2" if kwargs.get("post_rk_wsm6") else "prep_phase1"

        owner = "hexcore.cuda_arwen_physics_v841"
        self.wrap(arwen, "prepare_mpas_to_phys_cuda_v841", "prep", owner_name=owner, label_from=prep_label)
        self.wrap(arwen, "run_bl_ysu_gwdo_cuda_v841", "gwdo", owner_name=owner)

        owner = "hexcore.cuda_physics_v841"
        self.wrap(physics, "couple_raw_column_physics_v841", "couple", owner_name=owner)
        self.wrap(physics, "clamp_wsm6_scalars_in_place_v841", "clamp", owner_name=owner)
        self.wrap(physics, "recover_post_rk_wsm6_state_v841", "recover", owner_name=owner)

        driver = cuda_driver.CudaDryDycoreDriver
        owner = "CudaDryDycoreDriver"
        self.wrap(driver, "step_device_with_physics", "dynamics", owner_name=owner)
        self.wrap(driver, "commit_post_wsm6_candidate", "commit", owner_name=owner)
        self.wrap(driver, "_advance_dynamics_subcycle_v841", "rk_subcycle", owner_name=owner)
        self.wrap(driver, "_compute_moist_dynamics_coefficients_v841", "moist_coefficients", owner_name=owner)
        self.wrap(driver, "_copy_state", "copy_state", owner_name=owner)
        self.wrap(driver, "_copy_saved", "copy_saved", owner_name=owner)
        self.wrap(driver, "_rebuild_saved_diagnostics", "rebuild_saved", owner_name=owner)
        self.wrap(driver, "_launch_physics_driver", "gf_rate_kernel", owner_name=owner)

        owner = "hexcore.cuda_driver"
        self.wrap(cuda_driver, "validate_recovered_state_cuda_v841", "validate_recovered_global", owner_name=owner)
        self.wrap(cuda_driver, "accumulate_finite_array_cuda_v841", "finite_check", owner_name=owner)
        self.wrap(cuda_driver, "vector_momentum_tendency_cuda_v841", "momentum_tendency", owner_name=owner)
        self.wrap(cuda_driver, "advance_acoustic_step_cuda_v841", "acoustic_step_global", owner_name=owner)
        self.wrap(cuda_driver, "accumulate_split_flux_cuda_v841", "split_flux", owner_name=owner)
        self.wrap(cuda_driver, "finish_split_flux_cuda_v841", "split_flux", owner_name=owner)

        owner = "hexcore.cuda_regional_forecast_v841"
        self.wrap(regional, "advance_scalars_regional_cuda_v841", "scalar_transport", owner_name=owner)
        self.wrap(regional, "advance_acoustic_step_regional_cuda_v841", "acoustic_step", owner_name=owner)
        runtime = regional.CudaRegionalRuntimeV841
        owner = "CudaRegionalRuntimeV841"
        self.wrap(runtime, "begin_step", "lbc_begin_step", owner_name=owner)
        self.wrap(runtime, "adjust_dynamics_tendencies", "lbc_adjust_tendencies", owner_name=owner)
        self.wrap(runtime, "bdy_adjust_scalars", "bdy_adjust_scalars", owner_name=owner)
        self.wrap(runtime, "validate_recovered", "validate_recovered", owner_name=owner)
        self.wrap(runtime, "reset_speczone_values", "reset_speczone", owner_name=owner)
        self.wrap(runtime, "moist_coefficients", "moist_coefficients_regional", owner_name=owner)
        self.wrap(runtime, "clamp_negative_scalars", "clamp_negative_scalars", owner_name=owner)

        horiz = horizontal.CudaHorizontalV841
        owner = "CudaHorizontalV841"
        self.wrap(horiz, "solve_diagnostics", "solve_diagnostics", owner_name=owner)
        self.wrap(horiz, "pressure_gradient_euler_tendency", "pgf_tendency", owner_name=owner)
        self.wrap(horiz, "mass_flux_divergence", "mass_flux_divergence", owner_name=owner)
        self.wrap(horiz, "density_tendency", "density_tendency", owner_name=owner)
        self.wrap(horiz, "compute_dry_mixing_tendencies_v841", "mixing_tendencies", owner_name=owner)
        self.wrap(horiz, "recover_edge_fields", "recover_edge_fields", owner_name=owner)

    def patch_engine(self) -> None:
        if self.engine_patched:
            return
        self.engine_patched = True
        column_batch = sys.modules.get("gpuwm.core.mpas_column_batch")
        physics = sys.modules.get("gpuwm.core.physics")
        if column_batch is None or physics is None:
            self.missing.append("gpuwm.core.mpas_column_batch / gpuwm.core.physics not in sys.modules")
            return
        seam = column_batch.MpasColumnBatchPhysics
        owner = "MpasColumnBatchPhysics"
        self.wrap(seam, "export_state", "seam_export_state", owner_name=owner)
        self.wrap(seam, "run_phase1", "seam_phase1", owner_name=owner)
        self.wrap(seam, "run_phase2", "seam_phase2_wsm6", owner_name=owner)
        self.wrap(seam, "accumulated_precipitation", "seam_accumulated_precip", owner_name=owner)
        driver = physics.PhysicsDriver
        owner = "PhysicsDriver"
        self.wrap(driver, "_run_radiation", "radiation", owner_name=owner)
        self.wrap(driver, "_run_sfclay", "sfclay", owner_name=owner)
        self.wrap(driver, "_run_noahmp", "noahmp", owner_name=owner)
        self.wrap(driver, "_run_ysu", "ysu_pbl", owner_name=owner)
        self.wrap(driver, "_run_cumulus", "cumulus", owner_name=owner)
        self.wrap(driver, "accept_microphysics", "accept_microphysics", owner_name=owner)

    # ----------------------------------------------------------- report
    def write_report(self, rc: int, door_args: list[str]) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "gpuwm-hex.profile-forecast-step/v1",
            "rc": int(rc),
            "door_args": list(door_args),
            "nsys_window": self.nsys_window,
            "cprofile_window": self.cprofile_window,
            "nsys_marks": self.nsys_marks,
            "nvtx": self.nvtx is not None,
            "events": self.use_events,
            "baseline": self.baseline,
            "patched": self.patched,
            "missing": self.missing,
            "wall_seconds": time.perf_counter() - self.t_run_start,
            "steps_seen": self.step,
            "asnumpy_by_site": {
                site: {"calls": int(v[0]), "seconds": float(v[1]), "bytes": int(v[2])}
                for site, v in sorted(self.asnumpy_by_site.items(), key=lambda kv: -kv[1][1])
            },
            "preloop": self.preloop,
            "steps": self.steps,
        }
        path = self.report_dir / "profile-steps.json"
        path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return path


def _door_option(door_args: list[str], name: str) -> str | None:
    if name in door_args:
        index = door_args.index(name)
        if index + 1 < len(door_args):
            return door_args[index + 1]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--nsys-window", default=None, help="FIRST:LAST steps bracketed by cudaProfilerStart/Stop")
    parser.add_argument("--cprofile-window", default=None, help="FIRST:LAST steps under cProfile")
    parser.add_argument("--no-nvtx", action="store_true")
    parser.add_argument("--events", action="store_true", help="add device events around each region; default NVTX and host timers only")
    parser.add_argument("--baseline", action="store_true", help="real run with only reference retention and post-run final hashes; no timing, NVTX, copy census or profiler capture")
    parser.add_argument("--selftest", action="store_true", help="install every patch, run nothing, print what was found")
    parser.add_argument("door_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    door_args = list(args.door_args)
    if door_args and door_args[0] == "--":
        door_args = door_args[1:]
    nsys_window = _window(args.nsys_window)
    cprofile_window = _window(args.cprofile_window)
    if args.baseline and (nsys_window or cprofile_window or args.events):
        parser.error("--baseline cannot use capture windows or events")
    if nsys_window and cprofile_window and not (
        nsys_window[1] < cprofile_window[0] or cprofile_window[1] < nsys_window[0]
    ):
        raise SystemExit("the nsys and cProfile windows must not overlap")

    prof = StepProfiler(
        args.report,
        nsys_window=nsys_window,
        cprofile_window=cprofile_window,
        use_nvtx=not args.no_nvtx,
        use_events=args.events,
        baseline=args.baseline,
    )
    if not args.baseline:
        try:
            prof.attach()
        except ImportError as error:
            # The selftest installs patches and runs nothing, so it does not
            # need a device stack.  THE BREAKAGE THIS PREVENTS: the public CPU
            # tier (no CuPy, tools/battery/cpu_files.txt) ran this selftest and
            # failed at the cupy import before checking a single patch.
            if not args.selftest:
                raise
            print(f"[profile] selftest without a device stack ({error}); "
                  "the cupy hooks are not checked", file=sys.stderr, flush=True)
    prof.install()

    if args.selftest:
        import hexcore.forecast_door as door
        import hexcore.cuda_arwen_physics_v841 as arwen

        repo = _door_option(door_args, "--repo")
        door._load_drivers(door.resolve_repo(Path(repo) if repo else None))
        checkout = _door_option(door_args, "--gpuwm-checkout")
        if checkout:
            arwen._load_pinned_arwen_factory(checkout)
        print(json.dumps({"patched": prof.patched, "missing": prof.missing}, indent=1))
        return 0 if not prof.missing else 1

    from hexcore.cli import main as door_main

    started = time.perf_counter()
    rc = 1
    try:
        rc = int(door_main(["forecast", *door_args]))
    finally:
        prof.close_step(final=True)
        path = prof.write_report(rc, door_args)
        identity = prof.write_final_identity(rc)
    print(
        f"[profile] rc={rc} steps={prof.step} wall={time.perf_counter() - started:.1f}s "
        f"patched={len(prof.patched)} missing={len(prof.missing)} report={path} identity={identity}",
        flush=True,
    )
    if prof.missing:
        print("[profile] MISSING targets: " + ", ".join(prof.missing), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
