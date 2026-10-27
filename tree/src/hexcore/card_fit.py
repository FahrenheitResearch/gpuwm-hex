"""Global mesh predictions and process-scoped whole-run measurements.

Memory comes from device_admission. Runtime extrapolates a measured table
row by cell-steps and is explicitly not a measurement of the new geometry.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from .errors import MpasPortError


#: Headroom over the measured whole-run curve. The curve passes through each
#: run's sampled peak, which a 1 s sampler can miss by a subsecond spike, and
#: runs between two measured sizes are interpolated. 10 % puts every measured
#: run of today's build 10-12 % under its prediction (tests/test_global_vr_mesh_plan.py
#: test_shipped_curve_lands_10_to_15_percent_over_every_measured_run).
WHOLE_RUN_MARGIN = 0.10


def whole_run_curve(rows: Sequence[Mapping[str, Any]], count: int, *,
                    base_mib: Any) -> dict[str, Any] | None:
    """Peak MiB from this card's measured whole runs, plus WHOLE_RUN_MARGIN.

    THE BREAKAGE IT PREVENTS (WOOF run-controls, 2026-10-24): the nearest run
    at or below a size was held CONSTANT below the smallest run, so an
    80 -> 5 km zoom globe of 128,942 cells was priced at 25,272 MiB (the
    216,189-cell run) and peaked at 13,036 MiB; a 52,618-cell globe was priced
    the same and peaked at 7,472 MiB. Quotes, refusals and card leases were
    sized 2-3.4x too big. Before that, the largest floor over every run priced
    the 581,895-cell globe at 68.0 GB against 44,450 MiB (feasibility defect 3).

    Two or more runs: straight between neighbouring measured runs (peaks made
    non-decreasing in cells), the first segment's slope below the smallest and
    the last segment's slope past the largest. One run: device_admission's own
    slope through it below, cell-linear above (the old anchor's conservative
    side). Derived between and past the measured sizes, not measured there.
    """
    if not rows:
        return None
    by_cells: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        cells = int(row["cells"])
        if cells not in by_cells or row["peak_process_vram_mib"] > by_cells[cells]["peak_process_vram_mib"]:
            by_cells[cells] = row
    points = sorted(by_cells.values(), key=lambda row: row["cells"])
    xs = [float(row["cells"]) for row in points]
    ys: list[float] = []
    for row in points:
        ys.append(max([float(row["peak_process_vram_mib"]), *ys[-1:]]))
    x = float(count)
    if len(points) == 1:
        if x <= xs[0]:
            curve = ys[0] + float(base_mib(count)) - float(base_mib(int(xs[0])))
            basis = "one measured whole run: device_admission's slope through it below its size"
        else:
            curve = ys[0] * x / xs[0]
            basis = "one measured whole run: cell-linear past its size"
    else:
        segment = next((i for i in range(1, len(xs)) if x <= xs[i]), len(xs) - 1)
        x0, x1, y0, y1 = xs[segment - 1], xs[segment], ys[segment - 1], ys[segment]
        curve = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        basis = ("straight between the measured whole runs" if xs[0] <= x <= xs[-1] else
                 "the first measured segment's slope below the smallest run" if x < xs[0] else
                 "the last measured segment's slope past the largest run")
    nearest = min(points, key=lambda row: abs(math.log(x / float(row["cells"]))))
    floor = max(0.0, curve) * (1.0 + WHOLE_RUN_MARGIN)
    return {"observation": dict(nearest), "observations": [row["id"] if "id" in row else row.get("receipt_sha256")
                                                           for row in points],
            "curve_mib": round(curve, 3), "margin_fraction": WHOLE_RUN_MARGIN,
            "floor_mib": round(floor, 3),
            "basis": f"{basis}, plus {WHOLE_RUN_MARGIN:.0%}; derived at other sizes, not measured there"}


def predict(cells: float, *, card: Mapping[str, Any], levels: int,
            dt_seconds: float, hours: float, budget_mib: float | None = None,
            table: Mapping[str, Any] | None = None,
            observations: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    from . import device_admission as admission
    from .global_door import load_table, row_step

    if not all(math.isfinite(float(v)) and float(v) > 0
               for v in (cells, levels, dt_seconds, hours)):
        raise MpasPortError("card-fit requires finite positive cells, levels, timestep and hours")
    budget = float(card["nameplate_mib"] if budget_mib is None else budget_mib)
    if not math.isfinite(budget) or budget <= 0:
        raise MpasPortError("card-fit memory budget must be finite and positive")
    count = math.ceil(cells)
    profile = admission.KNOWN_CARDS[str(card["admission_card"])]
    model = admission.model_for_card(profile, configuration="global", levels=int(levels))
    base_peak = model.predict_bytes(count) / admission.MIB
    required = model.required_bytes(count) / admission.MIB
    margin = required - base_peak
    if observations is None:
        observations = json.loads((Path(__file__).with_name("data") /
            "card-fit-observations.v1.json").read_text())["observations"]
    applicable = [row for row in observations
                  if row["card"] == card["admission_card"] and row["levels"] == levels
                  and row["status"] == "complete"]
    calibration = None
    predicted_peak = base_peak
    curve = whole_run_curve(applicable, count,
                            base_mib=lambda cells: model.predict_bytes(cells) / admission.MIB)
    # Never lower device_admission: the curve is a floor on its peak.
    if curve is not None and curve["floor_mib"] > predicted_peak:
        predicted_peak = curve["floor_mib"]
        calibration = curve
    required = predicted_peak + margin
    table = load_table() if table is None else table
    references = [row for row in table.get("measured_seconds_per_forecast_hour", [])
                  if row["cards"] == 1 and str(row.get("card_match", row["card"])).lower()
                  in profile.name.lower()]
    runtime: dict[str, Any] = {"seconds": None, "basis": "no whole-run rate on this card"}
    if references and levels == admission.MODEL_VERTICAL_LEVELS:
        reference = min(references, key=lambda row: abs(math.log(
            count / table["meshes"][row["mesh"]]["cells"])))
        mesh = table["meshes"][reference["mesh"]]
        # Existing table's rate includes its physics cap. Keep that clock
        # even when a newer engine retires the cap for new forecasts.
        reference_dt = row_step(mesh, table["physics"]["arwen"], engine=frozenset())["dt_seconds"]
        factor = count / mesh["cells"] * reference_dt / dt_seconds
        runtime = {
            "seconds": round(reference["seconds"] * hours * factor, 3),
            "basis": "EXTRAPOLATED cell-steps from a measured whole-run row; excludes mesh, static, init and products; geometry, physics cadence, cache and contention can change runtime",
            "reference": dict(reference), "reference_cells": mesh["cells"],
            "reference_dt_seconds": reference_dt, "cell_step_factor": factor,
        }
    return {
        "schema": "gpuwm-hex.card-fit/v1", "cells": count, "levels": levels,
        "dt_seconds": dt_seconds, "hours": hours, "cards": 1,
        "predicted_peak_mib": round(predicted_peak, 3),
        "base_predicted_peak_mib": round(base_peak, 3),
        "whole_run_floor": calibration,
        "required_free_mib": round(required, 3), "budget_mib": budget,
        "fits": required <= budget, "model": model.as_dict(), "runtime": runtime,
        "scope": "default global physics; memory model terms outside 55 levels are not fully measured; launch rechecks actual free memory",
    }


def comparison(prediction: Mapping[str, Any], measured: Mapping[str, Any]) -> dict[str, Any]:
    peak = measured.get("peak_process_vram_mib")
    wall = measured["wall_seconds"]
    expected = prediction["runtime"]["seconds"]
    return {
        "predicted_peak_mib": prediction["predicted_peak_mib"],
        "measured_peak_process_vram_mib": peak,
        "memory_error_percent": None if peak is None or peak <= 0 else
            round(100 * (prediction["predicted_peak_mib"] / peak - 1), 3),
        "predicted_forecast_seconds": expected, "measured_forecast_seconds": wall,
        "runtime_error_percent": None if expected is None else round(100 * (expected / wall - 1), 3),
        "instrument": "1 s compute-process samples summed only over stage process group; subsecond peaks can be missed; wall includes process startup and JIT",
    }


def _process_group(group: int) -> tuple[set[int], float, int]:
    pids: set[int] = set()
    ticks = 0.0
    threads = 0
    hz = os.sysconf("SC_CLK_TCK")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                pids.add(int(entry.name))
                ticks += (int(fields[11]) + int(fields[12])) / hz
                threads += int(fields[17])
        except (OSError, ValueError, IndexError):
            continue
    return pids, ticks, threads


def _gpu_memory(pids: set[int]) -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if result.returncode:
            return None
        values = [float(parts[1]) for line in result.stdout.splitlines()
                  if len(parts := line.split(",")) == 2 and int(parts[0]) in pids]
        return sum(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class BackgroundStage:
    """A door stage started beside the foreground ones (the zoom globe's source fetch and decode beside its mesh
    and static, U-H6).

    No per-second sampling and no signal handlers: those belong to the foreground :func:`run_stage`. The child is
    reaped only in :meth:`finish`, so a foreground stage's RUSAGE_CHILDREN delta never counts this one's CPU; its own
    CPU comes from wait4, and its wall ends when it exits (waitid WNOWAIT), not when the door gets round to it."""

    def __init__(self, argv: Sequence[str], *, name: str, directory: Path,
                 env: Mapping[str, str] | None = None):
        import datetime
        import threading

        directory.mkdir(parents=True, exist_ok=True)
        self.name, self.argv = name, [str(a) for a in argv]
        self.log = directory / f"{name}.log"
        self._output = self.log.open("w")
        self._output.write("argv: " + repr(self.argv) + "\n")
        self._output.flush()
        self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._started = time.monotonic()
        self._ended: float | None = None
        self.process = subprocess.Popen(self.argv, stdout=self._output, stderr=subprocess.STDOUT,
                                        env=None if env is None else dict(env), start_new_session=True)
        self._watcher = threading.Thread(target=self._watch, name=f"stage-{name}", daemon=True)
        self._watcher.start()
        self._result: dict[str, Any] | None = None

    def _watch(self) -> None:
        try:
            os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOWAIT)
        except ChildProcessError:
            pass
        self._ended = time.monotonic()

    def finish(self) -> dict[str, Any]:
        """Wait for the stage, reap it and return its row (the shape :func:`run_stage` returns)."""
        if self._result is not None:
            return self._result
        self._watcher.join()
        _pid, status, usage = os.wait4(self.process.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        self.process.returncode = code
        self._output.close()
        wall = (self._ended or time.monotonic()) - self._started
        cpu = usage.ru_utime + usage.ru_stime
        self._result = {
            "stage": self.name, "argv": self.argv, "returncode": code, "started_utc": self.started_utc,
            "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
            "average_cpu_cores": round(cpu / max(wall, .001), 3), "peak_process_vram_mib": None,
            "log": str(self.log), "background": True}
        return self._result

    def cancel(self) -> None:
        """Stop the stage (the door failed before it needed it) and reap it."""
        if self._result is not None:
            return
        if self._ended is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.finish()


def start_stage(argv: Sequence[str], *, name: str, directory: Path,
                env: Mapping[str, str] | None = None) -> BackgroundStage:
    """Start a door stage beside the foreground ones; :meth:`BackgroundStage.finish` joins it."""
    return BackgroundStage(argv, name=name, directory=directory, env=env)


def run_stage(argv: Sequence[str], *, name: str, directory: Path,
              env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run an existing door with logs; never substitute whole-card usage."""
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{name}.log"
    samples = directory / f"{name}.samples.csv"
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak = None
    max_threads = 0
    last_ticks = 0.0
    last_time = started
    single_seconds = 0.0
    longest_single = 0.0
    with log.open("w") as output, samples.open("w", newline="") as stream:
        output.write("argv: " + repr(list(argv)) + "\n")
        output.flush()
        writer = csv.writer(stream)
        writer.writerow(["elapsed_seconds", "cpu_cores", "threads", "process_vram_mib", "pids"])
        process = subprocess.Popen(list(argv), stdout=output, stderr=subprocess.STDOUT,
                                   env=None if env is None else dict(env), start_new_session=True)
        def interrupted(signum, _frame):
            raise InterruptedError(f"stage {name} interrupted by signal {signum}")

        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                now = time.monotonic()
                pids, ticks, threads = _process_group(process.pid)
                gpu = _gpu_memory(pids)
                if gpu is not None:
                    peak = max(peak or 0, gpu)
                cores = max(0, ticks - last_ticks) / max(now - last_time, .001)
                if .7 <= cores <= 1.3 and gpu is None:
                    single_seconds += now - last_time
                else:
                    longest_single = max(longest_single, single_seconds)
                    single_seconds = 0
                max_threads = max(max_threads, threads)
                writer.writerow([round(now - started, 3), round(cores, 3), threads, gpu, " ".join(map(str, sorted(pids)))])
                stream.flush()
                last_ticks, last_time = ticks, now
                if process.poll() is not None:
                    break
                time.sleep(1)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.monotonic() - started
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    return {
        "stage": name, "argv": list(argv), "returncode": process.returncode,
        "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
        "average_cpu_cores": round(cpu / wall, 3), "maximum_sampled_threads": max_threads,
        "peak_process_vram_mib": peak, "log": str(log), "samples": str(samples),
        "longest_single_core_cpu_only_seconds": round(max(longest_single, single_seconds), 3),
        "parallelism_environment": {key: (env or os.environ).get(key) for key in
                                    ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
