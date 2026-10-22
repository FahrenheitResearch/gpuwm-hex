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
    for row in applicable:
        # One whole forecast is an anchor, not a two-size allocation fit.
        # Keep its observed fixed floor below the anchor and price all of
        # its footprint per cell above it. Never lower device_admission.
        floor = row["peak_process_vram_mib"] * max(1., count / row["cells"])
        if floor > predicted_peak:
            predicted_peak = floor
            calibration = {"observation": dict(row), "floor_mib": round(floor, 3),
                "basis": "single whole-run observed floor; constant below anchor cells, cell-linear above; derived at other shapes, not a measured multi-size fit"}
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
