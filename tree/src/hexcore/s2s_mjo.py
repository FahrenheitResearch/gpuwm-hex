"""S2S member binding, export and qualified intraseasonal diagnostics.

Python only schedules work. rw_s2s reads and reduces native NetCDF fields;
rw_wrfbatch draws the diagnostic figures. No EOF is fitted to the forecast.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .s2s_door import S2SError, cores, write_json

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


@contextmanager
def receipt_attempt(path, schema):
    """Invalidate old success before any work, including input validation."""
    started = time.monotonic()
    write_json(path, {"schema": schema, "status": "running"})
    try:
        yield
    except BaseException as error:
        write_json(path, {"schema": schema, "status": "failed",
                          "error": f"{type(error).__name__}: {error}",
                          "seconds": round(time.monotonic() - started, 3)})
        raise


def bind_histories(value, state, args):
    """A durable, complete common-clock roster over restart attempts."""
    from .s2s_products import validate_history
    start = datetime.strptime(value["start_time"], TIME_FORMAT)
    rows = []
    completed = set(state["completed"])
    for member in value["members"]:
        if member["run_id"] not in completed:
            raise S2SError(f"cannot bind incomplete S2S member {member['run_id']}")
        schedule = validate_history(member, state["attempts"], start, args.days, args.history_hours)
        rows.append({"member": member["run_id"], "init": member["init"],
                     "simulation_start": member["start_time"],
                     "frames": [{"valid_time": valid.strftime(TIME_FORMAT), "path": str(path.absolute()),
                                 "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                                for valid, path in schedule]})
    binding = {"schema": "gpuwm-hex.s2s-history-binding/v1", "configuration_sha256": value["signature"],
               "start": value["start_time"], "days": args.days, "cadence_hours": args.history_hours,
               "grid": value["paths"]["grid"], "members": rows,
               "overlap_policy": "newest attempt wins; atomic completed .nc files only"}
    binding["binding_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    target = Path(value["paths"]["work"]) / "s2s-history-binding.json"
    write_json(target, binding)
    return binding, target


def export_members(value, state, args, env):
    """Use the existing full-field export door once for every bound member."""
    receipt = Path(value["paths"]["work"]) / "s2s-export" / "receipt.json"
    with receipt_attempt(receipt, "gpuwm-hex.s2s-export/v1"):
        return _export_members(value, state, args, env)


def _export_members(value, state, args, env):
    from .global_ensemble import _cli
    started = time.monotonic()
    binding, binding_path = bind_histories(value, state, args)
    output = Path(value["paths"]["work"]) / "s2s-export"
    output.mkdir(parents=True, exist_ok=True)
    workers = min(cores(), len(binding["members"]))
    threads = max(1, cores() // workers)

    def export(member):
        target = output / member["member"]
        argv = ["export", "--history", *(r["path"] for r in member["frames"]),
                "--mesh", value["paths"]["grid"], "--init", member["init"],
                "--simulation-start", member["simulation_start"], "--out", str(target),
                "--window", args.s2s_export_window, "--field-set", "full", "--overwrite",
                "--jobs", "1"]
        log = output / (member["member"] + ".log")
        rc = _cli(argv, log, {**env, "RAYON_NUM_THREADS": str(threads), "OMP_NUM_THREADS": str(threads)})
        if rc:
            raise S2SError(f"S2S member export exited {rc}: {log}")
        receipt_path = target / "export-receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError) as error:
            raise S2SError(f"missing or invalid member export receipt {receipt_path}: {error}") from error
        if (receipt.get("schema") != "gpuwm-hex.export/v1"
                or receipt.get("histories") != [r["path"] for r in member["frames"]]
                or receipt.get("init") != member["init"]
                or receipt.get("simulation_start") != member["simulation_start"]
                or receipt.get("field_set") != "full"):
            raise S2SError(f"member export receipt differs from bound histories or init: {receipt_path}")
        if len(receipt.get("frames", [])) != len(member["frames"]):
            raise S2SError(f"member export lost scheduled frames: {receipt_path}")
        return {"member": member["member"], "receipt": str(receipt_path), "argv": argv,
                "frames": len(receipt["frames"]), "log": str(log)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(export, binding["members"]))
    result = {"schema": "gpuwm-hex.s2s-export/v1", "status": "complete",
              "binding": str(binding_path), "binding_sha256": binding["binding_sha256"],
              "members": rows, "seconds": round(time.monotonic() - started, 3),
              "parallelism": {"member_processes": workers, "rust_threads_per_process": threads}}
    path = output / "receipt.json"
    write_json(path, result)
    return {**result, "receipt": str(path)}


def backend():
    from .engines import EngineSpec, resolve
    return resolve(EngineSpec(name="rw_s2s", flag="native S2S bridge",
                   env_names=("GPUWM_HEX_RW_S2S",), gpuwm_env="GPUWM_RW_S2S",
                   subject="S2S diagnostics and scores", what_breaks="native S2S diagnostics cannot be reduced",
                   cargo_package="rw-s2s"))


def run_backend(mode, plan, output, env):
    schema = {"mjo": "gpuwm-hex.s2s-mjo/v1", "score": "gpuwm-hex.s2s-scores/v1"}[mode]
    with receipt_attempt(output / "receipt.json", schema):
        return _run_backend(mode, plan, output, env, schema)


def _run_backend(mode, plan, output, env, schema):
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / (mode + "-plan.json")
    write_json(plan_path, plan)
    argv = [str(backend()), mode, str(plan_path)]
    log = output / (mode + ".log")
    with log.open("w") as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, **env})
    if result.returncode:
        raise S2SError(f"rw_s2s {mode} exited {result.returncode}; {log}")
    receipt_path = output / "receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as error:
        raise S2SError(f"invalid rw_s2s receipt {receipt_path}: {error}") from error
    if receipt.get("status") != "complete" or receipt.get("schema") != schema:
        raise S2SError(f"rw_s2s did not complete: {receipt_path}")
    return {**receipt, "receipt": str(receipt_path), "argv": argv, "log": str(log)}


def build_mjo(value, state, args, env):
    output = Path(value["paths"]["work"]) / "s2s-mjo"
    stage_path = output / "stage-receipt.json"
    started = time.monotonic()
    with receipt_attempt(stage_path, "gpuwm-hex.s2s-mjo-products/v1"):
        result = _build_mjo(value, state, args, env)
        result = {**result, "stage_receipt": str(stage_path)}
        write_json(stage_path, {**result, "schema": "gpuwm-hex.s2s-mjo-products/v1",
                               "seconds": round(time.monotonic() - started, 3)})
        return result


def _build_mjo(value, state, args, env):
    output = Path(value["paths"]["work"]) / "s2s-mjo"
    if not args.mjo_reference:
        result = {"status": "unavailable", "reason": "--mjo-reference is required: fixed observed EOFs, "
                  "normalization, dated seasonal backgrounds and 120 prior daily anomalies"}
        write_json(output / "receipt.json", result)
        if args.stage == "diagnostics":
            raise S2SError(result["reason"])
        return result
    binding, binding_path = bind_histories(value, state, args)
    result = run_backend("mjo", {"binding": str(binding_path),
        "reference": str(Path(args.mjo_reference).absolute()), "output": str(output),
        "research_data_license": args.s2s_reference_license}, output, env)
    rendered = _render_mjo(result, output, env)
    return {**{key: value for key, value in result.items() if key != "series"},
            "series_rows": len(result["series"]), "binding": str(binding_path), **rendered}


def _render_mjo(result, output, env):
    # Resolve and launch through the engine's rustwx bridge. The small adapter
    # uses that module's renderer ladder and renderer environment, preserving
    # engine provenance and the existing headless renderer contract.
    from gpuwm import rustwx
    from .engines import RENDERER, resolve
    # Hex's wheel carries the diagnostics extension. Resolve that ladder first,
    # then let rustwx validate it through its explicit renderer override.
    selected = resolve(RENDERER)
    old_renderer = os.environ.get("GPUWM_RW_WRFBATCH")
    try:
        os.environ["GPUWM_RW_WRFBATCH"] = str(selected)
        renderer = rustwx.find_renderer()
    finally:
        if old_renderer is None:
            os.environ.pop("GPUWM_RW_WRFBATCH", None)
        else:
            os.environ["GPUWM_RW_WRFBATCH"] = old_renderer
    if renderer is None:
        raise S2SError("rw_wrfbatch is missing; install the S2S diagnostic renderer")
    argv = [str(renderer), "--s2s-diagnostics", result["receipt"], "--out-dir", str(output / "figures")]
    log = output / "render.log"
    render_env = rustwx.renderer_env()
    process_env = {**os.environ, **render_env, **env}
    thread_budget = int(process_env.get("RAYON_NUM_THREADS", "0")) or cores()
    render_started = time.monotonic()
    with log.open("w") as stream:
        rendered = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                  env=process_env)
    if rendered.returncode:
        raise S2SError(f"rw_wrfbatch S2S rendering exited {rendered.returncode}; {log}")
    # Only atomic writes acknowledged by this renderer invocation count. A
    # glob can accidentally adopt an old member/basis image after a retry.
    expected = set()
    for row in result["series"]:
        stem = f"{row['diagnostic']}-{row['member']}"
        expected.add(stem + "-phase.png")
        for field in row["fields"]:
            expected.add(stem + f"-{field}-longitude-time.png")
            if row["latitude_bins"] > 1:
                expected.add(stem + f"-{field}-latitude-time.png")
    images = [Path(line.split("\t", 1)[1]).absolute() for line in log.read_text().splitlines()
              if line.startswith("RENDERED\t")]
    directory = (output / "figures").absolute()
    if (not expected or len(images) != len(expected)
            or {p.name for p in images} != expected
            or any(p.parent != directory or not p.is_file() for p in images)):
        raise S2SError(f"rw_wrfbatch did not acknowledge every expected S2S figure; {log}")
    images = sorted(images)
    return {"images": [str(p) for p in images],
            "render_seconds": round(time.monotonic() - render_started, 6),
            "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in images},
            "renderer_argv": argv, "renderer_log": str(log),
            "parallelism": {"backend": "Rust / Rayon", "threads": thread_budget}}
