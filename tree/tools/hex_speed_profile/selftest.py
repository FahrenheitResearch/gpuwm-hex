#!/usr/bin/env python3
"""CPU-only checks of trace accounting, selection and missing device events."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    for path in ROOT.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    analyzer = load("analyze_nsys_profile")
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        create table StringIds(id integer, value text);
        create table TARGET_INFO_GPU(name text, smCount integer);
        insert into TARGET_INFO_GPU values('test card', 2);
        create table NVTX_EVENTS(start integer, end integer, text text, textId integer, globalTid integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer, end integer, correlationId integer, nameId integer, globalTid integer);
        create table CUPTI_ACTIVITY_KIND_DRIVER(start integer, end integer, correlationId integer, nameId integer, globalTid integer);
        create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer, correlationId integer, shortName integer, registersPerThread integer, gridX integer, gridY integer, gridZ integer, blockX integer, blockY integer, blockZ integer, staticSharedMemory integer, dynamicSharedMemory integer, localMemoryPerThread integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer, end integer, correlationId integer, bytes integer, copyKind integer);
        create table CUPTI_ACTIVITY_KIND_MEMSET(start integer, end integer, correlationId integer);
        create table ENUM_CUDA_MEMCPY_OPER(id integer, name text);
        insert into ENUM_CUDA_MEMCPY_OPER values(1, 'DtoH');
    """)
    connection.executemany("insert into StringIds values(?, ?)", [
        (1, "cudaLaunchKernel"), (2, "cuLaunchKernel"),
        (3, "cudaGetDeviceProperties"), (4, "cudaMemcpyAsync"),
        (5, "acoustic_test_v841"),
    ])
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, null, ?)", [
        (0, 2_000_000_000, "step 1", 1),
        (2_000_000_000, 3_000_000_000, "step 2", 1),
        (100, 1_500_000_000, "dycore", 1),
        (100, 1_500_000_000, "other thread", 2),
        (2_000_000_100, 2_500_000_000, "seam_export_state", 1),
    ])
    connection.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?, ?, ?, ?, ?)", [
        (200, 400, 10, 1, 1),
        (500, 700, 11, 3, 1),
        (2_000_000_200, 2_000_000_400, 20, 4, 1),
    ])
    connection.execute("insert into CUPTI_ACTIVITY_KIND_DRIVER values(220, 380, 10, 2, 1)")
    connection.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?, ?, ?, 5, 16, 2, 1, 1, 32, 1, 1, 0, 0, 0)", [
        (400, 800, 10), (4_000_000_000, 4_000_000_100, 30),
    ])
    connection.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(2000000400, 2000000900, 20, 64, 1)")
    outputs = []
    original_connect = analyzer.sqlite3.connect
    original_write = Path.write_text
    analyzer.sqlite3.connect = lambda *args, **kwargs: connection
    Path.write_text = lambda self, content, **kwargs: outputs.append(json.loads(content))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert analyzer.main([str(ROOT / "unused.sqlite"), "--points", "100", "--out", "unused.json"]) == 0
            assert analyzer.main([str(ROOT / "unused.sqlite"), "--first", "2", "--last", "2", "--out", "unused.json"]) == 0
            # A launch in step 1 keeps the GPU busy inside step 2. The
            # pre-selection launch also has an observed tail in step 1.
            connection.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?, ?, ?, 1, 1)", [
                (1_800_000_000, 1_800_000_200, 40), (-100, -80, 41),
            ])
            connection.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?, ?, ?, 5, 16, 2, 1, 1, 32, 1, 1, 0, 0, 0)", [
                (1_900_000_000, 2_100_000_000, 40), (-50, 50, 41),
            ])
            assert analyzer.main([str(ROOT / "unused.sqlite"), "--out", "unused.json"]) == 0
            assert analyzer.main([str(ROOT / "unused.sqlite"), "--first", "2", "--last", "2", "--out", "unused.json"]) == 0
    finally:
        analyzer.sqlite3.connect = original_connect
        Path.write_text = original_write
    result, selected, asynchronous, selected_async = outputs
    assert result["steps"]["steps"] == 2
    assert result["steps"]["wall_s_mean"] == 1.5
    assert result["steps"]["wall_ns_per_point_step"] == 15_000_000
    assert result["steps"]["launch_api_host_s_mean"] == 100 / 1e9
    assert result["kernels"][0]["instances_per_step"] == 0.5
    assert result["kernels"][0]["device_s_per_step"] == 200 / 1e9
    assert result["per_label"]["dycore"]["top_kernels"][0]["name"] == "acoustic_test_v841"
    assert not result["per_label"]["other thread"]["top_kernels"]
    assert result["per_label"]["seam_export_state"]["memcpy_bytes_per_step"]["DtoH"] == 32
    assert result["per_label"]["seam_export_state"]["memcpy_device_s_per_step"]["DtoH"] == 250 / 1e9
    assert result["runtime_api_per_step"]["cudaGetDeviceProperties"]["calls"] == 0.5
    assert selected["steps"]["steps"] == 1 and selected["steps"]["wall_s_mean"] == 1
    assert not selected["kernels"]
    assert selected["per_label"]["dycore"]["instances_per_step"] == 0
    assert asynchronous["per_step"][0]["device_busy_s"] == 100_000_450 / 1e9
    assert asynchronous["per_step"][1]["device_busy_s"] == 0.1
    assert selected_async["per_step"][0]["device_busy_s"] == 0.1
    assert not selected_async["kernels"]
    summary = load("summarize_forecast_profile")
    table = summary.region_table([{"regions": [
        {"label": "outer", "t0": 0, "t1": 5, "host_s": 5, "gpu_ms": None},
        {"label": "inner", "t0": 1, "t1": 3, "host_s": 2, "gpu_ms": None, "parent": "outer"},
    ]}])
    assert table["outer"]["exclusive_s"] == 3
    assert table["outer"]["device_span_s"] == 0
    wrapper = load("profile_forecast_step")
    receipt = {
        "schema": wrapper.DRIVER_SCHEMA,
        "status": "passed",
        "sources_unchanged": True,
        "authorities_unchanged": True,
        "arwen_checkout_unchanged": True,
        "forecast": {
            "refusal": None, "full_physics_cuda_executed": True,
            "steps_requested": 4, "steps_executed": 4,
            "schedule": {"steps": 4, "dt_seconds": 5.0},
            "walls": {"steps": 4, "forecast_seconds": 20.0},
        },
    }
    eligibility = lambda value, **overrides: wrapper.completion_evidence(**{
        "rc": 0, "completed_steps": 4, "receipt": value,
        "final_time_seconds": 20.0, "forcing_time_seconds": 20.0,
        **overrides,
    })
    assert eligibility(receipt)["eligible"]
    truncated = copy.deepcopy(receipt)
    truncated["status"] = "truncated_by_model_refusal"
    truncated["forecast"].update({"refusal": {"refused": True}, "steps_executed": 2})
    truncated["forecast"]["walls"].update({"steps": 2, "forecast_seconds": 10.0})
    assert not eligibility(truncated, completed_steps=2, final_time_seconds=10.0, forcing_time_seconds=10.0)["eligible"]
    assert not eligibility(None)["eligible"]
    assert not eligibility([])["eligible"]
    malformed = copy.deepcopy(receipt)
    del malformed["forecast"]["refusal"]
    assert not eligibility(malformed)["eligible"]
    assert not eligibility(receipt, completed_steps=3)["eligible"]
    assert not eligibility(receipt, forcing_time_seconds=None)["eligible"]
    assert not eligibility(receipt, forcing_time_seconds=15.0)["eligible"]
    assert not eligibility(receipt, final_time_seconds=15.0)["eligible"]
    assert not eligibility(receipt, rc=1)["eligible"]
    connection.close()
    print("PASS: syntax, all-step inclusion, thread attribution, launch union, copies, point normalization, selected windows, asynchronous execution tails, absent events, complete/truncated receipt eligibility, required forcing carrier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
