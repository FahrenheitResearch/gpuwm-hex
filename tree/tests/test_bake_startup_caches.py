"""The start-up cache bake fills on the first run and proves the second adds nothing."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "bake_startup_caches.py"

FAKE_FORECAST = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
always_compile = "--always-compile" in args
prep, cupy = Path(os.environ["GPUWM_HEX_PREP_CACHE"]), Path(os.environ["CUPY_CACHE_DIR"])
entry = prep / "ab" / ("ab" + "0" * 62)
runs = len(list(cupy.glob("*.cubin")))
first = not entry.exists()
if first:
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text("{}")
if first or always_compile:
    (cupy / f"module-{runs}.cubin").write_bytes(b"cubin")
out.mkdir(parents=True)
(out / "cuda-history.2026-10-03_12.00.00.nc").write_bytes(b"same frame bytes")
calls = [{"label": "unit", "event": "miss" if first else "hit-disk"}]
(out / "cuda-v841-forecast-receipt.json").write_text(json.dumps({"host_preparation_cache": {"calls": calls}}))
'''


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("bake_startup_caches", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_second_run_that_adds_nothing_is_baked(tool, tmp_path):
    rc = tool.main(["--cache-dir", str(tmp_path / "cache"), "--work", str(tmp_path / "work"),
                    "--", "-c", FAKE_FORECAST])
    record = json.loads((tmp_path / "cache" / "BAKE-RECORD.json").read_text())
    assert rc == 0 and record["verdict"]["baked"] is True
    assert record["prep_cache"]["added_by_fill"] == 1 and record["cupy_cache"]["added_by_fill"] == 1
    assert record["verdict"]["verify_prep_events"] == {"hit-disk": 1}
    assert record["runtime_environment"]["CUPY_CACHE_DIR"].endswith("cupy")


def test_a_verify_run_that_compiles_is_refused(tool, tmp_path):
    rc = tool.main(["--cache-dir", str(tmp_path / "cache"), "--work", str(tmp_path / "work"),
                    "--", "-c", FAKE_FORECAST, "--always-compile"])
    record = json.loads((tmp_path / "cache" / "BAKE-RECORD.json").read_text())
    assert rc == 1 and record["verdict"]["baked"] is False
    assert record["verdict"]["verify_new_cupy_modules"] == 1


def test_out_and_scratch_belong_to_the_tool(tool, tmp_path):
    assert tool.main(["--cache-dir", str(tmp_path), "--", "-c", "pass", "--out", "x"]) == 2
