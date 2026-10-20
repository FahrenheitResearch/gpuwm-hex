"""Checks for the narrow CLI hook in the lane-owned pinned engine install."""
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest


def staging():
    path = Path(__file__).resolve().parents[1] / "tools/hex_timeline/stage_timeline.py"
    spec = importlib.util.spec_from_file_location("hex_timeline_staging", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_staging_refuses_an_install_outside_the_owned_lane(tmp_path):
    import gpuwm

    cli = Path(gpuwm.__file__).parent / "cli.py"
    before = hashlib.sha256(cli.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="owned install"):
        staging().stage(tmp_path, tmp_path)
    assert hashlib.sha256(cli.read_bytes()).hexdigest() == before


def test_real_pinned_install_staging_is_idempotent():
    lane = os.environ.get("GPUWM_HEX_TIME_TEST_LANE")
    if not lane:
        pytest.skip("requires the explicit private 2.8.0 measurement install")
    lane = Path(lane)
    first = staging().stage(lane / "core", lane)
    second = staging().stage(lane / "core", lane)
    assert first["engine"] == second["engine"] == "2.8.0"
    assert first["manifest_fields"] == second["manifest_fields"] == 16
    assert first["numerical_seam_unchanged"] and second["numerical_seam_unchanged"]
    assert second["cli_before_sha256"] == second["cli_after_sha256"]


def test_real_calculator_reports_the_uncovered_tail(tmp_path):
    lane = os.environ.get("GPUWM_HEX_TIME_TEST_LANE")
    if not lane:
        pytest.skip("requires the explicit private measurement calculator")
    import json
    import subprocess
    import sys
    from hexcore import time_ledger

    path = tmp_path / "coverage.jsonl"
    saved = os.environ.get("GPUWM_TIME_LEDGER")
    os.environ["GPUWM_TIME_LEDGER"] = str(path)
    try:
        time_ledger.configure(tmp_path)
        time_ledger.emit("forecast.integrate", "begin", t_unix=1700000000, t_mono=100)
        time_ledger.emit("forecast.integrate", "end", t_unix=1700000008, t_mono=108)
        time_ledger.emit("finish", t_unix=1700000012, t_mono=112)
    finally:
        if saved is None:
            os.environ.pop("GPUWM_TIME_LEDGER")
        else:
            os.environ["GPUWM_TIME_LEDGER"] = saved
    output = tmp_path / "timeline.json"
    result = subprocess.run([sys.executable, "-m", "gpuwm.cli", "timeline", str(path),
        "--proxy-start", "1700000000", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["max_unnamed_gap_seconds"] == 4
    assert report["named_coverage_seconds"] == 8
    assert report["unnamed_gaps"] == [{"start": 1700000008, "end": 1700000012, "seconds": 4}]
