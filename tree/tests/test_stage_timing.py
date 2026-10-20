"""Timing survives stage errors without changing subprocess refusal behavior."""
import json
import sys

import pytest

from hexcore.moving_members import _run_logged
from hexcore.moving_case import MovingCaseError


@pytest.mark.parametrize("returncode", [0, 7])
def test_subprocess_timing_survives_failure(tmp_path, returncode):
    log = tmp_path / "stage.log"
    argv = [sys.executable, "-c", f"raise SystemExit({returncode})"]
    if returncode:
        with pytest.raises(MovingCaseError, match="exited 7"):
            _run_logged(argv, log, env={"RAYON_NUM_THREADS": "4"})
    else:
        assert _run_logged(argv, log, env={"RAYON_NUM_THREADS": "4"}) > 0
    row = json.loads(log.with_suffix(".timing.json").read_text())["stages"][0]
    assert row["returncode"] == returncode
    assert row["status"] == ("failed" if returncode else "done")
    assert row["started_utc"] <= row["ended_utc"]
    assert row["wall_seconds"] > 0
    assert row["parallelism"]["requested_rayon_threads"] == "4"
