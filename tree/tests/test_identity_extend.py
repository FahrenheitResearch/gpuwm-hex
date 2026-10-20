"""Bitwise native identity of the v8.4 scalar transport on a moist case.

The fixture is a native MPAS-A v8.4.2 GNU strict one-step run that also wrote
ruAvg and wwAvg (tools/native_dry_reference/run_native_case.sh with
EXTRA_STREAM_VARS="ruAvg wwAvg") plus the init it started from. With the
v8.2.3 vertical upwind association this replay differs from native in 43,513
of 2,252,910 qv words on the x1.40962 real-terrain case; with native's v8.4
order it differs in none. No tolerance: every 32-bit word must match.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/native_dry_reference/cpu_transport_replay.py"


@pytest.mark.slow
def test_scalar_transport_replays_native_fluxes_bit_for_bit(tmp_path):
    native = os.environ.get("HEXCORE_NATIVE_FLUX_RUN")
    init = os.environ.get("HEXCORE_NATIVE_FLUX_INIT")
    if not native or not init:
        pytest.skip("set HEXCORE_NATIVE_FLUX_RUN (native one-step run with ruAvg/wwAvg) and HEXCORE_NATIVE_FLUX_INIT")
    out = tmp_path / "replay.json"
    subprocess.run([sys.executable, str(TOOL), "--init", init, "--native", native, "--out", str(out)], check=True)
    report = json.loads(out.read_text())["report"]
    assert report["changed_by_native_step"] > 0, "the fixture must carry a moving nonzero scalar"
    assert report["identical"] is True and report["different_words"] == 0
