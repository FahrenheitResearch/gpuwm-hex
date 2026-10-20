"""The profiler selftest resolves drivers through the shipped forecast door."""
from pathlib import Path
import json
import subprocess
import sys
import importlib.util
import hashlib
import numpy as np

import pytest


@pytest.mark.parametrize("legacy_repo", [False, True])
def test_selftest_uses_packaged_drivers(tmp_path, legacy_repo):
    script = Path(__file__).parents[1] / "tools/hex_speed_profile/profile_forecast_step.py"
    args = [sys.executable, str(script), "--selftest", "--report", str(tmp_path / "report"), "--"]
    if legacy_repo:
        args.extend(["--repo", str(tmp_path / "absent-checkout")])
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["missing"] == []
    assert "driver.proof.execute_composite_step -> step bracket" in report["patched"]
    if legacy_repo:
        assert "is not used" in result.stderr


def test_profiled_background_writer_uses_the_bound_mesh_and_direct_bytes(tmp_path, monkeypatch):
    from hexcore import history_writer
    from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof
    from test_history_writer import _frame

    script = Path(__file__).parents[1] / "tools/hex_speed_profile/profile_forecast_step.py"
    spec = importlib.util.spec_from_file_location("profile_writer_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in (("N_CELLS",97),("N_EDGES",291),("N_LEVELS",7),("N_INTERFACES",8),("N_SOIL_LEVELS",4)):
        monkeypatch.setattr(proof, name, value)
    monkeypatch.delenv(history_writer.ENVIRONMENT, raising=False)
    original = proof.write_snapshot_netcdf
    frame = _frame(np.random.default_rng(3),97,291,7,4)
    original(tmp_path / "direct.nc", *frame)
    monkeypatch.setattr(proof, "write_snapshot_netcdf", original)
    prof = module.StepProfiler(tmp_path / "profile", nsys_window=None, cprofile_window=None, use_nvtx=False, use_events=False, baseline=False)
    prof.wrap(proof,"write_snapshot_netcdf","history_write",owner_name="proof")
    assert history_writer._unwrap(proof.write_snapshot_netcdf) is original
    writer = history_writer.open_writer(proof.write_snapshot_netcdf)
    try:
        writer.result(writer.submit(tmp_path / "background.nc", *frame))
    finally:
        writer.close()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(tmp_path / "direct.nc") == digest(tmp_path / "background.nc")
