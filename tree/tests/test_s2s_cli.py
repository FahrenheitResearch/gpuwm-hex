"""Every advertised row/source reaches public planning without compute or I/O."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


GUARDED_ENTRYPOINT = """
import runpy, sys
def no_network(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('dry-run attempted network access: ' + event)
class NoCuPy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise AssertionError('dry-run attempted CuPy import')
sys.addaudithook(no_network)
sys.meta_path.insert(0, NoCuPy())
sys.argv = ['gpuwm-hex'] + sys.argv[1:]
runpy.run_module('hexcore.cli', run_name='__main__')
"""


@pytest.mark.parametrize("source", ["gefs", "ifs-ens", "gfs", "era5", "cfs"])
@pytest.mark.parametrize("row,days", [("global-120km", 30), ("global-60km", 90)])
def test_public_source_row_dry_run_is_offline(tmp_path, source, row, days):
    destination = tmp_path / "not-created"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               GPUWM_HEX_CUDA_GRAPHS="0", GPUWM_HEX_CUDA_GRAPHS_REQUIRE_REPLAY="0")
    command = [sys.executable, "-B", "-c", GUARDED_ENTRYPOINT, "s2s", "--row", row,
               "--source", source, "--cycle", "2023-03-31T12", "--members", "4",
               "--days", str(days), "--work-dir", str(destination), "--dry-run"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert not list(tmp_path.iterdir())
    assert value["row"]["id"] == row and value["source"] == source
    assert len(value["members"]) == 4 and value["surface_updating"]
    start = datetime(2023, 3, 31, 12)
    assert value["surface"]["end"] == (start + timedelta(days=days)).strftime("%Y-%m-%d_%H:%M:%S")
    assert Path(value["surface"]["path"]) == destination / "surface_series.nc"
    for member in value["members"]:
        assert member["hours"] == days * 24
        assert member["valid_time"] == member["start_time"] == "2023-03-31_12:00:00"
        cycle = datetime.strptime(member["source_cycle"], "%Y-%m-%d_%H:%M:%S")
        assert cycle + timedelta(hours=member["lead_hours"]) == start
        assert member["checkpoint_hours"] == ",".join(str(h) for h in range(120, days * 24 + 1, 120))
    if source in ("gefs", "ifs-ens"):
        assert [member["source_member"] for member in value["members"]] == [1, 2, 3, 4]
    elif source == "era5":
        assert {member["lag_hours"] for member in value["members"]} == {0}
    else:
        assert [member["lag_hours"] for member in value["members"]] == [0, 6, 12, 18]
