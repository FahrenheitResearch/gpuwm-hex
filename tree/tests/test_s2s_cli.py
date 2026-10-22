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


@pytest.mark.parametrize("source", ["gefs", "ifs-ens", "gfs", "era5", "era5-arco", "cfs"])
@pytest.mark.parametrize("row,days", [("global-120km", 10), ("global-120km", 21),
                                     ("global-120km", 29), ("global-120km", 30), ("global-60km", 90)])
def test_public_source_row_dry_run_is_offline(tmp_path, source, row, days):
    destination = tmp_path / "not-created"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               GPUWM_HEX_CUDA_GRAPHS="0", GPUWM_HEX_CUDA_GRAPHS_REQUIRE_REPLAY="0")
    command = [sys.executable, "-B", "-c", GUARDED_ENTRYPOINT, "s2s", "--row", row,
               "--source", source, "--cycle", "2023-03-31T12", "--members", "4",
               "--days", str(days), "--work-dir", str(destination),
               "--product-region", "tropics:-25,25,30,130",
               "--research-data-license", "copernicus-products", "--dry-run"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert not list(tmp_path.iterdir())
    assert value["product_options"]["regions"] == ["tropics:-25,25,30,130"]
    assert value["product_options"]["research_data_license"] == "copernicus-products"
    assert value["row"]["id"] == row and value["source"] == source
    assert len(value["members"]) == 4 and value["surface_updating"]
    start = datetime(2023, 3, 31, 12)
    assert value["surface"]["end"] == (start + timedelta(days=days)).strftime("%Y-%m-%d_%H:%M:%S")
    # Default --ocean-mode prognostic (OCEAN-ICE.md): SST/ice evolve from each
    # member's initialization and no dated surface series is read.
    assert value["surface"]["ocean_mode"] == "prognostic"
    assert value["surface"]["owner"] == "prognostic-ocean" and value["surface"]["path"] is None
    for member in value["members"]:
        cycle = datetime.strptime(member["source_cycle"], "%Y-%m-%d_%H:%M:%S")
        spinup = member["provenance"].get("spinup_hours", 0)
        assert member["hours"] == days * 24 + spinup
        assert cycle + timedelta(hours=member["lead_hours"] + spinup) == start
        assert datetime.strptime(member["start_time"], "%Y-%m-%d_%H:%M:%S") + timedelta(hours=spinup) == start
        assert member["history_from_minutes"] == 60 * spinup
        assert member["checkpoint_hours"] == ",".join(str(h + spinup) for h in range(120, days * 24 + 1, 120))
    if source in ("gefs", "ifs-ens"):
        assert [member["source_member"] for member in value["members"]] == [1, 2, 3, 4]
    else:
        assert [member["lag_hours"] for member in value["members"]] == [0, 6, 12, 18]


def test_public_members_help_matches_count_parser(tmp_path):
    """The exact defect: S2S inherited native-member SPEC help with an int parser."""
    help_result = subprocess.run([sys.executable, "-m", "hexcore.cli", "s2s", "--help"],
                                 capture_output=True, text=True, timeout=20)
    assert help_result.returncode == 0, help_result.stderr
    help_text = " ".join(help_result.stdout.split())
    assert "--members COUNT" in help_text
    assert "total output member count (1..1000, default: 4)" in help_text
    assert "1-30" not in help_text and "0,1,2" not in help_text
    assert "forecast days (7 or more)" in help_text
    assert "the minimum is the week-1 product window" in help_text
    command = [sys.executable, "-B", "-c", GUARDED_ENTRYPOINT, "s2s", "--row", "global-120km",
               "--source", "era5", "--cycle", "2023-03-31T12", "--days", "10",
               "--work-dir", str(tmp_path / "not-created"), "--dry-run"]
    for count in (None, "1", "4"):
        result = subprocess.run(command + (["--members", count] if count else []),
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert len(json.loads(result.stdout)["members"]) == int(count or 4)
    for count in ("1-4", "0,1,2", "0", "1001"):
        result = subprocess.run(command + ["--members", count], capture_output=True, text=True, timeout=20)
        assert result.returncode != 0
        assert ("not a member list" if count in ("1-4", "0,1,2") else "--members must be in 1..1000") in result.stderr
    assert not list(tmp_path.iterdir())


def test_global_members_still_accepts_native_member_specs():
    from hexcore.cli import build_parser
    args = build_parser().parse_args(["global", "--members", "1-4"])
    assert args.members == "1-4"
