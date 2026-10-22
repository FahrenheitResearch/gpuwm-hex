"""Regional orchestration receipts must survive unsuccessful stages."""
import json
import os

import pytest

from hexcore import regional_door as door
from hexcore.cli import build_parser
from hexcore.errors import MpasPortError


def arguments(tmp_path):
    return build_parser().parse_args([
        "regional", "--point", "36.5,-97.5", "--cycle", "2024-05-06T18",
        "--hours", "3", "--work-dir", str(tmp_path / "run")])


@pytest.mark.parametrize("failure", [None, "returncode", "refusal", "exception"])
def test_receipt_records_stage_outcomes_and_restores_environment(tmp_path, monkeypatch, failure):
    args = arguments(tmp_path)
    monkeypatch.setenv("GPUWM_HEX_MESH_ROWS", "previous-rows")
    monkeypatch.setenv("GPUWM_HEX_REGIONAL_CONTRACT_DIR", "previous-contracts")
    mesh = {"row": "generated-cull", "grid": "grid.nc", "static": "static.nc"}
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": mesh})
    seen = []

    def runner(name):
        def run(args, value, status):
            live = json.loads((args.work_dir / "regional-receipt.json").read_text())
            assert live["stages"][name]["outcome"] == "running"
            assert os.environ["GPUWM_HEX_MESH_ROWS"].endswith(os.pathsep + "previous-rows")
            seen.append(name)
            if name == "products":
                if failure == "refusal":
                    raise MpasPortError("renderer missing")
                if failure == "exception":
                    raise RuntimeError("renderer crashed")
                return {"returncode": 1 if failure else 0,
                        "counts": {"rendered": 4, "skipped": 2, "failed": 1 if failure else 0}}
            return {"returncode": 0}
        return run

    monkeypatch.setattr(door, "_RUNNERS", {name: runner(name) for name in door.STAGES})
    if failure in ("refusal", "exception"):
        with pytest.raises((MpasPortError, RuntimeError)):
            door.run_regional(args)
    else:
        assert door.run_regional(args) == (1 if failure else 0)
    receipt = json.loads((args.work_dir / "regional-receipt.json").read_text())
    assert receipt["mesh"] == mesh
    assert receipt["outcome"] == ("refused" if failure == "refusal" else "failed" if failure else "done")
    assert receipt["seconds"] >= 0 and receipt["finished_utc"] >= receipt["started_utc"]
    assert seen == list(door.STAGES)
    for name, stage in receipt["stages"].items():
        assert stage["outcome"] == ("failed" if failure and name == "products" else "done")
        assert stage["seconds"] >= 0 and stage["finished_utc"] >= stage["started_utc"]
    if failure not in ("refusal", "exception"):
        assert receipt["products"]["counts"]["rendered"] == 4
    assert os.environ["GPUWM_HEX_MESH_ROWS"] == "previous-rows"
    assert os.environ["GPUWM_HEX_REGIONAL_CONTRACT_DIR"] == "previous-contracts"


def test_hrrr_is_limited_area_and_cannot_initialize_global_mesh():
    from hexcore.init_sources import load_table, resolve_source

    table = load_table()
    assert resolve_source(table, "hrrr-conus", coverage="limited_area")["intermediate"]["row"] == "hrrr"
    with pytest.raises(MpasPortError, match="covers a limited area"):
        resolve_source(table, "hrrr")
    with pytest.raises(MpasPortError, match="is a global source"):
        resolve_source(table, "gfs", coverage="limited_area")
