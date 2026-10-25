"""``gpuwm-hex regional --parent-*``: a region cut from a kept hex run (WOOF's "keep this run usable as a start").

The door plans from the parent's own files: the frame at the start, every frame through the window, and refuses by name
a start that is not one of the parent's frames and a window past its last frame.  The stages are the parent route's
(analysis, init, lbc from the parent), the caller's forecast flags reach the forecast door, and the parent's state at a
later frame carries the frame's time-varying fields into the init's layout.
"""
import json

import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore import parent_state
from hexcore import regional_door as door
from hexcore.cli import build_parser


def parent(tmp_path, hours=6, every=60):
    folder = tmp_path / "parent"
    (folder / "history").mkdir(parents=True)
    for name in ("grid.nc", "static.nc", "init.nc"):
        (folder / name).write_bytes(b"x")
    from datetime import datetime, timedelta
    for k in range(0, hours * 60 + 1, every):
        stamp = (datetime(2025, 5, 19, 18) + timedelta(minutes=k)).strftime("%Y-%m-%d_%H.%M.%S")
        (folder / "history" / f"cuda-history.{stamp}.nc").write_bytes(b"x")
    return folder


def arguments(tmp_path, folder, *extra, start="2025-05-19T19", hours="3"):
    return build_parser().parse_args([
        "regional", "--point", "35.47,-97.52", "--radius-km", "150", "--fine-dx-m", "3000", "--hours", hours,
        "--work-dir", str(tmp_path / "run"), "--parent-grid", str(folder / "grid.nc"),
        "--parent-static", str(folder / "static.nc"), "--parent-init", str(folder / "init.nc"),
        "--parent-history-dir", str(folder / "history"), "--parent-start", start, *extra])


def test_plan_reads_the_parent_frames_from_the_start_through_the_window(tmp_path):
    folder = parent(tmp_path)
    value = door.plan(arguments(tmp_path, folder))
    p = value["parent"]
    assert p["start"] == "2025-05-19T19:00"
    assert p["frame"].endswith("cuda-history.2025-05-19_19.00.00.nc")
    assert [h[-22:-3] for h in p["history"]] == ["2025-05-19_20.00.00", "2025-05-19_21.00.00", "2025-05-19_22.00.00"]
    assert value["start_time"] == "2025-05-19_19:00:00" and value["stop_time"] == "2025-05-19_22:00:00"
    assert value["source"]["id"] == "hex-parent"
    assert value["fine_dx_m"] == 3000.0 and value["radius_km"] == 150.0


def test_a_start_that_is_not_a_parent_frame_and_a_window_past_its_end_are_refused(tmp_path):
    folder = parent(tmp_path, hours=6)
    with pytest.raises(door.RegionalRefusal, match="no history frame at 2025-05-19 19:30"):
        door.plan(arguments(tmp_path, folder, start="2025-05-19T19:30"))
    with pytest.raises(door.RegionalRefusal, match="frames end at 2025-05-20 00:00, before this 6 h region"):
        door.plan(arguments(tmp_path, folder, start="2025-05-19T19", hours="6"))


def test_a_partial_parent_names_what_is_missing(tmp_path):
    folder = parent(tmp_path)
    args = build_parser().parse_args([
        "regional", "--point", "35.47,-97.52", "--hours", "3", "--work-dir", str(tmp_path / "run"),
        "--parent-grid", str(folder / "grid.nc"), "--parent-start", "2025-05-19T19"])
    with pytest.raises(door.RegionalRefusal, match="--parent-static, --parent-init, --parent-history-dir was not given"):
        door.plan(args)


def test_a_source_region_still_needs_its_cycle(tmp_path):
    args = build_parser().parse_args(["regional", "--point", "36.5,-97.5", "--hours", "3", "--work-dir", str(tmp_path / "run")])
    with pytest.raises(door.RegionalRefusal, match="--cycle was not given"):
        door.plan(args)


def test_the_parent_route_runs_its_own_stages(tmp_path, monkeypatch):
    folder = parent(tmp_path)
    args = arguments(tmp_path, folder)
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": {"row": "cull", "grid": "g.nc", "static": "s.nc"}})
    seen = []
    monkeypatch.setattr(door, "_RUNNERS", {name: (lambda name: lambda a, v, s: seen.append(name) or {"returncode": 0})(name)
                                           for name in door.PARENT_STAGES})
    assert door.run_regional(args) == 0
    assert seen == list(door.PARENT_STAGES)
    receipt = json.loads((tmp_path / "run" / "regional-receipt.json").read_text())
    assert receipt["source"] == "hex-parent" and receipt["outcome"] == "done"


def test_forecast_flags_reach_the_door_and_a_named_suite_keeps_the_row_step(tmp_path, monkeypatch):
    """Release review 20 (2.9.7): a named suite dropped `--dt row`, so the suite route took the bare 6 s/km rule
    instead of the Courant-stepped row step the contract stage minted, and the forecast refused an unminted class
    after the whole preparation.  The step the forecast door selects must be the contract's."""
    import argparse
    from types import SimpleNamespace

    from hexcore import forecast_door

    folder = parent(tmp_path)
    cull = {"row": "cull", "grid": "g.nc", "static": "s.nc", "lbc_dir": str(tmp_path / "lbc")}
    # 937.5 m: the bare rule gives 5 s; the Courant check stepped this mesh down to 4 s.
    contract_dt = 4.0
    monkeypatch.setattr(door, "_generated", lambda value: {"cull": cull, "parent": {"dt_seconds": contract_dt}})
    sent = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: sent.append(argv) or 0)
    monkeypatch.setattr(forecast_door, "resolve_repo", lambda repo: None)
    monkeypatch.setattr(forecast_door, "load_registry",
                        lambda repo: {"cull": SimpleNamespace(nominal_dx_m=937.5, dt_seconds=contract_dt)})
    from hexcore.mpasphys.forecast_v842 import namelist_option

    for extra in (["--forecast-arg=--physics=config_microp_scheme=mp_thompson"],
                  ["--forecast-arg=--physics-suite=convection_permitting"]):
        args = arguments(tmp_path, folder, *extra)
        value = door.plan(args)
        door._stage_forecast(args, value, None)
        argv = sent[-1]
        assert argv[argv.index("--dt") + 1] == "row"
        assert argv[-1] == extra[0].split("=", 1)[1]
        assert "--lbc-dir" in argv and argv[argv.index("--start-time") + 1] == "2025-05-19_19:00:00"
        # The suite route's own step selection, on the argv the region door sent.
        text = forecast_door.bind_native_step(
            argparse.Namespace(dt=argv[argv.index("--dt") + 1], mesh=cull["row"], repo=None),
            "&nhyd_model\n/\n")
        assert float(namelist_option(text, "nhyd_model", "config_dt")) == contract_dt

    # A physics namelist whose config_dt names another step is refused by name before the forecast.
    namelist = tmp_path / "namelist.atmosphere"
    namelist.write_text("&nhyd_model\n    config_dt = 5.0\n/\n", encoding="utf-8")
    args = arguments(tmp_path, folder, f"--forecast-arg=--physics-namelist={namelist}")
    with pytest.raises(door.RegionalRefusal, match="config_dt = 5 s"):
        door._stage_forecast(args, door.plan(args), None)
    namelist.write_text("&nhyd_model\n    config_dt = 4.0\n/\n", encoding="utf-8")
    door._stage_forecast(args, door.plan(args), None)
    assert sent[-1][sent[-1].index("--dt") + 1] == "row"


def test_the_parent_state_carries_the_frames_time_varying_fields(tmp_path):
    frame, init = tmp_path / "frame.nc", tmp_path / "init.nc"
    for path, value in ((frame, 2.0), (init, 1.0)):
        with Dataset(path, "w") as d:
            d.createDimension("Time", None)
            d.createDimension("nCells", 3)
            d.createDimension("nSoilLevels", 4)
            d.createVariable("skintemp", "f4", ("Time", "nCells"))[0] = np.full(3, value)
            d.createVariable("smois", "f4", ("Time", "nCells", "nSoilLevels"))[0] = np.full((3, 4), value)
            d.createVariable("ter", "f4", ("nCells",))[:] = np.full(3, value)
            d.createVariable("indexToCellID", "i4", ("nCells",))[:] = np.arange(1, 4)
            d.createVariable("theta_m", "f4", ("Time", "nCells"))[0] = np.full(3, value)
    with Dataset(init, "r+") as d, Dataset(frame) as f:
        carried = parent_state.carry_time_state(d, f, {"theta_m"})
        assert carried == ["skintemp", "smois"]
        assert float(d["skintemp"][0, 0]) == 2.0 and float(d["smois"][0, 0, 0]) == 2.0
        # Time-invariant fields (terrain), the index maps and the state already written stay the init's.
        assert float(d["ter"][0]) == 1.0 and float(d["theta_m"][0, 0]) == 1.0


def test_the_mesh_is_built_on_the_ladder_the_caller_names(tmp_path, monkeypatch):
    """Any spacing is a rung of some halving ladder: a 1 km region names the 128 km ladder (the mesh door's default 120 km
    ladder holds 937.5 m and 1.875 km, not 1 km), and the mesh stage passes it on. Without it, the region door could
    only build the 120 km ladder's rungs (WOOF open-ended hex resolution, 2026-10-24)."""
    folder = parent(tmp_path)
    value = door.plan(arguments(tmp_path, folder, "--fine-dx-m", "1000", "--background-km", "128"))
    assert value["fine_dx_m"] == 1000.0 and value["background_km"] == 128.0
    seen = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: seen.append(argv) or 0)
    door._stage_mesh(type("A", (), {"geog": None})(), value, None)
    argv = seen[0]
    assert argv[argv.index("--background-km") + 1] == "128.0" and argv[argv.index("--fine-dx-m") + 1] == "1000.0"
    # Not given: the mesh door's own default, as before.
    assert door.plan(arguments(tmp_path, folder))["background_km"] is None
