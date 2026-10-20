import copy
import json

import pytest

from hexcore.cli import build_parser
from hexcore.country_door import TABLE, plan, load_row
from hexcore.errors import MpasPortError


def arguments(tmp_path, table=TABLE, row="philippines"):
    return build_parser().parse_args(["country", "--row", row, "--table", str(table),
        "--cycle", "2025-11-03T00", "--work-dir", str(tmp_path / "run"), "--gpuwm-checkout", str(tmp_path / "engine")])


def test_row_connects_mesh_source_region_and_product_set(tmp_path):
    result = plan(arguments(tmp_path))
    assert PathExists(result["paths"]["spec"])
    assert result["commands"]["init"][2] == "gfs"
    assert result["commands"]["forecast"][2] == "ph48-3"
    assert "--init-source" in result["commands"]["forecast"]
    assert result["scorecard"]["region"]["west"] == 116.0
    assert "--region" in result["commands"]["products"]
    for command in result["commands"].values():
        build_parser().parse_args(command)


def PathExists(path):
    from pathlib import Path
    return Path(path).is_file()


def test_installed_engine_reaches_the_country_door_without_a_checkout(tmp_path):
    # A pip install has no gpuwm checkout. The forecast door already defaults
    # to the installed engine; the country door must not demand a checkout
    # the forecast it composes does not need.
    args = build_parser().parse_args(["country", "--row", "philippines", "--cycle", "2025-11-03T00",
                                      "--work-dir", str(tmp_path / "run")])
    forecast = plan(args)["commands"]["forecast"]
    assert "--gpuwm-checkout" not in forecast
    build_parser().parse_args(forecast)
    checked_out = plan(arguments(tmp_path))["commands"]["forecast"]
    assert checked_out[checked_out.index("--gpuwm-checkout") + 1] == str(tmp_path / "engine")


def test_an_unseen_country_is_only_a_row(tmp_path):
    data = json.loads(TABLE.read_text())
    row = copy.deepcopy(data["countries"][0])
    row["id"] = "new-area"
    row["mesh"]["id"] = "another-mesh"
    row["scorecard"]["region"] = {"west": -10, "south": 40, "east": 10, "north": 60}
    data["countries"] = [row]
    table = tmp_path / "table.json"
    table.write_text(json.dumps(data))
    result = plan(arguments(tmp_path, table, "new-area"))
    assert result["commands"]["forecast"][2] == "another-mesh"
    assert result["scorecard"]["region"]["west"] == -10


def test_disconnected_product_set_is_refused(tmp_path):
    data = json.loads(TABLE.read_text())
    data["countries"][0]["product_set"] = "missing"
    table = tmp_path / "table.json"
    table.write_text(json.dumps(data))
    with pytest.raises(MpasPortError, match="absent product set"):
        load_row("philippines", table)


def test_observation_file_override_does_not_move_the_country(tmp_path):
    args = arguments(tmp_path)
    args.observation_config = tmp_path / "observations.json"
    args.observation_config.write_text(json.dumps({"region": {"west": 0}, "observation_files": {"time": {"surface": "local.csv"}}}))
    result = plan(args)
    assert result["scorecard"]["region"]["west"] == 116.0
    assert result["scorecard"]["observation_files"]["time"]["surface"] == "local.csv"


def test_forecast_stage_leaves_destination_creation_to_driver(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from hexcore import cli
    from hexcore.country_door import run_country
    args = arguments(tmp_path)
    args.stage = "forecast"
    args.work_dir.mkdir()
    (args.work_dir / "mesh-rows.json").write_text('{}')
    def handler(argv):
        assert not (args.work_dir / "forecast").exists()
        assert Path(os.environ["GPUWM_HEX_OBSERVATION_CONFIG"]).is_file()
        return 0
    monkeypatch.setattr(cli, "main", handler)
    monkeypatch.setenv("GPUWM_HEX_OBSERVATION_CONFIG", "previous")
    assert run_country(args) == 0
    assert os.environ["GPUWM_HEX_OBSERVATION_CONFIG"] == "previous"


@pytest.mark.parametrize(("stage", "skip", "expected"), [
    ("all", [], ["prepare", "forecast", "products", "score"]),
    ("all", ["products"], ["prepare", "forecast", "score"]),
    ("all", ["prepare", "products"], ["forecast", "score"]),
    ("forecast", [], ["forecast"]),
    ("products", ["products"], []),
])
def test_country_dispatches_selected_stages_with_global_status(tmp_path, monkeypatch, stage, skip, expected):
    from hexcore import country_door
    args = arguments(tmp_path)
    assert args.stage == "all"
    args.stage, args.skip = stage, skip
    called = []
    def execute(name, args, value, status):
        called.append(name)
        return {"returncode": 0}
    monkeypatch.setattr(country_door, "_run_stage", execute)
    assert country_door.run_country(args) == 0
    assert called == expected
    events = [json.loads(line) for line in (args.work_dir / "events.jsonl").read_text().splitlines()]
    assert [e["stage"] for e in events if e["event"] == "stage-start"] == expected
    assert [e["stage"] for e in events if e["event"] == "stage-end"] == expected
    assert json.loads((args.work_dir / "status.json").read_text())["state"] == "done"
    receipt = json.loads((args.work_dir / "country-receipt.json").read_text())
    assert list(receipt["stages"]) == expected
    assert all(s["seconds"] >= 0 for s in receipt["stages"].values())


def test_country_failure_stops_following_stages_and_restores_environment(tmp_path, monkeypatch):
    import os
    from hexcore import country_door
    args = arguments(tmp_path)
    called = []
    def execute(name, args, value, status):
        called.append(name)
        assert os.environ["GPUWM_HEX_MESH_ROWS"].startswith(str(args.work_dir / "mesh-rows.json"))
        return {"returncode": 7 if name == "forecast" else 0}
    monkeypatch.setattr(country_door, "_run_stage", execute)
    monkeypatch.setenv("GPUWM_HEX_MESH_ROWS", "prior-rows")
    monkeypatch.setenv("GPUWM_HEX_OBSERVATION_CONFIG", "prior-config")
    assert country_door.run_country(args) == 7
    assert called == ["prepare", "forecast"]
    assert os.environ["GPUWM_HEX_MESH_ROWS"] == "prior-rows"
    assert os.environ["GPUWM_HEX_OBSERVATION_CONFIG"] == "prior-config"
    assert json.loads((args.work_dir / "status.json").read_text())["state"] == "failed"
    assert json.loads((args.work_dir / "country-receipt.json").read_text())["stages"]["forecast"]["returncode"] == 7


def test_country_plan_is_explicit_and_does_not_create_run(tmp_path, capsys):
    from hexcore.country_door import run_country
    args = arguments(tmp_path)
    args.stage = "plan"
    assert run_country(args) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "gpuwm-hex.country-plan/v1"
    assert not args.work_dir.exists()


def test_country_skip_is_repeatable_and_rejects_unknown_stages(tmp_path):
    args = arguments(tmp_path)
    command = ["country", "--row", "philippines", "--cycle", args.cycle, "--work-dir", str(args.work_dir)]
    assert build_parser().parse_args(command + ["--skip", "products", "--skip", "score"]).skip == ["products", "score"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(command + ["--skip", "typo"])


@pytest.mark.parametrize("verified", [False, True])
def test_country_partial_score_succeeds_only_when_card_is_verified(tmp_path, monkeypatch, verified):
    from hexcore import cli, country_door, observation_scorecard
    args = arguments(tmp_path)
    args.stage = "score"
    args.work_dir.mkdir()
    (args.work_dir / "mesh-rows.json").write_text('{}')
    monkeypatch.setattr(cli, "main", lambda argv: 3)
    calls = []
    def completed(run, config):
        calls.append(run)
        return {"returncode": 0, "status": "PARTIALLY_MEASURED", "note": "verified measured card"} if verified and len(calls) == 2 else None
    monkeypatch.setattr(observation_scorecard, "completed_scorecard", completed)
    monkeypatch.setattr(observation_scorecard, "join_automatic_scorer", lambda *args: None)
    assert country_door.run_country(args) == (0 if verified else 3)
    receipt = json.loads((args.work_dir / "country-receipt.json").read_text())
    if verified:
        assert receipt["stages"]["score"]["scorer_returncode"] == 3


def test_country_default_prepares_before_forecast_and_joins_one_scorer(tmp_path, monkeypatch):
    from hexcore import cli, country_door, mesh_rows, observation_scorecard
    args = arguments(tmp_path)
    args.grid, args.static, args.init = [tmp_path / name for name in ("grid.nc", "static.nc", "init.nc")]
    args.grid.touch()
    args.static.touch()
    stages = []
    monkeypatch.setattr(mesh_rows, "describe_generated", lambda **kw: kw)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: path.write_text('{}'))
    def door(argv):
        stages.append(argv[0])
        if argv[0] == "forecast":
            assert (args.work_dir / "mesh-rows.json").is_file()
            assert args.init.is_file()
        elif argv[0] == "init-source":
            args.init.touch()
        return 0
    monkeypatch.setattr(cli, "main", door)
    monkeypatch.setattr(observation_scorecard, "completed_scorecard", lambda *args: None)
    monkeypatch.setattr(observation_scorecard, "join_automatic_scorer", lambda *args: {"note": "joined once", "status": "PARTIALLY_MEASURED"})
    assert country_door.run_country(args) == 0
    assert stages == ["init-source", "forecast", "products"]


@pytest.mark.parametrize("override", [False, True])
def test_country_static_uses_helper_geography_default_or_explicit_override(tmp_path, monkeypatch, override):
    from hexcore import country_door, engines, mesh_rows, prep_store
    args = arguments(tmp_path)
    args.stage = "prepare"
    args.grid, args.static, args.init = [tmp_path / name for name in ("grid.nc", "static.nc", "init.nc")]
    args.grid.touch()
    args.init.touch()
    args.geog = tmp_path / "explicit-geography" if override else None
    calls = []
    monkeypatch.setattr(engines, "resolve", lambda spec: tmp_path / spec.name)
    monkeypatch.setattr(prep_store, "run_or_fetch", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(mesh_rows, "describe_generated", lambda **kw: kw)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: path.write_text('{}'))
    assert country_door.run_country(args) == 0
    assert len(calls) == 1
    static = calls[0]
    assert static[0].endswith("rw_mpas_static")
    if override:
        assert static[static.index("--geog") + 1] == str(args.geog)
    else:
        # Omit the flag so the native helper keeps its root lookup and its
        # fetch-geog diagnostic, instead of a second Python lookup ladder.
        assert "--geog" not in static


def test_country_missing_geography_keeps_fetch_command_and_refused_receipt(tmp_path, monkeypatch):
    import os
    import stat
    from hexcore import country_door, engines, prep_store
    if os.name == "nt":
        pytest.skip("shell stand-in")
    args = arguments(tmp_path)
    args.stage = "prepare"
    args.grid = tmp_path / "grid.nc"
    args.grid.touch()
    helper = tmp_path / "rw_mpas_static"
    helper.write_text("#!/bin/sh\necho 'the geography archive is missing required dataset(s)' >&2\n"
                      "echo 'run gpuwm fetch-geog --datasets mesh' >&2\nexit 2\n")
    helper.chmod(helper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(engines, "resolve", lambda spec: helper)
    monkeypatch.setattr(prep_store, "run_or_fetch", lambda argv, runner: runner(argv))
    with pytest.raises(MpasPortError, match="gpuwm fetch-geog --datasets mesh"):
        country_door.run_country(args)
    assert "geography archive is missing" in (args.work_dir / "static.log").read_text()
    receipt = json.loads((args.work_dir / "country-receipt.json").read_text())
    assert "gpuwm fetch-geog --datasets mesh" in receipt["refused"]
    assert json.loads((args.work_dir / "status.json").read_text())["state"] == "refused"
