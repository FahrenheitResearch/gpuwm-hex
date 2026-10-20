"""Public S2S plan, true lag clocks and interrupted-member recovery."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hexcore import s2s_door
from hexcore.cli import build_parser
from hexcore.forecast_door import _schedule


def arguments(tmp_path, *extra):
    return build_parser().parse_args(["s2s", "--row", "global-120km", "--source", "gfs",
        "--cycle", "2026-10-16T00", "--members", "4", "--days", "30", "--work-dir", str(tmp_path), *extra])


def test_dry_run_public_door_has_no_side_effects(tmp_path):
    destination = tmp_path / "not-created"
    argv = [sys.executable, "-m", "hexcore.cli", "s2s", "--row", "global-120km", "--source", "gfs",
            "--cycle", "2026-10-16T00", "--members", "4", "--days", "30", "--work-dir", str(destination), "--dry-run"]
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert not destination.exists()
    assert len(plan["members"]) == 4
    assert [m["lag_hours"] for m in plan["members"]] == [0, 6, 12, 18]
    last = plan["members"][-1]
    assert last["start_time"] == last["valid_time"] == "2026-10-16_00:00:00"
    assert last["source_cycle"] == "2026-10-15_06:00:00" and last["lead_hours"] == 18
    assert last["init_argv"][-2:] == ["--lead-hours", "18"]
    assert last["hours"] == 720
    assert last["history_from_minutes"] == 0
    assert last["checkpoint_hours"] == "120,240,360,480,600,720"
    assert plan["surface_updating"] and plan["stochastic"]["sppt"] and plan["stochastic"]["skeb"]
    assert plan["surface"]["source"] == "cfs"
    assert plan["surface"]["forecast_cycle"] == "2026-10-14T00"
    assert plan["surface"]["end"] == "2026-11-15_00:00:00"
    assert Path(plan["surface"]["path"]) == destination / "surface_series.nc"
    assert set(plan["parallelism"]) == set(s2s_door.STAGES)


def test_lagged_daily_history_has_common_valid_times():
    assert _schedule(738, 1440, 600, "global", 18 * 60) == (4428, 31)
    with pytest.raises(Exception, match="does not divide"):
        _schedule(737, 1440, 600, "global", 18 * 60)
    with pytest.raises(Exception, match="whole step"):
        _schedule(738, 1440, 600, "global", 0.5)


def test_bad_days_and_lags_refuse(tmp_path):
    args = arguments(tmp_path, "--days", "29")
    with pytest.raises(s2s_door.S2SError, match="30..90"):
        s2s_door.plan(args)
    for bad in ("-6", "1", "0,0", "junk"):
        with pytest.raises(s2s_door.S2SError):
            s2s_door.lag_hours(bad, 4, "gfs")


def test_checkpoint_search_retains_member_identity(tmp_path):
    a = tmp_path / "a1" / "mem001"
    b = tmp_path / "a2" / "mem001"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    first = a / "checkpoint.2026-10-21_00.00.00.pkl"
    latest = b / "checkpoint.2026-10-26_00.00.00.pkl"
    first.write_bytes(b"checkpoint")
    latest.write_bytes(b"checkpoint")
    (b / "checkpoint.2026-10-31_00.00.00.pkl.part").write_bytes(b"incomplete")
    attempts = [{"out": str(a.parent)}, {"out": str(b.parent)}]
    assert s2s_door.latest_checkpoint({"run_id": "mem001"}, attempts) == latest
    assert s2s_door.latest_checkpoint({"run_id": "mem002"}, attempts) is None


def test_signature_allows_different_card_placement(tmp_path):
    a = s2s_door.plan(arguments(tmp_path, "--devices", "0"))
    b = s2s_door.plan(arguments(tmp_path, "--devices", "0,1"))
    assert a["signature"] == b["signature"]


def test_interruption_after_final_checkpoint_restarts_before_end(tmp_path):
    root = tmp_path / "attempt" / "mem001"
    root.mkdir(parents=True)
    earlier = root / "checkpoint.2026-10-26_00.00.00.pkl"
    earlier.write_bytes(b"day-25")
    (root / "checkpoint.2026-10-31_00.00.00.pkl").write_bytes(b"day-30")
    member = {"run_id": "mem001", "start_time": "2026-10-01_00:00:00", "hours": 720}
    assert s2s_door.latest_checkpoint(member, [{"out": str(root.parent)}]) == earlier


def test_native_members_balanced_across_lags(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--source", "gefs", "--lagged", "0,24"))
    assert [(m["member"], m["lag_hours"]) for m in value["members"]] == [(1, 0), (1, 24), (2, 0), (2, 24)]


def test_lagged_source_forecasts_reach_one_clock_and_manifest(tmp_path):
    from datetime import datetime, timedelta
    value = s2s_door.plan(arguments(tmp_path, "--lagged", "0,6,12,18"))
    common = datetime(2026, 10, 16)
    s2s_door.update_manifests(value, {"attempts": [], "completed": []})
    for member in value["members"]:
        source_time = datetime.strptime(member["source_cycle"], "%Y-%m-%d_%H:%M:%S")
        assert source_time + timedelta(hours=member["lead_hours"]) == common
        assert member["start_time"] == common.strftime("%Y-%m-%d_%H:%M:%S")
        manifest = json.loads(Path(member["manifest"]).read_text())
        assert manifest["provenance"] == member["provenance"]
        assert manifest["source_member"] is None
        assert manifest["valid_time"] == member["valid_time"]


def test_analysis_only_source_defaults_to_common_init_with_stochastic_fill(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--source", "era5"))
    assert all(m["lag_hours"] == m["lead_hours"] == 0 for m in value["members"])
    assert len({m["init"] for m in value["members"]}) == 1
    assert all(m["stochastic"]["sppt"] and m["stochastic"]["skeb"] for m in value["members"])


def test_stochastic_policy_is_resolved_before_card_groups(tmp_path):
    four = s2s_door.plan(arguments(tmp_path))
    assert all(m["stochastic"]["sppt"] and m["stochastic"]["skeb"] for m in four["members"])
    assert len({m["stochastic"]["member"] for m in four["members"]}) == 4
    assert len({m["stochastic"]["seed"] for m in four["members"]}) == 4
    thirty = s2s_door.plan(arguments(tmp_path, "--source", "gefs", "--members", "30"))
    assert not thirty["stochastic"]["sppt"] and not thirty["stochastic"]["skeb"]
    assert s2s_door.science_arguments(four)[0] == "--stochastic"
    s2s_door.update_manifests(four, {"attempts": [], "completed": []})
    manifest = json.loads(Path(four["members"][0]["manifest"]).read_text())
    assert manifest["stochastic"] == json.loads(json.dumps(four["members"][0]["stochastic"]))


def test_explicit_stochastic_controls_reach_each_member(tmp_path):
    value = s2s_door.plan(arguments(tmp_path, "--stochastic", '{"amplitude":0.25,"skeb":false}'))
    assert all(m["stochastic"]["amplitude"] == 0.25 and not m["stochastic"]["skeb"]
               for m in value["members"])
    with pytest.raises(s2s_door.S2SError, match="roster"):
        s2s_door.plan(arguments(tmp_path, "--stochastic", '{"member":7}'))


def test_resident_members_discover_selected_surface_series(tmp_path, monkeypatch):
    from hexcore.surface_series_reader import discover
    source = tmp_path / "prepared.nc"
    source.write_bytes(b"native surface series")
    output = tmp_path / "attempts" / "a0001"
    for key, value in s2s_door.surface_environment({"surface": {"path": str(source)}}).items():
        monkeypatch.setenv(key, value)
    assert discover(output / "mem001").resolve() == source
    assert not output.exists()
    monkeypatch.setenv("GPUWM_HEX_SURFACE_SERIES", str(tmp_path / "missing.nc"))
    with pytest.raises(FileNotFoundError, match="selected surface series"):
        discover(output / "mem001")


def test_surface_restart_uses_fresh_destination_and_reuses_complete_receipt(tmp_path, monkeypatch):
    from hexcore import global_ensemble
    args = arguments(tmp_path)
    value = s2s_door.plan(args)
    assert Path(value["surface"]["path"]).name == "surface_series.nc"
    legacy = tmp_path / "surface-series.nc"
    legacy.write_bytes(b"existing series")
    assert s2s_door.plan(args)["surface"]["path"] == str(legacy)
    explicit = tmp_path / "selected.nc"
    assert s2s_door.plan(arguments(tmp_path, "--surface-series", str(explicit)))["surface"]["path"] == str(explicit)
    commands = []
    def producer(argv, log, env):
        commands.append(argv)
        if len(commands) == 1:
            return 1
        Path(argv[argv.index("--out") + 1]).write_bytes(b"native-series")
        s2s_door.write_json(Path(argv[argv.index("--receipt") + 1]), {"status": "complete"})
        return 0
    monkeypatch.setattr(global_ensemble, "_cli", producer)
    with pytest.raises(s2s_door.S2SError, match="surface-series exited 1"):
        s2s_door.surface_stage(value, args, {})
    result = s2s_door.surface_stage(value, args, {})
    assert not result["reused"]
    assert commands[0][commands[0].index("--out")+1] != commands[1][commands[1].index("--out")+1]
    assert "--forecast-cycle" in commands[1]
    assert s2s_door.surface_stage(value, args, {})["reused"]
    assert len(commands) == 2


def test_resume_reuses_init_with_source_lane_clock_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hexcore import global_ensemble
    args = arguments(tmp_path, "--members", "1", "--lagged", "0")
    value = s2s_door.plan(args)
    member = value["members"][0]
    init = Path(member["init"])
    init.parent.mkdir(parents=True)
    init.write_bytes(b"prepared")
    s2s_door.write_json(init.parent / "init-source-receipt.json", {
        "init": {"path": str(init)}, "source": "gfs", "member": None,
        "cycle": member["start_time"], "lead_hours": 0, "valid_time": member["start_time"]})
    def unexpected(*args, **kwargs):
        raise AssertionError("a matching init must not be rebuilt during resume")
    monkeypatch.setattr(global_ensemble, "_cli", unexpected)
    status = SimpleNamespace(stage=lambda *args, **kwargs: None, step=lambda *args, **kwargs: None)
    result = s2s_door._inits(value, args, {}, status)
    assert result["built"] == 0 and result["reused"] == 1


def test_prepared_lagged_init_checks_source_lead_and_common_valid_time(tmp_path):
    from types import SimpleNamespace
    init = tmp_path / "shared/init.nc"
    init.parent.mkdir()
    init.write_bytes(b"prepared-source-forecast")
    args = arguments(tmp_path / "run", "--members", "1", "--lagged", "6",
                     "--prepared-init", f"2026-10-15T18={init}")
    value = s2s_door.plan(args)
    receipt = {"source": "gfs", "member": None, "cycle": "2026-10-15_18:00:00",
               "source_cycle": "2026-10-15_18:00:00", "lead_hours": 6,
               "valid_time": "2026-10-16_00:00:00"}
    path = init.parent / "init-source-receipt.json"
    status = SimpleNamespace(stage=lambda *args, **kwargs: None, step=lambda *args, **kwargs: None)
    s2s_door.write_json(path, receipt)
    assert s2s_door._inits(value, args, {}, status)["reused"] == 1
    for changed in ({"lead_hours": 0}, {"valid_time": receipt["source_cycle"]},
                    {"source_cycle": "2026-10-15_12:00:00"}):
        s2s_door.write_json(path, {**receipt, **changed})
        with pytest.raises(s2s_door.S2SError, match="mismatch"):
            s2s_door._inits(value, args, {}, status)


def test_successful_recovery_clears_the_persisted_previous_error(tmp_path, monkeypatch):
    from hexcore import global_door
    args = arguments(tmp_path, '--stage', 'mesh')
    value = s2s_door.plan(args)
    state_path = tmp_path / 's2s-state.json'
    s2s_door.write_json(state_path, {'schema': s2s_door.SCHEMA,
        'signature': value['signature'], 'attempts': [], 'completed': [], 'stages': {},
        'error': 'S2SError: previous interrupted attempt'})
    monkeypatch.setattr(global_door, 'stage_mesh', lambda *args: {'reused': True})
    assert s2s_door.run(args) == 0
    assert 'error' not in json.loads(state_path.read_text())
    assert 'error' not in json.loads((tmp_path/'s2s-receipt.json').read_text())


def test_latest_cycle_stays_pinned_when_resuming_after_a_new_cycle_posts(tmp_path, monkeypatch, capsys):
    from hexcore import global_cycle, global_door
    resolutions = []

    def resolve(*args, **kwargs):
        cycle = "2026-10-16T00" if not resolutions else "2026-10-16T06"
        resolutions.append(cycle)
        return {"cycle": cycle}

    calls = []

    def mesh(*args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("interrupted mesh stage")
        return {"reused": True}

    monkeypatch.setattr(global_cycle, "resolve_latest", resolve)
    monkeypatch.setattr(global_door, "stage_mesh", mesh)
    with pytest.raises(RuntimeError, match="interrupted mesh stage"):
        s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh"))
    original = json.loads((tmp_path / "s2s-plan.json").read_text())
    assert s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh")) == 0
    resumed = json.loads((tmp_path / "s2s-plan.json").read_text())
    assert original["cycle"] == resumed["cycle"] == "2026-10-16T00"
    assert original["signature"] == resumed["signature"]
    assert resolutions == ["2026-10-16T00"]
    assert "error" not in json.loads((tmp_path / "s2s-state.json").read_text())
    # Preview the restart without contacting the provider or changing state.
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    capsys.readouterr()
    assert s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--dry-run")) == 0
    assert json.loads(capsys.readouterr().out)["signature"] == original["signature"]
    assert files == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert resolutions == ["2026-10-16T00"]
    with pytest.raises(s2s_door.S2SError, match="different S2S configuration"):
        s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh", "--days", "45"))
    assert len(calls) == 2


def test_latest_cycle_is_not_resolved_before_work_directory_admission(tmp_path, monkeypatch):
    import fcntl
    from hexcore import global_cycle

    def unexpected(*args, **kwargs):
        raise AssertionError("a second command must not contact the source while the work directory is locked")

    monkeypatch.setattr(global_cycle, "resolve_latest", unexpected)
    with (tmp_path / "s2s.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(s2s_door.S2SError, match="another s2s command holds"):
            s2s_door.run(arguments(tmp_path, "--cycle", "latest", "--stage", "mesh"))
