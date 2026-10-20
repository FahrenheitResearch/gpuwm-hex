"""The global ensemble: member inits, resident groups on the card, splits on refusal.

CPU only: the door commands are answered by a stand-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hexcore import device_admission, global_door, global_ensemble
from hexcore.global_ensemble import EnsembleDoorError, parse_members


def test_member_specs():
    assert parse_members("1-3") == [1, 2, 3]
    assert parse_members("0,5,7-8") == [0, 5, 7, 8]
    for bad in ("3-1", "1,1", ""):
        with pytest.raises(EnsembleDoorError):
            parse_members(bad)


def test_mutex_uuid_visibility_retains_four_member_admission(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(global_door.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="0, GPU-card-zero, 97887\n1, GPU-card-one, 97887\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-card-zero")
    assert global_door._visible_cards() == [97887 * 1024**2]
    value = {"mesh": {"cells": 40962}}
    assert global_ensemble.group_limit(value, SimpleNamespace(members_per_card="auto")) == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-card")
    assert global_door._visible_cards() == []  # Ambiguous UUID prefixes are not a device.


def test_numeric_and_empty_visibility_masks(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(global_door.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="0, GPU-zero, 97887\n1, GPU-one, 32768\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert global_door._visible_cards() == [32768 * 1024**2, 97887 * 1024**2]
    for mask in ("", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert global_door._visible_cards() == []


def test_global_meshes_have_co_tenancy_rows_and_the_floor_sum_still_binds():
    gib = 1024**3
    got = device_admission.admit_co_tenancy(mesh_kind="global", cells=40_962, levels=55, members=2,
                                            card_bytes=int(31.8 * gib), free_bytes=30 * gib,
                                            solo_required_bytes=12 * gib)
    # Since fix-admission-render (bbaca17f) the reported maximum is also
    # limited by measured free memory: 30 GiB // 12 GiB = 2. The table row
    # still allows 3, and three members still refuse on the floor sum.
    assert got["maximum_members"] == 2
    assert got["table_maximum_members"] == 3
    assert got["memory_limited_members"] == 2
    with pytest.raises(MemoryError, match="co-tenancy refuses 3"):
        device_admission.admit_co_tenancy(mesh_kind="global", cells=40_962, levels=55, members=3,
                                          card_bytes=int(31.8 * gib), free_bytes=30 * gib,
                                          solo_required_bytes=12 * gib)
    with pytest.raises(MemoryError, match="table limit 1"):
        device_admission.admit_co_tenancy(mesh_kind="global", cells=2_621_442, levels=55, members=2,
                                          card_bytes=96 * gib, free_bytes=90 * gib, solo_required_bytes=gib)


def _plan(tmp_path, **extra):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    global_door.add_global_parser(commands)
    argv = ["global", "--row", "global-120km", "--cycle", "2026-10-16T00", "--work-dir", str(tmp_path)]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return global_door.plan(parser.parse_args(argv)), parser.parse_args(argv)


def test_each_member_gets_its_own_init(tmp_path):
    value, _ = _plan(tmp_path, source="gefs", members="1-3")
    assert [m["run_id"] for m in value["members"]] == ["mem001", "mem002", "mem003"]
    for row in value["members"]:
        argv = row["init_argv"]
        assert argv[argv.index("--member") + 1] == str(row["member"])
        assert argv[argv.index("--out-dir") + 1] == row["init_dir"]
        assert row["init_dir"].startswith(str(tmp_path / "members"))


def test_one_member_and_a_one_state_source(tmp_path):
    value, _ = _plan(tmp_path, source="gefs", member=7)
    assert value["commands"]["init"][-2:] == ["--member", "7"]
    assert "member 7" in " ".join(value["commands"]["forecast"])
    with pytest.raises(global_door.GlobalDoorError, match="publishes one state"):
        _plan(tmp_path, members="1-3")


def test_stochastic_member_defaults_are_decided_before_card_grouping(tmp_path):
    small, _ = _plan(tmp_path, source='gefs', members='1-3', stochastic_seed=71)
    large, _ = _plan(tmp_path, source='gefs', members='1-10')
    assert all(m['stochastic']['sppt'] and m['stochastic']['skeb'] for m in small['members'])
    assert [m['stochastic']['member'] for m in small['members']] == [1,2,3]
    assert all(m['stochastic']['seed'] == 71 for m in small['members'])
    assert all(not m['stochastic']['sppt'] for m in large['members'])
    single, _ = _plan(tmp_path, source='gefs', member=1, stochastic='{"sppt":true,"seed":71}')
    command = single['commands']['forecast']
    assert json.loads(command[command.index('--stochastic')+1]) == {'sppt':True,'seed':71}


def test_a_group_runs_resident_through_the_roster_on_one_card(tmp_path):
    value, _ = _plan(tmp_path, source="gefs", members="1-2", devices="0")
    roster = tmp_path / "g01.roster.json"
    argv = global_ensemble.forecast_argv(value, roster, tmp_path / "out", tmp_path / "scr", "/x/init.nc")
    assert argv[-2:] == ["--members", str(roster)] and "--samehost-devices" not in argv
    assert argv[argv.index("--init") + 1] == "/x/init.nc"


class _Status:
    def __init__(self):
        self.notes = []

    def stage(self, *a, **k):
        pass

    def step(self, *a, note=None, **k):
        self.notes.append(note)

    def note(self, text):
        self.notes.append(text)


def test_a_group_the_card_refuses_is_split_and_every_member_runs(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source="gefs", members="1-3", members_per_card=3, devices="0")
    members = value["members"]
    calls = []

    def fake(argv, log, env):
        roster = json.loads(Path(argv[argv.index("--members") + 1]).read_text())
        calls.append([m["run_id"] for m in roster])
        log.parent.mkdir(parents=True, exist_ok=True)
        if len(roster) == 3:
            log.write_text("MemoryError: co-tenancy refuses 3 members: table limit 3 ...")
            return 1
        log.write_text("ok")
        return 0

    monkeypatch.setattr(global_ensemble, "_cli", fake)
    status = _Status()
    result = global_ensemble.run_forecasts(value, members, args, status, {})
    assert calls[0] == ["mem001", "mem002", "mem003"] and sorted(sum(calls[1:], [])) == ["mem001", "mem002", "mem003"]
    assert result["failed"] == [] and any(g.get("split") for g in result["groups"])
    assert all(m["forecast"].endswith(m["run_id"]) for m in members)


def test_init_jobs_auto_is_bounded_by_cores_and_host_ram(monkeypatch):
    """--init-jobs auto (unified image, 2026-10-16: a fixed 4 left members waiting on a 24-core box)."""
    value = {"mesh": {"cells": 40_962}}
    monkeypatch.setattr(global_ensemble.os, "sched_getaffinity", lambda pid: set(range(24)))
    monkeypatch.setattr(global_ensemble, "_available_host_bytes", lambda: 100 * 1024**3)
    assert global_ensemble.init_jobs("auto", value, 6) == 6
    assert global_ensemble.init_jobs("auto", value, 30) == 6
    assert global_ensemble.init_jobs("2", value, 6) == 2
    per_init = 40_962 * global_ensemble.INIT_BYTES_PER_CELL
    monkeypatch.setattr(global_ensemble, "_available_host_bytes", lambda: 2 * per_init + 1)
    assert global_ensemble.init_jobs("auto", value, 6) == 2
    monkeypatch.setattr(global_ensemble, "_available_host_bytes", lambda: 1)
    assert global_ensemble.init_jobs("auto", value, 6) == 1
    with pytest.raises(EnsembleDoorError):
        global_ensemble.init_jobs("0", value, 6)


def test_resumed_member_products_use_saved_forecast_outputs(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source='gefs', members='1-2')
    saved = []
    for m in value['members']:
        forecast = tmp_path / 'ensemble/g03' / m['run_id']
        forecast.mkdir(parents=True)
        saved.append({**m, 'forecast': str(forecast), 'group': 'g03', 'card': '0', 'forecast_returncode': 0})
    (tmp_path / 'ensemble.json').write_text(json.dumps({'source': 'gefs', 'cycle': value['cycle'],
        'row': value['row']['id'], 'members': saved}))
    calls = []
    monkeypatch.setattr(global_door, '_run_door', lambda argv, env: calls.append(argv) or 0)
    record = global_ensemble.run_ensemble(value, args, ['products', 'export'], _Status(), {})
    assert [a[1] for a in calls] == [m['forecast'] for m in saved] * 2
    assert all('/ensemble/g03/' in a[1] for a in calls)
    assert all(m['products_returncode'] == 0 for m in record['members'])


def test_member_postprocessing_refuses_foreign_manifest(tmp_path):
    value, args = _plan(tmp_path, source='gefs', members='1-2')
    (tmp_path / 'ensemble.json').write_text(json.dumps({'source': 'other', 'cycle': value['cycle'],
        'row': value['row']['id'], 'members': []}))
    with pytest.raises(EnsembleDoorError, match='different row, source or cycle'):
        global_ensemble.run_ensemble(value, args, ['products'], _Status(), {})


@pytest.mark.parametrize('scorer_rc,verified,expected_rc', [(3, True, 0), (3, False, 3), (2, True, 2)])
def test_member_scores_only_complete_on_verified_measurements(tmp_path, monkeypatch, scorer_rc, verified, expected_rc):
    from hexcore import observation_scorecard

    value, args = _plan(tmp_path, source='gefs', members='1-2')
    saved = []
    for member in value['members']:
        forecast = tmp_path / 'ensemble/g01' / member['run_id']
        forecast.mkdir(parents=True)
        Path(member['init_dir']).mkdir(parents=True)
        saved.append({**member, 'forecast': str(forecast), 'forecast_returncode': 0})
    (tmp_path / 'ensemble.json').write_text(json.dumps({'source': value['source'], 'cycle': value['cycle'],
        'row': value['row']['id'], 'members': saved}))
    checked = []

    def completed(forecast, config):
        checked.append((forecast, json.loads(config.read_text())))
        return {'status': 'PARTIALLY_MEASURED', 'measured_metrics': 7, 'note': 'measured partial coverage'} if verified else None

    monkeypatch.setattr(global_door, '_run_door', lambda argv, env: scorer_rc)
    monkeypatch.setattr(observation_scorecard, 'completed_scorecard', completed)
    if expected_rc:
        with pytest.raises(EnsembleDoorError, match=f'score of mem001 exited {expected_rc}'):
            global_ensemble.run_ensemble(value, args, ['score'], _Status(), {})
    else:
        record = global_ensemble.run_ensemble(value, args, ['score'], _Status(), {})
        assert all(m['score_returncode'] == 0 and m['scorer_returncode'] == 3 for m in record['members'])
        assert all(m['measured_metrics'] == 7 for m in record['members'])
    assert len(checked) == (2 if expected_rc == 0 else 1 if scorer_rc == 3 else 0)
    for (forecast, config), member in zip(checked, saved):
        assert forecast == member['forecast'] and config['init'] == member['init']
