"""Graceful recovery in the global ensemble door (WOOF 2026-10-26, run_2de68ba40c93de79): a member stage that fails is
retried once, a member that still fails is recorded missing with its own reason, and the ensemble goes on.

CPU only: the door commands are answered by stand-ins."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hexcore import global_door, global_ensemble
from hexcore.global_ensemble import EnsembleDoorError

from test_global_ensemble import _Status, _plan


@pytest.fixture(autouse=True)
def fresh_plan(monkeypatch):
    monkeypatch.setattr(global_ensemble, "_FAULT_PLAN", None)
    monkeypatch.delenv(global_ensemble.FAULT_ENV, raising=False)


def _saved(tmp_path, value):
    saved = []
    for m in value["members"]:
        forecast = tmp_path / "ensemble/g01" / m["run_id"]
        forecast.mkdir(parents=True)
        saved.append({**m, "forecast": str(forecast), "group": "g01", "card": "0", "forecast_returncode": 0})
    (tmp_path / "ensemble.json").write_text(json.dumps({"source": value["source"], "cycle": value["cycle"],
                                                         "row": value["row"]["id"], "members": saved}))
    return saved


def test_a_member_whose_products_fail_once_is_retried_and_recovers(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source="gefs", members="1-3")
    _saved(tmp_path, value)
    calls = []

    def door(argv, env):
        calls.append(argv[1])
        if argv[1].endswith("mem002") and calls.count(argv[1]) == 1:
            print("rw_hex_points: refused: history q2 is not finite", flush=True)
            return 1
        return 0

    monkeypatch.setattr(global_door, "_run_door", door)
    record = global_ensemble.run_ensemble(value, args, ["products"], _Status(), {})
    assert [Path(c).name for c in calls] == ["mem001", "mem002", "mem002", "mem003"]
    assert all(m["products_returncode"] == 0 for m in record["members"])
    assert record["missing"] == []
    [retry] = record["recovery"]
    assert retry["run_id"] == "mem002" and retry["recovered"] and "q2 is not finite" in retry["reason"]


def test_a_member_whose_products_always_fail_is_missing_and_the_rest_finish(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source="gefs", members="1-3")
    _saved(tmp_path, value)

    def door(argv, env):
        if argv[0] == "products" and argv[1].endswith("mem001"):
            print("rw_hex_points: refused: history q2 is not finite", flush=True)
            return 1
        return 0

    monkeypatch.setattr(global_door, "_run_door", door)
    record = global_ensemble.run_ensemble(value, args, ["products", "export"], _Status(), {})
    by = {m["run_id"]: m for m in record["members"]}
    assert by["mem001"]["products_returncode"] == 1 and by["mem001"]["export_returncode"] == 0
    assert by["mem002"]["products_returncode"] == by["mem003"]["products_returncode"] == 0
    [lost] = record["missing"]
    assert lost["run_id"] == "mem001" and lost["stage"] == "products" and "q2 is not finite" in lost["reason"]
    assert json.loads((tmp_path / "ensemble.json").read_text())["missing"][0]["run_id"] == "mem001"


def test_every_member_failing_still_refuses(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source="gefs", members="1-2")
    _saved(tmp_path, value)
    monkeypatch.setattr(global_door, "_run_door", lambda argv, env: 1)
    with pytest.raises(EnsembleDoorError, match="products failed for every member"):
        global_ensemble.run_ensemble(value, args, ["products"], _Status(), {})


def test_the_fault_plan_fails_a_named_member_stage(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source="gefs", members="1-3")
    _saved(tmp_path, value)
    monkeypatch.setenv(global_ensemble.FAULT_ENV, json.dumps(
        [{"unit": "products", "member": 1, "hour": 15, "times": 9, "kind": "nonfinite"}]))
    monkeypatch.setattr(global_door, "_run_door", lambda argv, env: 0)
    record = global_ensemble.run_ensemble(value, args, ["products"], _Status(), {})
    [lost] = record["missing"]
    assert lost["run_id"] == "mem001" and lost["reason"] == "FAULT INJECTION: history q2 is not finite at f015"


def test_a_failed_member_forecast_is_retried_alone_with_its_own_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    value, args = _plan(tmp_path, source="era5", members="1-4", members_per_card=4, devices="0")
    for m in value["members"]:
        m["init_returncode"] = 0
    launched = []
    monkeypatch.setenv(global_ensemble.FAULT_ENV, json.dumps([{"unit": "forecast", "member": 3, "times": 1,
                                                               "kind": "nonfinite", "hour": 3}]))

    def cli(argv, log, env):
        roster = json.loads(Path(argv[argv.index("--members") + 1]).read_text())
        launched.append([(m["run_id"], m["stochastic"]["member"], m["stochastic"].get("ensemble_size")) for m in roster])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ok")
        return 0

    monkeypatch.setattr(global_ensemble, "_cli", cli)
    record = global_ensemble.run_ensemble(value, args, ["forecast"], _Status(), {})
    # the first group (all four) failed on the injected member before it ran; each was retried alone
    assert sorted(x[0][0] for x in launched) == ["mem001", "mem002", "mem003", "mem004"]
    assert all(len(x) == 1 for x in launched)
    assert record["missing"] == [] and all(m["forecast_returncode"] == 0 for m in record["members"])
    assert all(Path(m["forecast"]).parent.name.startswith("r1g") for m in record["members"])
    assert {r["run_id"] for r in record["recovery"]} == {"mem001", "mem002", "mem003", "mem004"}
    assert all("not finite at f003" in r["reason"] for r in record["recovery"])


def test_a_member_forecast_that_fails_again_is_missing_and_the_ensemble_goes_on(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    value, args = _plan(tmp_path, source="era5", members="1-3", members_per_card=1, devices="0")
    for m in value["members"]:
        m["init_returncode"] = 0

    def cli(argv, log, env):
        roster = json.loads(Path(argv[argv.index("--members") + 1]).read_text())
        log.parent.mkdir(parents=True, exist_ok=True)
        if roster[0]["run_id"] == "mem002":
            log.write_text("step 412: FloatingPointError: non-finite theta\n")
            return 1
        log.write_text("ok")
        return 0

    calls = []
    monkeypatch.setattr(global_ensemble, "_cli", cli)
    monkeypatch.setattr(global_door, "_run_door", lambda argv, env: calls.append(argv) or 0)
    record = global_ensemble.run_ensemble(value, args, ["forecast", "products"], _Status(), {})
    [lost] = record["missing"]
    assert lost["run_id"] == "mem002" and lost["stage"] == "forecast" and "non-finite theta" in lost["reason"]
    assert sorted(Path(a[1]).name for a in calls) == ["mem001", "mem003"]


def test_a_failed_init_is_retried_then_its_members_are_missing(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source="gefs", members="1-3")
    tries = {}

    def cli(argv, log, env):
        member = argv[argv.index("--member") + 1]
        tries[member] = tries.get(member, 0) + 1
        log.parent.mkdir(parents=True, exist_ok=True)
        if member == "2":
            log.write_text("init-source: HTTP 500 from the archive\n")
            return 1
        if member == "3" and tries[member] == 1:
            log.write_text("Connection reset by peer\n")
            return 1
        log.write_text("ok")
        return 0

    monkeypatch.setattr(global_ensemble, "_cli", cli)
    result = global_ensemble.run_inits(value, value["members"], args, _Status(), {})
    assert tries == {"1": 1, "2": 2, "3": 2}
    assert result["missing"] == ["mem002"]
    assert value["members"][1]["missing"]["reason"] == "init-source: HTTP 500 from the archive"
    assert [r["recovered"] for r in sorted(result["retried"], key=lambda r: r["run_id"])] == [False, True]
