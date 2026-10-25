"""The shapes 2-3 checker's defects (2026-10-23), each tested exactly.

1. add-nest on a LIVE run: the facts come from the run's start record / started stage.
2. the hold: a nest is stopped at its hold; a trigger's firings never pass its hold.
3. a trigger nest lands on a storm object, not between two storms.
4. two quick firings cannot both pass the card fit (atomic reservation).
"""

import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from hexcore import add_nest_door as door
from hexcore import nest_budget
from hexcore import nest_triggers as nt
from hexcore.card_ledger import CardLedger


# --------------------------------------------------------------------------- 1. live parent facts

def test_add_nest_reads_a_live_runs_facts_from_its_start_record(tmp_path):
    run = tmp_path / "hex"
    forecast = run / "forecast"
    forecast.mkdir(parents=True)
    # The mesh-plan receipt mid-run: the forecast stage has STARTED, not finished.
    (run / "global-vr-receipt.json").write_text(json.dumps({
        "status": "running", "active_stage": "forecast",
        "stages": [{"stage": "mesh", "argv": ["x"]}, {"stage": "init-source", "argv": ["y"]}]}))
    # The forecast driver's start record (written before step 1).
    (forecast / "forecast-start.json").write_text(json.dumps({
        "schema": "gpuwm-hex.forecast-start/1", "grid": "/w/hex/mesh.grid.nc", "static": "/w/hex/mesh.static.nc",
        "init": "/w/hex/init/init.nc", "start_time": "2025-05-19_12:00:00", "forecast_hours": 15.0,
        "history_every_minutes": 60.0, "history_from_minutes": 0.0}))
    parent = door.parent_run(run)
    assert parent["forecast"] == str(forecast)
    assert parent["mesh"] == "/w/hex/mesh.grid.nc" and parent["init"] == "/w/hex/init/init.nc"
    assert parent["simulation_start"] == "2025-05-19_12:00:00" and parent["hours"] == 15.0


def test_add_nest_reads_a_started_stage_argv_before_the_forecast_writes_anything(tmp_path):
    run = tmp_path / "hex"
    run.mkdir()
    argv = ["python", "-m", "hexcore.cli", "forecast", "--grid", "/w/g.nc", "--init", "/w/i.nc",
            "--start-time", "2025-05-19_12:00:00", "--hours", "15", "--history-every-minutes", "60",
            "--out", str(run / "forecast")]
    (run / "global-vr-receipt.json").write_text(json.dumps({
        "status": "running", "stages": [], "started_stages": {"forecast": {"argv": argv}}}))
    parent = door.parent_run(run)
    assert [parent[k] for k in ("forecast", "mesh", "init", "simulation_start", "hours")] == \
        [str(run / "forecast"), "/w/g.nc", "/w/i.nc", "2025-05-19_12:00:00", 15.0]


def test_the_refusal_names_the_real_flags(tmp_path):
    parser = door.argparse.ArgumentParser()
    sub = parser.add_subparsers()
    door.add_add_nest_parser(sub)
    args = parser.parse_args(["add-nest", str(tmp_path), "--lat", "35", "--lon", "-97", "--wait-minutes", "0"])
    with pytest.raises(door.AddNestRefusal) as refused:
        door.add_nest(args)
    text = str(refused.value)
    assert "--forecast-dir" in text and "--parent-hours" in text
    assert "--forecast " not in text and "--hours" not in text


def test_a_nest_given_at_launch_waits_for_the_forecast_to_start(tmp_path, monkeypatch):
    """Shape 1: the nest is given with the hex run, before mesh, static and init have built the forecast."""
    import threading

    run = tmp_path / "hex"
    run.mkdir()
    argv = ["python", "-m", "hexcore.cli", "forecast", "--grid", "/w/g.nc", "--init", "/w/i.nc",
            "--start-time", "2025-05-19_12:00:00", "--hours", "15", "--out", str(run / "forecast")]

    def start_later():
        import time
        time.sleep(1.5)
        (run / "global-vr-receipt.json").write_text(json.dumps({"started_stages": {"forecast": {"argv": argv}}}))

    class Reached(Exception):
        pass

    def stop_here(*_a, **_k):
        raise Reached
    monkeypatch.setattr(door, "choose_start", stop_here)
    parser = door.argparse.ArgumentParser()
    door.add_add_nest_parser(parser.add_subparsers())
    args = parser.parse_args(["add-nest", str(run), "--lat", "35", "--lon", "-97", "--wait-minutes", "1",
                              "--start", "2025-05-19_12:00:00", "--out", str(tmp_path / "nests")])
    threading.Thread(target=start_later).start()
    with pytest.raises(Reached):
        door.add_nest(args)


# --------------------------------------------------------------------------- 2. the hold

def test_a_nest_is_stopped_at_its_hold_and_charged_no_less_than_it_cost(tmp_path):
    # 1.0 s of box at USD 36/h is exactly 1 cent: a 1-cent budget is 1 s.
    budget = door.hold_budget_seconds(1.0, 36.0)
    assert abs(budget - 1.0) < 1e-9
    sim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    timeline = door.Timeline(tmp_path / "t.jsonl")
    clicked = time.time()
    stopped = door.watch_nest({"sim": sim}, clicked=clicked, budget_seconds=budget, timeline=timeline,
                              poll_seconds=0.05)
    assert stopped and sim.poll() is not None
    measured = door.measured_block(clicked, time.time(), 36.0, stopped)
    assert measured["stopped_at_hold"] and measured["wall_seconds"] < 30
    # Hold 125 cents, overrun 1.25 -> the nest is stopped at 100 cents of cost. Every
    # cost up to (and, if a stop lags, past) that point is charged at least its cost.
    for cost in (1.0, 50.0, 90.0, 100.0, 104.0, 130.0):
        charge = nest_budget.committed_cents(cost, budget_cents=125, quote_cents=60, overrun_factor=1.25)
        assert charge >= cost
        if cost <= 100.0:
            assert charge <= 125


def _hold(tmp_path, **over):
    hold = {"hold_cents": 2 * 152 + 31, "reserve_cents": 31, "per_firing_cents": 152,
            "per_firing_quote_cents": 76, "overrun_factor": 1.25, "usd_per_hour": 3.998}
    hold.update(over)
    (tmp_path / nest_budget.HOLD_NAME).write_text(json.dumps(hold))
    return hold


def test_a_triggers_firings_never_pass_its_hold(tmp_path):
    _hold(tmp_path)
    firings = []
    check = nest_budget.hold_left_hook(tmp_path, lambda: firings)
    for index in range(3):
        firing = {"index": index}
        ok, why = check(firing)
        if ok:
            firings.append(firing)
    assert len(firings) == 2, "two full per-firing holds fit; the third is refused"
    assert sum(f["budget_cents"] for f in firings) <= 2 * 152
    # nest-addon.ts nestStopCents: the hold over the overrun, less the 120 s upload tail at the hold's rate.
    assert all(abs(f["stop_after_cents"] - (152 / 1.25 - 120 / 3600 * 3.998 * 100)) < 1e-3 for f in firings)


def test_a_finished_firing_returns_its_unspent_hold(tmp_path):
    _hold(tmp_path, hold_cents=152 + 76 + 31)  # room for one full firing and a cheap one after it ends
    firings = [{"index": 0, "budget_cents": 152}]
    check = nest_budget.hold_left_hook(tmp_path, lambda: firings)
    assert not check({"index": 1})[0], "while firing 0 runs, its whole hold is committed"
    nest_budget.write_spent(tmp_path, 0, 76)  # it ended at its quote
    assert check({"index": 1})[0]


# --------------------------------------------------------------------------- 3. on a storm object

def _grid():
    lat = np.arange(30.0, 40.0, 0.1)
    lon = np.arange(-104.0, -92.0, 0.1)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ny, nx = la.shape
    idx = np.arange(la.size).reshape(ny, nx)
    coc = -np.ones((la.size, 6), int)
    for k, (dy, dx) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
        coc[:, k] = np.roll(idx, (-dy, -dx), axis=(0, 1)).ravel()
    area = np.full(la.size, (11.1e3) ** 2) * np.cos(np.radians(la.ravel()))
    return la.ravel(), lo.ravel(), coc, np.full(la.size, 4), area


def test_a_cover_trigger_lands_on_a_storm_and_the_second_firing_on_the_other_storm(tmp_path):
    la, lo, coc, ne, area = _grid()
    table = nt.resolve({"max_firings": 2, "max_live": 2, "sustain_looks": 1, "cooldown_s": 0,
                        "area": {"box": {"south": 30, "north": 40, "west": -104, "east": -92}},
                        "nest": {"dx_km": 3, "size_km": 600, "hours": 6},
                        "when": [{"all": [{"field": "refl", "op": ">", "value": 45, "cover_km2": 500}]}]})
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    trig = nt.LoopTrigger(table, run_dir=tmp_path, run_start="2025-05-19_12:00:00", dt_s=60.0,
                          geometry=geo, xp=np, available_sources=["refl10cm"])
    refl = np.zeros((2, la.size), np.float32)
    big = (np.abs(la - 36.0) < 0.35) & (np.abs(lo + 100.0) < 0.35)    # western Oklahoma
    small = (np.abs(la - 33.0) < 0.25) & (np.abs(lo + 95.0) < 0.25)   # northeast Texas, ~700 km away
    refl[1, big | small] = 55.0
    fired = []
    for step in (15, 30):
        trig.look(step, {"refl10cm": refl})
        firing = trig.poll()
        if firing:
            fired.append(firing)
    # The old centroid of every meeting cell sat between them, near (34.9, -98.2).
    assert len(fired) == 2
    assert abs(fired[0]["lat"] - 36.0) < 0.1 and abs(fired[0]["lon"] + 100.0) < 0.1
    assert abs(fired[1]["lat"] - 33.0) < 0.1 and abs(fired[1]["lon"] + 95.0) < 0.1


# --------------------------------------------------------------------------- 4. atomic card fit

def test_two_quick_firings_cannot_both_pass_the_card_fit(tmp_path):
    gib = 2**30

    def slow_free():
        time.sleep(0.2)  # both claims read free memory in the same window
        return 10 * gib

    ledger_a = CardLedger(tmp_path / "card.json", free_bytes=slow_free, used_bytes=lambda pids: 0,
                          alive=lambda pid: True)
    ledger_b = CardLedger(tmp_path / "card.json", free_bytes=slow_free, used_bytes=lambda pids: 0,
                          alive=lambda pid: True)
    verdicts = {}

    def claim(ledger, name):
        verdicts[name] = ledger.reserve(name, 6 * gib, [1])

    threads = [threading.Thread(target=claim, args=(ledger_a, "trigger-a")),
               threading.Thread(target=claim, args=(ledger_b, "trigger-b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(v["fits"] for v in verdicts.values()) == [False, True]


def test_the_loops_card_fit_hook_reserves_and_add_nest_adopts_it(tmp_path):
    gib = 2**30
    ledger = CardLedger(tmp_path / "card.json", free_bytes=lambda: 20 * gib, used_bytes=lambda pids: 0,
                        alive=lambda pid: True)
    (tmp_path / nest_budget.FIRE_BUDGET_NAME).write_text(json.dumps({"nest_bytes": 8 * gib}))
    fits = nest_budget.card_fit_hook(tmp_path, ledger, owner_pid=1)
    first, second, third = {"index": 0}, {"index": 1}, {"index": 2}
    assert fits(first)[0] and fits(second)[0]
    assert not fits(third)[0], "8.8 GiB + 8.8 GiB claimed; a third does not fit in 20 GiB"
    placed = door.place_on_card(ledger, "x", 8 * gib, forecast_dir=tmp_path,
                                reservation=first["card_reservation"])
    assert placed["adopted"]
    ledger.release(first["card_reservation"])
    assert fits(third)[0]


def test_the_measure_is_box_time_added_not_gpu_busy_and_not_time_spent_waiting_on_hex(tmp_path):
    # 1,000 s from the click: 300 s waiting on the parent's next frame (the engine's seam-wait log), the
    # rest busy beside hex or running past hex's end -- both are box time the nest adds.
    nest = tmp_path / "nest"
    nest.mkdir()
    rows = [{"event": "source_wait_started", "emitted_unix_ms": 1_100_000},
            {"event": "source_wait_finished", "waited_seconds": 200.0, "emitted_unix_ms": 1_300_000},
            {"event": "boundary_wait_started", "emitted_unix_ms": 1_900_000}]
    (nest / door.WAIT_LOG_NAME).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert door.nest_wait_seconds(nest, now=2000.0) == 200.0 + 100.0  # the open wait counts to now
    assert door.box_seconds(1000.0, 2000.0, nest) == 700.0
    measured = door.measured_block(1000.0, 2000.0, 36.0, False, nest_out=nest)
    assert measured["box_seconds"] == 700.0 and measured["cost_cents"] == 700.0  # USD 36/h is 1 cent a second


def test_the_geography_is_staged_with_the_engines_own_flag(tmp_path, monkeypatch):
    # The real test runs: `gpuwm fetch-geog --geog-root` is not a flag (it is --root), so every add stopped at
    # its domain step before this.
    seen = []

    def fake_run(argv, log):
        seen.append([str(a) for a in argv])
        if "domain" in seen[-1]:
            (tmp_path / "n.toml").write_text("[experiment]\nrestart_interval_s = 3600.0\n")
    monkeypatch.setattr(door, "_run", fake_run)
    door.nest_config("gpuwm", tmp_path, "n", lat=35.5, lon=-97.5, dx_km=3, size_km=600, hours=6,
                     cycle="2025-05-19T15", cadence_h=1.0, geog="/data/WPS_GEOG")
    fetch = next(a for a in seen if "fetch-geog" in a)
    assert fetch[fetch.index("--root") + 1] == "/data/WPS_GEOG" and "--geog-root" not in fetch


def test_a_firing_whose_nest_fails_frees_its_card_claim_and_its_hold(tmp_path, monkeypatch):
    run = tmp_path / "hexfc"
    run.mkdir()
    (run / "forecast-start.json").write_text(json.dumps({"grid": "g", "init": "i", "start_time": "2025-05-19_12:00:00",
                                                         "forecast_hours": 15.0}))
    _hold(run)
    ledger_path = tmp_path / "card.json"
    CardLedger(ledger_path, free_bytes=lambda: 50 * 2**30, used_bytes=lambda p: 0,
               alive=lambda p: True).reserve("trigger-x-0", 2**30, [1])
    firing = {"index": 0, "budget_cents": 152, "card_reservation": "trigger-x-0", "card_ledger": str(ledger_path)}
    parser = door.argparse.ArgumentParser()
    door.add_add_nest_parser(parser.add_subparsers())
    args = parser.parse_args(["add-nest", str(run), "--lat", "35", "--lon", "-97", "--name", "trigger-0",
                              "--usd-per-hour", "3.998", "--firing-json", json.dumps(firing)])

    def broken(*a, **k):
        (run / "nests" / "trigger-0").mkdir(parents=True)
        raise door.AddNestRefusal("fetch-geog failed")
    monkeypatch.setattr(door, "add_nest", broken)
    with pytest.raises(door.AddNestRefusal):
        door.run_add_nest(args)
    assert "trigger-x-0" not in json.loads(ledger_path.read_text())["reservations"].__repr__()
    assert nest_budget.read_spent(run)[0] <= 1, "charged its few seconds of cost, not its quote or hold"
    receipt = json.loads((run / "nests" / "trigger-0" / door.RECEIPT_NAME).read_text())
    assert receipt["measured"]["failed"]


def test_the_nest_forecast_is_started_with_the_engines_sim_command():
    # The test run: `gpuwm sim prepared DIR` is refused (exit 2, unrecognized arguments); the command is `gpuwm sim DIR`.
    import inspect
    source = inspect.getsource(door.add_nest)
    assert '"sim", prepared' in source and '"sim", "prepared"' not in source


def test_the_wait_log_is_read_from_the_sims_run_folder(tmp_path):
    """`gpuwm sim --outdir OUT` writes OUT/run-<stamp>/waits.jsonl (whole test run, 2026-10-23)."""
    nest = tmp_path / "nest"
    run = nest / "run-20261024-001308Z"
    run.mkdir(parents=True)
    rows = [{"event": "source_wait_started", "emitted_unix_ms": 1_100_000},
            {"event": "source_wait_finished", "waited_seconds": 80.0, "emitted_unix_ms": 1_180_000}]
    (run / door.WAIT_LOG_NAME).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert door.nest_wait_seconds(nest, now=2000.0) == 80.0
    assert door.box_seconds(1000.0, 2000.0, nest) == 920.0
