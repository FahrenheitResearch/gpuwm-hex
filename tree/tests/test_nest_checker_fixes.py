"""The shapes 1-3 checker's defects (the run of record, 2026-10-24), each tested exactly."""

import json
import math
import sys
import time
import types

import numpy as np
import pytest

from hexcore import add_nest_door as door
from hexcore import nest_budget
from hexcore import nest_triggers as nt


def _grid():
    lat = np.arange(30.0, 40.0, 0.05)
    lon = np.arange(-104.0, -92.0, 0.05)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ny, nx = la.shape
    idx = np.arange(la.size).reshape(ny, nx)
    coc = -np.ones((la.size, 6), int)
    for k, (dy, dx) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
        coc[:, k] = np.roll(idx, (-dy, -dx), axis=(0, 1)).ravel()
    area = np.full(la.size, (5.55e3) ** 2) * np.cos(np.radians(la.ravel()))
    return la.ravel(), lo.ravel(), coc, np.full(la.size, 4), area


def _trigger(run_dir, **over):
    raw = {"max_firings": 2, "sustain_looks": 1, "cooldown_s": 0,
           "area": {"box": {"south": 30, "north": 40, "west": -104, "east": -94}},
           "nest": {"dx_km": 3, "size_km": 600, "hours": 6},
           "when": [{"all": [{"field": "refl", "op": ">", "value": 45, "cover_km2": 500}]}]}
    raw.update(over)
    table = nt.resolve(raw)
    la, lo, coc, ne, area = _grid()
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    trig = nt.LoopTrigger(table, run_dir=run_dir, run_start="2025-05-19_12:00:00", dt_s=60.0,
                          geometry=geo, xp=np, available_sources=["refl10cm"])
    return trig, la, lo, area


def _okc_nest(run_dir):
    nt.note_live_nest(run_dir, {"event": "started", "name": "s1-okc", "lat": 35.5, "lon": -97.5, "size_km": 600,
                                "start_valid": "2025-05-19_12:00:00", "hours": 15, "firing": None})


def test_echo_inside_a_live_nest_does_not_count_toward_the_trigger(tmp_path):
    """20:45Z: 1,441 km2 inside shape 1 plus a 197 km2 Kansas echo fired a 500 km2 trigger."""
    _okc_nest(tmp_path)
    trig, la, lo, area = _trigger(tmp_path)
    refl = np.zeros((1, la.size), np.float32)
    inside = (np.abs(la - 35.6) < 0.18) & (np.abs(lo + 97.6) < 0.22)     # Oklahoma, inside shape 1
    kansas = (np.abs(la - 38.47) < 0.06) & (np.abs(lo + 97.57) < 0.08)   # small, outside it
    refl[0, inside | kansas] = 55.0
    assert area[inside].sum() / 1e6 > 1000 and area[kansas].sum() / 1e6 < 500
    trig.look(15, {"refl10cm": refl})
    assert trig.poll() is None, "the Kansas echo alone is below 500 km2"
    look = [json.loads(x) for x in (tmp_path / nt.LOOKS_NAME).read_text().splitlines()][-1]
    block = look["blocks"][0]
    assert not block["held"] and block["size_km2"] < 500 and block["all_km2"] > 1000


def test_a_storm_in_a_nests_corner_is_covered(tmp_path):
    """The footprint is the 600 km square: a corner storm (about 380 km from the centre) is inside it."""
    _okc_nest(tmp_path)
    trig, la, lo, area = _trigger(tmp_path)
    # About 250 km east and 250 km north of the centre: outside the 300 km circle, inside the square.
    corner = (np.abs(la - 37.75) < 0.15) & (np.abs(lo + 94.7) < 0.2)
    refl = np.zeros((1, la.size), np.float32)
    refl[0, corner] = 55.0
    assert area[corner].sum() / 1e6 > 500
    trig.look(15, {"refl10cm": refl})
    assert trig.poll() is None
    # The same storm with no nest there fires.
    bare = tmp_path / "bare"
    bare.mkdir()
    trig2, *_ = _trigger(bare)
    trig2.look(15, {"refl10cm": refl})
    assert trig2.poll() is not None


def test_a_fifth_storm_outside_the_nests_can_be_picked(tmp_path):
    """Covered storms take no object slot: four big storms inside the nest, a smaller one outside."""
    _okc_nest(tmp_path)
    trig, la, lo, area = _trigger(tmp_path)
    refl = np.zeros((1, la.size), np.float32)
    for dlat, dlon in ((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)):
        refl[0, (np.abs(la - 35.5 - dlat) < 0.3) & (np.abs(lo + 97.5 - dlon) < 0.3)] = 55.0
    small = (np.abs(la - 31.5) < 0.15) & (np.abs(lo + 101.5) < 0.15)
    refl[0, small] = 55.0
    trig.look(15, {"refl10cm": refl})
    firing = trig.poll()
    assert firing is not None and abs(firing["storm"]["lat"] - 31.5) < 0.1


def test_a_moving_storm_gets_a_nest_centred_on_its_track_and_stays_inside(tmp_path):
    """Trigger-0's storm grew to the south-east against the edges: the nest goes half its run ahead."""
    trig, la, lo, area = _trigger(tmp_path, sustain_looks=3)
    # 15 m/s towards the south-east, looks every 15 min: about 9.5 km south and 9.5 km east a look.
    step_km = 15.0 * 900 / 1000 / math.sqrt(2)
    firing = None
    for k in range(3):
        lat = 38.5 - k * step_km / 111.2
        lon = -97.6 + k * step_km / (111.2 * math.cos(math.radians(38.4)))
        refl = np.zeros((1, la.size), np.float32)
        refl[0, (np.abs(la - lat) < 0.15) & (np.abs(lo - lon) < 0.2)] = 55.0
        trig.look(15 * (k + 1), {"refl10cm": refl})
        firing = trig.poll() or firing
    assert firing is not None
    placement = firing["placement"]
    assert placement["track_looks"] == 3
    east, north = placement["shift_km"]
    assert east > 50 and north < -50, "the centre moves down the storm's track (south-east)"
    # The storm as it is now stays at least 15 % of the size inside every edge.
    se, sn = nt._offset_km(firing["lat"], firing["lon"], firing["storm"]["lat"], firing["storm"]["lon"])
    radius = math.sqrt(firing["object_km2"] / math.pi)
    assert max(abs(se), abs(sn)) + radius <= 300 - 0.15 * 600 + 1.0


def test_a_lone_look_centres_the_nest_on_the_storm():
    placed = nt.place_on_track({"lat": 38.0, "lon": -97.0, "km2": 600}, [], block=0, model_seconds=900.0,
                               size_km=600, hours=6)
    assert (placed["lat"], placed["lon"]) == (38.0, -97.0)


def test_a_retried_runs_open_wait_is_not_subtracted(tmp_path):
    """c369adcf summed every run folder's waits.jsonl: an older run's open wait counted to now."""
    nest = tmp_path / "nest"
    old, new = nest / "run-20261024-001308Z", nest / "run-20261024-002000Z"
    old.mkdir(parents=True)
    new.mkdir()
    (old / door.WAIT_LOG_NAME).write_text(json.dumps({"event": "boundary_wait_started",
                                                      "emitted_unix_ms": 1_000_000}) + "\n")
    (new / door.WAIT_LOG_NAME).write_text(
        json.dumps({"event": "source_wait_started", "emitted_unix_ms": 1_500_000}) + "\n"
        + json.dumps({"event": "source_wait_finished", "waited_seconds": 50.0, "emitted_unix_ms": 1_550_000}) + "\n")
    (nest / "latest-run.txt").write_text(old.name)  # still names the old run: the new one has drawn nothing yet
    assert door.current_run_folder(nest) == new
    assert door.nest_wait_seconds(nest, now=2000.0) == 50.0


def test_an_engine_without_a_peak_envelope_is_not_read_for_its_pool_number(monkeypatch, capsys, tmp_path):
    fake = types.SimpleNamespace(alloc_estimate_bytes=2_000_000_000)
    monkeypatch.setitem(sys.modules, "gpuwm.core.preflight",
                        types.SimpleNamespace(estimate_experiment=lambda e: fake))
    monkeypatch.setitem(sys.modules, "gpuwm.domain_wizard",
                        types.SimpleNamespace(experiment_from_text=lambda t, source: None))
    config = tmp_path / "n.toml"
    config.write_text("")
    assert door.nest_estimate_bytes(config) is None
    assert "no peak_envelope_bytes" in capsys.readouterr().out


def _args(run, *extra):
    parser = door.argparse.ArgumentParser()
    door.add_add_nest_parser(parser.add_subparsers())
    return parser.parse_args(["add-nest", str(run), *extra])


def _run(tmp_path):
    run = tmp_path / "hexfc"
    run.mkdir()
    (run / "forecast-start.json").write_text(json.dumps({"grid": "g", "init": "i", "start_time": "2025-05-19_12:00:00",
                                                         "forecast_hours": 15.0}))
    return run


def test_a_nest_that_dies_after_its_first_map_is_failed_and_charged_machine_time_capped_at_its_quote(tmp_path):
    """Shapes 1 and 2 exited 2 and 1 at their last hours and were reported done at the full quote."""
    run = _run(tmp_path)
    (run / nest_budget.HOLD_NAME).write_text(json.dumps({
        "hold_cents": 469, "reserve_cents": 37, "per_firing_cents": 216, "per_firing_quote_cents": 108,
        "overrun_factor": 1.25, "usd_per_hour": 3.998}))
    out = tmp_path / "nests" / "trigger-0"
    (out / "nest").mkdir(parents=True)
    receipt = {}
    ledger = door.CardLedger(tmp_path / "card.json", free_bytes=lambda: 2**34, used_bytes=lambda p: 0,
                             alive=lambda p: True)
    door._account(receipt, out=out, clicked=time.time() - 3600.0, rate=3.998, stopped=False, ledger=ledger,
                  card={}, firing={"index": 0}, forecast_dir=run, timeline=door.Timeline(out / "t.jsonl"),
                  failed=True, partial=True, returncode=2)
    m = receipt["measured"]
    assert m["failed"] and m["partial"] and m["returncode"] == 2
    assert m["cost_cents"] > 108, "an hour at USD 3.998 costs about 400 cents"
    assert nest_budget.read_spent(run)[0] == 108, "machine time, capped at the quote"


def test_add_nest_exits_non_zero_for_a_failed_nest(tmp_path, monkeypatch):
    run = _run(tmp_path)
    monkeypatch.setattr(door, "add_nest", lambda *a, **k: {"failed": True})
    assert door.run_add_nest(_args(run, "--lat", "35", "--lon", "-97", "--name", "s1")) == 2
    monkeypatch.setattr(door, "add_nest", lambda *a, **k: {})
    assert door.run_add_nest(_args(run, "--lat", "35", "--lon", "-97", "--name", "s1")) == 0


def test_an_add_that_fails_without_a_name_still_ends_its_live_row(tmp_path, monkeypatch):
    """f534af2c: no --name meant no "ended" row, so the failed nest blocked firings until its planned end."""
    run = _run(tmp_path)
    seen = {}

    def broken(arguments, *, firing=None, name=None, **k):
        seen["name"] = name
        nt.note_live_nest(run, {"event": "started", "name": name, "lat": 35, "lon": -97, "size_km": 600,
                                "start_valid": "2025-05-19_12:00:00", "hours": 15, "firing": None})
        raise door.AddNestRefusal("prep ended before its head")
    monkeypatch.setattr(door, "add_nest", broken)
    with pytest.raises(door.AddNestRefusal):
        door.run_add_nest(_args(run, "--lat", "35", "--lon", "-97"))
    assert seen["name"] == "nest-+35p00-97p00"
    assert nt.read_other_nests(run, "2025-05-19_12:00:00") == []


def test_a_nest_whose_output_does_not_fit_on_the_disk_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(door.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=2 * 10**9))
    with pytest.raises(door.AddNestRefusal, match="full disk"):
        door.disk_check(tmp_path, size_km=600, dx_km=3, hours=12)
    monkeypatch.setattr(door.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=50 * 10**9))
    assert door.disk_check(tmp_path, size_km=600, dx_km=3, hours=12)["need_bytes"] < 50 * 10**9


def test_a_nest_writes_no_restart_checkpoints(tmp_path):
    """Test run 2: 323 MB a hour, all kept, 5.8 GB per 18 h nest; nothing restarts a nest."""
    config = tmp_path / "n.toml"
    config.write_text("[experiment]\nrun_seconds = 64800.0\nrestart_interval_s = 3600.0\n\n[physics]\nmp_physics = 10\n")
    door.no_restarts(config)
    text = config.read_text()
    assert "restart_interval_s = 0.0" in text and "3600.0" not in text and "mp_physics = 10" in text
    bare = tmp_path / "m.toml"
    bare.write_text("[experiment]\nrun_seconds = 3600.0\n")
    door.no_restarts(bare)
    assert "[experiment]\nrestart_interval_s = 0.0\nrun_seconds" in bare.read_text()


def test_a_nest_draws_the_met_set_not_all_by_default(tmp_path):
    """2.9.5/2.9.6: `all` includes the hub-height maps, which die on a 1.5 km nest's dotted folder name."""
    args = _args(_run(tmp_path), "--lat", "35", "--lon", "-97")
    assert args.render_products == door.NEST_PRODUCTS
    assert "all" not in args.render_products.split(",") and not any(p.startswith("hub") for p in args.render_products.split(","))
    assert "composite_reflectivity" in args.render_products.split(",")


def test_the_box_stop_leaves_room_for_the_upload_tail():
    assert abs(nest_budget.stop_cents(216, overrun_factor=1.25, usd_per_hour=3.998) - 159.473) < 0.01
