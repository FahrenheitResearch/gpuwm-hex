"""[hex_nests]: a storm that appears in the area fires one nest at its centroid, on time."""

import json

import numpy as np
import pytest

from hexcore import nest_triggers as nt


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


def _table(**over):
    raw = {"max_firings": 2, "sustain_looks": 2,
           "area": {"box": {"south": 31, "north": 39, "west": -103, "east": -93}},
           "nest": {"dx_km": 3, "size_km": 400, "hours": 6},
           "when": [{"all": [{"field": "refl", "op": ">=", "value": 45, "cover_km2": 500}]}]}
    raw.update(over)
    return nt.resolve(raw)


def test_fires_once_at_the_storm_after_sustain(tmp_path):
    la, lo, coc, ne, area = _grid()
    table = _table()
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    trig = nt.LoopTrigger(table, run_dir=tmp_path, run_start="2025-05-19_12:00:00", dt_s=60.0,
                          geometry=geo, xp=np, available_sources=["refl10cm"])
    refl = np.zeros((3, la.size), np.float32)
    storm = (np.abs(la - 35.5) < 0.15) & (np.abs(lo + 97.5) < 0.15)
    fired = []
    for step in range(15, 300, 15):
        if step >= 90:
            refl[1, storm] = 50.0
        trig.look(step, {"refl10cm": refl})
        firing = trig.poll()
        if firing:
            fired.append(firing)
    assert len(fired) == 1, "a storm inside a live nest is not a second firing"
    assert fired[0]["look_valid"] == "2025-05-19T13:45:00Z"  # second look holding
    assert abs(fired[0]["lat"] - 35.5) < 0.05 and abs(fired[0]["lon"] + 97.5) < 0.05
    assert nt.read_firings(tmp_path)[0]["index"] == 0


def test_object_test_picks_the_biggest_object():
    la, lo, coc, ne, area = _grid()
    table = nt.resolve({"sustain_looks": 1, "when": [{"all": [
        {"field": "storm_object", "op": ">=", "value": 45, "object_km2": 500}]}]})
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    ev = nt.DeviceEvaluator(table, geo, nt.load_field_table())
    refl = np.zeros((1, la.size), np.float32)
    refl[0, (np.abs(la - 35.5) < 0.15) & (np.abs(lo + 97.5) < 0.15)] = 50.0
    refl[0, (np.abs(la - 33.0) < 0.05) & (np.abs(lo + 95.0) < 0.05)] = 50.0
    ev.launch({"refl10cm": refl}, step=1, model_seconds=60.0)
    record, _ = ev.ready()
    lat, lon = nt._latlon(record[0, nt.RECORD_HEAD:nt.RECORD_HEAD + 3])
    assert record[0, 0] == 1.0 and abs(lat - 35.5) < 0.05 and abs(lon + 97.5) < 0.05


def test_a_field_the_route_does_not_form_is_refused(tmp_path):
    la, lo, coc, ne, area = _grid()
    table = nt.resolve({"when": [{"all": [{"field": "mucape", "op": ">=", "value": 2000}]}]})
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    with pytest.raises(nt.NestTriggerRefusal, match="could never fire"):
        nt.LoopTrigger(table, run_dir=tmp_path, run_start="2025-05-19_12:00:00", dt_s=60.0,
                       geometry=geo, xp=np, available_sources=["refl10cm"])


def test_unknown_key_is_refused_by_name():
    with pytest.raises(nt.NestTriggerRefusal, match="does not have key"):
        nt.resolve({"when": [{"all": [{"field": "refl", "value": 40}]}], "bogus": 1})


def test_a_storm_inside_a_hand_placed_nest_is_skipped(tmp_path):
    """A nest placed at launch or added by hand covers its storm: the firing goes to the next storm."""
    la, lo, coc, ne, area = _grid()
    table = _table(sustain_looks=1)
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    nt.note_live_nest(tmp_path, {"event": "started", "name": "okc", "lat": 35.5, "lon": -97.5, "size_km": 600,
                                 "start_valid": "2025-05-19_12:00:00", "hours": 15, "firing": None})
    trig = nt.LoopTrigger(table, run_dir=tmp_path, run_start="2025-05-19_12:00:00", dt_s=60.0,
                          geometry=geo, xp=np, available_sources=["refl10cm"])
    refl = np.zeros((3, la.size), np.float32)
    big = (np.abs(la - 35.6) < 0.3) & (np.abs(lo + 97.6) < 0.3)      # inside the OKC nest, the biggest
    small = (np.abs(la - 31.5) < 0.15) & (np.abs(lo + 101.5) < 0.15)  # west Texas, outside it
    refl[1, big] = 50.0
    refl[1, small] = 50.0
    trig.look(15, {"refl10cm": refl})
    firing = trig.poll()
    assert firing is not None
    assert abs(firing["lat"] - 31.5) < 0.1 and abs(firing["lon"] + 101.5) < 0.1
    # Once the hand nest ends, its storm is fair game again.
    nt.note_live_nest(tmp_path, {"event": "ended", "name": "okc"})
    assert nt.read_other_nests(tmp_path, "2025-05-19_12:00:00") == []


def test_no_firing_whose_nest_would_outlive_the_parent(tmp_path):
    """A 6 h nest is not fired 1 h before the parent ends (it would run 1 h and be charged as 6 h)."""
    la, lo, coc, ne, area = _grid()
    table = _table(sustain_looks=1)
    geo = nt.area_geometry(table["area"], lat_rad=np.radians(la), lon_rad=np.radians(lo),
                           area_cell=area, cells_on_cell=coc, n_edges_on_cell=ne)
    trig = nt.LoopTrigger(table, run_dir=tmp_path, run_start="2025-05-19_12:00:00", dt_s=60.0,
                          geometry=geo, xp=np, available_sources=["refl10cm"], parent_seconds=15 * 3600.0)
    refl = np.zeros((3, la.size), np.float32)
    refl[1, (np.abs(la - 35.5) < 0.15) & (np.abs(lo + 97.5) < 0.15)] = 50.0
    trig.look(14 * 60, {"refl10cm": refl})   # +14 h: a 6 h nest would end at +20 h
    assert trig.poll() is None
    last = [r for r in nt.read_firings(tmp_path)]
    assert last == []
    trig.look(9 * 60, {"refl10cm": refl})    # +9 h: ends at +15 h, exactly the parent's end
    assert trig.poll() is not None
