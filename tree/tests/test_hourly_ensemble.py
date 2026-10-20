"""Science and resource invariants of the hourly ensemble door (CPU only)."""
import json
from pathlib import Path
import threading
import time

import numpy as np
from netCDF4 import Dataset
import pytest

from hexcore.ensemble import (SCHEMA, EnsembleRefusal, execute_jobs, export_native,
                              packing, read_manifest, recenter, spread, validate_inputs)


def state(path, offset=0, reversed_order=False):
    with Dataset(path, "w") as ds:
        for name, size in (("Time", 1), ("nCells", 3), ("nEdges", 3),
                           ("nVertLevels", 2), ("nVertLevelsP1", 3), ("StrLen", 64)):
            ds.createDimension(name, size)
        ds.createVariable("nominalMinDc", "f4", ())[:] = 60000
        for name in ("indexToCellID", "indexToEdgeID"):
            ds.createVariable(name, "i4", ("nCells" if name.endswith("CellID") else "nEdges",))[:] = [3,2,1] if reversed_order else [1,2,3]
        for name in ("latCell", "lonCell", "latEdge", "lonEdge"):
            ds.createVariable(name, "f8", ("nCells" if name.endswith("Cell") else "nEdges",))[:] = [0.2,0.1,0] if reversed_order else [0,0.1,0.2]
        for name, dims, value in (("areaCell", ("nCells",), [1,2,3]),
                                  ("dcEdge", ("nEdges",), [1,1,1]),
                                  ("dvEdge", ("nEdges",), [1,1,1]),
                                  ("zgrid", ("nCells", "nVertLevelsP1"), 1),
                                  ("rho_base", ("Time", "nCells", "nVertLevels"), 1),
                                  ("theta_base", ("Time", "nCells", "nVertLevels"), 300)):
            ds.createVariable(name, "f4", dims)[:] = value
        from hexcore.ensemble import FIELDS
        for name in FIELDS:
            dims = ("Time", "nEdges" if name == "u" else "nCells")
            if name != "surface_pressure":
                dims += ("nVertLevelsP1" if name == "w" else "nVertLevels",)
            base = {"theta": 300, "rho": 1, "pressure": 100000, "surface_pressure": 100000}.get(name, 2)
            ds.createVariable(name, "f4", dims)[:] = base + offset
        ds.createVariable("xtime", "S1", ("Time", "StrLen"))[0] = np.frombuffer(b"2026-10-10_06:00:00".ljust(64,b" "), dtype="S1")
    return path


def test_spread_sample_variance_and_missing_value(tmp_path):
    a, b = state(tmp_path / "a.nc"), state(tmp_path / "b.nc", 2)
    result = spread([a,b], variables=("theta", "u", "w"), chunk=2)
    for row in result:
        assert np.allclose(row["spread_per_level"], np.sqrt(2))
    with Dataset(b, "r+") as ds:
        ds["theta"][0,1,0] = np.nan
    with pytest.raises(EnsembleRefusal, match="nonfinite"):
        spread([a,b], variables=("theta",))


def test_native_export_identity_not_sort_by_value(tmp_path):
    template = state(tmp_path / "template.nc")
    history = state(tmp_path / "history.nc", reversed_order=True)
    from hexcore.ensemble import ALIASES
    with Dataset(history, "r+") as ds:
        for target, alias in ALIASES.items():
            ds.renameVariable(target, alias)
        ds["theta"][0,:,0] = [303,302,301]
    output = tmp_path / "out.nc"
    export_native(history, template, output, "2026-10-10_07:00:00")
    with Dataset(output) as ds:
        assert list(ds["theta"][0,:,0]) == [301,302,303]
        assert b"".join(ds["xtime"][0]).strip() == b"2026-10-10_07:00:00"
        assert np.array_equal(ds["indexToCellID"][:], [1,2,3])
        assert np.allclose(ds["pressure_p"][:] + ds["pressure_base"][:], ds["pressure"][:], atol=.01, rtol=0)


def test_native_export_accepts_equivalent_longitude_branches(tmp_path):
    template, history = state(tmp_path / "template.nc"), state(tmp_path / "history.nc")
    from hexcore.ensemble import ALIASES
    with Dataset(history, "r+") as ds:
        for target,alias in ALIASES.items():
            ds.renameVariable(target,alias)
        ds["lonCell"][:] = ds["lonCell"][:] - 2*np.pi
        ds["lonEdge"][:] = ds["lonEdge"][:] - 2*np.pi
    export_native(history,template,tmp_path / "out.nc","2026-10-10_07:00:00")


def test_recenter_preserves_anomalies_and_mean(tmp_path):
    a, b = state(tmp_path / "a.nc"), state(tmp_path / "b.nc", 2)
    base = state(tmp_path / "base.nc", 4)
    result = recenter([a,b], base, tmp_path / "recenter")
    with Dataset(result[0]) as x, Dataset(result[1]) as y, Dataset(base) as z:
        for name in ("theta", "rho", "qv", "qi", "qs", "qg", "u", "w"):
            assert np.allclose((x[name][:]+y[name][:])/2, z[name][:])
            assert np.allclose(y[name][:]-x[name][:], 2)


def test_export_refuses_same_ids_with_different_geometry(tmp_path):
    template = state(tmp_path / "template.nc")
    history = state(tmp_path / "history.nc")
    with Dataset(history, "r+") as ds:
        ds["latCell"][0] = .3
    with pytest.raises(EnsembleRefusal, match="different meshes"):
        export_native(history, template, tmp_path / "out.nc", "2026-10-10_07:00:00")
    assert not (tmp_path / "out.nc").exists()
    assert not (tmp_path / "out.partial.nc").exists()


def test_packing_and_worker_slot_bound():
    assert packing([100,50], 25, 20) == [0,1,0,1,0,0]
    assert packing([100], 25, 1, 1) == [0]
    with pytest.raises(EnsembleRefusal, match="invalid"):
        packing([100], 25, 0)
    with pytest.raises(EnsembleRefusal, match="exhaust"):
        packing([50], 25, 20, 3)
    running = {0:0,1:0}; peak = {0:0,1:0}; lock = threading.Lock()
    def launch(row, card):
        with lock:
            running[card] += 1; peak[card] = max(peak[card],running[card])
        time.sleep(.005)
        with lock:
            running[card] -= 1
        return {"id": row, "returncode": 0}
    results = execute_jobs(list(range(80)), [0,0,1], launch)
    assert len(results) == 80 and len({r["id"] for r in results}) == 80
    assert peak[0] <= 2 and peak[1] <= 1


def test_manifest_80_members_and_duplicate_rejection(tmp_path):
    files = [state(tmp_path / f"member{i}.nc") for i in range(80)]
    data = {"schema": SCHEMA, "resolution_km": 60, "start_time": "2026-10-10_06:00:00",
            **{k:str(files[0]) for k in ("grid", "static", "mesh_rows", "native_template")},
            "members": [{"id":f"m{i}","init":str(p),"source":"prepared EnKF"} for i,p in enumerate(files)]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(data))
    assert len(read_manifest(path, 80)["members"]) == 80
    data["members"][1]["init"] = data["members"][0]["init"]
    path.write_text(json.dumps(data))
    with pytest.raises(EnsembleRefusal, match="duplicate"):
        read_manifest(path, 80)


def test_duplicate_physical_states_cannot_be_hidden_by_file_names(tmp_path):
    a, b = state(tmp_path / "a.nc"), state(tmp_path / "b.nc")
    data = {"grid":str(a),"static":str(a),"resolution_km":60,"start_time":"2026-10-10_06:00:00",
            "members":[{"id":"a","init":str(a)},{"id":"b","init":str(b)}]}
    with pytest.raises(EnsembleRefusal, match="duplicate physical"):
        validate_inputs(data)


def history_frame(path, reversed_order=False, valid="2026-10-10_12.00.00"):
    """A tiny forecast-history-shaped frame (aliases, pressure, condensates)."""
    state(path, reversed_order=reversed_order)
    from hexcore.ensemble import ALIASES
    with Dataset(path, "r+") as ds:
        for target, alias in ALIASES.items():
            ds.renameVariable(target, alias)
        ds["theta"][0, :, 0] = [303, 302, 301] if reversed_order else [301, 302, 303]
        ds["pressure"][0, :, :] = [[90000, 40000]] * 3
        ds["qv"][0, :, :] = 0.01
    return path


def test_jedi_member_has_ensemble_stream_fields_in_native_order(tmp_path):
    from hexcore.ensemble import JEDI_ENSEMBLE_FIELDS, JEDI_SCALARS, export_jedi_member
    template = state(tmp_path / "template.nc")
    history = history_frame(tmp_path / "history.nc", reversed_order=True)
    out = tmp_path / "member.nc"
    export_jedi_member(history, template, out, "2026-10-10_12:00:00")
    with Dataset(out) as ds:
        assert ds.file_format == "NETCDF3_64BIT_DATA"
        for name in JEDI_ENSEMBLE_FIELDS + JEDI_SCALARS:
            assert name in ds.variables, name
        assert list(ds["theta"][0, :, 0]) == [301, 302, 303]
        total = ds["pressure_p"][0].astype(np.float64) + ds["pressure_base"][0]
        assert np.allclose(total, [[90000, 40000]] * 3, atol=0.05, rtol=0)
        assert b"".join(ds["xtime"][0]).strip() == b"2026-10-10_12:00:00"
    with pytest.raises(EnsembleRefusal, match="exists"):
        export_jedi_member(history, template, out, "2026-10-10_12:00:00")


def test_pressure_level_extract_interpolates_in_log_pressure(tmp_path):
    from hexcore.ensemble import extract_pressure_levels
    history = history_frame(tmp_path / "history.nc")
    with Dataset(history, "r+") as ds:
        ds["pressure"][0, :, :] = [[100000, 25000]] * 3
    zmid = np.array([[0.0, 10000.0]] * 3)
    out = extract_pressure_levels(history, zmid, tmp_path / "x.npz")
    data = np.load(out["path"])
    expected = 10000.0 * np.log(50000 / 100000) / np.log(25000 / 100000)
    assert np.allclose(data["z500"], expected, rtol=1e-6)
    assert np.isnan(data["z250"]).all()


def test_frame_consumer_waits_for_durable_marker_and_discards(tmp_path):
    from hexcore.ensemble import consume_frames
    template = state(tmp_path / "template.nc")
    hist = tmp_path / "history"; hist.mkdir()
    frame = history_frame(hist / "cuda-history.2026-10-10_12.00.00.nc")
    done = threading.Event(); done.set()
    # no durable marker yet: nothing is consumed, nothing is deleted
    assert consume_frames(hist, "2026-10-10_06:00:00", done, export_hours=[6], template=template,
                          export_dir=tmp_path, member="m1", discard=True) == []
    assert frame.exists()
    (hist / (frame.name + ".durable.json")).write_text("{}")
    rows = consume_frames(hist, "2026-10-10_06:00:00", done, export_hours=[6], template=template,
                          export_dir=tmp_path, member="m1", discard=True)
    assert len(rows) == 1 and rows[0]["forecast_hour"] == 6 and rows[0]["frame_deleted"]
    assert (tmp_path / "mpas.m1.2026-10-10_12.00.00.nc").exists() and not frame.exists()


def test_streamed_member_prepare_must_be_argv(tmp_path):
    files = [state(tmp_path / f"member{i}.nc") for i in range(2)]
    data = {"schema": SCHEMA, "resolution_km": 60, "start_time": "2026-10-10_06:00:00",
            **{k: str(files[0]) for k in ("grid", "static", "mesh_rows", "native_template")},
            "members": [{"id": "a", "init": str(tmp_path / "later-a.nc"), "source": "s", "prepare": ["true"]},
                        {"id": "b", "init": str(tmp_path / "later-b.nc"), "source": "s", "prepare": "rm -rf /"}]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(data))
    with pytest.raises(EnsembleRefusal, match="argv"):
        read_manifest(path, 2)
    data["members"][1]["prepare"] = ["true"]
    path.write_text(json.dumps(data))
    assert len(read_manifest(path, 2)["members"]) == 2


def test_regridded_extract_keeps_below_ground_nan(tmp_path):
    from hexcore.ensemble import extract_pressure_levels, latlon_regrid_weights
    history = history_frame(tmp_path / "history.nc")
    with Dataset(history, "r+") as ds:
        ds["pressure"][0, :, :] = [[100000, 25000]] * 3
    zmid = np.array([[0.0, 10000.0]] * 3)
    regrid = latlon_regrid_weights(np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2]), 45.0, neighbours=2)
    data = np.load(extract_pressure_levels(history, zmid, tmp_path / "g.npz", regrid)["path"])
    assert data["z500"].shape == (5, 8)
    assert np.allclose(data["z500"], 10000.0 * np.log(0.5) / np.log(0.25), rtol=1e-6)
    assert np.isnan(data["z250"]).all()


def test_deadline_admission_leaves_members_unlaunched():
    launched = []
    def launch(row, card):
        launched.append(row)
        return {"id": row, "returncode": 0}
    budget = {"left": 3}
    def admit(row):
        budget["left"] -= 1
        return budget["left"] >= 0
    results = execute_jobs(list(range(10)), [0, 1], launch, admit)
    assert len(results) == 3 and len(set(launched)) == 3


def test_frame_products_run_in_a_spawned_process(tmp_path):
    """netCDF is not thread-safe: the door publishes frames from spawned processes."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from hexcore.ensemble import consume_frames
    template = state(tmp_path / "template.nc")
    hist = tmp_path / "history"; hist.mkdir()
    frame = history_frame(hist / "cuda-history.2026-10-10_12.00.00.nc")
    with Dataset(frame, "r+") as ds:
        ds["pressure"][0, :, :] = [[100000, 25000]] * 3
    (hist / (frame.name + ".durable.json")).write_text("{}")
    weights = tmp_path / "w.npz"
    np.savez(weights, zmid=np.array([[0.0, 10000.0]] * 3))
    (tmp_path / "x").mkdir()
    done = threading.Event(); done.set()
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = consume_frames(hist, "2026-10-10_06:00:00", done, export_hours=[6], template=template,
                              export_dir=tmp_path, member="m1", extract_dir=tmp_path / "x", discard=True,
                              pool=pool, weights_path=str(weights))
    assert rows[0]["frame_deleted"] and "jedi" in rows[0] and "extract" in rows[0]
    assert np.allclose(np.load(tmp_path / "x" / "2026-10-10_12.00.00.npz")["z500"], 5000.0)


def test_member_forecast_flags_are_validated_and_kept(tmp_path):
    files = [state(tmp_path / f"member{i}.nc") for i in range(2)]
    data = {"schema": SCHEMA, "resolution_km": 60, "start_time": "2026-10-10_06:00:00",
            **{k: str(files[0]) for k in ("grid", "static", "mesh_rows", "native_template")},
            "forecast_flags": ["--pbl-cadence", "step"],
            "members": [{"id": "a", "init": str(files[0]), "source": "s"},
                        {"id": "b", "init": str(files[1]), "source": "s",
                         "forecast_flags": ["--convection", "kain-fritsch"]}]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(data))
    rows = read_manifest(path, 2)["members"]
    assert rows[0]["forecast_flags"] == [] and rows[1]["forecast_flags"] == ["--convection", "kain-fritsch"]
    assert all(row['stochastic']['sppt'] and row['stochastic']['skeb'] for row in rows)
    before = {row['id']: row['stochastic'] for row in rows}
    data['members'].reverse(); path.write_text(json.dumps(data))
    assert {row['id']: row['stochastic'] for row in read_manifest(path,2)['members']} == before
    data['members'].reverse()
    for bad, match in ((["--init", "other.nc"], "replace a member"), (["--hours=99"], "replace a member"),
                       (["--samehost-devices", "0,1"], "replace a member"), ("--convection off", "argv"),
                       # since the products door the forecast door has --no-products; the
                       # ensemble owns the member's product stage, so it is a controlled flag
                       (["--no-products"],"replace a member"), (["--products-detached"],"replace a member"),
                       (["--fullstep-lts-mode"],"not executable")):
        data["members"][1]["forecast_flags"] = bad
        path.write_text(json.dumps(data))
        with pytest.raises(EnsembleRefusal, match=match):
            read_manifest(path, 2)


@pytest.mark.parametrize('joined', [False, True])
def test_member_stochastic_object_overrides_manifest_flags(tmp_path, joined):
    # Only configuration is consumed here; no weather fields are prepared.
    files = [tmp_path/f'member{i}.nc' for i in range(2)]
    for path in files:
        path.touch()
    common = json.dumps(dict(sppt=True, skeb=False, seed=711, degree=8,
                             time_s=600, amplitude=.4))
    flags = ['--stochastic='+common] if joined else ['--stochastic', common]
    data = dict(schema=SCHEMA, resolution_km=60, start_time='2026-10-10_06:00:00',
                **{key: str(files[0]) for key in ('grid','static','mesh_rows','native_template')},
                forecast_flags=flags, members=[
                    dict(id='a', init=str(files[0]), source='configuration fixture',
                         stochastic=dict(sppt=False, seed=712, length_m=1500000),
                         forecast_flags=['--stochastic', '{"skeb":true,"amplitude":0.2}']),
                    dict(id='b', init=str(files[1]), source='configuration fixture')])
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(data))
    rows = {row['id']: row['stochastic'] for row in read_manifest(path, 2)['members']}
    assert not rows['a']['sppt']
    assert rows['a']['seed'] == 712
    assert rows['a']['skeb'] and rows['a']['amplitude'] == .2
    assert rows['a']['degree'] == 8 and rows['a']['time_s'] == 600
    assert rows['a']['length_m'] == 1500000
    assert rows['b']['sppt'] and not rows['b']['skeb']
    assert rows['b']['seed'] == 711 and rows['b']['amplitude'] == .4
    data['members'].reverse()
    path.write_text(json.dumps(data))
    assert {row['id']: row['stochastic'] for row in read_manifest(path, 2)['members']} == rows


def test_patch_one_member_is_a_forecast_pilot_not_covariance(tmp_path):
    f = state(tmp_path / "control.nc")
    data = {"schema": SCHEMA, "resolution_km": 3, "start_time": "2026-10-10_06:00:00",
            **{k: str(f) for k in ("grid", "static", "mesh_rows", "native_template")},
            "patch": {"center_deg": [42.5, -87], "core_radius_km": 650},
            "forecast_flags": ["--dt", "18.75", "--fullstep-local-timestep", "--fullstep-lts-mode", "active"],
            "members": [{"id": "control", "init": str(f), "source": "GEFS control",
                         "reference_init": "control.nc"}]}
    p = tmp_path / "case.json"; p.write_text(json.dumps(data))
    with pytest.raises(EnsembleRefusal, match="two members"):
        read_manifest(p, 1, 3)
    result = read_manifest(p, 1, 3, forecast_only=True)
    assert result["patch"] == data["patch"]
    assert result["forecast_flags"] == data["forecast_flags"]
    assert len(result["members"]) == 1
    assert result["members"][0]["reference_init"] == str(f)
    from hexcore.cli import build_parser
    args = build_parser().parse_args(["ensemble", "--manifest", str(p), "--out", str(tmp_path / "out"),
        "--members", "1", "--resolution-km", "3", "--forecast-only", "--members-per-card", "1"])
    assert args.forecast_only and args.resolution_km == 3


def test_members_validate_their_own_mesh_orders(tmp_path):
    a = state(tmp_path / "a.nc")
    b = state(tmp_path / "b.nc", 2, reversed_order=True)
    data = {"resolution_km": 60, "start_time": "2026-10-10_06:00:00", "grid": str(a), "static": str(a),
            "members": [{"id": "a", "init": str(a)},
                        {"id": "b", "init": str(b), "grid": str(b), "static": str(b)}]}
    from hexcore.ensemble import validate_inputs
    assert validate_inputs(data, state_hashes=set())["cells"] == 3
    assert data["members"][1]["dimensions"]["levels"] == 2
    data["members"][1]["grid"] = str(a)
    with pytest.raises(EnsembleRefusal, match="static .* differs"):
        validate_inputs(data, state_hashes=set())


def test_discard_waits_for_forecast_command_closure(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hexcore.ensemble import consume_frames
    template = state(tmp_path / "template.nc")
    hist = tmp_path / "history"; hist.mkdir()
    frame = history_frame(hist / "cuda-history.2026-10-10_07.00.00.nc")
    (hist / (frame.name + ".durable.json")).write_text("{}")
    done = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(consume_frames, hist, "2026-10-10_06:00:00", done,
                             export_hours=[], template=template, export_dir=tmp_path,
                             member="a", discard=True, poll=0.02)
        time.sleep(0.1)
        assert frame.exists() and not future.done()
        done.set()
        records = future.result(timeout=5)
    assert records[0]["frame_deleted"] and not frame.exists()


def test_lts_timings_come_from_this_members_full_macro_receipt(tmp_path):
    from hexcore.ensemble import lts_run_facts
    p = tmp_path/'fullstep-lts-receipt.json'
    record = dict(devices=[0], macros=[dict(seconds=300),dict(seconds=20)],
                  memory_pool_peak_bytes_per_card={'0':123456})
    p.write_text(json.dumps(record))
    facts = lts_run_facts(tmp_path)
    assert facts['integration_seconds'] == 320
    assert facts['resident_devices'] == [0]
    assert facts['memory_pool_peak_bytes_per_card'] == {'0':123456}
    record['devices'] = [0,1]
    p.write_text(json.dumps(record))
    with pytest.raises(EnsembleRefusal, match='expected one'):
        lts_run_facts(tmp_path)


def test_onecard_lts_v1_receipt_has_no_device_partition_or_pool_peak(tmp_path):
    from hexcore.ensemble import lts_run_facts
    p = tmp_path/'fullstep-lts-receipt.json'
    record = dict(schema='gpuwm-hex.fullstep-lts-forecast/v1', dispatch='rate classes',
                  macros=[dict(seconds=7), dict(seconds=11)])
    p.write_text(json.dumps(record))
    facts = lts_run_facts(tmp_path)
    assert facts['integration_seconds'] == 18
    assert facts['resident_devices'] == [0]
    assert facts['memory_pool_peak_bytes_per_card'] is None
    record['dispatch'] = 'unknown'
    p.write_text(json.dumps(record))
    with pytest.raises(EnsembleRefusal, match='dispatch'):
        lts_run_facts(tmp_path)
