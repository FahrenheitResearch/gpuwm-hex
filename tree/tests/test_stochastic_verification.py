"""Paired-deck validation plus independent, opt-in GPU reduction checks."""
from copy import deepcopy
from datetime import datetime, timedelta
import os
import sys

import pytest

from hexcore.stochastic_verification import (
    SCHEMA, forecast_fields_gpu, qualify_daily, score_pair_gpu,
)

CYCLE = "2026-10-16T00:00:00+00:00"


def synthetic_receipts():
    """Small score fixtures only, never forecast evidence."""
    cycle = datetime.fromisoformat(CYCLE)
    return [dict(schema=SCHEMA, lead_hours=lead,
                 valid_time=(cycle+timedelta(hours=lead)).isoformat(),
                 members=[f"member-{i}" for i in range(10)],
                 analysis_source="GFS", analysis_id=f"synthetic-fixture-{lead}", variance_ddof=1,
                 fields={field: dict(unit=unit, valid_cells=20, area_fraction=1.,
                                    analysis_area_fraction=1., forecast_missing_cells=0,
                                    control=dict(spread=1., rmse_mean=2., ratio=.5),
                                    stochastic=dict(spread=1.5, rmse_mean=2., ratio=.75))
                         for field, unit in (("z500", "m"), ("t2m", "K"))})
            for lead in range(24, 241, 24)]


def test_daily_pass_does_not_claim_complete_qualification():
    result = qualify_daily(list(reversed(synthetic_receipts())), cycle=CYCLE)
    assert result["spread_skill_passed"]
    assert not result["qualification_complete"]
    assert result["fields"]["t2m"]["daily"][-1]["mean_not_worse"]
    assert "visual inspection" in result["pending"]


@pytest.mark.parametrize("field", ["z500", "t2m"])
@pytest.mark.parametrize("key,value", [("spread", 1.), ("rmse_mean", 2.1)])
def test_one_day_or_field_failing_fails_the_bar(field, key, value):
    rows = synthetic_receipts()
    metric = rows[4]["fields"][field]["stochastic"]
    metric[key] = value
    metric["ratio"] = metric["spread"]/metric["rmse_mean"]
    result = qualify_daily(rows, cycle=CYCLE)
    assert not result["spread_skill_passed"]
    assert not result["fields"][field]["passed"]


@pytest.mark.parametrize("change,match", [
    (lambda r: r.pop(), "each daily lead"),
    (lambda r: r.append(deepcopy(r[0])), "each daily lead"),
    (lambda r: r[1].update(members=list(reversed(r[1]["members"]))), "roster changes"),
    (lambda r: r[0].update(members=["x"]*10), "ten distinct"),
    (lambda r: r[0].update(valid_time="2026-10-18T00:00:00Z"), "valid time"),
    (lambda r: r[0].update(variance_ddof=0), "sample variance"),
    (lambda r: r[0].update(analysis_source="forecast"), "shared GFS"),
    (lambda r: r[0]["fields"]["t2m"].update(unit="C"), "units differ"),
    (lambda r: r[0]["fields"]["t2m"].update(area_fraction=0), "common verification support"),
    (lambda r: r[0]["fields"]["z500"].update(forecast_missing_cells=1), "shared terrain mask"),
    (lambda r: r[0]["fields"]["z500"]["control"].update(spread=float("nan")), "nonfinite"),
    (lambda r: r[0]["fields"]["z500"]["control"].update(ratio=1), "ratio differs"),
])
def test_incomplete_or_inconsistent_receipts_are_rejected(change, match):
    rows = synthetic_receipts()
    change(rows)
    with pytest.raises(ValueError, match=match):
        qualify_daily(rows, cycle=CYCLE)


def test_times_require_explicit_utc():
    with pytest.raises(ValueError, match="explicit UTC"):
        qualify_daily(synthetic_receipts(), cycle="2026-10-16T00:00:00")


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "negative_stride"])
def test_forecast_extraction_analytic_log_pressure_oracle(monkeypatch, layout):
    # NumPy supplies a tiny independent analytic oracle through the CuPy
    # interface for CPU CI only. Production exposes no host-field backend.
    import numpy as np
    monkeypatch.setitem(sys.modules, "cupy", np)
    p = np.array([[100000., 60000., 25000.],
                  [50000., 40000., 20000.],
                  [100000., 80000., 50000.],
                  [80000., 50000., 20000.]])
    h = 12000.-600.*np.log(p)
    if layout == "fortran":
        p, h = np.asfortranarray(p), np.asfortranarray(h)
    elif layout == "negative_stride":
        p, h = p[:, ::-1].copy()[:, ::-1], h[:, ::-1].copy()[:, ::-1]
    t2m = np.array([280., 290., np.nan, 270.], dtype=np.float32)
    with np.errstate(all="raise"):
        fields = forecast_fields_gpu(p, h, t2m)
    assert fields["z500"] == pytest.approx(np.full(4, 12000.-600.*np.log(50000.)))
    assert fields["t2m"] is t2m  # Actual surface diagnostic, no lowest-level proxy.


def test_forecast_extraction_invalid_columns_stay_missing(monkeypatch):
    import numpy as np
    monkeypatch.setitem(sys.modules, "cupy", np)
    p = np.array([[40000., 30000., 20000.],   # Below the lowest model level.
                  [100000., 80000., 60000.], # Above the model top.
                  [100000., 100000., 20000.],
                  [100000., 20000., 60000.],
                  [100000., 0., 20000.],
                  [100000., -1., 20000.],
                  [100000., np.nan, 20000.],
                  [np.inf, 60000., 20000.],
                  [100000., 60000., 20000.],
                  [100000., 60000., 20000.]])
    h = np.broadcast_to([100., 4000., 11000.], p.shape).copy()
    h[-2, 1], h[-1, :] = np.nan, np.inf
    with np.errstate(all="raise"):
        fields = forecast_fields_gpu(p, h, np.full(10, 280.))
    assert np.isnan(fields["z500"]).all()


def test_forecast_extraction_requires_matching_float_device_contract(monkeypatch):
    import numpy as np
    monkeypatch.setitem(sys.modules, "cupy", np)
    with pytest.raises(TypeError, match="floating-point GPU"):
        forecast_fields_gpu(np.ones((2, 3), dtype=np.int32), np.ones((2, 3)), np.ones(2))
    with pytest.raises(TypeError, match="floating-point GPU"):
        forecast_fields_gpu([[100000., 20000.]], np.ones((1, 2)), np.ones(1))
    for p, h, t in ((np.ones((2, 1)), np.ones((2, 1)), np.ones(2)),
                    (np.ones((2, 3)), np.ones((3, 2)), np.ones(2)),
                    (np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 1))),
                    (np.ones((0, 3)), np.ones((0, 3)), np.ones(0))):
        with pytest.raises(ValueError, match="matching"):
            forecast_fields_gpu(p, h, t)


@pytest.mark.gpu
def test_gpu_forecast_extraction_into_paired_scoring():
    if os.environ.get("HEX_STOCHASTIC_GPU_TESTS") != "1":
        pytest.skip("set HEX_STOCHASTIC_GPU_TESTS=1 only on a mutex-granted card")
    cp = pytest.importorskip("cupy")
    p = cp.asarray([[100000., 60000., 25000.], [50000., 40000., 20000.],
                    [100000., 80000., 50000.], [100000., 60000., 20000.]])
    h = 12000.-600.*cp.log(p)
    h[-1, 1] = cp.nan
    t2 = cp.full(4, 280.)
    control, stochastic = {}, {}
    for member, offset in (("a", -1.), ("b", 1.)):
        control[member] = forecast_fields_gpu(p[:, ::-1].copy()[:, ::-1], h+offset, t2+offset)
        stochastic[member] = forecast_fields_gpu(p, h+2*offset, t2+2*offset)
    observed = dict(z500=cp.full(4, 12000.-600.*cp.log(50000.)), t2m=t2)
    result = score_pair_gpu(control, stochastic, observed, cp.ones(4),
                            lead_hours=24, valid_time="2026-10-17T00:00:00Z", analysis_id="fixture")
    assert isinstance(control["a"]["z500"], cp.ndarray)
    z = result["fields"]["z500"]
    assert z["valid_cells"] == 3 and z["forecast_missing_cells"] == 1
    assert z["control"]["spread"] == pytest.approx(2**.5)
    assert z["stochastic"]["spread"] == pytest.approx(8**.5)
    assert z["control"]["rmse_mean"] < 1e-10
    assert result["fields"]["t2m"]["valid_cells"] == 4


@pytest.mark.gpu
def test_gpu_paired_area_weighting_sample_variance_and_common_support():
    if os.environ.get("HEX_STOCHASTIC_GPU_TESTS") != "1":
        pytest.skip("set HEX_STOCHASTIC_GPU_TESTS=1 only on a mutex-granted card")
    cp = pytest.importorskip("cupy")
    # The third cell is missing in one arm, so both arms use weights 1/4, 3/4.
    # Two samples [-1,+1] have sample variance 2. Stochastic spread is twice that.
    control = {m: {f: cp.asarray([offset, offset+2, 0.]) for f in ("z500", "t2m")}
               for m, offset in (("b", 1.), ("a", -1.))}
    stochastic = {m: {f: cp.asarray([offset, offset, cp.nan]) for f in ("z500", "t2m")}
                  for m, offset in (("a", -2.), ("b", 2.))}
    analysis = {f: cp.zeros(3) for f in ("z500", "t2m")}
    result = score_pair_gpu(control, stochastic, analysis, cp.asarray([1., 3., 4.]),
                            lead_hours=24, valid_time="2026-10-17T00:00:00Z", analysis_id="fixture")
    row = result["fields"]["z500"]
    assert result["members"] == ["a", "b"]
    assert row["valid_cells"] == 2 and row["area_fraction"] == .5
    assert row["analysis_area_fraction"] == 1 and row["forecast_missing_cells"] == 1
    assert row["control"]["spread"] == pytest.approx(2**.5)
    assert row["control"]["rmse_mean"] == pytest.approx(3**.5)
    assert row["stochastic"]["spread"] == pytest.approx(8**.5)
    assert row["stochastic"]["rmse_mean"] == 0
    assert row["stochastic"]["ratio"] is None
    assert result["gpu_elapsed_ms"] >= 0
    with pytest.raises(ValueError, match="same two or more"):
        score_pair_gpu(control, {"a": stochastic["a"]}, analysis, cp.ones(3),
                       lead_hours=24, valid_time="2026-10-17T00:00:00Z", analysis_id="fixture")
    with pytest.raises(ValueError, match="positive"):
        score_pair_gpu(control, stochastic, analysis, cp.asarray([-1., 3., 4.]),
                       lead_hours=24, valid_time="2026-10-17T00:00:00Z", analysis_id="fixture")


def test_rust_transport_upload_and_cleanup_on_success_and_decode_error(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from hexcore import stochastic_day as day
    monkeypatch.setitem(sys.modules, "cupy", np)
    audit = []
    def dump(command, **kwargs):
        folder = Path(command[3])
        np.array([2., 3.], dtype="<f8").tofile(folder/"0000.f64")
        (folder/"metadata.json").write_text(json.dumps(dict(
            schema="gpuwm-rw-netcdf-dump-v1", variables=[dict(name="x", filename="0000.f64",
            shape=[2], dimensions=["nCells"], dtype="<f8", units="K", cf=dict(applied=True))])))
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(day.subprocess, "run", dump)
    fields = day.decode_gpu("fixture.nc", ["x", "x"], reader="rw_netcdf", work=tmp_path, audit=audit)
    assert fields["x"][0].tolist() == [2., 3.]
    assert audit[0]["command"][-1:] == ["x"]
    assert sum(p["bytes"] for p in audit[0]["decoded_files_deleted"]) > 16
    assert list(tmp_path.iterdir()) == []
    assert audit[0]["observed_threads"] is None
    for change, match in ((dict(filename="../escape"), "inside its temporary"),
                          (dict(dtype="<f4"), "dtype"),
                          (dict(shape=[3]), "byte count"),
                          (dict(cf=dict(applied=False)), "CF unpacking")):
        def bad_dump(command, **kwargs):
            result = dump(command, **kwargs)
            path = Path(command[3])/"metadata.json"
            document = json.loads(path.read_text())
            document["variables"][0].update(change)
            path.write_text(json.dumps(document))
            return result
        monkeypatch.setattr(day.subprocess, "run", bad_dump)
        with pytest.raises(ValueError, match=match):
            day.decode_gpu("bad-contract.nc", ["x"], reader="rw_netcdf", work=tmp_path, audit=audit)
        assert list(tmp_path.iterdir()) == []
    def fail(command, **kwargs):
        (Path(command[3])/"partial").write_bytes(b"incomplete")
        return SimpleNamespace(returncode=2, stderr="rw_netcdf: truncated fixture")
    monkeypatch.setattr(day.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="truncated fixture"):
        day.decode_gpu("broken.nc", ["x"], reader="rw_netcdf", work=tmp_path, audit=audit)
    assert audit[-1]["exit_code"] == 2
    assert audit[-1]["decoded_files_deleted"][0]["bytes"] == 10
    assert list(tmp_path.iterdir()) == []
    # Keep a tiny deletion inventory for the lane receipt runner; payloads
    # themselves are gone. This is generated protocol data, not weather.
    (tmp_path/"transport-audit.json").write_text(json.dumps(audit))


def test_named_dimensions_units_and_embedded_clocks():
    import numpy as np
    from hexcore.stochastic_day import field, check_clock
    cube = np.arange(6.).reshape(3, 1, 2)
    fields = {"p": (cube, dict(dimensions=["nVertLevels", "Time", "nCells"], units="Pa")),
              "xtime": (b"2026-10-17_00:00:00", dict(dtype="|S1", shape=[1, 19])),
              "time": (None, dict(dtype="<f8", times=["2026-10-17T00:00:00Z"]))}
    assert field(fields, "p", ("nCells", "nVertLevels"), ("Pa",)).tolist() == [[0., 2., 4.], [1., 3., 5.]]
    for clock in ("xtime", "time"):
        check_clock(fields, clock, "2026-10-17T00:00:00Z")
        with pytest.raises(ValueError, match="embedded input time"):
            check_clock(fields, clock, "2026-10-18T00:00:00Z")
    with pytest.raises(ValueError, match="expected units"):
        field(fields, "p", ("nCells", "nVertLevels"), ("hPa",))
    fields["p"] = (np.ones((3, 2, 2)), fields["p"][1])
    with pytest.raises(ValueError, match="one history record"):
        field(fields, "p", ("nCells", "nVertLevels"), ("Pa",))


def day_inputs(xp):
    """Tiny decoded fixtures, never forecast or GFS evidence."""
    def f(values, dims, units=None):
        return xp.asarray(values, dtype=xp.float64), dict(dimensions=list(dims), units=units)
    clock = (b"2026-10-17_00:00:00", dict(dtype="|S1", shape=[1, 19]))
    cell = f([11., 12.], ("nCells",))
    inputs = {"geo": {"indexToCellID": cell,
        "areaCell": f([1., 3.], ("nCells",), "m2"),
        "zgrid": f([[0., 2000., 8000., 14000.]]*2, ("nCells", "nVertLevelsP1"), "m")}}
    for arm in ("control", "stochastic"):
        for member, offset in (("a", -1.), ("b", 1.)):
            inputs[f"{arm}-{member}"] = {"indexToCellID": cell, "xtime": clock,
                "pressure": f([[100000., 50000., 20000.]]*2, ("nCells", "nVertLevels"), "Pa")}
            inputs[f"{arm}-{member}-diag"] = {"indexToCellID": cell, "xtime": clock,
                "t2": f([280.+offset*(1 if arm == "control" else 2)]*2, ("nCells",), "K")}
    inputs["analysis"] = {"indexToCellID": cell, "xtime": clock,
        "z500": f([5000.*9.80616/9.80665]*2, ("nCells",), "m"),
        "t2m": f([280.]*2, ("nCells",), "K"),
        "support": f([1., 0.], ("nCells",), "1")}
    request = dict(schema="hex-stochastic-day-v1", lead_hours=24, valid_time="2026-10-17T00:00:00Z",
        geometry=dict(path="geo", height_variable="zgrid", height_kind="mpas_constant_gravity_interfaces",
                      model_gravity_m_s2=9.80616),
        analysis=dict(path="analysis", source="GFS", analysis_id="synthetic-fixture",
                      remapping_provenance="fixture identity", terrain_mask_variable="support"),
        **{arm: {m: dict(path=f"{arm}-{m}", diagnostics=f"{arm}-{m}-diag") for m in ("a", "b")}
           for arm in ("control", "stochastic")})
    return inputs, request


@pytest.mark.parametrize("height_units", ["m", "m MSL"])
@pytest.mark.parametrize("id_units", [None, "unitless", "1", "-"])
def test_daily_consumer_geometry_surface_diagnostics_and_shared_mask(monkeypatch, tmp_path, height_units, id_units):
    import numpy as np
    from hexcore import stochastic_day as day
    monkeypatch.setitem(sys.modules, "cupy", np)
    inputs, request = day_inputs(np)
    inputs["geo"]["zgrid"][1]["units"] = height_units
    inputs["geo"]["indexToCellID"][1]["units"] = id_units
    def read(path, names, **kwargs):
        return {name: inputs[path][name] for name in names}
    monkeypatch.setattr(day, "decode_gpu", read)
    def score(control, stochastic, analysis, area, **kwargs):
        assert control["a"]["z500"] == pytest.approx(np.full(2, 5000.*9.80616/9.80665))
        assert control["a"]["t2m"].tolist() == [279., 279.]
        assert stochastic["b"]["t2m"].tolist() == [282., 282.]
        assert np.isfinite(analysis["z500"][0]) and np.isnan(analysis["z500"][1])
        assert analysis["t2m"].tolist() == [280., 280.]
        assert area.tolist() == [1., 3.]
        return dict(members=sorted(control), **kwargs)
    monkeypatch.setattr(day, "score_pair_gpu", score)
    result = day.score_day(request, reader="rw_netcdf", work=tmp_path)
    assert not result["qualification_complete"]
    assert result["shared_terrain_mask"] == "support"
    inputs["stochastic-b"]["indexToCellID"] = (np.array([12., 11.]), dict(dimensions=["nCells"]))
    with pytest.raises(ValueError, match="cell identities/order"):
        day.score_day(request, reader="rw_netcdf", work=tmp_path)
    inputs["stochastic-b"]["indexToCellID"] = inputs["geo"]["indexToCellID"]
    inputs["stochastic-b-diag"]["xtime"] = (b"2026-10-18_00:00:00", dict(dtype="|S1", shape=[1, 19]))
    with pytest.raises(ValueError, match="embedded input time"):
        day.score_day(request, reader="rw_netcdf", work=tmp_path)


def test_daily_cli_preserves_failure_audit(monkeypatch, tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    path = Path(__file__).parents[1]/"tools/stochastic_verify_day.py"
    spec = importlib.util.spec_from_file_location("stochastic_verify_day", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    request, output = tmp_path/"request.json", tmp_path/"error.json"
    request.write_text("{}")
    def failed_score(request, **kwargs):
        kwargs["audit"].append(dict(exit_code=2, decoded_files_deleted=[dict(path="fixture", bytes=10)]))
        raise RuntimeError("truncated fixture")
    monkeypatch.setattr(cli, "score_day", failed_score)
    monkeypatch.setattr(sys, "argv", [str(path), str(request), "--reader", "rw_netcdf",
                                     "--work", str(tmp_path), "--out", str(output)])
    with pytest.raises(RuntimeError, match="truncated fixture"):
        cli.main()
    result = json.loads(output.read_text())
    assert result["error"] == "truncated fixture"
    assert result["input_reads"][0]["exit_code"] == 2
    assert result["input_reads"][0]["decoded_files_deleted"][0]["bytes"] == 10
    assert not result["qualification_complete"] and "fields" not in result


@pytest.mark.gpu
def test_gpu_daily_consumer_with_decoded_fixtures(monkeypatch, tmp_path):
    if os.environ.get("HEX_STOCHASTIC_GPU_TESTS") != "1":
        pytest.skip("set HEX_STOCHASTIC_GPU_TESTS=1 only on a mutex-granted card")
    cp = pytest.importorskip("cupy")
    from hexcore import stochastic_day as day
    inputs, request = day_inputs(cp)
    monkeypatch.setattr(day, "decode_gpu", lambda path, names, **kw: {n: inputs[path][n] for n in names})
    result = day.score_day(request, reader="rw_netcdf", work=tmp_path)
    assert result["fields"]["z500"]["valid_cells"] == 1
    assert result["fields"]["z500"]["forecast_missing_cells"] == 0
    assert result["fields"]["t2m"]["stochastic"]["spread"] == pytest.approx(8**.5)
    assert result["fields"]["t2m"]["control"]["rmse_mean"] == 0


def gfs_grid(scan=0):
    return dict(template=0, nx=4, ny=3, scan_mode=scan, dlat=90., dlon=90.,
                lat1=-90. if scan & 64 else 90., lat2=90. if scan & 64 else -90.,
                lon1=270. if scan & 128 else 0., lon2=0. if scan & 128 else 270.)


@pytest.mark.parametrize("scan", [0, 64, 128, 192])
@pytest.mark.parametrize("coordinate_dtype", ["float64", "float32"])
def test_gfs_gpu_remap_scan_order_seam_poles_and_missing_corners(monkeypatch, scan, coordinate_dtype):
    import numpy as np
    from hexcore.stochastic_gfs import remap_gpu
    monkeypatch.setitem(sys.modules, "cupy", np)  # Analytic oracle, no production CPU backend.
    canonical = np.array([90., 0., -90.])[:, None]+np.array([0., 10., 20., 30.])[None, :]
    def stored(a):
        return a[::(-1 if scan & 64 else 1), ::(-1 if scan & 128 else 1)]
    lat = np.deg2rad(np.array([45., 0., -45., 90., -90.]))
    lon = np.deg2rad(np.array([45., -45., 315., 360., -720.]))
    # The Rust reader widens the init's encoded float32 coordinates to f64.
    lat, lon = [a.astype(coordinate_dtype).astype(np.float64) for a in (lat, lon)]
    result = remap_gpu(stored(canonical), gfs_grid(scan), lat, lon)
    assert result == pytest.approx([50., 15., -30., 90., -90.])
    canonical[1, 1] = np.nan
    with np.errstate(all="raise"):
        result = remap_gpu(stored(canonical), gfs_grid(scan), np.zeros(3), np.deg2rad(np.array([0., 45., 90.])))
    assert result[0] == 0 and np.isnan(result[1:]).all()


@pytest.mark.parametrize("change", [dict(nx=5), dict(scan_mode=16), dict(template=30),
                                  dict(lat1=89.), dict(lat2=-89.), dict(lon2=271.),
                                  dict(dlat=0.), dict(dlon=float("nan"))])
def test_gfs_grid_inconsistency_is_rejected(change):
    from hexcore.stochastic_gfs import regular_grid
    with pytest.raises(ValueError, match="grid"):
        regular_grid({**gfs_grid(), **change})


def test_gfs_remap_requires_valid_device_coordinates_and_shapes(monkeypatch):
    import numpy as np
    from hexcore.stochastic_gfs import remap_gpu
    monkeypatch.setitem(sys.modules, "cupy", np)
    with pytest.raises(TypeError, match="GPU"):
        remap_gpu(np.ones((3, 4), dtype=np.int32), gfs_grid(), np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="shapes"):
        remap_gpu(np.ones((3, 4)), gfs_grid(), np.zeros(2), np.zeros(1))
    beyond_pole = float(np.nextafter(np.float32(np.pi/2), np.float32(np.inf)))
    for latitude, longitude in (([2.], [0.]), ([np.nan], [0.]), ([0.], [np.inf]),
                                ([beyond_pole], [0.]), ([-beyond_pole], [0.])):
        with pytest.raises(ValueError, match="finite radians"):
            remap_gpu(np.ones((3, 4)), gfs_grid(), np.array(latitude), np.array(longitude))


def gfs_index():
    return ("1:0:d=2026101700:HGT:500 mb:anl:\n"
            "2:10:d=2026101700:TMP:2 m above ground:anl:\n"
            "3:20:d=2026101700:PRES:surface:anl:\n"
            "4:30:d=2026101700:UGRD:10 m above ground:anl:\n")


def test_gfs_range_selection_exact_analysis_and_bounded_offsets():
    from hexcore.stochastic_gfs import index_ranges
    assert index_ranges(gfs_index(), "2026-10-17T00:00:00Z") == dict(z500=(0, 9), t2m=(10, 19), psfc=(20, 29))
    for text in (gfs_index().replace(":anl:", ":0 hour fcst:"),
                 gfs_index().replace("2026101700", "2026101600"),
                 gfs_index().replace("2:10:", "2:0:"),
                 gfs_index().replace("4:30:", "4:999999999:"),
                 "\n".join(gfs_index().splitlines()[:3]),
                 gfs_index()+"5:40:d=2026101700:HGT:500 mb:anl:\n"):
        with pytest.raises(ValueError, match="GFS"):
            index_ranges(text, "2026-10-17T00:00:00Z")


def test_gfs_https_range_transport_checks_headers_lengths_and_audit(monkeypatch, tmp_path):
    import io
    from hexcore import stochastic_gfs as gfs
    class Response(io.BytesIO):
        status = 206
    defect = None
    def open_url(request, **kwargs):
        span = request.headers.get("Range")
        body = gfs_index().encode() if span is None else b"0123456789"
        result = Response(body[:-1] if defect == "truncated" and span else body)
        result.headers = {"Content-Range": (span.replace("=", " ")+"/40") if span else ""}
        if defect == "headers":
            result.headers = {}
        if defect == "status":
            result.status = 200
        return result
    monkeypatch.setattr(gfs.urllib.request, "urlopen", open_url)
    audit = {}
    paths = gfs.fetch_ranges("https://example.invalid/gfs", "2026-10-17T00:00:00Z", tmp_path, audit)
    assert set(paths) == set(gfs.SELECTORS)
    assert all(path.read_bytes() == b"0123456789" for path in paths.values())
    assert all(row["bytes"] == 10 and len(row["sha256"]) == 64 for row in audit["downloads"].values())
    for defect in ("truncated", "headers", "status"):
        with pytest.raises(ValueError, match="GFS"):
            gfs.fetch_ranges("https://example.invalid/gfs", "2026-10-17T00:00:00Z", tmp_path, {})


def test_concurrent_rust_stack_protocol_remap_and_error_cleanup(monkeypatch, tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import threading
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.ingest import cpu_backend
    from hexcore import stochastic_gfs as gfs
    class Event:
        def record(self):
            pass
        def synchronize(self):
            pass
    fake_cupy = SimpleNamespace(**{k: getattr(np, k) for k in (
        "fromfile", "where", "isfinite", "nan", "ndarray", "all", "abs", "clip", "minimum",
        "floor", "int64", "zeros", "float64", "ones")},
        cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda *args: 0.))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    bridge, source = tmp_path/"mock-reader.so", tmp_path/"source.grib2"
    bridge.write_bytes(b"mock native library, never loaded")
    source.write_bytes(b"mock source, never decoded")
    barrier, defect = threading.Barrier(3), None
    def decode(input_path, spec_path, output):
        barrier.wait(timeout=5)  # The orchestration must enter all three native calls concurrently.
        key = json.loads(spec_path.read_text())["select"][0]["key"]
        output.mkdir()
        values = (np.full((3, 4), 5000.) if key == "z500" else
                  np.arange(12.).reshape(3, 4)+280. if key == "t2m" else np.full((3, 4), 100000.))
        if key == "psfc":
            values[1, 1] = 40000.
        payload = output/"field.f32le"
        values.astype("<f4").tofile(payload)
        row = dict(key=key, file=payload.name, sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
            levels=gfs.selector(key)["levels"], pdt=0, forecast_step=0,
            selector_values=[{k: v for k, v in gfs.selector(key).items() if k not in ("key", "levels")}],
            valid_time="2026-10-17T00:00:00Z", reference_time="2026-10-17T00:00:00Z", grid=gfs_grid())
        if key == "z500" and defect:
            row.update(defect)
        (output/"stack.json").write_text(json.dumps(dict(schema="gpuwm-grib2-stack-v1", outputs=[row])))
    monkeypatch.setattr(cpu_backend, "CpuPreprocessBackend", lambda path: SimpleNamespace(path=bridge, grib2_stack=decode))
    audits = []
    fields, identity, provenance = gfs.analysis_gpu(dict(path=str(source)), "2026-10-17T00:00:00Z",
        np.zeros(2), np.array([0., np.pi/2]), work=tmp_path, audit=audits, bridge=bridge)
    assert fields["z500"][0] == 5000. and np.isnan(fields["z500"][1])
    assert fields["t2m"].tolist() == [284., 285.]
    assert identity.startswith("GFS-f000-") and len(identity) == 73
    assert "surface pressure" in provenance["terrain_support"]
    assert len({r["native_thread_id"] for r in audits[0]["decodes"].values()}) == 3
    for defect in (dict(pdt=8), dict(forecast_step=6), dict(levels=[85000.]),
                   dict(valid_time="2026-10-17T06:00:00Z"), dict(selector_values=[]),
                   dict(sha256="0"*64), dict(file="../escape"), dict(grid=gfs_grid(64))):
        with pytest.raises(ValueError, match="GFS"):
            gfs.analysis_gpu(dict(path=str(source)), "2026-10-17T00:00:00Z", np.zeros(2), np.zeros(2),
                             work=tmp_path, audit=audits, bridge=bridge)
    defect = None
    def fetch(url, valid_time, folder, audit):
        paths = {key: folder/(key+".grib2") for key in gfs.SELECTORS}
        for path in paths.values():
            path.write_bytes(b"mock selected GRIB")
        return paths
    monkeypatch.setattr(gfs, "fetch_ranges", fetch)
    gfs.analysis_gpu(dict(url="https://example.invalid/gfs"), "2026-10-17T00:00:00Z",
        np.zeros(2), np.zeros(2), work=tmp_path, audit=audits, bridge=bridge)
    assert sum(r["path"].endswith(".grib2") for r in audits[-1]["decoded_files_deleted"]) == 3
    assert source.exists() and bridge.exists()
    assert not list(tmp_path.glob("stochastic-gfs-*"))
    assert all(not Path(row["path"]).exists() for audit in audits for row in audit["decoded_files_deleted"])
    (tmp_path/"gfs-transport-audit.json").write_text(json.dumps(audits))


def test_daily_consumer_raw_gfs_path_uses_geometry_radians_and_generated_provenance(monkeypatch, tmp_path):
    import numpy as np
    from hexcore import stochastic_day as day, stochastic_gfs as gfs
    monkeypatch.setitem(sys.modules, "cupy", np)
    inputs, request = day_inputs(np)
    request["analysis"] = dict(format="gfs_grib2", source="GFS", path="raw.grib2")
    for name in ("latCell", "lonCell"):
        inputs["geo"][name] = (np.zeros(2), dict(dimensions=["nCells"], units="radian"))
    monkeypatch.setattr(day, "decode_gpu", lambda path, names, **kw: {n: inputs[path][n] for n in names})
    def remap(spec, valid, latitude, longitude, **kwargs):
        assert valid == request["valid_time"] and latitude.tolist() == [0., 0.]
        assert kwargs["bridge"] == "native.so"
        return dict(z500=np.ones(2), t2m=np.full(2, 280.)), "GFS-f000-fixture", dict(method="fixture")
    monkeypatch.setattr(gfs, "analysis_gpu", remap)
    monkeypatch.setattr(day, "score_pair_gpu", lambda *a, **kw: dict(members=["a", "b"], **kw))
    result = day.score_day(request, reader="rw_netcdf", work=tmp_path, grib_bridge="native.so")
    assert result["analysis_id"] == "GFS-f000-fixture"
    assert result["gfs_remapping_provenance"] == dict(method="fixture")


@pytest.mark.gpu
def test_gpu_gfs_remap_periodic_longitude_and_missing_support():
    if os.environ.get("HEX_STOCHASTIC_GPU_TESTS") != "1":
        pytest.skip("set HEX_STOCHASTIC_GPU_TESTS=1 only on a mutex-granted card")
    cp = pytest.importorskip("cupy")
    from hexcore.stochastic_gfs import remap_gpu
    values = cp.asarray([[90., 100., 110., 120.], [0., 10., 20., 30.], [-90., -80., -70., -60.]])
    result = remap_gpu(values, gfs_grid(), cp.asarray([0., 0.]), cp.asarray([-cp.pi/4, cp.pi/4]))
    assert cp.asnumpy(result).tolist() == pytest.approx([15., 5.])
    values[1, 1] = cp.nan
    result = remap_gpu(values, gfs_grid(), cp.zeros(2), cp.asarray([0., cp.pi/4]))
    assert float(result[0]) == 0 and bool(cp.isnan(result[1]))


def synthetic_gfs_record(key, values, scan):
    """Tiny GRIB2 IEEE/bitmap fixture, following engine test_grib2_stack.py.

    No forecast or GFS observation data. Real Rust parses these sections.
    """
    import math
    import struct
    from hexcore.stochastic_gfs import selector
    def section(number, size):
        result = bytearray(size)
        struct.pack_into(">I", result, 0, size)
        result[4] = number
        return result
    spec, grid = selector(key), gfs_grid(scan)
    s1 = section(1, 21)
    struct.pack_into(">H", s1, 5, 7)  # NCEP centre, synthetic analysis clock.
    struct.pack_into(">H", s1, 12, 2026)
    s1[14:16] = bytes([10, 17])
    s3 = section(3, 72)
    struct.pack_into(">I", s3, 6, 12)
    s3[14] = 6
    struct.pack_into(">II", s3, 30, 4, 3)
    for offset, name in ((46, "lat1"), (50, "lon1"), (55, "lat2"), (59, "lon2")):
        value = round(grid[name]*1e6)
        struct.pack_into(">I", s3, offset, abs(value) | (0x80000000 if value < 0 else 0))
    struct.pack_into(">II", s3, 63, 90000000, 90000000)
    s3[71] = scan
    s4 = section(4, 34)
    s4[9:11] = bytes([spec["category"], spec["parameter"]])
    s4[17] = 1
    s4[22] = spec["level_type"]
    struct.pack_into(">I", s4, 24, int(spec["levels"][0]))
    s4[28] = 255
    finite = [math.isfinite(v) for v in values]
    s5 = section(5, 12)
    struct.pack_into(">I", s5, 5, sum(finite))
    struct.pack_into(">H", s5, 9, 4)  # IEEE template, 64-bit floats.
    s5[11] = 2
    s6 = section(6, 8)
    s6[6] = sum(int(v) << (7-i) for i, v in enumerate(finite[:8]))
    s6[7] = sum(int(v) << (7-i) for i, v in enumerate(finite[8:]))
    packed = b"".join(struct.pack(">d", v) for v in values if math.isfinite(v))
    s7 = section(7, 5+len(packed))
    s7[5:] = packed
    body = b"".join((s1, s3, s4, s5, s6, s7))
    return b"GRIB"+bytes([0, 0, 0, 2])+struct.pack(">Q", len(body)+20)+body+b"7777"


@pytest.mark.parametrize("scan", [0, 64, 128, 192])
def test_native_rust_gfs_stack_selectors_grid_clock_bitmap_and_words(tmp_path, scan):
    if os.environ.get("HEX_STOCHASTIC_RUST_TESTS") != "1":
        pytest.skip("set HEX_STOCHASTIC_RUST_TESTS=1 with an explicitly built engine Rust library")
    import hashlib
    import json
    import math
    from pathlib import Path
    import struct
    import time
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend, CPU_BRIDGE_ENV
    from hexcore.stochastic_gfs import SELECTORS, decode_stacks
    backend = CpuPreprocessBackend(Path(os.environ[CPU_BRIDGE_ENV]))
    expected = {key: [base+i*.125 for i in range(12)]
                for key, base in (("z500", 5000.), ("t2m", 280.), ("psfc", 100000.))}
    expected["z500"][5] = math.nan
    source = tmp_path/"synthetic-analysis.grib2"
    source.write_bytes(b"".join(synthetic_gfs_record(key, values, scan) for key, values in expected.items()))
    passes, started = [], time.perf_counter()
    for repeat in range(2):
        folder = tmp_path/str(repeat)
        folder.mkdir()
        audit = {}
        decoded = decode_stacks({key: source for key in SELECTORS}, backend, folder, audit,
                                "2026-10-17T00:00:00Z")
        words = {}
        for key, payload, record in decoded:
            actual = struct.unpack("<12f", payload.read_bytes())
            assert all((math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(actual, expected[key]))
            assert record["grid"] == gfs_grid(scan)
            words[key] = hashlib.sha256(payload.read_bytes()).hexdigest()
        passes.append(dict(audit=audit, payload_sha256=words))
    assert passes[0]["payload_sha256"] == passes[1]["payload_sha256"]
    (tmp_path/"native-rust-audit.json").write_text(json.dumps(dict(
        synthetic_grib_only=True, real_rust=True, no_gpu_execution=True, no_network=True,
        scan_mode=scan, source_bytes=source.stat().st_size, reader=str(backend.path),
        passes=passes, wall_seconds=time.perf_counter()-started), indent=2))


@pytest.mark.parametrize("container", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF4"])
def test_native_netcdf_transport_cf_clocks_words_and_cleanup(tmp_path, container):
    if os.environ.get("HEX_STOCHASTIC_NETCDF_TESTS") != "1":
        pytest.skip("set HEX_STOCHASTIC_NETCDF_TESTS=1 with an existing engine rw_netcdf")
    import hashlib
    import json
    import math
    import struct
    import time
    import numpy as np
    from hexcore.stochastic_day import decoded_transport, check_clock
    nc = pytest.importorskip("netCDF4")
    source, work = tmp_path/"synthetic-history.nc", tmp_path/"work"
    work.mkdir()
    started = time.perf_counter()
    # Independent C library writes a fixture; Rust is the only reader.
    with nc.Dataset(source, "w", format=container) as dataset:
        for dim, size in (("Time", None), ("nCells", 2), ("nVertLevels", 3), ("StrLen", 64)):
            dataset.createDimension(dim, size)
        p = dataset.createVariable("pressure", "i2", ("Time", "nCells", "nVertLevels"),
                                   fill_value=-32768, zlib=container == "NETCDF4")
        p.units, p.scale_factor, p.add_offset, p.missing_value = "Pa", .5, 100000., np.int16(-32767)
        p.set_auto_maskandscale(False)
        p[0] = np.array([[0, 2, 4], [6, -32768, -32767]], dtype="i2")
        t2 = dataset.createVariable("t2", "f8", ("Time", "nCells"))
        t2.units, t2[0] = "K", [279., 281.]
        ids = dataset.createVariable("indexToCellID", "i4", ("nCells",))
        ids[:] = [11, 12]
        xtime = dataset.createVariable("xtime", "S1", ("Time", "StrLen"))
        xtime[0] = np.frombuffer(b"2026-10-17_00:00:00".ljust(64, b" "), dtype="S1")
        clock = dataset.createVariable("time", "f8", ("Time",))
        clock.units = "hours since 2026-10-16 00:00:00"
        clock[0] = 24.
    names = ["pressure", "t2", "indexToCellID", "xtime", "time"]
    audit, repeats = [], []
    arguments = dict(reader=os.environ["HEX_STOCHASTIC_NETCDF_READER"], work=work, audit=audit)
    expected = dict(pressure=[100000., 100001., 100002., 100003., math.nan, math.nan],
                    t2=[279., 281.], indexToCellID=[11., 12.], time=[24.])
    for repeat in range(2):
        hashes, fields = {}, {}
        with decoded_transport(source, names, **arguments) as payloads:
            assert [record["name"] for _, record in payloads] == names
            for payload, record in payloads:
                raw, name = payload.read_bytes(), record["name"]
                hashes[name] = hashlib.sha256(raw).hexdigest()
                fields[name] = (raw, record)
                if name != "xtime":
                    values = struct.unpack("<"+"d"*len(expected[name]), raw)
                    assert all((math.isnan(a) and math.isnan(b)) or a == b
                               for a, b in zip(values, expected[name]))
            assert fields["pressure"][1]["dimensions"] == ["Time", "nCells", "nVertLevels"]
            assert fields["pressure"][1]["shape"] == [1, 2, 3]
            assert fields["pressure"][1]["units"] == "Pa"
            for name in ("xtime", "time"):
                check_clock(fields, name, "2026-10-17T00:00:00Z")
        assert list(work.iterdir()) == []
        repeats.append(hashes)
    assert repeats[0] == repeats[1]
    # Cleanup must also cover a Rust partial decode and a failing consumer.
    with pytest.raises(RuntimeError, match="Rust NetCDF decode failed"):
        with decoded_transport(source, ["pressure", "absent"], **arguments):
            pytest.fail("missing variable must fail before yielding")
    assert audit[-1]["exit_code"] != 0
    assert list(work.iterdir()) == []
    with pytest.raises(RuntimeError, match="consumer fixture failure"):
        with decoded_transport(source, ["t2"], **arguments):
            raise RuntimeError("consumer fixture failure")
    assert list(work.iterdir()) == []
    (tmp_path/"native-netcdf-audit.json").write_text(json.dumps(dict(
        real_rust=True, synthetic_netcdf_only=True, no_gpu_execution=True,
        container=container, source_bytes=source.stat().st_size,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reader=arguments["reader"], audit=audit, repeated_payload_sha256=repeats,
        wall_seconds=time.perf_counter()-started), indent=2))
