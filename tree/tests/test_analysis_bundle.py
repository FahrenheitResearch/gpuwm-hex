"""The MPAS analysis bundle (hexcore.analysis_bundle): its manifest rules, its byte checks, the
increment and background arithmetic, and the run checks, on tiny host files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore import analysis_bundle as ab

N_CELLS, N_EDGES, N_LEV = 4, 6, 3
VALID = "2022-06-13_23:15:00"
#: An absolute home path a manifest must refuse, built from pieces so the release leak scan
#: (which refuses home paths in shipped bytes) does not read the test's own input as one.
HOME = "/" + "home" + "/someone"


def write_state(path: Path, *, seed: int, valid: str = VALID, zgrid_shift: float = 0.0,
                extra: tuple[str, ...] = (), order: str = "cells-first", numbers: bool = False,
                same_w_as: int | None = None) -> Path:
    """An init-form MPAS file small enough to read by eye."""

    rng = np.random.default_rng(seed)
    with Dataset(path, "w", format="NETCDF3_64BIT_DATA") as d:
        for name, size in (("Time", None), ("StrLen", 64), ("nCells", N_CELLS), ("nEdges", N_EDGES),
                           ("nVertLevels", N_LEV), ("nVertLevelsP1", N_LEV + 1)):
            d.createDimension(name, size)
        x = d.createVariable("xtime", "S1", ("Time", "StrLen"))
        x[0, :] = np.array(list(valid.ljust(64)), dtype="S1")
        cell_dims = ("Time", "nCells", "nVertLevels") if order == "cells-first" else ("Time", "nVertLevels", "nCells")
        fields = {"rho": 1.0 + 0.1 * rng.random((N_CELLS, N_LEV)),
                  "theta": 300.0 + rng.random((N_CELLS, N_LEV)),
                  "qv": 0.01 * rng.random((N_CELLS, N_LEV))}
        for name in ("qc", "qr", "qi", "qs", "qg"):
            fields[name] = 1e-4 * rng.random((N_CELLS, N_LEV))
        for name, values in fields.items():
            v = d.createVariable(name, "f4", cell_dims)
            v[0] = values if order == "cells-first" else values.T
        u = d.createVariable("u", "f4", ("Time", "nEdges", "nVertLevels"))
        u[0] = 10 * rng.standard_normal((N_EDGES, N_LEV))
        w = d.createVariable("w", "f4", ("Time", "nCells", "nVertLevelsP1"))
        w_rng = rng if same_w_as is None else np.random.default_rng(same_w_as)
        w[0] = w_rng.standard_normal((N_CELLS, N_LEV + 1))
        if numbers:
            nr = d.createVariable("nr", "f4", ("Time", "nCells", "nVertLevels"))
            nr[0] = 1e4 * rng.random((N_CELLS, N_LEV))
        land = d.createVariable("skintemp", "f4", ("Time", "nCells"))
        land[0] = 290.0 + rng.random(N_CELLS)
        z = d.createVariable("zgrid", "f4", ("nCells", "nVertLevelsP1"))
        z[:] = np.tile(np.array([0.0, 1000.0, 3000.0, 6000.0]) + zgrid_shift, (N_CELLS, 1))
        for name in ("zz", "rho_base", "theta_base"):
            v = d.createVariable(name, "f4", ("Time", "nCells", "nVertLevels") if name != "zz" else ("nCells", "nVertLevels"))
            if name == "zz":
                v[:] = np.ones((N_CELLS, N_LEV))
            else:
                v[0] = np.ones((N_CELLS, N_LEV))
        ids = d.createVariable("indexToCellID", "i4", ("nCells",))
        ids[:] = np.arange(1, N_CELLS + 1)
        for name in extra:
            v = d.createVariable(name, "f4", ("Time", "nCells", "nVertLevels"))
            v[0] = np.ones((N_CELLS, N_LEV))
    return path


def write_grid(path: Path, cells: int = N_CELLS) -> Path:
    with Dataset(path, "w") as d:
        d.createDimension("nCells", cells)
        d.createVariable("areaCell", "f8", ("nCells",))[:] = np.arange(cells) + 1.0
    return path


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    return {"analysis": write_state(root / "an.nc", seed=1), "background": write_state(root / "bg.nc", seed=2),
            "grid": write_grid(root / "mesh.grid.nc"), "static": _bytes(root / "mesh.static.nc", b"static")}


def _bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def make(tmp_path, files, *, increment=True, name="bundle", **overrides):
    kwargs = dict(analysis=files["analysis"], grid=files["grid"], static=files["static"], mesh_id="test-mesh",
                  static_builder="rw_mpas_static test", physics_suite="arwen_mpas_column", physics_mode="correct",
                  kind="mean", members=36, recipe_id="recipe.v1", recipe_sha256="a" * 64, cycles=5, window=900.0,
                  producer="tests", background=files["background"] if increment else None,
                  iau_window_s=600.0 if increment else None)
    kwargs.update(overrides)
    return ab.make_bundle(tmp_path / name, **kwargs)


def test_a_made_bundle_reads_back_with_every_file_checked(tmp_path, files):
    bundle = make(tmp_path, files)
    assert bundle.manifest["schema"] == ab.SCHEMA
    assert bundle.valid_time == VALID
    assert set(bundle.files) == {ab.ANALYSIS, ab.INCREMENT}
    assert bundle.iau_window_seconds == 600.0
    assert bundle.form == "init"
    reread = ab.load_bundle(bundle.root / ab.MANIFEST)
    assert reread.manifest_sha256 == bundle.manifest_sha256
    assert "valid 2022-06-13_23:15:00" in bundle.provenance_sentence()


def test_the_increment_is_analysis_minus_background_in_mpas_iau_form(tmp_path, files):
    bundle = make(tmp_path, files)
    increments = ab.read_increment(bundle.increment, n_cells=N_CELLS, n_edges=N_EDGES, n_levels=N_LEV)
    assert set(increments) == {name for name, _, _ in ab.IAU_FIELDS}
    with Dataset(files["analysis"]) as a, Dataset(files["background"]) as b:
        for name, source, entity in ab.IAU_FIELDS:
            expect = (np.asarray(a[source][0], np.float64) - np.asarray(b[source][0], np.float64)).T
            np.testing.assert_array_equal(increments[name], expect.astype(np.float32))
    with Dataset(bundle.increment) as d:
        assert d["u_amb"].dimensions == ("Time", "nEdges", "nVertLevels")
        assert d["rho_amb"].dimensions == ("Time", "nCells", "nVertLevels")
        assert d["u_amb"].dtype == np.float64, "float64 by default: the background is rebuilt exactly"
        # every other differing prognostic carries its own X_amb; a 2-D land field never does
        assert d["w_amb"].dimensions == ("Time", "nCells", "nVertLevelsP1")
        assert "skintemp_amb" not in d.variables
    assert ab.increment_variables(bundle.increment, files["analysis"])["w_amb"] == "w"


def test_the_background_is_rebuilt_bit_for_bit_from_a_float64_increment(tmp_path, files):
    bundle = make(tmp_path, files)
    out = tmp_path / "rebuilt.nc"
    receipt = ab.write_background(bundle.analysis, bundle.increment, out)
    assert set(receipt["changed"]) == {"u", "rho", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "w"}
    with Dataset(out) as r, Dataset(files["background"]) as b, Dataset(files["analysis"]) as a:
        for name in receipt["changed"]:
            assert np.asarray(r[name][0]).tobytes() == np.asarray(b[name][0]).tobytes(), name
        # a field with no increment (the land state) is the analysis's own
        np.testing.assert_array_equal(np.asarray(r["skintemp"][0]), np.asarray(a["skintemp"][0]))


def test_a_float32_increment_rebuilds_the_background_to_rounding(tmp_path, files):
    bundle = make(tmp_path, files, increment_precision="f4")
    with Dataset(bundle.increment) as d:
        assert d["theta_amb"].dtype == np.float32
    out = tmp_path / "rebuilt.nc"
    ab.write_background(bundle.analysis, bundle.increment, out)
    with Dataset(out) as r, Dataset(files["background"]) as b:
        for name in ("theta", "u", "qv", "w"):
            np.testing.assert_allclose(np.asarray(r[name][0]), np.asarray(b[name][0]), rtol=1e-6, atol=1e-9)


def test_number_concentrations_carry_their_own_increment(tmp_path):
    root = tmp_path / "n"
    root.mkdir()
    an = write_state(root / "an.nc", seed=1, numbers=True)
    bg = write_state(root / "bg.nc", seed=2, numbers=True)
    receipt = ab.write_increment(an, bg, root / "inc.nc")
    assert "nr_amb" in receipt["rms"] and receipt["exact_background_all"] is True
    mpas_only = ab.write_increment(an, bg, root / "inc-mpas.nc", fields="mpas")
    assert set(mpas_only["rms"]) == {name for name, _, _ in ab.IAU_FIELDS}
    out = root / "rebuilt.nc"
    ab.write_background(an, root / "inc.nc", out)
    with Dataset(out) as r, Dataset(bg) as b:
        assert np.asarray(r["nr"][0]).tobytes() == np.asarray(b["nr"][0]).tobytes()


def test_an_increment_for_a_variable_the_analysis_lacks_is_refused(tmp_path, files):
    root = tmp_path / "n"
    root.mkdir()
    an = write_state(root / "an.nc", seed=1, numbers=True)
    bg = write_state(root / "bg.nc", seed=2, numbers=True)
    ab.write_increment(an, bg, root / "inc.nc")
    with pytest.raises(ab.BundleRefusal, match="has no nr"):
        ab.write_background(files["analysis"], root / "inc.nc", root / "bg-out.nc")


def test_any_changed_byte_is_refused_by_name(tmp_path, files):
    bundle = make(tmp_path, files)
    data = bytearray(bundle.analysis.read_bytes())
    data[-1] ^= 1
    bundle.analysis.write_bytes(bytes(data))
    with pytest.raises(ab.BundleRefusal, match="analysis.nc has SHA-256"):
        ab.load_bundle(bundle.root)


def test_a_truncated_file_is_refused_by_size(tmp_path, files):
    bundle = make(tmp_path, files)
    bundle.increment.write_bytes(bundle.increment.read_bytes()[:-8])
    with pytest.raises(ab.BundleRefusal, match="increment.nc is .* bytes"):
        ab.load_bundle(bundle.root)


def test_an_unlisted_bundle_file_is_refused(tmp_path, files):
    bundle = make(tmp_path, files, increment=False)
    _bytes(bundle.root / ab.NODA, b"noda")
    with pytest.raises(ab.BundleRefusal, match="not listed in the manifest"):
        ab.load_bundle(bundle.root)


def _manifest(bundle):
    return json.loads((bundle.root / ab.MANIFEST).read_text())


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(schema="mpas-analysis-bundle/2"), "schema"),
    (lambda m: m.update(extra=1), "does not know"),
    (lambda m: m["analysis"].update(recipe_id=HOME + "/recipe"), "path or address"),
    (lambda m: m["static"].update(builder="C:\\builds\\static"), "path or address"),
    (lambda m: m["mesh"].update(sha256="XYZ"), "SHA-256"),
    (lambda m: m.update(valid_time="2022-06-13T23:15Z"), "valid_time"),
    (lambda m: m.update(notes=["a list"]), "notes must be an object"),
    (lambda m: m.update(forecast={"args": ["--samehost-devices", "0,1"]}), "only model configuration"),
    (lambda m: m.update(forecast={"args": ["--dt"]}), "takes a value"),
    (lambda m: m.update(forecast={"args": ["--physics", HOME + "/namelist"]}), "path or address"),
    (lambda m: m.update(forecast={"args": "--dt 18"}), "list of strings"),
    (lambda m: m.update(forecast={"args": [], "extra": 1}), "does not know"),
    (lambda m: m.update(notes={"where": HOME + "/run"}), "path or address"),
    (lambda m: m["physics"].update(mode="loose"), "physics.mode"),
    (lambda m: m["analysis"].update(kind="median"), "analysis.kind"),
    (lambda m: m.pop("iau"), "go together"),
    (lambda m: m["mesh"]["vertical_spec"].update(levels=4), "differ"),
    (lambda m: m["files"].update({"other.nc": {"sha256": "b" * 64, "bytes": 1}}), "only"),
])
def test_manifest_rules(tmp_path, files, mutate, match):
    bundle = make(tmp_path, files)
    manifest = _manifest(bundle)
    mutate(manifest)
    with pytest.raises(ab.BundleRefusal, match=match):
        ab.validate_manifest(manifest)


def test_an_mpasout_form_analysis_carries_no_increment(tmp_path, files):
    bundle = make(tmp_path, files)
    manifest = _manifest(bundle)
    manifest["analysis"]["form"] = "mpasout"
    with pytest.raises(ab.BundleRefusal, match="mpasout-form"):
        ab.validate_manifest(manifest)


def test_the_run_checks_name_the_mesh_the_static_and_the_vertical(tmp_path, files):
    bundle = make(tmp_path, files)
    checks = ab.check_run(bundle, grid=files["grid"], static=files["static"], levels=N_LEV)
    assert checks["grid_sha256"] == bundle.manifest["mesh"]["sha256"]
    other_grid = write_grid(tmp_path / "other.grid.nc", cells=5)
    with pytest.raises(ab.BundleRefusal, match="the analysis is on mesh test-mesh"):
        ab.check_run(bundle, grid=other_grid, static=files["static"])
    with pytest.raises(ab.BundleRefusal, match="static SHA-256"):
        ab.check_run(bundle, grid=files["grid"], static=_bytes(tmp_path / "s2.nc", b"other"))
    with pytest.raises(ab.BundleRefusal, match="levels"):
        ab.check_run(bundle, grid=files["grid"], static=files["static"], levels=N_LEV + 1)
    taller = write_state(tmp_path / "taller.nc", seed=1, zgrid_shift=50.0)
    with pytest.raises(ab.BundleRefusal, match="vertical grid"):
        ab.check_run(bundle, grid=files["grid"], static=files["static"], vertical_sources=[taller])


def test_the_vertical_fingerprint_ignores_storage_order(tmp_path):
    a = write_state(tmp_path / "a.nc", seed=1)
    b = write_state(tmp_path / "b.nc", seed=3, order="levels-first")
    assert ab.vertical_fingerprint(a)["zgrid_sha256"] == ab.vertical_fingerprint(b)["zgrid_sha256"]


def test_negative_water_left_by_a_float32_increment_is_zeroed_and_counted(tmp_path, files):
    bundle = make(tmp_path, files, increment_precision="f4")
    increment = tmp_path / "big.nc"
    with Dataset(bundle.increment) as src, Dataset(increment, "w", format="NETCDF3_64BIT_DATA") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, var in src.variables.items():
            out = dst.createVariable(name, var.dtype, var.dimensions)
            out[:] = var[:]
        dst["qc_amb"][0] = np.full((N_CELLS, N_LEV), 1.0, np.float32)
    receipt = ab.write_background(bundle.analysis, increment, tmp_path / "bg.nc")
    assert receipt["clipped_negative"]["qc"]["values"] == N_CELLS * N_LEV
    with Dataset(tmp_path / "bg.nc") as d:
        assert float(np.min(d["qc"][0])) == 0.0


def test_a_coupled_alias_in_the_analysis_is_refused_for_a_background(tmp_path, files):
    analysis = write_state(tmp_path / "coupled.nc", seed=1, extra=("rho_zz",))
    out = tmp_path / "inc.nc"
    ab.write_increment(analysis, files["background"], out)
    with pytest.raises(ab.BundleRefusal, match="rho_zz"):
        ab.write_background(analysis, out, tmp_path / "bg.nc")


def test_an_analysis_at_another_time_is_refused(tmp_path, files):
    write_state(files["analysis"], seed=1, valid="2022-06-13_23:30:00")
    with pytest.raises(ab.BundleRefusal, match="no record at the valid time"):
        ab.check_analysis_file(files["analysis"], valid_time=VALID)


def test_the_command_line_verify_reports_refusals(tmp_path, files, capsys):
    bundle = make(tmp_path, files)
    assert ab.main(["verify", str(bundle.root)]) == 0
    bundle.analysis.write_bytes(b"x")
    assert ab.main(["verify", str(bundle.root)]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_the_manifest_writes_iso_utc_and_reads_either_form(tmp_path, files):
    bundle = make(tmp_path, files)
    manifest = _manifest(bundle)
    assert manifest["valid_time"] == "2022-06-13T23:15:00Z"
    assert bundle.valid_time == VALID, "the door's --start-time is MPAS's clock text"
    manifest["valid_time"] = VALID
    assert ab.validate_manifest(manifest)["valid_time"] == VALID


def test_notes_are_recorded_and_never_read(tmp_path, files):
    bundle = make(tmp_path, files, notes={"producer_run": "abc", "cycles_kept": [1, 2]})
    assert bundle.receipt()["notes"] == {"producer_run": "abc", "cycles_kept": [1, 2]}


def _rows_document(tmp_path, files, name="test-mesh"):
    from hexcore.mesh_rows import MeshRow, write_rows

    receipt = tmp_path / "generator.json"
    receipt.write_text("{}")
    row = MeshRow.from_dict(dict(
        kind="generated-global", name=name, n_cells=N_CELLS, n_edges=N_EDGES, n_levels=N_LEV,
        n_interfaces=N_LEV + 1, n_soil_levels=4, nominal_dx_m=3000.0, dt_seconds=18.75,
        grid=str(files["grid"]), grid_bytes=files["grid"].stat().st_size, grid_sha256=ab.sha256_file(files["grid"]),
        static=str(files["static"]), static_bytes=files["static"].stat().st_size,
        static_sha256=ab.sha256_file(files["static"]), spec_sha256="d" * 64, generator_receipt=str(receipt),
        static_receipt=None, point_deg=(41.8, -85.2), fine_dx_m=3000.0, core_radius_km=450.0,
        background_km=96.0, admission={"scope": "test"}))
    return write_rows(tmp_path / "rows" / "mesh-rows.json", [row])


def test_a_bundle_carries_its_mesh_and_the_row_that_binds_it(tmp_path, files):
    rows = _rows_document(tmp_path, files)
    bundle = make(tmp_path, files, carry_mesh=True, mesh_rows=rows)
    assert set(bundle.files) == {ab.ANALYSIS, ab.INCREMENT, ab.GRID, ab.STATIC, ab.MESH_ROW}
    carried = json.loads(bundle.carried_row.read_text())["rows"][0]
    assert carried["grid"] == ab.GRID and carried["static"] == ab.STATIC
    assert carried["generator_receipt"].startswith("sha256:"), "a carried row holds no producer paths"
    bound = ab.bind_carried_mesh(bundle, tmp_path / "work")
    from hexcore.mesh_rows import read_rows

    (row,) = read_rows(Path(bound["rows"]))
    assert row.name == "test-mesh" and Path(row.grid) == bundle.carried_grid
    assert Path(row.static) == bundle.carried_static and Path(row.static).is_absolute()


def test_a_carried_static_that_is_also_the_grid_needs_no_grid_file(tmp_path, files):
    same = dict(files, static=files["grid"])
    bundle = make(tmp_path, same, carry_mesh=True)
    assert ab.GRID not in bundle.files and bundle.carried_grid == bundle.carried_static


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["files"].pop(ab.STATIC), "without static.nc"),
    (lambda m: m["static"].update(sha256="e" * 64), "static.sha256 says"),
    (lambda m: m["files"].pop(ab.GRID), "serves as the grid only"),
])
def test_carried_mesh_rules(tmp_path, files, mutate, match):
    bundle = make(tmp_path, files, carry_mesh=True, mesh_rows=_rows_document(tmp_path, files))
    manifest = _manifest(bundle)
    mutate(manifest)
    with pytest.raises(ab.BundleRefusal, match=match):
        ab.validate_manifest(manifest)


def test_a_carried_row_with_a_path_or_another_name_is_refused(tmp_path, files):
    bundle = make(tmp_path, files, carry_mesh=True, mesh_rows=_rows_document(tmp_path, files))
    manifest = _manifest(bundle)
    document = json.loads(bundle.carried_row.read_text())
    document["rows"][0]["generator_receipt"] = HOME + "/receipt.json"
    bundle.carried_row.write_text(json.dumps(document))
    with pytest.raises(ab.BundleRefusal, match="path or address"):
        ab.read_carried_row(bundle.carried_row, manifest, bundle.root)
    document["rows"][0].update(generator_receipt="sha256:" + "f" * 64, name="another-mesh")
    bundle.carried_row.write_text(json.dumps(document))
    with pytest.raises(ab.BundleRefusal, match="names row 'another-mesh'"):
        ab.read_carried_row(bundle.carried_row, manifest, bundle.root)


def test_a_float64_increment_keeps_the_producers_own_negative_roundoff(tmp_path, files):
    # transport leaves mixing ratios a hair below zero in a forecast-made background where the
    # analysis has none; the rebuilt background must be that background, not a clipped copy
    for path, value in ((files["background"], -2.0e-19), (files["analysis"], 0.0)):
        with Dataset(path, "r+") as d:
            values = np.asarray(d["qr"][0])
            values[0, 0] = value
            d["qr"][0] = values
    bundle = make(tmp_path, files)
    out = tmp_path / "rebuilt.nc"
    receipt = ab.write_background(bundle.analysis, bundle.increment, out)
    assert receipt["negative_kept_from_producer"]["qr"]["values"] == 1 and not receipt["clipped_negative"]
    with Dataset(out) as r, Dataset(files["background"]) as b:
        assert np.asarray(r["qr"][0]).tobytes() == np.asarray(b["qr"][0]).tobytes()


def test_exactness_is_measured_per_variable_where_float64_cannot_carry_the_background(tmp_path, files):
    # a background value below the analysis value's float64 resolution (rain where the background
    # had none) cannot survive analysis - increment: the receipt says so instead of claiming it
    for path, value in ((files["background"], 1.0e-20), (files["analysis"], 6.7e-3)):
        with Dataset(path, "r+") as d:
            values = np.asarray(d["qr"][0])
            values[0, 0] = value
            d["qr"][0] = values
    receipt = ab.write_increment(files["analysis"], files["background"], tmp_path / "inc.nc")
    assert receipt["exact_background"]["qr_amb"] is False and receipt["exact_background"]["theta_amb"] is True
    assert receipt["exact_background_all"] is False


def test_the_producers_forecast_configuration_is_carried(tmp_path, files):
    args = ["--dt", "18.75", "--fullstep-local-timestep", "--fullstep-lts-mode", "active",
            "--fullstep-lts-max-rate", "8", "--physics", "config_microp_scheme=mp_thompson"]
    bundle = make(tmp_path, files, forecast_args=args)
    assert bundle.manifest["forecast"] == {"args": args}
    assert ab.parse_forecast_args(args)[1] == ("--fullstep-local-timestep", None)
