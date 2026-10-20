"""The RRFS v2 JEDI file contract (stage H6, tier T1): writer, reader, soil map.

The tiny arm always runs on the CPU.  The real-init arm (the DA-cycling
recoupling against hex's own cold-start coupling on a real MPAS init) runs
when HEXCORE_DA_INIT names an init.nc and HEXCORE_DA_GRID / HEXCORE_DA_STATIC
name its mesh files.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore import jedi_contract as jc


def test_stream_lists_are_the_public_ones():
    assert len(jc.BACKGROUND_FIELDS) == 27
    assert len(jc.ANALYSIS_FIELDS) == 12
    assert set(jc.ANALYSIS_FIELDS) <= set(jc.BACKGROUND_FIELDS)
    assert set(jc.RECOUPLED_INPUTS) <= set(jc.ANALYSIS_FIELDS)
    assert set(jc.FIELD_ROLES) == set(jc.BACKGROUND_FIELDS)


def test_interim_soil_mapping_is_labelled_bounded_and_exact_on_constants():
    rng = np.random.default_rng(3)
    four = rng.uniform(0.1, 0.4, size=(4, 11)).astype(np.float32)
    four[:, 0] = 0.25
    nine = jc.interim_soil_levels(four)
    assert nine.shape == (9, 11) and nine.dtype == np.float32
    assert np.all(nine[:, 0] == np.float32(0.25))
    assert np.all(nine >= four.min(axis=0) - 1e-7)
    assert np.all(nine <= four.max(axis=0) + 1e-7)
    # 0, 0.01 and 0.04 m lie above the first centre (0.05 m): held constant.
    assert np.array_equal(nine[0], four[0]) and np.array_equal(nine[2], four[0])
    # 3.0 m lies below the last centre (1.5 m): held constant.
    assert np.array_equal(nine[8], four[3])
    # 0.10 m: a fifth of the way from 0.05 to 0.25.
    np.testing.assert_allclose(nine[3], four[0] + 0.25 * (four[1] - four[0]), rtol=1e-6)
    assert "INTERIM" in jc.INTERIM_SOIL_LABEL
    with pytest.raises(jc.ContractRefusal):
        jc.interim_soil_levels(np.zeros((9, 3), np.float32))


def test_monthly_to_date_is_mid_month_linear():
    table = np.arange(12, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)
    # 2026-10-03: 18 of 30 days from 15 Sep (index 8) to 15 Oct (index 9).
    value = jc.monthly_to_date(table, datetime(2026, 10, 3, 12))
    np.testing.assert_allclose(value, 8.0 + 18.0 / 30.0, rtol=1e-6)
    value = jc.monthly_to_date(table, datetime(2026, 10, 15))
    np.testing.assert_allclose(value, 9.0, rtol=1e-6)


def _tiny_fields(n_cells=7, n_edges=19, n_levels=3, soil=9, seed=0, native_soil=4):
    rng = np.random.default_rng(seed)
    cell3 = (n_levels, n_cells)
    snap = {name: rng.uniform(1e-6, 1e-2, cell3).astype(np.float32) for name in jc.DEFAULT_SCALAR_NAMES}
    snap.update(
        theta=rng.uniform(280, 320, cell3).astype(np.float32),
        rho=rng.uniform(0.3, 1.2, cell3).astype(np.float32),
        surface_pressure=rng.uniform(9e4, 1.02e5, n_cells).astype(np.float32),
        normal_u=rng.normal(0, 10, (n_levels, n_edges + 1)).astype(np.float32),
        w=rng.normal(0, 0.1, (n_levels + 1, n_cells)).astype(np.float32),
        u_zonal=rng.normal(0, 10, cell3).astype(np.float32),
        v_meridional=rng.normal(0, 10, cell3).astype(np.float32),
        tsk=rng.uniform(270, 310, n_cells).astype(np.float32),
        t2=rng.uniform(270, 310, n_cells).astype(np.float32),
        q2=rng.uniform(1e-3, 2e-2, n_cells).astype(np.float32),
        u10=rng.normal(0, 5, n_cells).astype(np.float32),
        v10=rng.normal(0, 5, n_cells).astype(np.float32),
        refl10cm=rng.uniform(-35, 50, cell3).astype(np.float32),
        smois=rng.uniform(0.05, 0.45, (native_soil, n_cells)).astype(np.float32),
        tslb=rng.uniform(270, 300, (native_soil, n_cells)).astype(np.float32),
    )
    seam = {
        "snowh": rng.uniform(0, 1, (1, 1, n_cells + 1)).astype(np.float32),
        "cldfra": rng.uniform(0, 1, (n_levels, 1, n_cells)).astype(np.float32),
    }
    fallback = {
        "snowc": np.zeros(n_cells, np.float32),
        "xice": np.zeros(n_cells, np.float32),
        "vegfra": np.full(n_cells, 40.0, np.float32),
        "lai12m": np.ones((12, n_cells), np.float32),
    }
    fields, provenance = jc.assemble_background_fields(
        snapshot_arrays=snap,
        pressure_perturbation=rng.normal(0, 100, (n_levels, n_cells + 1)).astype(np.float32),
        pressure_base=rng.uniform(2e4, 1e5, cell3).astype(np.float32),
        seam_fields=seam,
        init_fallback=fallback,
        scalar_names=jc.DEFAULT_SCALAR_NAMES,
        n_cells=n_cells,
        n_edges=n_edges,
        n_levels=n_levels,
        soil_levels=soil,
        valid=datetime(2026, 10, 3, 15),
    )
    return fields, provenance, snap


def test_assemble_names_every_field_and_its_source():
    fields, provenance, snap = _tiny_fields()
    expected = [name for name in jc.BACKGROUND_FIELDS if name != "scalars"] + list(jc.DEFAULT_SCALAR_NAMES) + list(jc.RRFS_EXTRA_SCALARS)
    assert sorted(fields) == sorted(expected)
    assert set(provenance) == set(fields)
    assert provenance["snowh"].startswith("physics: hex physics field")
    assert provenance["cldfrac"].startswith("physics: hex physics field")
    assert provenance["lai"].startswith("init: lai12m")
    assert provenance["re_cloud"].startswith("absent:")
    assert "INTERIM" in provenance["smois"]
    assert fields["snowh"].shape == (7,)  # garbage column stripped
    assert np.array_equal(fields["u"], snap["normal_u"][:, :19])  # garbage edge stripped
    assert fields["pressure_p"].shape == (3, 7)


def test_native_nine_level_soil_is_written_unmapped_and_labelled_native(tmp_path):
    """A suite row (stage H4a) integrates RUC's nine levels: no interim map."""

    from netCDF4 import Dataset

    fields, provenance, snap = _tiny_fields(native_soil=9)
    assert np.array_equal(fields["smois"], snap["smois"])
    assert np.array_equal(fields["tslb"], snap["tslb"])
    assert "INTERIM" not in provenance["smois"] and "no mapping" in provenance["smois"]
    record = jc.write_background(
        tmp_path / "mpasout.nc", fields=fields, provenance=provenance,
        valid=datetime(2026, 10, 3, 15), n_cells=7, n_edges=19, n_levels=3,
        soil_levels=9, scalar_names=jc.DEFAULT_SCALAR_NAMES,
    )
    with Dataset(record.path) as dataset:
        label = dataset.getncattr("hex_soil_mapping")
    assert "INTERIM" not in label and "9 levels" in label
    four, four_provenance, _ = _tiny_fields()
    assert jc.soil_mapping_label(four_provenance) == jc.INTERIM_SOIL_LABEL


def test_background_writes_no_negative_zero(tmp_path):
    """A zero increment added by JEDI returns +0.0; the file must already hold it."""

    from netCDF4 import Dataset

    fields, provenance, _ = _tiny_fields()
    fields["u"] = fields["u"].copy()
    fields["u"][:, :5] = np.float32(-0.0)
    record = jc.write_background(
        tmp_path / "mpasout.nc", fields=fields, provenance=provenance,
        valid=datetime(2026, 10, 3, 15), n_cells=7, n_edges=19, n_levels=3,
        soil_levels=9, scalar_names=jc.DEFAULT_SCALAR_NAMES,
    )
    with Dataset(record.path) as dataset:
        u = np.asarray(dataset.variables["u"][:])
    assert not np.any(np.signbit(u) & (u == 0))
    assert np.array_equal(u[0].T.view(np.uint32), (fields["u"] + np.float32(0)).view(np.uint32))


def test_background_is_native_mpas_cdf5_and_reads_back_exactly(tmp_path):
    from netCDF4 import Dataset

    fields, provenance, _ = _tiny_fields()
    record = jc.write_background(
        tmp_path / "mpasout.nc",
        fields=fields,
        provenance=provenance,
        valid=datetime(2026, 10, 3, 15),
        n_cells=7,
        n_edges=19,
        n_levels=3,
        soil_levels=9,
        scalar_names=jc.DEFAULT_SCALAR_NAMES,
    )
    path = Path(record.path)
    assert not list(tmp_path.glob("*.part"))
    with Dataset(path) as dataset:
        assert dataset.data_model == "NETCDF3_64BIT_DATA"
        assert dataset.dimensions["Time"].isunlimited()
        assert {name: len(dim) for name, dim in dataset.dimensions.items()} == {
            "Time": 1, "StrLen": 64, "nCells": 7, "nEdges": 19,
            "nVertLevels": 3, "nVertLevelsP1": 4, "nSoilLevels": 9,
            "nLags": 140, "nOznLevels": 59,
        }
        assert dataset.variables["u"].dimensions == ("Time", "nEdges", "nVertLevels")
        assert dataset.variables["w"].dimensions == ("Time", "nCells", "nVertLevelsP1")
        assert dataset.variables["smois"].dimensions == ("Time", "nCells", "nSoilLevels")
        assert dataset.variables["t2m"].dimensions == ("Time", "nCells")
        xtime = b"".join(dataset.variables["xtime"][0].data).decode().strip()
        assert xtime == "2026-10-03_15:00:00"
        assert "INTERIM" in dataset.variables["tslb"].hex_provenance
    back = jc.read_contract_fields(
        path, jc.BACKGROUND_FIELDS, scalar_names=jc.DEFAULT_SCALAR_NAMES + jc.RRFS_EXTRA_SCALARS
    )
    for name, value in fields.items():
        assert np.array_equal(back[name].view(np.uint32), value.view(np.uint32)), name
    receipt = json.loads(Path(record.receipt_path).read_text())
    assert receipt["sha256"] == record.sha256
    assert set(receipt["fields"]) == set(fields)
    with pytest.raises(FileExistsError):
        jc.write_background(
            path, fields=fields, provenance=provenance,
            valid=datetime(2026, 10, 3, 15), n_cells=7, n_edges=19,
            n_levels=3, soil_levels=9, scalar_names=jc.DEFAULT_SCALAR_NAMES,
        )


def test_reader_refuses_a_missing_contract_field(tmp_path):
    from netCDF4 import Dataset

    with Dataset(tmp_path / "x.nc", "w", format="NETCDF3_64BIT_DATA") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("nCells", 2)
        dataset.createVariable("t2m", "f4", ("Time", "nCells"))[:] = np.ones((1, 2))
    with pytest.raises(jc.ContractRefusal, match="q2"):
        jc.read_contract_fields(tmp_path / "x.nc", ("t2m", "q2"))


def test_compare_fields_counts_ulps():
    a = np.array([1.0, 2.0, -3.0], np.float32)
    b = a.copy()
    b[1] = np.nextafter(b[1], np.float32(10))
    b[2] = np.nextafter(np.nextafter(b[2], np.float32(10)), np.float32(10))
    report = jc.compare_fields({"x": a}, {"x": b})["x"]
    assert not report["identical"]
    assert report["changed_values"] == 2
    assert report["max_ulp"] == 2
    assert jc.compare_fields({"x": a}, {"x": a.copy()})["x"]["identical"]


@pytest.mark.skipif(
    not all(os.environ.get(key) for key in ("HEXCORE_DA_INIT", "HEXCORE_DA_GRID", "HEXCORE_DA_STATIC")),
    reason="set HEXCORE_DA_INIT/GRID/STATIC to a real MPAS init and its mesh files",
)
def test_da_cycling_recouple_reproduces_the_cold_start_coupling(tmp_path):
    """Recoupling the init's own uncoupled fields gives the cold start's state.

    The cold start consumes the init's pressure_p while the DA-cycling path
    recomputes it (upstream atm_init_coupled_diagnostics does), so the saved
    pressure perturbation is held to a rounding bound; every prognostic and
    every other saved diagnostic must be bit-identical.
    """

    from netCDF4 import Dataset

    from hexcore.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from hexcore.mesh import load_precision_preserving_mesh_pair

    init = Path(os.environ["HEXCORE_DA_INIT"])
    mesh, _, _ = load_precision_preserving_mesh_pair(
        Path(os.environ["HEXCORE_DA_GRID"]), Path(os.environ["HEXCORE_DA_STATIC"])
    )
    regional = bool(getattr(mesh, "is_regional", False))
    native = load_mpas_vertical_grid(init, mesh, allow_regional_sentinels=regional)
    names = jc.DEFAULT_SCALAR_NAMES
    state0, reference0, saved0 = load_mpas_initial_state(
        init, mesh, native.vertical_grid, scalar_names=names,
        terrain_metrics=native.terrain_metrics, return_saved_diagnostics=True,
        allow_regional_sentinels=regional,
    )
    with Dataset(init) as dataset:
        n_cells = len(dataset.dimensions["nCells"])
        n_edges = len(dataset.dimensions["nEdges"])
        n_levels = len(dataset.dimensions["nVertLevels"])
        valid_time = b"".join(dataset.variables["xtime"][0]).decode().strip()
    # The analysis is the complete init: its own uncoupled fields, its xtime and
    # its element identity, as a JEDI write-back into a background carries them.
    path = tmp_path / "analysis.nc"
    import shutil

    shutil.copyfile(init, path)
    common = dict(
        init_path=init, mesh=mesh, vertical=native.vertical_grid,
        terrain_metrics=native.terrain_metrics, expected_reference=reference0,
        scalar_names=names, n_cells=n_cells, n_edges=n_edges, n_levels=n_levels,
        allow_regional_sentinels=regional, time_seconds=0.0, work_dir=tmp_path,
    )
    # OU1: another valid time is refused before anything is written.
    later = (datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S") + timedelta(hours=1)).strftime("%Y-%m-%d_%H:%M:%S")
    with pytest.raises(jc.ContractRefusal, match="expected exactly one record"):
        jc.recouple_da_cycling(path, expected_valid_time=later, **common)
    assert not (tmp_path / "da-cycling-uncoupled.nc").exists()
    state1, saved1, receipt = jc.recouple_da_cycling(path, expected_valid_time=valid_time, **common)
    assert receipt["analysis_record"] == 0 and receipt["analysis_identity_equals_init"] is True
    for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
        assert np.array_equal(getattr(state0, name), getattr(state1, name)), name
    for name in ("theta_m", "exner", "density_perturbation", "rho_theta_perturbation",
                 "normal_velocity", "vertical_velocity"):
        assert np.array_equal(getattr(saved0, name), getattr(saved1, name)), name
    report = jc.compare_fields(
        {"pressure_p": saved0.pressure_perturbation}, {"pressure_p": saved1.pressure_perturbation}
    )["pressure_p"]
    (tmp_path / "recouple-report.json").write_text(json.dumps(report))
    print("RECOUPLE pressure_p", json.dumps(report))
    assert report["max_abs"] < 1.0
    assert receipt["reference_identical_to_run"] is True


def test_uncoupled_state_is_written_by_the_rust_writer_and_reads_back_exactly(tmp_path):
    """The DA-cycling restart's temporary file goes through the Rust writer.

    netCDF4-python reads it here only as an independent oracle; the product
    path writes and reads through the engine's Rust bridges.
    """

    from netCDF4 import Dataset

    rng = np.random.default_rng(3)
    names = jc.DEFAULT_SCALAR_NAMES
    analysis = {
        "rho": rng.random((3, 7), dtype=np.float32),
        "theta": rng.random((3, 7), dtype=np.float32),
        "u": rng.random((3, 19), dtype=np.float32),
        "w": rng.random((4, 7), dtype=np.float32),
        **{name: rng.random((3, 7), dtype=np.float32) for name in names},
    }
    path = tmp_path / "uncoupled.nc"
    jc.write_uncoupled_state(path, analysis, scalar_names=names, n_cells=7, n_edges=19, n_levels=3)
    with Dataset(path) as dataset:
        assert dataset.data_model == "NETCDF3_64BIT_DATA"
        assert dataset.variables["u"].dimensions == ("Time", "nEdges", "nVertLevels")
        for name, value in analysis.items():
            stored = np.asarray(dataset.variables[name][0]).T
            assert np.array_equal(stored.view(np.uint32), value.view(np.uint32)), name
    back = jc.read_contract_fields(path, jc.RECOUPLED_INPUTS, scalar_names=names)
    for name, value in analysis.items():
        assert np.array_equal(back[name].view(np.uint32), value.view(np.uint32)), name


# -- OU1: the DA-cycling restart consumes THIS run's analysis, on THIS run's elements ----------
# Breakage prevented (reproduced 2026-10-13 before the fix, on the real x1.2562 init through the
# engine's Rust bridges): the restart accepted an analysis at another valid time, read record 0 of a
# two-record file, and recoupled an analysis whose cells or edges were in another order.


def _tiny_init(path, n_cells=7, n_edges=19, *, cell_ids=None):
    from netCDF4 import Dataset

    rng = np.random.default_rng(9)
    with Dataset(path, "w", format="NETCDF3_64BIT_DATA") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("nCells", n_cells)
        dataset.createDimension("nEdges", n_edges)
        dataset.createDimension("StrLen", 64)
        dataset.createVariable("xtime", "S1", ("Time", "StrLen"))[0] = np.array(list("2026-10-03_15:00:00".ljust(64)), "S1")
        ids = np.arange(1, n_cells + 1, dtype=np.int32) if cell_ids is None else np.asarray(cell_ids, np.int32)
        dataset.createVariable("indexToCellID", "i4", ("nCells",))[:] = ids
        dataset.createVariable("indexToEdgeID", "i4", ("nEdges",))[:] = np.arange(1, n_edges + 1, dtype=np.int32)
        for name, dim, n in (("latCell", "nCells", n_cells), ("lonCell", "nCells", n_cells),
                             ("latEdge", "nEdges", n_edges), ("lonEdge", "nEdges", n_edges)):
            dataset.createVariable(name, "f8", (dim,))[:] = rng.uniform(-1.0, 1.0, n)
    return path


def _identity_background(tmp_path, init):
    fields, provenance, _ = _tiny_fields()
    return jc.write_background(
        tmp_path / "mpasout.nc", fields=fields, provenance=provenance,
        valid=datetime(2026, 10, 3, 15), n_cells=7, n_edges=19, n_levels=3,
        soil_levels=9, scalar_names=jc.DEFAULT_SCALAR_NAMES, identity_path=init,
    ), fields


def test_ou1_background_carries_the_init_identity_in_its_stored_types(tmp_path):
    init = _tiny_init(tmp_path / "init.nc")
    record, _ = _identity_background(tmp_path, init)
    mine, theirs = jc.read_identity(Path(record.path)), jc.read_identity(init)
    for name in jc.IDENTITY_FIELDS:
        assert mine[name].dtype == theirs[name].dtype and np.array_equal(mine[name], theirs[name]), name
    receipt = json.loads(Path(record.receipt_path).read_text())
    assert set(receipt["identity"]["fields"]) == set(jc.IDENTITY_FIELDS)
    assert jc.analysis_record_for_restart(
        Path(record.path), init_path=init, expected_valid_time="2026-10-03_15:00:00", n_cells=7, n_edges=19
    ) == 0


def test_ou1_restart_refuses_another_valid_time(tmp_path):
    init = _tiny_init(tmp_path / "init.nc")
    record, _ = _identity_background(tmp_path, init)
    with pytest.raises(jc.ContractRefusal, match="expected exactly one record"):
        jc.analysis_record_for_restart(
            Path(record.path), init_path=init, expected_valid_time="2026-10-03_16:00:00", n_cells=7, n_edges=19
        )


def test_ou1_restart_reads_the_record_at_its_valid_time_not_record_zero(tmp_path):
    from netCDF4 import Dataset

    init = _tiny_init(tmp_path / "init.nc")
    record, fields = _identity_background(tmp_path, init)
    with Dataset(record.path, "r+") as dataset:   # MPAS appends a record for another time
        dataset.variables["xtime"][1] = np.array(list("2026-10-03_16:00:00".ljust(64)), "S1")
        dataset.variables["rho"][1] = np.full((7, 3), 2.5, np.float32)
    later = jc.analysis_record_for_restart(
        Path(record.path), init_path=init, expected_valid_time="2026-10-03_16:00:00", n_cells=7, n_edges=19
    )
    assert later == 1
    rho = jc.read_contract_fields(Path(record.path), ("rho",), time_index=later)["rho"]
    assert np.all(rho == np.float32(2.5))
    assert jc.analysis_record_for_restart(
        Path(record.path), init_path=init, expected_valid_time="2026-10-03_15:00:00", n_cells=7, n_edges=19
    ) == 0


@pytest.mark.parametrize("name", ["indexToCellID", "lonCell", "indexToEdgeID", "latEdge"])
def test_ou1_restart_refuses_another_element_order(tmp_path, name):
    from netCDF4 import Dataset

    init = _tiny_init(tmp_path / "init.nc")
    record, _ = _identity_background(tmp_path, init)
    with Dataset(record.path, "r+") as dataset:
        dataset.variables[name][:] = dataset.variables[name][:][::-1]
    with pytest.raises(jc.ContractRefusal, match=f"{name} differs from the init"):
        jc.analysis_record_for_restart(
            Path(record.path), init_path=init, expected_valid_time="2026-10-03_15:00:00", n_cells=7, n_edges=19
        )


def test_ou1_restart_refuses_a_background_without_identity(tmp_path):
    init = _tiny_init(tmp_path / "init.nc")
    fields, provenance, _ = _tiny_fields()
    record = jc.write_background(
        tmp_path / "mpasout.nc", fields=fields, provenance=provenance,
        valid=datetime(2026, 10, 3, 15), n_cells=7, n_edges=19, n_levels=3,
        soil_levels=9, scalar_names=jc.DEFAULT_SCALAR_NAMES,
    )
    with pytest.raises(jc.ContractRefusal, match="no indexToCellID"):
        jc.analysis_record_for_restart(
            Path(record.path), init_path=init, expected_valid_time="2026-10-03_15:00:00", n_cells=7, n_edges=19
        )


def test_ou1_background_refuses_an_init_outside_mpas_global_order(tmp_path):
    init = _tiny_init(tmp_path / "init.nc", cell_ids=[2, 1, 3, 4, 5, 6, 7])
    with pytest.raises(jc.ContractRefusal, match="not 1..7 in file order"):
        _identity_background(tmp_path, init)
    assert not (tmp_path / "mpasout.nc").exists()
