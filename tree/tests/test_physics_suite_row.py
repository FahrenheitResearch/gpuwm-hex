"""The column-suite row: a second in-tree backend row that moves nothing.

Holds the row and its adapter to what they claim, on the host only (no card):
the frozen row and its digests are untouched; the profile table names every
selector and the soil geometry; the suite's sealed constructor takes the
land surface's own soil column and sea-ice switch and the drag's exact
statistics; the configuration a suite run records names the selectors it
executes and still validates every frozen value through the base type; and
the orographic reader decodes through the engine's Rust bridge.  Needs the
engine's ``gpuwm.core.mpas_column_suite`` importable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gpuwm.core.mpas_column_suite")

from hexcore import physics_backend_admission as admission  # noqa: E402
from hexcore import cuda_suite_physics_v841 as suite  # noqa: E402
from hexcore.errors import ConfigurationRefusal  # noqa: E402

ROW = "mynn_ruc_gsl_column"
NCOL, NLEV = 6, 10


def test_optional_cell_areas_are_sealed_as_native_float64_static_inputs():
    from hexcore.physics_suites import resolve_physics
    row = resolve_physics("rrfs_v2_tempo_candidate", mesh="conus12km")
    values = _frozen_values()
    values.update(microphysics_scheme="tempo", cumulus_scheme="ntiedtke", gf_ishallow=0)
    profile = row.column_profile
    from gpuwm.core.mpas_column_suite import orographic_statistic_names
    oro = {name: np.ones(NCOL, np.float32)
           for name in orographic_statistic_names("gsl")}
    area = np.full(NCOL, 144.e6, np.float64)
    mapped = suite.constructor_values_for(values, row_name=row.backend_row,
                                         orographic_statistics=oro, cell_area_m2=area)
    sealed = suite.seal_constructor(mapped)
    assert not sealed._values["cell_area_m2"].flags.writeable
    assert sealed._values["cell_area_m2"].dtype == np.float64
    area[0] = 1.
    assert sealed._values["cell_area_m2"][0] == 144.e6
    mapped["cell_area_m2"] = np.full(NCOL, -1., np.float64)
    with pytest.raises(ValueError, match="positive"):
        suite.seal_constructor(mapped)


def _frozen_values(soil_levels=9):
    rng = np.random.default_rng(3)
    f32 = lambda value: np.full(NCOL, value, np.float32)  # noqa: E731
    return {
        "n_levels": NLEV, "n_columns": NCOL, "dt": 60.0,
        "radiation_seconds": 1800.0, "surface_pbl_seconds": 60.0,
        "cumulus_seconds": 60.0, "cumulus_scheme": "gf",
        "microphysics_scheme": "thompson_aero",
        "start_time": datetime(2026, 10, 3, 12),
        "p_top_pa": 2500.0, "dx_m": 12000.0, "gf_ishallow": 1,
        "wsm6_hail_opt": 0, "xice_threshold": 0.02,
        "latitude_deg": f32(38.0), "longitude_deg": f32(-100.0),
        "terrain_height_m": rng.uniform(0, 2000, NCOL).astype(np.float32),
        "z_interface_nominal_m": np.linspace(0.0, 25000.0, NLEV + 1),
        "landmask": f32(1.0), "xland": f32(1.0),
        "ivgtyp": np.full(NCOL, 10, np.int32), "isltyp": np.full(NCOL, 6, np.int32),
        "vegfra": f32(40.0), "tsk": f32(295.0), "tmn": f32(287.0),
        "xice": f32(0.0), "snow": f32(0.0), "snow_depth": f32(0.0),
        "soil_temperature": np.full((soil_levels, NCOL), 290.0, np.float32),
        "soil_moisture": np.full((soil_levels, NCOL), 0.25, np.float32),
        "dx_column_m": f32(12000.0),
    }


def _oro():
    from gpuwm.core.mpas_column_suite import orographic_statistic_names

    return {name: np.zeros(NCOL, np.float32) for name in orographic_statistic_names("gsl")}


def test_the_frozen_row_is_untouched_and_still_the_default():
    frozen = admission.resolve_backend()
    assert frozen is admission.FROZEN_WSM6_COLUMN_ROW
    assert frozen.batch_module == "gpuwm.core.mpas_column_batch"
    assert frozen.adapter_module == "hexcore.cuda_arwen_physics_v841"
    assert frozen.driven_scalar_names == ("qv", "qc", "qr", "qi", "qs", "qg")


def test_the_suite_row_resolves_and_names_its_profile():
    row = admission.resolve_backend(ROW)
    assert row is admission.MYNN_RUC_GSL_COLUMN_ROW
    assert row.anchored is False and "MRMS and ASOS" in row.unanchored_remedy
    assert row.load_adapter() is suite
    assert suite.scheme_alias_for(ROW) == "mp_thompson_aerosols"
    assert suite.soil_levels_for(ROW) == 9
    record = suite.physics_suite_record(ROW)
    assert record["selection"] == {
        "radiation_scheme": "rrtmg_legacy", "surface_layer_scheme": "mynn",
        "land_surface_scheme": "ruc", "pbl_scheme": "mynn", "drag_scheme": "gsl"}
    assert record["drag_reads_statistics"] is True
    assert len(record["differences_from_shaped_after"]) >= 5


def test_the_species_block_is_the_thompson_aerosol_row():
    from hexcore.species_row import species_row_for_scheme

    row = admission.MYNN_RUC_GSL_COLUMN_ROW
    names = species_row_for_scheme("mp_thompson_aerosols").names()
    assert row.model_scalar_names() == tuple(names)


def test_seam_options_refuse_a_point_source_table():
    assert suite.seam_options_for(source_table=None) == {}
    with pytest.raises(ConfigurationRefusal):
        suite.seam_options_for(source_table="table.csv")


def test_the_door_mapping_becomes_the_suite_mapping():
    mapped = suite.constructor_values_for(
        _frozen_values(), row_name=ROW, orographic_statistics=_oro())
    assert "xice_threshold" not in mapped
    assert mapped["fractional_seaice"] == 1
    assert mapped["land_surface_scheme"] == "ruc"
    sealed = suite.seal_constructor(mapped)
    assert sealed.n_soil_levels == 9
    assert sealed.selection["drag_scheme"] == "gsl"
    kwargs = sealed.arwen_kwargs()
    assert kwargs["soil_temperature"].shape == (9, NCOL)
    assert sorted(kwargs["orographic_statistics"]) == sorted(_oro())
    assert set(sealed.expected_surface_classification()) == {
        "xland_source", "xland_land_columns", "xland_water_columns"}


def test_the_sealed_constructor_refuses_a_soil_column_for_another_land_surface():
    mapped = suite.constructor_values_for(
        _frozen_values(soil_levels=4), row_name=ROW, orographic_statistics=_oro())
    with pytest.raises(ValueError, match="shape"):
        suite.seal_constructor(mapped)


def test_the_sealed_constructor_refuses_missing_or_extra_drag_statistics():
    partial = _oro()
    partial.pop("VARSS")
    mapped = suite.constructor_values_for(
        _frozen_values(), row_name=ROW, orographic_statistics=partial)
    with pytest.raises(ValueError, match="orographic statistics"):
        suite.seal_constructor(mapped)
    mapped = suite.constructor_values_for(
        _frozen_values(), row_name=ROW, orographic_statistics=None)
    with pytest.raises(ValueError, match="needs"):
        suite.seal_constructor(mapped)


def test_the_suite_backend_refuses_a_hex_side_gwdo():
    mapped = suite.constructor_values_for(
        _frozen_values(), row_name=ROW, orographic_statistics=_oro())
    with pytest.raises(ConfigurationRefusal, match="twice"):
        suite.PersistentSuiteCudaPhysicsBackendV841(
            constructor=suite.seal_constructor(mapped), prep_geometry=None,
            kernel_cache=None, profile_id="mynn-ruc-gsl", gwdo_static=object())


def test_the_recorded_configuration_names_the_executed_selectors():
    from hexcore.drivers import run_cuda_v841_forecast as forecast

    from hexcore.config_v841 import (
        V841MpasColumnPhysicsConfig, V841MpasColumnPhysicsGwdoConfig)

    base = forecast.build_forecast_config(dt_seconds=120.0)
    recorded = suite.suite_config(base, ROW)
    # Not a GWDO type: the dycore's receipt lane reads that type as the
    # hex-side YSU-GWDO composer, which the suite does not run.
    assert isinstance(base, V841MpasColumnPhysicsGwdoConfig)
    assert isinstance(recorded, V841MpasColumnPhysicsConfig)
    assert not isinstance(recorded, V841MpasColumnPhysicsGwdoConfig)
    for name in ("config_dt", "config_horiz_mixing", "config_monotonic",
                 "config_time_integration_order", "config_radt_seconds"):
        assert getattr(recorded, name) == getattr(base, name), name
    assert recorded.config_pbl_scheme == "bl_mynn"
    assert recorded.config_lsm_scheme == "sf_ruc"
    assert recorded.config_gwdo_scheme == "bl_ugwp_gwdo"
    assert recorded.config_physics_suite_profile == "mynn-ruc-gsl"
    recorded.validate()
    from dataclasses import replace

    with pytest.raises(ConfigurationRefusal):
        replace(recorded, config_pbl_scheme="bl_ysu").validate()
    # The base type still holds its own literals: a frozen-row run is unmoved.
    assert base.config_pbl_scheme == "bl_ysu"
    base.validate()


def test_the_orographic_reader_uses_the_rust_bridge(tmp_path):
    source = Path(__file__).resolve().parent / "fixtures" / "conus12km.ugwp_oro_data.nc"
    if not source.is_file():
        pytest.skip("the conus12km statistics fixture is staged by the box run")
    values, receipt = suite.read_orographic_statistics(
        source, row_name=ROW, n_cells=157859)
    assert receipt["reader"].startswith("gpuwm.netcdf_bridge")
    assert sorted(values) == sorted(_oro())
    assert all(value.dtype == np.float32 and value.shape == (157859,)
               for value in values.values())
    assert float(values["VARLS"].max()) > 100.0
