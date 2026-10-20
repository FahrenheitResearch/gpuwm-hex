"""Any physics option WOOF's registry declares is selectable on hex.

THE BREAKAGE THIS FILE PREVENTS (ruled 2026-10-11, the arbitrary acceptance
test): hex's physics selection was a hand-kept table.  A scheme WOOF ran but
the table did not list was refused with "no seam runs it", so every new
scheme was a hex code change.  These tests hold the resolver to the engine's
own registry: the options are read from ``gpuwm/physics_registry_v2.json``
in the run's engine tree, a selection outside the two slot-named seams is
handed to the generic column seam as RunConfig assignments, and what is
still refused is refused for a reason WOOF itself would give.

The registry tests run against a small registry written by the test (no
engine needed); the plan tests need an engine carrying the generic seam and
skip without one.
"""

from __future__ import annotations

import json

import pytest

from hexcore import convection_admission, engine_registry
from hexcore import physics_suites as ps
from hexcore import species_row as sr

REGISTRY = {
    "components": {
        "microphysics": {
            "selector_keys": ["mp_physics"],
            "options": {
                "wsm6-mp6": {"selectors": {"mp_physics": 6}, "implemented": True, "label": "WSM6"},
                "newscheme-mp77": {"selectors": {"mp_physics": 77}, "implemented": True,
                                   "label": "A scheme added after this test was written",
                                   "maturity": "implemented-unverified"},
                "planned-one": {"selectors": {}, "implemented": False, "maturity": "planned"},
            },
        },
        "cumulus": {
            "selector_keys": ["cu_physics"],
            "options": {
                "off": {"selectors": {"cu_physics": 0}, "implemented": True},
                "kain-fritsch": {"selectors": {"cu_physics": 1}, "implemented": True, "label": "KF"},
            },
        },
        "radiation": {
            "selector_keys": ["ra_lw_physics", "ra_sw_physics"],
            "options": {
                "rte-rrtmgp": {"selectors": {"ra_lw_physics": 4, "ra_sw_physics": 4}, "implemented": True},
                "dudhia-shortwave": {"selectors": {"ra_lw_physics": 0, "ra_sw_physics": 1}, "implemented": True},
            },
        },
        "turbulence": {
            "selector_keys": ["km_opt"],
            "options": {"smagorinsky-2d": {"selectors": {"km_opt": 4}, "implemented": True}},
        },
        "urban": {
            "selector_keys": ["sf_urban_physics"],
            "options": {"slucm": {"selectors": {"sf_urban_physics": 1}, "implemented": True}},
        },
    },
    "parameters": {
        "gwd_opt": {"type": "integer", "enum": [0, 1, 3], "default": 0},
        "diff_6th_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0,
                         "consuming_read": "gpuwm/core/dycore.py"},
        "topo_wind": {"type": "integer", "enum": [0, 1, 2], "default": 0},
        "aercu_opt": {"type": "integer", "implemented": False,
                      "unimplemented_reason": "the multiscale Kain-Fritsch is not ported"},
    },
}


@pytest.fixture
def engine_tree(tmp_path):
    """An engine tree carrying only a registry: enough to resolve names."""

    root = tmp_path / "engine"
    (root / "gpuwm").mkdir(parents=True)
    (root / "gpuwm" / "physics_registry_v2.json").write_text(json.dumps(REGISTRY))
    return root


def _plan_engine():
    """The importable engine, when it carries the generic column seam."""

    suite = pytest.importorskip("gpuwm.core.mpas_column_suite")
    if not hasattr(suite, engine_registry.GENERIC_SEAM_FUNCTION):
        pytest.skip("the importable engine predates the generic column seam")
    return suite


# ---------------------------------------------------------------------------
# the registry is the list
# ---------------------------------------------------------------------------
def test_a_scheme_the_registry_adds_is_selectable_with_no_hex_row(engine_tree):
    route = {"component": "microphysics", "selector": "mp_physics"}
    for spelling in ("newscheme-mp77", "77", 77):
        found = engine_registry.resolve_selector_value(route, spelling, engine_tree)
        assert dict(found.assignments) == {"mp_physics": 77}
        assert found.option_id == "newscheme-mp77"
    assert "newscheme-mp77" not in json.dumps(ps._thaw(ps.load_table()))


def test_an_option_the_registry_does_not_implement_is_refused_in_its_words(engine_tree):
    route = {"component": "microphysics", "selector": "mp_physics"}
    with pytest.raises(engine_registry.EngineValueRefusal, match="does not implement"):
        engine_registry.resolve_selector_value(route, "planned-one", engine_tree)
    with pytest.raises(engine_registry.EngineValueRefusal, match="declares no microphysics option"):
        engine_registry.resolve_selector_value(route, "mp_never_heard_of", engine_tree)


def test_a_parameter_is_typed_and_bounded_by_its_registry_row(engine_tree):
    assert engine_registry.setting_value("topo_wind", "2", engine_tree) == 2
    assert engine_registry.setting_value("sf_urban_physics", "slucm", engine_tree) == 1
    with pytest.raises(engine_registry.EngineValueRefusal, match=r"topo_wind in \[0, 1, 2\]"):
        engine_registry.setting_value("topo_wind", "7", engine_tree)
    with pytest.raises(engine_registry.EngineValueRefusal, match="not ported"):
        engine_registry.setting_value("aercu_opt", "1", engine_tree)


def test_an_engine_tree_without_the_generic_seam_is_named(engine_tree):
    problem = engine_registry.generic_seam_problem(engine_tree)
    assert problem is not None and "mpas_column_suite" in problem
    with pytest.raises(ps.PhysicsSelectionRefusal, match="generic column seam"):
        ps.resolve_physics(
            "arwen_mpas_column", {"mp_physics": "77"}, engine_checkout=engine_tree)


def test_a_tree_with_no_registry_is_named(tmp_path):
    with pytest.raises(engine_registry.EngineRegistryUnavailable, match="physics_registry_v2.json"):
        engine_registry.engine_root(tmp_path)


def test_woofs_own_selector_names_are_the_mpas_keys(engine_tree):
    aliases = ps.engine_selector_aliases()
    assert aliases["mp_physics"] == "config_microp_scheme"
    assert aliases["cu_physics"] == "config_convection_scheme"
    assert aliases["gwd_opt"] == "config_gwdo_scheme"
    # one scheme, one name: WOOF's number resolves to the MPAS spelling
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics(
            "arwen_mpas_column", {"mp_physics": "6", "cu_physics": "1"},
            engine_checkout=engine_tree)
    assert [item.knob for item in caught.value.refusals] == ["physics"]


def test_the_default_never_reads_the_registry(tmp_path):
    """The default suite resolves with no engine at all, exactly as before."""

    resolved = ps.resolve_physics(None, engine_checkout=tmp_path / "no-engine-here")
    assert resolved.is_default()
    assert (resolved.seam, resolved.backend_row, resolved.convection) == (
        "frozen", "wsm6_column", "auto")
    assert resolved.engine_physics is None


def test_every_alias_row_names_an_engine_assignment():
    table = ps.load_table()
    for key, rows in table["options"].items():
        for value, row in rows.items():
            if "refused" in row or value == "auto":
                continue
            assert row["engine"], (key, value)
    assert set(table["engine_selectors"]["selectors"]) == set(table["selector_keys"])


def test_what_is_still_refused_is_something_woof_cannot_run():
    """The refused rows left in the table, each for a reason that is WOOF's."""

    refused = {
        (key, value)
        for key, rows in ps.load_table()["options"].items()
        for value, row in rows.items() if "refused" in row
    }
    assert refused == {
        ("config_convection_scheme", "cu_gf_monan"),
        ("config_radt_lw_scheme", "cam_lw"),
        ("config_radt_sw_scheme", "cam_sw"),
        ("config_radt_cld_scheme", "cld_fraction_thompson"),
    }


def test_a_seam_owned_field_is_refused_by_name(engine_tree):
    with pytest.raises(ps.PhysicsSelectionRefusal, match="step counters"):
        ps.resolve_physics("arwen_mpas_column", {"dt": "30"}, engine_checkout=engine_tree)


@pytest.mark.parametrize("override", [{"diff_6th_opt": "2"}])
def test_a_setting_only_woofs_dycore_reads_is_refused(engine_tree, override):
    # Breakage: these resolved on 2026-10-11, moved the run to the generic
    # seam and were recorded; hex runs the MPAS-A dycore and never read them.
    with pytest.raises(ps.PhysicsSelectionRefusal, match="never read") as caught:
        ps.resolve_physics("arwen_mpas_column", override, engine_checkout=engine_tree)
    assert [item.knob for item in caught.value.refusals] == list(override)
    # km_opt stays WOOF's call (its validator pairs it with the PBL scheme):
    # it reaches the seam question, not this refusal.
    with pytest.raises(ps.PhysicsSelectionRefusal, match="generic column seam"):
        ps.resolve_physics("arwen_mpas_column", {"km_opt": "4"}, engine_checkout=engine_tree)


# ---------------------------------------------------------------------------
# cumulus schemes register themselves
# ---------------------------------------------------------------------------
def test_a_registry_cumulus_scheme_gets_its_names_without_a_row():
    convection_admission.register_engine_scheme(
        "cu_test_only_scheme", constructor_name="test-only", cu_physics=4242, label="Test Only")
    assert convection_admission.constructor_scheme("cu_test_only_scheme") == "test-only"
    assert convection_admission.cu_physics_for_constructor_scheme("test-only") == 4242
    assert convection_admission.label("cu_test_only_scheme") == "Test Only"
    assert convection_admission.gf_ishallow("cu_test_only_scheme") == 0
    decision = convection_admission.convection_decision(nominal_dx_m=30_000.0, requested="test-only")
    assert decision["scheme"] == "cu_test_only_scheme" and decision["source"] == "explicit"
    with pytest.raises(convection_admission.ConvectionAdmissionError, match="already registered"):
        convection_admission.register_engine_scheme(
            "cu_test_only_scheme", constructor_name="other", cu_physics=1, label="x")
    # the three ruled schemes keep their numbers
    assert convection_admission.cu_physics_for_constructor_scheme("gf") == 3
    assert convection_admission.cu_physics_for_constructor_scheme("ntiedtke") == 16
    assert convection_admission.cu_physics_for_constructor_scheme(None) == 0


# ---------------------------------------------------------------------------
# species rows from the engine's plan
# ---------------------------------------------------------------------------
def test_a_species_row_is_derived_from_the_engines_plan():
    row = sr.engine_species_row(
        scheme_name="test-derived-mp77", mp_physics=77,
        species=("qv", "qc", "qr", "qnr", "qh", "qi", "qvolh"),
        mass_species=("qv", "qc", "qr", "qh", "qi"),
        surface_accumulators=("RAINNC", "HAILNC"), radius_names=("effc",))
    # loading species first (the six in their canonical order, then others)
    assert row.names() == ("qv", "qc", "qr", "qi", "qh", "qnr", "qvolh")
    assert row.n_mass_loading == 5
    assert row.declaration("qvolh").species_class == sr.VOLUME
    assert row.declaration("qnr").species_class == sr.NUMBER
    assert [item.history_name for item in row.surface_accumulators] == ["RAINNC", "HAILNC"]
    assert sr.register_engine_row(row) is row
    assert sr.species_row_for_scheme("test-derived-mp77") is row
    assert sr.species_row_for_names(row.names()) is row
    assert sr.species_row_for_mp_physics(77) is row
    other = sr.engine_species_row(
        scheme_name="test-derived-mp77", mp_physics=77, species=("qv", "qc"),
        mass_species=("qv", "qc"))
    with pytest.raises(Exception, match="not replaceable"):
        sr.register_engine_row(other)
    with pytest.raises(Exception, match="declared species row"):
        sr.register_engine_row(sr.engine_species_row(
            scheme_name="mp_wsm6", mp_physics=6, species=("qv", "qc"), mass_species=("qv", "qc")))


def test_the_rows_once_declared_and_refused_now_name_the_engine_selector():
    assert sr.KESSLER_SPECIES_ROW.require_engine_scheme() == "mp_physics=1"
    assert sr.THOMPSON_SPECIES_ROW.require_engine_scheme() == "mp_physics=8"


# ---------------------------------------------------------------------------
# against the engine's own plan
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overrides", [
    {"config_microp_scheme": "mp_kessler"},
    {"mp_physics": "10"},
    {"config_microp_scheme": "mp_thompson"},
    {"config_microp_scheme": "mp_p3"},
    {"config_microp_scheme": "mp_nssl2m"},
    {"config_microp_scheme": "mp_tempo"},
    {"config_convection_scheme": "cu_kain_fritsch"},
    {"config_convection_scheme": "6"},
    {"config_pbl_scheme": "bl_myj", "config_sfclayer_scheme": "2"},
    {"config_sfclayer_scheme": "sf_monin_obukhov"},
    {"config_lsm_scheme": "sf_noah"},
    {"config_lsm_scheme": "sf_ruc"},
    {"config_gwdo_scheme": "off"},
    {"config_gwdo_scheme": "bl_ugwp_gwdo"},
    {"config_radt_lw_scheme": "rrtmgp_lw", "config_radt_sw_scheme": "rrtmgp_sw"},
])
def test_what_no_seam_ran_now_resolves_to_the_generic_seam(overrides):
    _plan_engine()
    resolved = ps.resolve_physics("arwen_mpas_column", overrides)
    assert resolved.seam == "engine"
    profile = resolved.column_profile
    assert profile["generic"] and profile["engine_batch"]["entrypoint"] == "run_mpas_column_physics"
    assert dict(profile["physics"]) == dict(resolved.engine_physics)
    row = sr.species_row_for_scheme(profile["microphysics"])
    assert list(row.names()) == list(profile["species"]) and row.names()[0] == "qv"
    # the same request resolves to the same row (the door and the driver agree)
    assert ps.resolve_physics("arwen_mpas_column", overrides).backend_row == resolved.backend_row
    from hexcore.physics_backend_admission import resolve_backend

    backend = resolve_backend(resolved.backend_row)
    assert backend.batch_entrypoint == "run_mpas_column_physics"
    assert (*backend.driven_scalar_names, *backend.appended_scalar_names) == row.names()


def _generic_seam_values(n_soil, ncol=6, nlev=20):
    from datetime import datetime

    import numpy as np

    f32 = lambda value: np.full(ncol, value, np.float32)  # noqa: E731
    values = {
        "n_levels": nlev, "n_columns": ncol, "dt": 180.0,
        "radiation_seconds": 1800.0, "surface_pbl_seconds": 180.0,
        "cumulus_seconds": 180.0, "cumulus_scheme": "gf",
        "microphysics_scheme": "wsm6", "start_time": datetime(2026, 10, 3, 12),
        "p_top_pa": 2500.0, "dx_m": 30000.0,
        # what the driver derives from the scheme name (MPAS's hardwire)
        "gf_ishallow": 1,
        "wsm6_hail_opt": 0, "xice_threshold": 0.02,
        "latitude_deg": f32(10.0), "longitude_deg": f32(-60.0),
        "terrain_height_m": f32(100.0),
        "z_interface_nominal_m": np.linspace(0.0, 25000.0, nlev + 1),
        "landmask": f32(1.0), "xland": f32(1.0),
        "ivgtyp": np.full(ncol, 10, np.int32), "isltyp": np.full(ncol, 6, np.int32),
        "vegfra": f32(40.0), "tsk": f32(295.0), "tmn": f32(287.0),
        "xice": f32(0.0), "snow": f32(0.0), "snow_depth": f32(0.0),
        "dx_column_m": f32(30000.0),
    }
    if n_soil:
        values["soil_temperature"] = np.full((n_soil, ncol), 290.0, np.float32)
        values["soil_moisture"] = np.full((n_soil, ncol), 0.25, np.float32)
    return values


@pytest.mark.parametrize("given, ran", [("0", 0), ("1", 1), (None, 1)])
def test_the_seal_carries_the_shallow_switch_the_engine_is_handed(given, ran):
    """Measured 2026-10-11 (global 30 km, two cards): cu_grell_freitas with
    ishallow=0 ran the engine with the shallow arm off while the seal and the
    receipt's convection record said gf_ishallow=1."""

    _plan_engine()
    from hexcore import cuda_suite_physics_v841 as suite

    overrides = {"config_convection_scheme": "cu_grell_freitas", "config_gwdo_scheme": "off"}
    if given is not None:
        overrides["ishallow"] = given
    resolved = ps.resolve_physics("arwen_mpas_column", overrides)
    assert resolved.seam == "engine"
    values = _generic_seam_values(int(resolved.column_profile["n_soil_levels"]))
    mapped = suite.constructor_values_for(
        values, row_name=resolved.backend_row, orographic_statistics=None)
    sealed = suite.seal_constructor(mapped)
    assert sealed._values["physics"]["ishallow"] == ran
    assert sealed._values["gf_ishallow"] == ran


def test_mpas_own_reference_suite_resolves():
    _plan_engine()
    resolved = ps.resolve_physics("mesoscale_reference")
    assert resolved.seam == "engine" and resolved.convection == "ntiedtke"
    assert resolved.engine_physics["sf_surface_physics"] == 2
    assert resolved.column_profile["orographic_statistics"]["file"] == "static"


def test_a_combination_woof_refuses_is_refused_in_woofs_words():
    _plan_engine()
    with pytest.raises(ps.PhysicsSelectionRefusal, match="WOOF itself refuses"):
        ps.resolve_physics("arwen_mpas_column", {"config_sfclayer_scheme": "sf_mynn"})
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics("arwen_mpas_column", {"config_not_a_field": "1"})
    assert [item.knob for item in caught.value.refusals] == ["config_not_a_field"]


def test_statics_no_mesh_stages_are_a_named_missing_input():
    _plan_engine()
    with pytest.raises(ps.PhysicsSelectionRefusal, match="missing static input"):
        ps.resolve_physics("arwen_mpas_column", {"topo_wind": "1"})


def test_the_anchored_seams_keep_their_rows():
    _plan_engine()
    assert ps.resolve_physics("arwen_thompson_column").backend_row == "thompson_aero_column"
    rrfs = ps.resolve_physics("rrfs_v2_dev")
    assert (rrfs.seam, rrfs.backend_row) == ("column_suite", "mynn_ruc_gsl_column")
    assert rrfs.engine_physics is None


@pytest.mark.parametrize("value", ["mp_kessler", "off"])
def test_a_three_species_block_resolves_to_the_generic_seam(value):
    """Kessler and microphysics off transport (qv, qc, qr) and nothing else.

    THE BREAKAGE THIS PREVENTS: guard hexphys-short-scalar-block refused
    every block shorter than six because the commit kernel copied six
    planes whatever the row's length and stopped the card with
    cudaErrorIllegalAddress on a three-plane buffer.  The kernel now takes
    the row's length (tests/test_physics_carrier_kernels.py holds it on the
    card), so the guard is retired and the door must hand these selections
    to the seam again."""

    _plan_engine()
    resolved = ps.resolve_physics("arwen_mpas_column", {"config_microp_scheme": value})
    assert resolved.seam == "engine"
    assert tuple(resolved.column_profile["species"]) == ("qv", "qc", "qr")
    row = sr.species_row_for_scheme(resolved.column_profile["microphysics"])
    assert row.names() == ("qv", "qc", "qr")


@pytest.mark.parametrize("value,expected", [
    ("mp_kessler", {"RAINC", "RAINNC"}),
    ("off", {"RAINC"}),
])
def test_the_suite_boundary_requires_the_rows_buckets_and_no_others(value, expected):
    """Microphysics off has no grid-scale bucket in the engine's plan.

    THE BREAKAGE THIS PREVENTS: the generic seam's committed-boundary
    snapshot required RAINNC whatever the row said, so microphysics off
    stopped at rank construction with "suite precipitation inventory
    changed: ['RAINC']" (60 km global, 2026-10-11)."""

    from hexcore.cuda_suite_physics_v841 import required_precipitation_keys

    _plan_engine()
    resolved = ps.resolve_physics("arwen_mpas_column", {"config_microp_scheme": value})
    keys = required_precipitation_keys(
        generic=True, microphysics=resolved.column_profile["microphysics"])
    assert keys == expected
    assert required_precipitation_keys(generic=False, microphysics="mp_wsm6") == {
        "RAINC", "RAINNC", "SNOWNC"}


def test_the_generic_backend_takes_its_row_from_the_selection_not_the_names():
    """Kessler and microphysics off carry the same (qv, qc, qr) block.

    THE BREAKAGE THIS PREVENTS: phase two looked the row up by names, got
    Kessler's for microphysics off, and refused the off seam's buckets
    ("cumulative precipitation keys changed: ['RAINC'], expected ['RAINC',
    'RAINNC']", 60 km and 30 km global, 2026-10-11)."""

    from types import SimpleNamespace

    from hexcore.cuda_suite_physics_v841 import PersistentSuiteCudaPhysicsBackendV841

    _plan_engine()
    rows = {}
    for value in ("mp_kessler", "off"):
        resolved = ps.resolve_physics("arwen_mpas_column", {"config_microp_scheme": value})
        backend = SimpleNamespace(
            _constructor=SimpleNamespace(physics=dict(resolved.engine_physics)),
            _profile={"microphysics": resolved.column_profile["microphysics"]})
        rows[value] = PersistentSuiteCudaPhysicsBackendV841._species_row(
            backend, ("qv", "qc", "qr"))
    assert [i.history_name for i in rows["mp_kessler"].surface_accumulators] == ["RAINNC"]
    assert rows["off"].surface_accumulators == ()
    assert sr.species_row_for_names(("qv", "qc", "qr")) is rows["mp_kessler"]
    with pytest.raises(ValueError, match="not the scalar block"):
        PersistentSuiteCudaPhysicsBackendV841._species_row(backend, ("qv", "qc"))


def test_microphysics_off_forms_no_reflectivity_and_every_scheme_does():
    """The engine refuses refl_10cm_due with mp_physics=0 ("no scheme
    computes a reflectivity"), so a microphysics-off run stopped at its
    first history step.  The generic backend now says so, by the engine's
    own rule, and the history frame carries the allocated default."""

    from types import SimpleNamespace

    from hexcore.cuda_suite_physics_v841 import PersistentSuiteCudaPhysicsBackendV841

    _plan_engine()
    answers = {}
    for value in ("mp_kessler", "off"):
        resolved = ps.resolve_physics("arwen_mpas_column", {"config_microp_scheme": value})
        backend = SimpleNamespace(
            _constructor=SimpleNamespace(physics=dict(resolved.engine_physics)))
        answers[value] = PersistentSuiteCudaPhysicsBackendV841.computes_reflectivity(backend)
    assert answers == {"mp_kessler": True, "off": False}


# ---------------------------------------------------------------------------
# the claim sentence names the schemes the run executes
# ---------------------------------------------------------------------------
def test_a_selection_is_labelled_by_the_registry(engine_tree):
    labels = engine_registry.selection_labels(
        {"mp_physics": 77, "cu_physics": 1, "ra_lw_physics": 4, "ra_sw_physics": 4,
         "gwd_opt": 1, "ra_rrtmg_variant": "rrtmg_legacy"},
        engine_tree, skip=("cu_physics",))
    assert labels == (
        "A scheme added after this test was written", "rte-rrtmgp",
        "gwd_opt=1", "ra_rrtmg_variant=rrtmg_legacy")


def test_the_claim_names_the_selected_schemes_not_the_frozen_suite():
    from hexcore.drivers.mpas_mesh_binding import claim_for_selection

    claim = ("one forecast using WSM6 + GF + YSU + legacy RRTMG, at dt = 180 s, "
             "from the init named in this receipt")
    rewritten = claim_for_selection(claim, ["Thompson / MP8", "New Tiedtke", "YSU PBL"])
    assert rewritten == (
        "one forecast using Thompson / MP8 + New Tiedtke + YSU PBL, at dt = 180 s, "
        "from the init named in this receipt")
    assert "WSM6" not in rewritten
    with pytest.raises(ValueError, match="no 'using <schemes>, at dt = ' term"):
        claim_for_selection("a sentence of another shape", ["x"])


def test_every_seam_carries_the_assignments_its_claim_is_labelled_from():
    from hexcore.drivers.mpas_mesh_binding import claim_scheme_terms

    frozen = ps.resolve_physics(
        "arwen_mpas_column", {"config_microp_scheme": "mp_thompson_aerosols"})
    assert frozen.seam == "frozen" and not frozen.is_default()
    assert dict(frozen.engine_assignments["config_microp_scheme"]) == {"mp_physics": 28}
    assert "config_convection_scheme" not in frozen.engine_assignments
    assert claim_scheme_terms("x using A + B + C, at dt = 1 s") == ("A", "B", "C")


_BOUND_CLAIM = (
    "one uninterrupted real-initialized x1.655362 MPAS-A v8.4.2 CUDA forecast "
    "using WSM6 + New Tiedtke + YSU + external YSU-GWDO + revised-MO + NoahMP "
    "(with the CUDA glacier path) + cloud fraction + legacy RRTMG, at "
    "dt = 180 s, from the init named in this receipt"
)


def test_the_claim_names_the_selection_that_ran_not_the_frozen_batch():
    # Breakage: the 30 km RRTM + Dudhia run of 2026-10-11 carried a receipt
    # claiming legacy RRTMG and the external YSU drag; neither ran.
    from types import SimpleNamespace

    from hexcore.drivers.run_cuda_v841_forecast import claim_for_physics

    resolved = SimpleNamespace(
        seam="engine",
        selectors={"config_convection_scheme": "auto", "config_pbl_scheme": "bl_ysu",
                   "config_radt_lw_scheme": "off", "config_radt_sw_scheme": "dudhia-shortwave"},
        engine_physics={"ra_sw_physics": 1, "ra_lw_physics": 0, "gwd_opt": 1})
    claim = claim_for_physics(_BOUND_CLAIM, resolved)
    assert "config_convection_scheme" not in claim
    assert "config_radt_sw_scheme=dudhia-shortwave" in claim
    assert "gwd_opt=1, ra_lw_physics=0, ra_sw_physics=1" in claim
    assert "legacy RRTMG" not in claim and "external YSU-GWDO" not in claim
    assert "using New Tiedtke for convection" in claim
    assert claim.endswith(", at dt = 180 s, from the init named in this receipt")
    # Naming it twice changes nothing; a frozen-seam run keeps its sentence.
    assert claim_for_physics(claim, resolved) == claim
    assert claim_for_physics(_BOUND_CLAIM, ps.resolve_physics(None, {})) is _BOUND_CLAIM
    assert claim_for_physics(_BOUND_CLAIM, None) is _BOUND_CLAIM
