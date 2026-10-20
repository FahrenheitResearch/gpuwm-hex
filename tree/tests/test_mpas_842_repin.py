"""The dycore reference is MPAS-A v8.4.2, and what moved with the pin.

Re-pinned from v8.4.1 on 2026-10-11.  The two tags differ in twelve files.
Held here:

* every record of the pin names the same tag, tag object and commit, and the
  digests of the three ``core_atmosphere`` files whose bytes changed;
* the one default that moved, ``config_number_of_sub_steps`` 2 -> 4, is what
  a bare run gets, and 2 is still selectable;
* the v8.4.2 fixes that have no hex counterpart stay without one: the
  prognostic-TKE closure (whose eddy-viscosity floor is the only arithmetic
  change in the dycore directory) is not ported and is refused by name, and
  lateral-boundary species are matched by name, never by the Registry's
  declaration order (the order of ``lbc_ni`` and ``lbc_nr`` is what v8.4.2
  corrected upstream).

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pytest

from hexcore import forecast_door as door, mpas_defaults
from hexcore.config_v841 import V841DryDycoreConfig, V841MpasColumnPhysicsConfig
from hexcore.errors import ConfigurationRefusal
from hexcore.timestep_admission import SplitExplicitSchedule

TREE = Path(__file__).resolve().parents[1]
PIN = mpas_defaults.UPSTREAM_PIN
CITED = TREE / "verification" / "mpas-v8.4.2-registry" / "cited-lines.txt"


# ---------------------------------------------------------------------------
# one pin, named the same everywhere
# ---------------------------------------------------------------------------
def test_the_pin_is_v8_4_2_and_remembers_what_it_replaced():
    assert PIN["tag"] == "v8.4.2"
    assert PIN["commit"] == "f34984b2c89353cf31c235b08d3b4acd96b37bb3"
    assert PIN["tag_object"] == "2eee75b147d83d67835cecfb30c6fb3793e3042b"
    assert PIN["previous"]["tag"] == "v8.4.1"
    assert PIN["previous"]["commit"] == "91c5eac175eebeaf4206bacd5cb50c39dff3c152"
    assert set(PIN["changed_files"]) == {
        "src/core_atmosphere/Registry.xml",
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        "src/core_atmosphere/dynamics/mpas_atm_dissipation_models.F",
    }
    for row in PIN["changed_files"].values():
        assert row["sha256"] != row["previous_sha256"] and row["change"]


def test_the_source_profile_table_names_the_same_commit():
    from hexcore.source_profiles import load_source_profiles

    row = load_source_profiles()["mpas-a-v8.4.2"]
    assert row["role"] == "dycore_pin"
    assert row["model_source"]["tag"] == PIN["tag"]
    assert row["model_source"]["commit"] == PIN["commit"]
    assert row["model_source"]["registry_version"] == "8.4.2"


def test_the_provenance_strings_name_the_pinned_commit():
    from hexcore import damping_v841, driver, offcentering_v841, physics_seam_v841

    for text in (driver.V841_SOURCE, offcentering_v841.V841_OFFCENTERING_SOURCE,
                 damping_v841.V841_DSS_SOURCE, physics_seam_v841.V841_PHYSICS_SOURCE):
        assert PIN["commit"] in text
        assert PIN["previous"]["commit"] not in text
    for text in (driver.V841_SOURCE, physics_seam_v841.V841_PHYSICS_SOURCE):
        assert PIN["tag_object"] in text and PIN["archive_sha256"] in text


def test_the_cited_lines_record_carries_the_pinned_digests():
    head = CITED.read_text(encoding="utf-8").split("== ", 1)[0]
    digests = dict(re.findall(r"#\s+(src/\S+)\s+sha256 ([0-9a-f]{64})", head))
    for path in ("src/core_atmosphere/Registry.xml",
                 "src/core_atmosphere/dynamics/mpas_atm_time_integration.F"):
        assert digests[path] == PIN["changed_files"][path]["sha256"]
    assert PIN["changed_files"]["src/core_atmosphere/Registry.xml"]["sha256"] in (
        mpas_defaults.REGISTRY_SOURCE)
    assert PIN["commit"] in head and PIN["tag_object"] in head


def test_the_registry_defaults_table_reads_the_pinned_registry():
    from hexcore import physics_suites as ps

    defaults = ps.load_table()["namelist"]["registry_defaults"]
    assert defaults["default_registry"] == "mpas-a-v8.4.2"
    source = defaults["registries"]["mpas-a-v8.4.2"]["source"]
    assert PIN["changed_files"]["src/core_atmosphere/Registry.xml"]["sha256"] in source


# ---------------------------------------------------------------------------
# the default that moved: four acoustic sub-steps, two still selectable
# ---------------------------------------------------------------------------
def test_a_bare_run_takes_four_acoustic_substeps():
    assert mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS == 4
    assert mpas_defaults.PREVIOUS_ACOUSTIC_SUBSTEPS == 2
    assert mpas_defaults.REGISTRY_DEFAULTS["config_number_of_sub_steps"]["value"] == 4
    block = CITED.read_text(encoding="utf-8").split("== Registry.xml:104-107\n", 1)[1]
    assert 'name="config_number_of_sub_steps" type="integer" default_value="4"' in (
        block.split("== ", 1)[0])
    assert V841MpasColumnPhysicsConfig().config_number_of_sub_steps == 4
    assert SplitExplicitSchedule().acoustic_substeps == 4
    bare = door.resolve_schedule(argparse.Namespace(), None)
    assert bare["substeps"] == 4 and bare["sources"]["substeps"] != "explicit"
    parser = door.build_parser() if hasattr(door, "build_parser") else None
    if parser is not None:
        action = next(a for a in parser._actions if "--acoustic-substeps" in a.option_strings)
        assert action.default is None  # the door resolves it from mpas_defaults


def test_the_driver_parser_and_the_binding_default_to_the_same_value():
    source = (TREE / "src/hexcore/drivers/run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--acoustic-substeps", type=int, default=4)' in source
    assert "acoustic_substeps: int = 2" not in source
    assert "acoustic_substeps: int = 2" not in (
        TREE / "src/hexcore/forecast_door.py").read_text(encoding="utf-8")


def test_two_acoustic_substeps_is_still_selectable():
    chosen = door.resolve_schedule(argparse.Namespace(acoustic_substeps=2), None)
    assert chosen["substeps"] == 2 and chosen["sources"]["substeps"] == "explicit"
    config = V841MpasColumnPhysicsConfig(config_number_of_sub_steps=2)
    config.validate()
    assert config.config_number_of_sub_steps == 2
    # the stability bound follows the selection: 2 sub-steps take an acoustic
    # step twice as long as 4, so the same dt sits at twice the acoustic number
    assert SplitExplicitSchedule(2, 2).largest_acoustic_fraction() == pytest.approx(
        2.0 * SplitExplicitSchedule(2, 4).largest_acoustic_fraction())


def test_the_rk2_certificates_cover_both_counts():
    table = json.loads((TREE / "src/hexcore/data/dycore-step-anchors.v1.json").read_text())
    assert {row["acoustic_substeps"] for row in table["anchors"]} == {2, 4}


# ---------------------------------------------------------------------------
# v8.4.2 fixes with no hex counterpart
# ---------------------------------------------------------------------------
def test_the_prognostic_tke_closure_is_not_ported_and_is_refused_by_name():
    """v8.4.2 floors the eddy viscosities of the 1.5-order TKE closure
    (mpas_atm_dissipation_models.F:410,413).  hex has no LES closure; a run
    asking for one is refused rather than run without it."""

    with pytest.raises(ConfigurationRefusal, match="config_les_model"):
        V841DryDycoreConfig(config_les_model="3d_smagorinsky").validate()
    with pytest.raises(ConfigurationRefusal, match="config_les_model"):
        V841DryDycoreConfig(config_les_model="prognostic_1.5_order").validate()


def test_boundary_species_are_matched_by_name_not_by_declaration_order(tmp_path):
    """v8.4.2 swapped the declarations of lbc_nr and lbc_ni in Registry.xml so
    the boundary array's order matches the model's (ni before nr).  hex reads
    each boundary species by its name, so the order a file declares them in
    cannot route one species' boundary values to another."""

    netCDF4 = pytest.importorskip("netCDF4")
    from hexcore.drivers.run_cuda_v841_forecast import select_lbc_scalar_names

    shape = ("Time", "nCells", "nVertLevels")
    paths = []
    for label, order in (("v841", ("lbc_nr", "lbc_ni")), ("v842", ("lbc_ni", "lbc_nr"))):
        path = tmp_path / f"lbc.{label}.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as dataset:
            dataset.createDimension("Time", None)
            dataset.createDimension("nCells", 3)
            dataset.createDimension("nVertLevels", 2)
            for name in ("lbc_qr", "lbc_qv", *order, "lbc_qc"):
                dataset.createVariable(name, "f4", shape)[:] = np.zeros((1, 3, 2))
        paths.append(str(path))
    mass = ("qv", "qc", "qr")
    for path in paths:
        assert select_lbc_scalar_names([path], mass) == ("lbc_qv", "lbc_qc", "lbc_qr")
    # MPAS-A v8.4.2 Registry.xml:2188-2191 selects both number
    # concentrations. Configured order stays independent of file declaration.
    for path in paths:
        assert select_lbc_scalar_names([path], (*mass, "ni", "nr")) == (
            "lbc_qv", "lbc_qc", "lbc_qr", "lbc_ni", "lbc_nr")
