"""The thompson_aero_column backend row: table work on the frozen batch.

Breakage each check prevents is named inline.
"""

from __future__ import annotations

import pytest

from hexcore.errors import ConfigurationRefusal
from hexcore.physics_backend_admission import (
    DEFAULT_BACKEND, THOMPSON_AERO_COLUMN_ROW, resolve_backend,
)
from hexcore.species_row import species_row_for_scheme


def test_the_row_resolves_and_leaves_the_default_alone():
    assert resolve_backend("thompson_aero_column") is THOMPSON_AERO_COLUMN_ROW
    assert resolve_backend(DEFAULT_BACKEND).name == "wsm6_column"


def test_model_order_is_the_engine_species_row():
    # A different order would hand the engine's mp=28 seam the wrong tracer
    # in each slot (the batch reads its scalars by position).
    assert THOMPSON_AERO_COLUMN_ROW.model_scalar_names() == tuple(
        species_row_for_scheme("mp_thompson_aerosols").names())


def test_the_boundary_drives_only_the_six_masses():
    # An MPAS LBC stream carries lbc_qv..lbc_qg and the two aerosols but no
    # ni/nr/nc; driving all eleven refuses at the door ("noncontiguous model
    # species"), measured on conus12km 2026-10-03.
    assert THOMPSON_AERO_COLUMN_ROW.driven_scalar_names == (
        "qv", "qc", "qr", "qi", "qs", "qg")
    assert THOMPSON_AERO_COLUMN_ROW.appended_scalar_names == (
        "ni", "nr", "nc", "nwfa", "nifa")


def test_same_frozen_batch_and_scheme_alias():
    row = THOMPSON_AERO_COLUMN_ROW
    assert (row.batch_module, row.batch_entrypoint) == (
        "gpuwm.core.mpas_column_batch", "run_mpas_column_batch")
    adapter = row.load_adapter()
    assert adapter.scheme_alias_for(row.name) == "mp_thompson_aerosols"
    assert adapter.seam_options_for() == {}


def test_a_source_table_is_refused_not_ignored():
    adapter = THOMPSON_AERO_COLUMN_ROW.load_adapter()
    with pytest.raises(ConfigurationRefusal):
        adapter.seam_options_for(source_table="points.json")
