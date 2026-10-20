"""Convection admission on a variable-resolution mesh: each cell decides.

THE BREAKAGE THIS PREVENTS (2026-10-11, the 48-3 km patch mesh from GFS
2026-10-03 12Z): the whole-mesh reading took the finest cell for every cell,
so ``min(dcEdge) = 2778.75 m`` switched cumulus off in the 48 km cells too
and the run made zero convective rain anywhere.  MPAS-A hands each scheme
the cell's own ``len_disp / meshDensity**0.25`` and lets the scale-aware
formulation fade the closure toward fine cells; the 3 km ruling is applied
per cell.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore import convection_admission
from hexcore.convection_admission import (
    CONVECTION_OFF_BELOW_M,
    SCHEME_GRELL_FREITAS,
    SCHEME_NTIEDTKE,
    SCHEME_OFF,
    ConvectionAdmissionError,
)


def _density_for(spacing_m: np.ndarray, len_disp_m: float) -> np.ndarray:
    return np.asarray((len_disp_m / spacing_m) ** 4, dtype=np.float32)


def test_cell_spacing_is_mpas_len_disp_over_density_to_the_quarter():
    density = np.asarray([1.0, 1.0 / 16.0, 1.0 / 256.0], dtype=np.float32)
    spacing = convection_admission.cell_spacing_m(density, 3_000.0)
    assert spacing.dtype == np.float32
    assert spacing.tolist() == [3_000.0, 6_000.0, 12_000.0]


def test_cell_spacing_refuses_a_density_outside_the_unit_interval():
    with pytest.raises(ConvectionAdmissionError, match="meshDensity"):
        convection_admission.cell_spacing_m(np.asarray([0.0, 1.0]), 3_000.0)
    with pytest.raises(ConvectionAdmissionError, match="meshDensity"):
        convection_admission.cell_spacing_m(np.asarray([1.5]), 3_000.0)


def test_a_48_to_3_km_mesh_runs_the_scheme_in_every_cell():
    """The measured case: finest cell 3000.28 m, min(dcEdge) 2778.75 m."""

    spacing = np.asarray([3_000.28, 6_000.0, 24_000.0, 48_000.0])
    decision = convection_admission.convection_decision(
        nominal_dx_m=3_000.0,
        minimum_dc_edge_m=2_778.75,
        cell_spacing_m=spacing,
    )
    assert decision["scheme"] == SCHEME_GRELL_FREITAS
    assert decision["constructor_scheme"] == "gf"
    assert decision["gf_ishallow"] == 1
    assert decision["source"] == "resolution"
    assert decision["below_threshold"] is False
    assert decision["finest_spacing_m"] == pytest.approx(3_000.28)
    variable = decision["variable_resolution"]
    assert variable["coarsest_cell_spacing_m"] == 48_000.0
    assert variable["cells_below_threshold"] == 0
    # The whole-mesh reading of the same mesh is the defect: off everywhere.
    whole = convection_admission.convection_decision(
        nominal_dx_m=3_000.0, minimum_dc_edge_m=2_778.75
    )
    assert whole["scheme"] == SCHEME_OFF


def test_the_60_to_3_km_and_15_to_3_km_meshes_run_grell_freitas():
    for spacing in ([3_034.9, 9_000.0, 60_000.0], [3_000.0, 9_000.0, 15_000.0]):
        decision = convection_admission.convection_decision(
            nominal_dx_m=3_000.0,
            minimum_dc_edge_m=2_700.0,
            cell_spacing_m=np.asarray(spacing),
        )
        assert decision["scheme"] == SCHEME_GRELL_FREITAS


def test_a_variable_mesh_whose_finest_cell_is_20_km_or_coarser_runs_new_tiedtke():
    """x4.163842-shaped: 25-100 km, every cell in the New Tiedtke band."""

    decision = convection_admission.convection_decision(
        nominal_dx_m=25_000.0,
        minimum_dc_edge_m=21_000.0,
        cell_spacing_m=np.asarray([25_000.0, 50_000.0, 100_000.0]),
    )
    assert decision["scheme"] == SCHEME_NTIEDTKE
    assert decision["coarse_band"] is True


def test_a_registered_swath_keeps_the_scheme_it_was_anchored_with():
    """v6.75.114257: finest 7.936 km, coarsest 84 km, bound (20 s, gf)."""

    decision = convection_admission.convection_decision(
        nominal_dx_m=7_936.0,
        minimum_dc_edge_m=7_400.0,
        cell_spacing_m=np.asarray([7_936.0, 30_000.0, 84_007.5]),
    )
    whole = convection_admission.convection_decision(
        nominal_dx_m=7_936.0, minimum_dc_edge_m=7_400.0
    )
    assert decision["scheme"] == whole["scheme"] == SCHEME_GRELL_FREITAS


def test_a_variable_mesh_with_cells_finer_than_3_km_stays_off_and_says_why():
    """No per-column switch exists, so the 3 km ruling keeps the whole mesh off."""

    spacing = np.asarray([1_000.0, 2_000.0, 15_000.0, 60_000.0])
    decision = convection_admission.convection_decision(
        nominal_dx_m=1_000.0, minimum_dc_edge_m=900.0, cell_spacing_m=spacing
    )
    assert decision["scheme"] == SCHEME_OFF
    assert decision["below_threshold"] is True
    status = decision["variable_resolution"]["status"]
    assert "2 cells are finer than" in status
    # its coarse cells (60 km) would take New Tiedtke
    assert "New Tiedtke" in status


def test_a_uniform_mesh_is_decided_exactly_as_before():
    """meshDensity 1 everywhere: per-cell spacing changes nothing, byte for byte."""

    for nominal, edge in ((3_000.0, 2_750.0), (15_000.0, 13_000.0), (120_000.0, 97_076.0)):
        spacing = np.full(7, nominal)
        with_cells = convection_admission.convection_decision(
            nominal_dx_m=nominal, minimum_dc_edge_m=edge, cell_spacing_m=spacing
        )
        without = convection_admission.convection_decision(
            nominal_dx_m=nominal, minimum_dc_edge_m=edge
        )
        assert with_cells == without
        assert "variable_resolution" not in with_cells


def test_preflight_from_the_row_alone_agrees_with_the_bind():
    spacing = np.asarray([3_000.28, 48_000.0])
    bind = convection_admission.convection_decision(
        nominal_dx_m=3_000.0, minimum_dc_edge_m=2_778.75, cell_spacing_m=spacing
    )
    row = convection_admission.convection_decision(
        nominal_dx_m=3_000.0, coarsest_dx_m=48_000.0
    )
    assert row["scheme"] == bind["scheme"] == SCHEME_GRELL_FREITAS
    fine_row = convection_admission.convection_decision(
        nominal_dx_m=1_000.0, coarsest_dx_m=48_000.0
    )
    assert fine_row["scheme"] == SCHEME_OFF


def test_explicit_requests_still_override_and_say_so():
    spacing = np.asarray([3_000.0, 60_000.0])
    off = convection_admission.convection_decision(
        nominal_dx_m=3_000.0, cell_spacing_m=spacing, requested="off"
    )
    assert off["scheme"] == SCHEME_OFF
    assert off["source"] == "explicit"
    assert "OVERRIDING" in off["note"]
    gf = convection_admission.convection_decision(
        nominal_dx_m=3_000.0, cell_spacing_m=spacing, requested="gf"
    )
    assert gf["scheme"] == SCHEME_GRELL_FREITAS
    with pytest.raises(ConvectionAdmissionError):
        convection_admission.convection_decision(
            nominal_dx_m=3_000.0, cell_spacing_m=spacing, requested="nope"
        )


def test_the_threshold_is_applied_per_cell_at_the_ruling_value():
    """A cell exactly at 3 km is not below it; one a metre finer is."""

    at = convection_admission.convection_decision(
        nominal_dx_m=3_000.0,
        cell_spacing_m=np.asarray([CONVECTION_OFF_BELOW_M, 60_000.0]),
    )
    assert at["scheme"] == SCHEME_GRELL_FREITAS
    under = convection_admission.convection_decision(
        nominal_dx_m=3_000.0,
        cell_spacing_m=np.asarray([CONVECTION_OFF_BELOW_M - 1.0, 60_000.0]),
    )
    assert under["scheme"] == SCHEME_OFF


def test_density_round_trip_matches_the_column_physics_construction():
    """The admission classifies the same per-cell length the scheme receives."""

    from hexcore.cuda_gwdo_v841 import native_cell_dx_m

    spacing = np.asarray([3_000.0, 4_500.0, 12_000.0, 48_000.0])
    density = _density_for(spacing, 3_000.0)
    admission = convection_admission.cell_spacing_m(density, 3_000.0)
    physics = native_cell_dx_m(density, 3_000.0)
    np.testing.assert_allclose(admission, physics, rtol=1e-6)
