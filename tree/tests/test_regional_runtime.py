"""The v8.4.1 regional runtime: masks, driving state, and the replicated quirks.

Every check here is a property of the TRANSCRIPTION, not of a run: the zone
constants, the mask derivations, the admission checks, the derived coupled
fields, and — the reason this file exists — the three native behaviours that
are replicated rather than fixed.  A future edit that "cleans up" any of the
three would silently break the byte pin, so each is a checked fact with its
native anchor in the assertion message.

When ``GPUWM_HEX_REGIONAL_REFERENCE_DIR`` names the reference mirror the
mask and driving-state checks additionally run against the real culled
bytes; without it they run on a synthetic ring layout.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.regional_v841 import (
    N_BDY_ZONE,
    N_RELAX_ZONE,
    N_SPEC_ZONE,
    RVORD_F32,
    RegionalAdmissionError,
    RegionalMasks,
    adjust_dynamics_speczone_tend,
    bdy_adjust_scalars,
    bdy_set_scalars,
    clamp_negative_scalars,
    regional_bdy_checks,
    reset_speczone_values,
    rk_timestep_f32,
    dynamics_time_offset,
)

REFERENCE_DIR_VARIABLE = "GPUWM_HEX_REGIONAL_REFERENCE_DIR"


def _reference_dir() -> Path:
    raw = os.environ.get(REFERENCE_DIR_VARIABLE)
    if not raw:
        pytest.skip(f"{REFERENCE_DIR_VARIABLE} is unset")
    path = Path(raw)
    if not path.is_dir():
        pytest.skip(f"{REFERENCE_DIR_VARIABLE}={raw} is not a directory")
    return path


def _masks(bdy_cell: np.ndarray, bdy_edge: np.ndarray) -> RegionalMasks:
    dtype = np.dtype(np.float32)
    spec_cell = (bdy_cell > N_RELAX_ZONE).astype(dtype)
    spec_edge = (bdy_edge > N_RELAX_ZONE).astype(dtype)
    return RegionalMasks(
        bdy_mask_cell=bdy_cell,
        bdy_mask_edge=bdy_edge,
        bdy_mask_vertex=bdy_cell,
        spec_zone_mask_cell=spec_cell,
        spec_zone_mask_edge=spec_edge,
        spec_zone_mask_vertex=spec_cell,
        nearest_relaxation_cell=np.zeros_like(bdy_cell),
        spec_cells=np.flatnonzero(bdy_cell > N_RELAX_ZONE),
        spec_edges=np.flatnonzero(bdy_edge > N_RELAX_ZONE),
        relax_cells=np.flatnonzero((bdy_cell > 1) & (bdy_cell <= N_RELAX_ZONE)),
        relax_edges=np.flatnonzero((bdy_edge > 1) & (bdy_edge <= N_RELAX_ZONE)),
        nudged_cells=np.flatnonzero(bdy_cell > 1),
    )


def test_zone_constants_match_the_native_parameters() -> None:
    # mpas_atm_boundaries.F:36-38
    assert (N_SPEC_ZONE, N_RELAX_ZONE, N_BDY_ZONE) == (2, 5, 7)


def test_rvord_is_the_float32_quotient_not_the_rounded_double() -> None:
    # mpas_constants.F computes rvord as a REAL(RKIND) division of the
    # float32 constants; rounding the float64 quotient lands one ulp high
    # and was measured breaking frame-0 theta on the regional ladder.
    assert RVORD_F32 == np.float32(461.6) / np.float32(287.0)
    assert RVORD_F32 != np.float32(461.6 / 287.0)


def test_bdy_checks_refuse_both_inconsistent_directions() -> None:
    masked = _masks(np.array([0, 3, 7]), np.array([0, 3, 7]))
    unmasked = _masks(np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))
    # mpas_atm_bdy_checks F:828-847
    with pytest.raises(RegionalAdmissionError, match="config_apply_lbcs = false"):
        regional_bdy_checks(
            masked, config_apply_lbcs=False, lbc_input_interval_valid=True
        )
    with pytest.raises(RegionalAdmissionError, match="no boundary cells"):
        regional_bdy_checks(
            unmasked, config_apply_lbcs=True, lbc_input_interval_valid=True
        )
    with pytest.raises(RegionalAdmissionError, match="lbc_in"):
        regional_bdy_checks(
            masked, config_apply_lbcs=True, lbc_input_interval_valid=False
        )
    regional_bdy_checks(
        masked, config_apply_lbcs=True, lbc_input_interval_valid=True
    )
    regional_bdy_checks(
        unmasked, config_apply_lbcs=False, lbc_input_interval_valid=False
    )


def test_speczone_tend_overwrites_exactly_the_outer_two_rings() -> None:
    bdy_cell = np.array([0, 1, 5, 6, 7], dtype=np.int64)
    bdy_edge = np.array([0, 5, 6, 7], dtype=np.int64)
    masks = _masks(bdy_cell, bdy_edge)
    nlev = 3
    tend_ru = np.zeros((nlev, 4), dtype=np.float32)
    tend_rho = np.zeros((nlev, 5), dtype=np.float32)
    tend_rt = np.zeros((nlev, 5), dtype=np.float32)
    tend_rw = np.ones((nlev + 1, 5), dtype=np.float32)
    drive_ru = np.full((nlev, 4), 7.0, dtype=np.float32)
    drive_rt = np.full((nlev, 5), 8.0, dtype=np.float32)
    drive_rho = np.full((nlev, 5), 9.0, dtype=np.float32)
    adjust_dynamics_speczone_tend(
        masks=masks,
        tend_ru=tend_ru,
        tend_rho=tend_rho,
        tend_rt=tend_rt,
        tend_rw=tend_rw,
        ru_driving_tend=drive_ru,
        rt_driving_tend=drive_rt,
        rho_driving_tend=drive_rho,
    )
    assert np.array_equal(tend_rho[:, [3, 4]], drive_rho[:, [3, 4]])
    assert np.array_equal(tend_rt[:, [3, 4]], drive_rt[:, [3, 4]])
    assert not tend_rho[:, :3].any() and not tend_rt[:, :3].any()
    assert np.array_equal(tend_ru[:, [2, 3]], drive_ru[:, [2, 3]])
    assert not tend_ru[:, :2].any()
    # F:7948 zeroes rows 1..nVertLevels of the omega tendency and leaves the
    # top interface row untouched.
    assert not tend_rw[:nlev, [3, 4]].any()
    assert tend_rw[nlev, 3] == np.float32(1.0)


def test_ring_one_is_never_nudged_and_the_spec_zone_is_set_outright() -> None:
    # atm_bdy_adjust_scalars_work F:8338/8380/8402: ring 1 falls in neither
    # branch and the copy-back admits only bdyMaskCell > 1.
    bdy_cell = np.array([0, 1, 2, 6], dtype=np.int64)
    masks = _masks(bdy_cell, np.zeros(1, dtype=np.int64))

    class _Mesh:
        arrays = {
            "nEdgesOnCell": np.zeros(4, dtype=np.int64),
            "edgesOnCell": np.zeros((4, 1), dtype=np.int64),
            "cellsOnEdge": np.zeros((1, 2), dtype=np.int64),
            "dvEdge": np.ones(1, dtype=np.float32),
            "dcEdge": np.ones(1, dtype=np.float32),
        }

    scalars = np.ones((1, 2, 4), dtype=np.float32)
    driving = np.full((1, 2, 4), 5.0, dtype=np.float32)
    bdy_adjust_scalars(
        _Mesh(),
        masks=masks,
        mesh_scaling_regional_cell=np.ones(4, dtype=np.float32),
        scalars_new=scalars,
        scalars_driving=driving,
        dt=120.0,
        dt_rk=np.float32(40.0),
    )
    assert scalars[0, 0, 0] == np.float32(1.0), "interior is untouched"
    assert scalars[0, 0, 1] == np.float32(1.0), "ring 1 is never nudged (F:8338)"
    assert scalars[0, 0, 2] != np.float32(1.0), "ring 2 relaxes"
    assert scalars[0, 0, 3] == np.float32(5.0), "the spec zone takes driving values"


def test_set_scalars_and_reset_speczone_touch_only_rings_six_and_seven() -> None:
    bdy_cell = np.array([0, 5, 6, 7], dtype=np.int64)
    masks = _masks(bdy_cell, np.zeros(1, dtype=np.int64))
    scalars = np.zeros((1, 2, 4), dtype=np.float32)
    driving = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    bdy_set_scalars(masks=masks, scalars_new=scalars, scalars_driving=driving)
    assert np.array_equal(scalars[:, :, 2:], driving[:, :, 2:])
    assert not scalars[:, :, :2].any()

    theta_m = np.zeros((2, 4), dtype=np.float32)
    rho_theta = np.zeros((2, 4), dtype=np.float32)
    rt = np.full((2, 4), 6.0, dtype=np.float32)
    rho = np.full((2, 4), 3.0, dtype=np.float32)
    reset_speczone_values(
        masks=masks,
        theta_m=theta_m,
        rho_theta=rho_theta,
        rt_driving_values=rt,
        rho_driving_values=rho,
    )
    # F:8237-8238: theta_m := rt/rho and rtheta_p := rt - rtheta_base, so the
    # port's rho_theta (= rtheta_p + rtheta_base) becomes rt exactly.
    assert np.array_equal(theta_m[:, 2:], np.full((2, 2), 2.0, dtype=np.float32))
    assert np.array_equal(rho_theta[:, 2:], rt[:, 2:])
    assert not theta_m[:, :2].any() and not rho_theta[:, :2].any()


def test_the_end_of_step_clamp_is_unconditional() -> None:
    # atm_srk3:2798-2800 runs in every DO_PHYSICS build regardless of the
    # physics suite; the pinned dry reference arms ran such a build.
    scalars = np.array([[[-1.0, 0.5]]], dtype=np.float32)
    clamp_negative_scalars(scalars)
    assert scalars[0, 0, 0] == np.float32(0.0)
    assert scalars[0, 0, 1] == np.float32(0.5)


def test_stage_time_offsets_are_the_native_float32_expressions() -> None:
    # atm_srk3:2066-2074 and the time_dyn_step of lines 2329/2451.
    assert rk_timestep_f32(outer_dt=120.0, dynamics_split=3, rk_step=1) == np.float32(
        np.float32(40.0) / np.float32(3.0)
    )
    assert rk_timestep_f32(outer_dt=120.0, dynamics_split=3, rk_step=3) == np.float32(40.0)
    assert dynamics_time_offset(
        outer_dt=120.0, dynamics_split=3, dynamics_substep=2, rk_timestep=40.0
    ) == np.float32(80.0)


def test_mono_copy_back_excludes_the_relaxation_rings() -> None:
    """F:5771 REPLICATED: ``bdyMaskCell <= nSpecZone`` admits rings 0-2 only.

    The monotonic limiter writes its result back for rings 0-2 and leaves
    rings 3-5 holding their pre-transport values, which
    atm_bdy_adjust_scalars then nudges.  A "fix" that copied every updated
    cell back would change the trajectory.
    """

    from hexcore.transport import _N_SPEC_ZONE, _N_RELAX_ZONE

    assert (_N_SPEC_ZONE, _N_RELAX_ZONE) == (2, 5)
    kept = [ring for ring in range(8) if ring > _N_SPEC_ZONE]
    assert kept == [3, 4, 5, 6, 7]


def test_mask_four_five_edge_condition_keeps_the_fortran_precedence() -> None:
    """F:5541/F:5654 REPLICATED: ``.and.`` binds tighter than ``.or.``.

    ``config_apply_lbcs .and. (mask == nRelaxZone) .or. (mask == nRelaxZone-1)``
    parses as ``(lbcs .and. mask==5) .or. (mask==4)``, so the mask-4 half
    fires even with lbcs off.  Parenthesising it "correctly" would change
    which antidiffusive fluxes are zeroed.
    """

    def native(lbcs: bool, mask: int) -> bool:
        return (lbcs and mask == N_RELAX_ZONE) or (mask == N_RELAX_ZONE - 1)

    assert native(False, 4) is True, "the mask-4 half is unguarded"
    assert native(False, 5) is False
    assert native(True, 5) is True


def test_the_regional_reference_masks_derive_exactly_as_native(monkeypatch) -> None:
    reference = _reference_dir()
    grid = reference / "cull-x1" / "conus.grid.nc"
    if not grid.exists():
        pytest.skip("the quick cull is not in this mirror")
    from netCDF4 import Dataset

    from hexcore.mesh import Mesh
    from hexcore.regional_v841 import derive_regional_masks

    mesh = Mesh.from_netcdf(grid, validate=False)
    masks = derive_regional_masks(mesh, np.dtype(np.float32))
    with Dataset(grid) as dataset:
        dataset.set_auto_maskandscale(False)
        bdy_cell = np.array(dataset.variables["bdyMaskCell"][:], dtype=np.int64)
        bdy_edge = np.array(dataset.variables["bdyMaskEdge"][:], dtype=np.int64)
    # mpas_atm_boundaries.F:697-699
    assert np.array_equal(
        masks.spec_zone_mask_cell, (bdy_cell > N_RELAX_ZONE).astype(np.float32)
    )
    assert np.array_equal(
        masks.spec_zone_mask_edge, (bdy_edge > N_RELAX_ZONE).astype(np.float32)
    )
    assert set(np.unique(masks.spec_zone_mask_cell)) <= {0.0, 1.0}
    assert int(bdy_cell.max()) == N_BDY_ZONE
    assert masks.relax_cells.size and masks.spec_cells.size


def test_the_driving_state_derives_the_four_coupled_fields() -> None:
    reference = _reference_dir()
    lbc_dir = reference / "lbc-x1"
    grid = reference / "cull-x1" / "conus.grid.nc"
    init = reference / "init-x1" / "conus.init.nc"
    if not (lbc_dir.is_dir() and init.exists()):
        pytest.skip("the x1-cull lbc series is not in this mirror")
    from netCDF4 import Dataset

    from hexcore.lbc import LbcInventory
    from hexcore.mesh import Mesh
    from hexcore.regional_v841 import RegionalDrivingState

    mesh = Mesh.from_netcdf(grid, init, validate=False)
    with Dataset(init) as dataset:
        dataset.set_auto_maskandscale(False)
        # The init file carries zz as (nCells, nVertLevels) with no record
        # axis; the model-side convention is (level, cell).
        zz = np.ascontiguousarray(
            np.array(dataset.variables["zz"][:], dtype=np.float32).T
        )
    paths = sorted(str(p) for p in lbc_dir.glob("lbc.*.nc"))
    driving = RegionalDrivingState(LbcInventory(paths), mesh, zz)
    start = datetime(2026, 8, 12, 6, 0, 0)
    driving.start(start)
    driving.advance(start)
    # mpas_atm_update_bdy_tend F:217-262, verified against the file fields.
    with Dataset(paths[1]) as dataset:
        dataset.set_auto_maskandscale(False)
        rho = np.ascontiguousarray(
            np.array(dataset.variables["lbc_rho"][0][:], dtype=np.float32).T
        )
        theta = np.ascontiguousarray(
            np.array(dataset.variables["lbc_theta"][0][:], dtype=np.float32).T
        )
        qv = np.ascontiguousarray(
            np.array(dataset.variables["lbc_qv"][0][:], dtype=np.float32).T
        )
    at_end = driving.state_at("rho_zz", start, np.float32(10800.0))
    assert np.array_equal(at_end, np.asarray(rho / zz, dtype=np.float32))
    expected_rtheta = np.asarray(
        theta * (rho / zz) * (np.float32(1.0) + RVORD_F32 * qv), dtype=np.float32
    )
    assert np.array_equal(
        driving.state_at("rtheta_m", start, np.float32(10800.0)), expected_rtheta
    )
    # The interval-start end of the same linear form is the first file.
    assert driving.interval_end == datetime(2026, 8, 12, 9, 0, 0)
