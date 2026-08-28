"""The lbc reader, the admission rules, and the two-level pool.

Two arms.  The synthetic arm always runs: it writes miniature but
schema-correct lbc files and proves the refusals, the admission ordering and
the float32 pool arithmetic.  The oracle arm runs against the three real
native case-9 files (2026-08-25 regional oracle) when
``GPUWM_HEX_LBC_ORACLE_DIR`` names the directory holding them, and is what
ties the module to actual producer bytes rather than to its own fixture —
the cross-lane-fixture lesson applied.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hexcore.lbc import (
    LBC_REQUIRED_VARIABLES,
    LbcAdmissionError,
    LbcFileError,
    LbcInventory,
    LbcPool,
    read_lbc_file,
    read_lbc_valid_time,
)

netCDF4 = pytest.importorskip("netCDF4")


# ---------------------------------------------------------------------------
# synthetic fixtures: tiny, schema-correct
# ---------------------------------------------------------------------------

N_CELLS, N_EDGES, NZ = 3, 4, 2


def write_lbc(path: Path, xtime: str, fill: float, **overrides) -> Path:
    """A miniature lbc file with the real stream's variables and dimensions."""

    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as d:
        d.createDimension("nVertLevels", NZ)
        d.createDimension("nCells", N_CELLS)
        d.createDimension("Time", None)
        d.createDimension("nEdges", N_EDGES)
        d.createDimension("nVertLevelsP1", NZ + 1)
        d.createDimension("StrLen", 64)
        for name, dims in LBC_REQUIRED_VARIABLES.items():
            if overrides.get("drop") == name:
                continue
            if overrides.get("transpose") == name:
                dims = (dims[0],) + tuple(reversed(dims[1:]))
            dtype = "f8" if overrides.get("widen") == name else "f4"
            v = d.createVariable(name, dtype, dims)
            shape = tuple(d.dimensions[n].size for n in dims[1:])
            v[0] = np.full(shape, fill, dtype=dtype)
        xt = d.createVariable("xtime", "S1", ("Time", "StrLen"))
        xt[0] = np.frombuffer(xtime.ljust(64).encode("ascii"), dtype="S1")
    return path


def series(tmp_path: Path) -> list[Path]:
    return [
        write_lbc(tmp_path / "lbc.a.nc", "2026-08-12_06:00:00", 1.0),
        write_lbc(tmp_path / "lbc.b.nc", "2026-08-12_09:00:00", 4.0),
        write_lbc(tmp_path / "lbc.c.nc", "2026-08-12_12:00:00", 2.0),
    ]


def at(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S")


# ---------------------------------------------------------------------------
# reader refusals
# ---------------------------------------------------------------------------


def test_the_valid_time_comes_from_xtime_not_the_filename(tmp_path):
    path = write_lbc(tmp_path / "misleading-name.nc", "2026-08-12_09:00:00", 1.0)
    assert read_lbc_valid_time(path) == at("2026-08-12_09:00:00")


def test_a_missing_variable_is_refused_by_name(tmp_path):
    path = write_lbc(tmp_path / "x.nc", "2026-08-12_06:00:00", 1.0, drop="lbc_w")
    with pytest.raises(LbcFileError, match="lbc_w"):
        read_lbc_file(path)


def test_transposed_dimensions_are_refused_not_reshaped(tmp_path):
    path = write_lbc(tmp_path / "x.nc", "2026-08-12_06:00:00", 1.0, transpose="lbc_u")
    with pytest.raises(LbcFileError, match="lbc_u"):
        read_lbc_file(path)


def test_a_widened_payload_is_refused(tmp_path):
    path = write_lbc(tmp_path / "x.nc", "2026-08-12_06:00:00", 1.0, widen="lbc_rho")
    with pytest.raises(LbcFileError, match="float32"):
        read_lbc_file(path)


def test_a_well_formed_file_reads_with_squeezed_float32_fields(tmp_path):
    f = read_lbc_file(series(tmp_path)[0])
    assert f.valid_time == at("2026-08-12_06:00:00")
    assert f.fields["lbc_u"].shape == (N_EDGES, NZ)
    assert f.fields["lbc_w"].shape == (N_CELLS, NZ + 1)
    assert f.fields["lbc_theta"].dtype == np.float32
    assert not f.fields["lbc_theta"].flags.writeable


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------


def test_latest_before_takes_the_file_at_or_before_the_model_time(tmp_path):
    inv = LbcInventory(series(tmp_path))
    assert inv.latest_before(at("2026-08-12_06:00:00")).name == "lbc.a.nc"
    assert inv.latest_before(at("2026-08-12_08:59:59")).name == "lbc.a.nc"
    assert inv.latest_before(at("2026-08-12_09:00:00")).name == "lbc.b.nc"


def test_latest_before_the_first_file_refuses_naming_rule_and_timeline(tmp_path):
    inv = LbcInventory(series(tmp_path))
    with pytest.raises(LbcAdmissionError) as error:
        inv.latest_before(at("2026-08-12_05:00:00"))
    message = str(error.value)
    assert "LATEST_BEFORE" in message
    assert "2026-08-12_05:00:00" in message
    assert "2026-08-12_06:00:00" in message  # the inventory it searched


def test_earliest_strictly_after_excludes_the_current_boundary(tmp_path):
    inv = LbcInventory(series(tmp_path))
    assert inv.earliest_strictly_after(at("2026-08-12_06:00:00")).name == "lbc.b.nc"
    assert inv.earliest_strictly_after(at("2026-08-12_09:00:00")).name == "lbc.c.nc"


def test_running_off_the_end_of_the_inventory_refuses_by_name(tmp_path):
    inv = LbcInventory(series(tmp_path))
    with pytest.raises(LbcAdmissionError) as error:
        inv.earliest_strictly_after(at("2026-08-12_12:00:00"))
    message = str(error.value)
    assert "EARLIEST_STRICTLY_AFTER" in message
    assert "2026-08-12_12:00:00" in message


def test_an_empty_inventory_refuses(tmp_path):
    with pytest.raises(LbcAdmissionError, match="empty"):
        LbcInventory([])


def test_duplicate_valid_times_refuse_rather_than_depend_on_order(tmp_path):
    a = write_lbc(tmp_path / "one.nc", "2026-08-12_06:00:00", 1.0)
    b = write_lbc(tmp_path / "two.nc", "2026-08-12_06:00:00", 2.0)
    with pytest.raises(LbcAdmissionError, match="same valid time"):
        LbcInventory([a, b])


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------


def test_the_pool_tendency_is_the_float32_interval_slope(tmp_path):
    pool = LbcPool(LbcInventory(series(tmp_path)))
    pool.start(at("2026-08-12_06:30:00"))
    assert pool.interval_end == at("2026-08-12_06:00:00")
    pool.advance()
    assert pool.interval_start == at("2026-08-12_06:00:00")
    assert pool.interval_end == at("2026-08-12_09:00:00")
    # (4 - 1) * float32(1/10800), formed exactly as the reference forms it.
    expected = np.float32(3.0) * (np.float32(1.0) / np.float32(10800.0))
    tend = pool.tendency("lbc_theta")
    assert tend.dtype == np.float32
    assert np.all(tend == expected)


def test_state_interpolates_linearly_backward_from_the_interval_end(tmp_path):
    pool = LbcPool(LbcInventory(series(tmp_path)))
    pool.start(at("2026-08-12_06:00:00"))
    pool.advance()
    # At the interval end the state is the admitted file, exactly.
    assert np.all(pool.state_at("lbc_theta", at("2026-08-12_09:00:00")) == np.float32(4.0))
    # Inside the interval: state(end) - (end - t) * tend, all float32.
    tend = pool.tendency("lbc_theta")[0, 0]
    expected = np.float32(4.0) - np.float32(5400.0) * tend
    got = pool.state_at("lbc_theta", at("2026-08-12_07:30:00"))
    assert got.dtype == np.float32
    assert np.all(got == expected)


def test_the_second_advance_swaps_the_interval_and_the_slope_sign(tmp_path):
    pool = LbcPool(LbcInventory(series(tmp_path)))
    pool.start(at("2026-08-12_06:00:00"))
    pool.advance()
    pool.advance()
    assert pool.interval_start == at("2026-08-12_09:00:00")
    assert pool.interval_end == at("2026-08-12_12:00:00")
    expected = np.float32(-2.0) * (np.float32(1.0) / np.float32(10800.0))
    assert np.all(pool.tendency("lbc_theta") == expected)


def test_a_pool_without_a_complete_interval_refuses_state_and_tendency(tmp_path):
    pool = LbcPool(LbcInventory(series(tmp_path)))
    with pytest.raises(LbcAdmissionError, match="never started"):
        pool.advance()
    pool.start(at("2026-08-12_06:00:00"))
    with pytest.raises(LbcAdmissionError, match="tendencies"):
        pool.tendency("lbc_theta")
    with pytest.raises(LbcAdmissionError, match="complete"):
        pool.state_at("lbc_theta", at("2026-08-12_06:30:00"))


def test_advancing_past_the_last_file_refuses_with_the_admission_rule(tmp_path):
    pool = LbcPool(LbcInventory(series(tmp_path)))
    pool.start(at("2026-08-12_06:00:00"))
    pool.advance()
    pool.advance()
    with pytest.raises(LbcAdmissionError, match="EARLIEST_STRICTLY_AFTER"):
        pool.advance()


def test_a_derived_field_name_is_refused_with_the_wiring_boundary_named(tmp_path):
    pool = LbcPool(LbcInventory(series(tmp_path)))
    pool.start(at("2026-08-12_06:00:00"))
    pool.advance()
    with pytest.raises(LbcFileError, match="lbc_rho_zz"):
        pool.state_at("lbc_rho_zz", at("2026-08-12_07:00:00"))


# ---------------------------------------------------------------------------
# the oracle arm: the three real native case-9 files
# ---------------------------------------------------------------------------

ORACLE_VAR = "GPUWM_HEX_LBC_ORACLE_DIR"
ORACLE_NAMES = (
    "lbc.2026-08-12_06.00.00.nc",
    "lbc.2026-08-12_09.00.00.nc",
    "lbc.2026-08-12_12.00.00.nc",
)


def oracle_paths() -> list[Path]:
    root = os.environ.get(ORACLE_VAR, "")
    if not root:
        pytest.skip(
            f"{ORACLE_VAR} is not set; the oracle arm needs the three native "
            f"case-9 lbc files of the 2026-08-25 regional record"
        )
    paths = [Path(root) / name for name in ORACLE_NAMES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"{ORACLE_VAR} is set but missing: {missing}")
    return paths


def test_oracle_files_read_with_the_measured_regional_shapes():
    files = [read_lbc_file(p) for p in oracle_paths()]
    for f in files:
        assert f.n_cells == 44770
        assert f.n_edges == 135107
        assert f.n_vert_levels == 55
        assert f.fields["lbc_w"].shape == (44770, 56)
        for name in LBC_REQUIRED_VARIABLES:
            assert f.fields[name].dtype == np.float32
        # The scalar package carries qc and qr as computed-zero fields.
        assert np.all(f.fields["lbc_qc"] == 0.0)
        assert np.all(f.fields["lbc_qr"] == 0.0)
    assert [f.valid_time for f in files] == [
        at("2026-08-12_06:00:00"),
        at("2026-08-12_09:00:00"),
        at("2026-08-12_12:00:00"),
    ]


def test_oracle_admission_walks_the_three_hour_series():
    paths = oracle_paths()
    inv = LbcInventory(paths)
    assert inv.latest_before(at("2026-08-12_07:12:00")) == paths[0]
    assert inv.earliest_strictly_after(at("2026-08-12_06:00:00")) == paths[1]
    with pytest.raises(LbcAdmissionError, match="EARLIEST_STRICTLY_AFTER"):
        inv.earliest_strictly_after(at("2026-08-12_12:00:00"))
    with pytest.raises(LbcAdmissionError, match="LATEST_BEFORE"):
        inv.latest_before(at("2026-08-12_05:59:59"))


def test_oracle_pool_reproduces_the_reference_tendency_arithmetic():
    paths = oracle_paths()
    pool = LbcPool(LbcInventory(paths))
    pool.start(at("2026-08-12_06:45:00"))
    pool.advance()

    with netCDF4.Dataset(paths[0]) as d0, netCDF4.Dataset(paths[1]) as d1:
        d0.set_auto_maskandscale(False)
        d1.set_auto_maskandscale(False)
        u0 = np.array(d0.variables["lbc_u"][0][:], dtype=np.float32)
        u1 = np.array(d1.variables["lbc_u"][0][:], dtype=np.float32)
    inv_dt = np.float32(np.float32(1.0) / np.float32(10800.0))
    expected_tend = (u1 - u0) * inv_dt
    assert np.array_equal(pool.tendency("lbc_u"), expected_tend)

    # At the interval end, the state is the 09z file bit-for-bit.
    assert np.array_equal(
        pool.state_at("lbc_u", at("2026-08-12_09:00:00")), u1
    )
    # Midway, the backward-from-the-end form, float32.
    expected_mid = u1 - np.float32(5400.0) * expected_tend
    assert np.array_equal(
        pool.state_at("lbc_u", at("2026-08-12_07:30:00")), expected_mid
    )

    # The second interval spans 09z -> 12z.
    pool.advance()
    assert pool.interval_start == at("2026-08-12_09:00:00")
    assert pool.interval_end == at("2026-08-12_12:00:00")
