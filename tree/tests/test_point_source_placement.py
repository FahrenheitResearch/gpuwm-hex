"""The point-source placement instrument: the table it writes, the census it takes, the line it places.

Everything here runs on a CPU-only box and starts no forecast.  A synthetic
leg is written by hand -- a 41 x 41 patch of cells on a 0.02-degree
latitude/longitude lattice, seven levels, two valid times -- with cloud water
at sub-zero temperatures in ONE north-south strip and nowhere else, and a
uniform westerly wind.  The tool is run over it exactly as a user runs it.

WHY THE TABLE NUMBERS ARE ASSERTED OUTRIGHT.  The table is what the engine
reads; an epoch off by a day, a pass gap off by a minute or an ``on`` flag
on the wrong waypoint changes the injection without any error.  The first
epoch asserted below is the first row of the 2026-09-02 3.75 km proof's own
table (``303289200.0`` for 2026-08-12 07:00Z), so the arithmetic here is
pinned to the table that was proven to respond.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import numpy
import pytest
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "point_source_placement.py"


def _load():
    spec = importlib.util.spec_from_file_location("point_source_placement", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load()

# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

PROOF_RELEASE = dt.datetime(2026, 8, 12, 7, 0, 0)
PROOF_FIRST_EPOCH = 303289200.0  # the 3.75 km proof's first row, 2026-08-12 07:00:00Z


def test_the_epoch_is_the_engines_2017_epoch() -> None:
    assert tool.epoch_seconds(PROOF_RELEASE) == PROOF_FIRST_EPOCH
    assert tool.epoch_seconds(dt.datetime(2026, 9, 10, 19, 0, 0)) == 305838000.0
    assert tool.epoch_seconds(tool.EPOCH) == 0.0


def test_the_table_rows_carry_the_proven_pass_structure() -> None:
    rows = tool.table_rows(43.55, -84.25, 5000.0, PROOF_RELEASE)
    assert len(rows) == 15
    epochs = [row[0] for row in rows]
    assert epochs[0] == PROOF_FIRST_EPOCH
    # five waypoints 180 s apart inside a pass; passes 21 min apart
    for pass_index in range(3):
        block = epochs[5 * pass_index:5 * pass_index + 5]
        assert [b - block[0] for b in block] == [0.0, 180.0, 360.0, 540.0, 720.0]
        assert block[0] - epochs[0] == 1260.0 * pass_index
    # pass 1 south to north, pass 2 north to south, pass 3 south to north
    lats = [row[1] for row in rows]
    assert lats[0:5] == [43.55, 43.65, 43.75, 43.85, 43.95]
    assert lats[5:10] == [43.95, 43.85, 43.75, 43.65, 43.55]
    assert lats[10:15] == lats[0:5]
    # the last waypoint of every pass is off; longitude, altitude and rate never move
    assert [row[4] for row in rows] == [1, 1, 1, 1, 0] * 3
    assert {row[2] for row in rows} == {-84.25}
    assert {row[3] for row in rows} == {5000.0}
    assert {row[5] for row in rows} == {tool.RATE_DEFAULT}
    assert [row[6] for row in rows] == [1] * 5 + [2] * 5 + [3] * 5


def test_the_table_text_reads_back_to_its_rows_and_says_synthetic() -> None:
    text = tool.table_text(43.55, -84.25, 5000.0, PROOF_RELEASE, notes=("placed by hand",))
    lines = text.splitlines()
    comments = [line for line in lines if line.startswith("#")]
    data = [line for line in lines if not line.startswith("#")]
    assert "SYNTHETIC" in comments[1]
    assert "07:00-07:54Z" in "\n".join(comments)
    assert "# placed by hand" in comments
    assert comments[-1].split() == ["#", "epoch_sec", "lat_deg", "lon_deg", "alt_m_MSL", "on",
                                    "rate_part_per_s", "src_id"]
    assert len(data) == 15
    parsed = [tuple(float(field) for field in line.split()) for line in data]
    expected = tool.table_rows(43.55, -84.25, 5000.0, PROOF_RELEASE)
    for got, want in zip(parsed, expected):
        assert got[0] == want[0]
        assert got[1] == pytest.approx(want[1])
        assert got[2] == pytest.approx(want[2])
        assert got[3] == want[3]
        assert int(got[4]) == want[4]
        assert got[5] == pytest.approx(want[5])
        assert int(got[6]) == want[6]


def test_the_upwind_shift_moves_against_the_wind() -> None:
    dlat, dlon = tool.upwind_shift(35.0, 10.0, 0.0, 1800.0)
    assert dlat == 0.0
    assert dlon == pytest.approx(-10.0 * 1800.0 / (111_000.0 * numpy.cos(numpy.radians(35.0))))
    assert dlon < 0  # a westerly carries the plume east, so upwind is west
    dlat, dlon = tool.upwind_shift(35.0, 0.0, -5.0, 1800.0)
    assert dlon == 0.0
    assert dlat == pytest.approx(5.0 * 1800.0 / 111_000.0)  # a northerly: upwind is north


def test_the_table_command_writes_the_file_and_its_placement_record(tmp_path: Path) -> None:
    out = tmp_path / "table.txt"
    rc = tool.main(["table", "--lat0", "34.958", "--lon", "-96.78", "--alt-m", "6200",
                    "--release", "2026-09-10_19:00:00", "--out", str(out),
                    "--case", "a synthetic case", "--note", "one note"])
    assert rc == 0
    text = out.read_text()
    assert text.splitlines()[0] == "# Point-source table, a synthetic case."
    data = [line for line in text.splitlines() if not line.startswith("#")]
    assert data[0].split() == ["305838000.0", "34.958", "-96.78", "6200", "1", "8.5e+13", "1"]
    record = json.loads((tmp_path / "table.txt.placement.json").read_text())
    assert record["lat1"] == 35.358
    assert record["release"] == "2026-09-10_19:00:00"
    assert record["notes"] == ["one note"]


# --------------------------------------------------------------------------
# a synthetic leg: one supercooled strip, a westerly wind
# --------------------------------------------------------------------------

N = 41
LEVELS = 7
LAT0, LON0, STEP = 34.6, -97.4, 0.02
CENTER = (35.0, -97.0)
STRIP_LON = -96.9
STRIP_LAT = (34.8, 35.2)
WARM_THETA, COLD_THETA = 280.0, 265.0  # pressure is P0, so T = theta
U, V = 10.0, 0.0


def _lattice():
    lat = LAT0 + STEP * numpy.repeat(numpy.arange(N), N)
    lon = LON0 + STEP * numpy.tile(numpy.arange(N), N)
    return lat, lon


def _strip(lat, lon):
    return ((numpy.abs(lon - STRIP_LON) <= 0.03 + 1e-9)
            & (lat >= STRIP_LAT[0] - 1e-9) & (lat <= STRIP_LAT[1] + 1e-9))


def _write_leg(root: Path):
    lat, lon = _lattice()
    cells = lat.size
    grid = root / "grid.nc"
    with Dataset(str(grid), "w", format="NETCDF4") as ds:
        ds.createDimension("nCells", cells)
        ds.createVariable("latCell", "f8", ("nCells",))[:] = numpy.radians(lat)
        ds.createVariable("lonCell", "f8", ("nCells",))[:] = numpy.radians(lon)
    init = root / "init.nc"
    with Dataset(str(init), "w", format="NETCDF4") as ds:
        ds.createDimension("nCells", cells)
        ds.createDimension("nVertLevelsP1", LEVELS + 1)
        z = ds.createVariable("zgrid", "f8", ("nCells", "nVertLevelsP1"))
        z[:] = numpy.tile(1000.0 * numpy.arange(LEVELS + 1), (cells, 1))
    out = root / "out"
    out.mkdir()
    strip = _strip(lat, lon)
    for stamp in ("2026-09-10_19.30.00", "2026-09-10_20.00.00"):
        path = out / f"cuda-history.{stamp}.nc"
        with Dataset(str(path), "w", format="NETCDF4") as ds:
            ds.createDimension("Time", None)
            ds.createDimension("nCells", cells)
            ds.createDimension("nVertLevels", LEVELS)
            theta = numpy.full((cells, LEVELS), WARM_THETA)
            theta[:, 5:] = COLD_THETA
            qc = numpy.zeros((cells, LEVELS))
            qc[strip, 5:] = 2.0e-4
            def var(name, values):
                variable = ds.createVariable(name, "f4", ("Time", "nCells", "nVertLevels"))
                variable[0, :, :] = values
            var("qc", qc)
            var("theta", theta)
            var("pressure", numpy.full((cells, LEVELS), tool.P0))
            var("u_zonal", numpy.full((cells, LEVELS), U))
            var("v_meridional", numpy.full((cells, LEVELS), V))
    return grid, init, out, int(strip.sum())


def test_the_census_counts_the_strip_and_reads_the_wind(tmp_path: Path, capsys) -> None:
    grid, init, out, strip_cells = _write_leg(tmp_path)
    assert strip_cells == 63  # three lattice columns by twenty-one rows
    report = tmp_path / "census.json"
    rc = tool.main(["census", "--leg-out", str(out), "--grid", str(grid), "--init", str(init),
                    "--center", str(CENTER[0]), str(CENTER[1]), "--radius-km", "100",
                    "--json", str(report)])
    assert rc == 0
    rows = json.loads(report.read_text())
    assert [row["frame"] for row in rows] == ["2026-09-10_19.30.00", "2026-09-10_20.00.00"]
    for row in rows:
        assert row["slw_cells"] == 2 * strip_cells  # two cold levels in every strip cell
        assert row["slw_cols"] == strip_cells
        assert row["qc_cols"] == strip_cells
        assert row["T_min"] == pytest.approx(COLD_THETA - 273.15, abs=1e-3)
        assert row["wlon"] == pytest.approx(STRIP_LON, abs=1e-6)
        assert row["wlat"] == pytest.approx(35.0, abs=1e-6)
        assert row["wz"] == pytest.approx(6000.0)  # levels 5 and 6 at 5,500 and 6,500 m
        assert row["u"] == pytest.approx(U)
        assert row["v"] == pytest.approx(V)
    printed = capsys.readouterr().out
    assert "core cells within 100 km" in printed


def test_the_placement_finds_the_strip_and_shifts_the_line_upwind(tmp_path: Path) -> None:
    grid, init, out, _ = _write_leg(tmp_path)
    report = tmp_path / "placement.json"
    rc = tool.main(["place", "--leg-out", str(out), "--grid", str(grid), "--init", str(init),
                    "--center", str(CENTER[0]), str(CENTER[1]),
                    "--frames", "2026-09-10_19.30.00,2026-09-10_20.00.00",
                    "--lead-min", "30", "--json", str(report)])
    assert rc == 0
    result = json.loads(report.read_text())
    target = result["target_segment"]
    assert abs(target["lon"] - STRIP_LON) <= 0.05
    assert abs(target["lat0"] - STRIP_LAT[0]) <= 0.1
    assert target["alt_m_weighted"] == 6000.0
    assert target["u"] == pytest.approx(U)
    assert target["v"] == pytest.approx(V)
    assert target["T_wtd_C"] == pytest.approx(COLD_THETA - 273.15, abs=1e-3)
    line = result["line"]
    assert line["alt_source"] == "qc-weighted"
    assert line["alt_m"] == 6000.0
    expected_dlon = tool.upwind_shift(target["lat0"] + 0.2, U, V, 1800.0)[1]
    assert line["lon"] == pytest.approx(target["lon"] + expected_dlon, abs=0.006)
    assert line["lat0"] == pytest.approx(target["lat0"], abs=1e-6)
    assert line["lat1"] == pytest.approx(line["lat0"] + 0.4, abs=1e-6)
    # the line sits 0.2 deg west of the strip, in clear air; the target is in the strip
    for frame in result["check_line"].values():
        assert frame["slw_frac"] == 0.0
        assert frame["u"] == pytest.approx(U)
    for frame in result["check_target"].values():
        assert frame["slw_frac"] == 1.0
        assert frame["T_max"] < 0.0
    assert result["result_sentence"].startswith("a model result of the control leg")


def test_the_placement_can_be_told_a_colder_level(tmp_path: Path) -> None:
    grid, init, out, _ = _write_leg(tmp_path)
    report = tmp_path / "placement.json"
    tool.main(["place", "--leg-out", str(out), "--grid", str(grid), "--init", str(init),
               "--center", str(CENTER[0]), str(CENTER[1]),
               "--frames", "2026-09-10_19.30.00", "--alt-m", "6500", "--json", str(report)])
    result = json.loads(report.read_text())
    assert result["line"]["alt_source"] == "explicit --alt-m"
    assert result["line"]["alt_m"] == 6500.0
    profile = result["target_profile"][0]
    assert len(profile["slw_cells_per_level"]) == LEVELS
    assert profile["slw_cells_per_level"][:5] == [0, 0, 0, 0, 0]
    assert profile["slw_cells_per_level"][5] > 0 and profile["slw_cells_per_level"][6] > 0


def test_the_placement_refuses_a_leg_with_nothing_to_seed(tmp_path: Path) -> None:
    grid, init, out, _ = _write_leg(tmp_path)
    with pytest.raises(SystemExit, match="nothing to place"):
        tool.main(["place", "--leg-out", str(out), "--grid", str(grid), "--init", str(init),
                   "--center", str(CENTER[0]), str(CENTER[1]),
                   "--frames", "2026-09-10_19.30.00", "--alt-band", "0", "4000",
                   "--json", str(tmp_path / "none.json")])


def test_the_module_imports_without_the_scientific_stack_being_touched() -> None:
    # census and place import numpy and netCDF4 inside themselves; the module
    # itself and the table command are stdlib only.
    source = TOOL.read_text()
    header = source.split("# the table (stdlib only")[0]
    assert "import numpy" not in header
    assert "import netCDF4" not in header
