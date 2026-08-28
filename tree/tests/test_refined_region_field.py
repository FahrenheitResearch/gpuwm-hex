"""The refined-region instrument reads the field it is asked for, on the axis
the mesh says is the cell axis.

THE BREAKAGE THESE PREVENT.  The instrument used to read ``w`` and nothing
else, collapsing "the long axis is the cell axis" by comparing the two
dimension lengths.  Both assumptions are wrong in this history stream:

* a 4.6 km grid placed on an atmospheric river buys moisture transport and
  rainfall, not ascent, so a hard-wired ``w`` reports "unimpressive" about a
  grid that was doing its job in a variable the tool refused to read;
* a per-cell surface field is one-dimensional, and a column field on a mesh
  with fewer cells than levels -- a regional cull is one -- has the CELL axis
  shorter than the level axis, so "the long axis is the cell axis" silently
  reports the maximum over a column as the maximum over the domain.

The cell axis is now identified by LENGTH AGAINST THE MESH, and a field whose
shape has no such axis is refused by name rather than reduced along a guess.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import measure_refined_region_w as instrument  # noqa: E402


def test_a_surface_field_is_already_per_cell():
    values = np.arange(7.0)
    assert instrument._collapse(values, 7).tolist() == values.tolist()


def test_a_column_field_collapses_to_the_column_extremum():
    # (nCells, nVertLevels): cell 1 has the tallest column.
    values = np.array([[1.0, 2.0], [9.0, 3.0], [4.0, 4.0]])
    assert instrument._collapse(values, 3).tolist() == [2.0, 9.0, 4.0]


def test_the_cell_axis_is_found_by_length_not_by_being_longer():
    # A regional cull with 3 cells and 5 levels: the CELL axis is the SHORT
    # one, which is exactly the case the old "longer axis wins" rule got
    # backwards.
    values = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 7.0, 0.0, 0.0],
    ])
    assert instrument._collapse(values, 3).tolist() == [5.0, 6.0, 7.0]


def test_a_leading_time_axis_of_one_is_stripped():
    values = np.array([[[1.0, 8.0], [3.0, 2.0]]])   # (1, nCells=2, nLevels=2)
    assert instrument._collapse(values, 2).tolist() == [8.0, 3.0]


def test_a_field_with_no_cell_axis_is_refused_by_name():
    values = np.zeros((4, 5))
    with pytest.raises(SystemExit) as excinfo:
        instrument._collapse(values, 9)
    assert "9 cells" in str(excinfo.value)


def test_a_missing_field_names_what_the_frame_does_carry():
    class _Fake:
        variables = {"w": None, "t2": None}

    with pytest.raises(SystemExit) as excinfo:
        instrument._read(_Fake(), "rainnc", Path("cuda-history.nc"))
    message = str(excinfo.value)
    assert "rainnc" in message and "t2" in message and "w" in message


def test_two_names_are_a_vector_magnitude(monkeypatch, tmp_path):
    frame = tmp_path / "cuda-history.2026-08-12_09.00.00.nc"

    class _Dataset:
        variables = {"u10": np.array([3.0, 0.0]), "v10": np.array([4.0, -5.0])}

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setitem(sys.modules, "netCDF4",
                        type("m", (), {"Dataset": _Dataset}))
    speed = instrument._frame_field(frame, "u10,v10", 2)
    assert speed.tolist() == [5.0, 5.0]


def test_more_than_two_names_is_refused(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        instrument._frame_field(tmp_path / "f.nc", "u10,v10,w", 2)
    assert "one published name, or two" in str(excinfo.value)
