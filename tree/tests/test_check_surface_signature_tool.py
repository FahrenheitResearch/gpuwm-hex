"""tools/check_surface_signature.py admits or refuses an init from its four surface fields alone."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_surface_signature.py"
ISICE = 15


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_surface_signature", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_init(path: Path, rows, *, tsk_threshold=271.0):
    """rows: (xland, xice, ivgtyp, landmask, lat_deg, lon_deg)."""

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("nCells", len(rows))
        for name, dtype, dims in (
            ("xland", "f4", ("Time", "nCells")),
            ("xice", "f4", ("Time", "nCells")),
            ("ivgtyp", "i4", ("nCells",)),
            ("landmask", "i4", ("nCells",)),
            ("latCell", "f8", ("nCells",)),
            ("lonCell", "f8", ("nCells",)),
        ):
            dataset.createVariable(name, dtype, dims)
        dataset["xland"][0, :] = np.asarray([r[0] for r in rows], dtype=np.float32)
        dataset["xice"][0, :] = np.asarray([r[1] for r in rows], dtype=np.float32)
        dataset["ivgtyp"][:] = np.asarray([r[2] for r in rows], dtype=np.int32)
        dataset["landmask"][:] = np.asarray([r[3] for r in rows], dtype=np.int32)
        dataset["latCell"][:] = np.asarray([math.radians(r[4]) for r in rows])
        dataset["lonCell"][:] = np.asarray([math.radians(r[5]) for r in rows])
        dataset.setncattr("config_tsk_seaice_threshold", float(tsk_threshold))


def test_conus3km_shape_is_admitted_and_its_cells_named(tmp_path, capsys):
    rows = [
        (1.0, 0.0, 10, 1, 40.0, -100.0),  # grassland
        (2.0, 0.0, 17, 0, 30.0, -80.0),  # ocean
        (1.0, 0.9, ISICE, 0, 70.0, -150.0),  # sea ice
        (1.0, 0.0, ISICE, 1, -80.0, 0.0),  # Antarctic glacier
        (1.0, 0.0, ISICE, 0, 52.2, -117.4),  # frozen lake, native skin-temperature branch
        (1.0, 0.0, ISICE, 0, 52.25, -117.45),  # second frozen lake cell
    ]
    init = tmp_path / "init.nc"
    _write_init(init, rows)
    tool = _load_tool()
    assert tool.main([str(init)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["admitted"] is True
    assert report["n_cells"] == 6
    assert report["config_tsk_seaice_threshold"] == 271.0
    counts = report["surface_classification"]
    assert counts["glacier_columns"] == 3
    assert counts["glacier_columns_on_landmask_0"] == 2
    assert counts["sea_ice_columns_on_landmask_1"] == 0
    named = report["measured_cells"]["glacier_on_landmask_0"]
    assert [row["cell"] for row in named] == [4, 5]
    assert named[0]["lat_deg"] == pytest.approx(52.2, abs=1e-3)
    assert named[0]["lon_deg"] == pytest.approx(-117.4, abs=1e-3)


def test_lake_cells_fold_to_open_water_as_the_door_folds_them(tmp_path, capsys):
    # The door turns ivgtyp 21 into open water (xland 2, landmask 0) before it
    # classifies; the tool must count the same way or its report disagrees
    # with the door's receipt by the lake count.
    rows = [
        (1.0, 0.0, 10, 1, 40.0, -100.0),  # grassland
        (1.0, 0.0, 21, 1, 44.0, -87.0),  # unfrozen lake, still category 21
        (1.0, 0.0, 21, 0, 46.0, -85.0),  # lake on a water landmask
        (1.0, 0.0, ISICE, 0, 52.2, -117.4),  # frozen lake native converted
    ]
    init = tmp_path / "init.nc"
    _write_init(init, rows)
    tool = _load_tool()
    assert tool.main([str(init)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["lake_fold"] == {"lake_category": 21, "lake_columns": 2}
    counts = report["surface_classification"]
    assert counts["open_water_columns"] == 2
    assert counts["sflx_land_columns"] == 1
    assert counts["glacier_columns"] == 1
    assert counts["xland_water_columns"] == 2
    assert counts["glacier_columns_on_landmask_0"] == 1


def test_unconverted_sea_ice_is_refused_with_exit_2(tmp_path, capsys):
    rows = [
        (1.0, 0.0, 10, 1, 40.0, -100.0),
        (2.0, 0.6, 17, 0, 60.0, -150.0),  # sea ice still xland 2: native did not convert it
    ]
    init = tmp_path / "init.nc"
    _write_init(init, rows)
    tool = _load_tool()
    assert tool.main([str(init)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["admitted"] is False
    assert "xland 2 instead of native's 1" in captured.err
    assert "cells [1]" in captured.err
