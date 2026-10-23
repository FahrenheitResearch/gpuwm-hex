"""The opening maps: content first, reflectivity from its first echo (products_door.first_maps).

THE BREAKAGE THIS PREVENTS (hex 1 km LAM fast start, GFS 2025-05-19 18Z, 2026-10-23): a cold start carries no
hydrometeors, so the f000 composite reflectivity, the first map a viewer showed, was an empty map.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hexcore import products_door as pd

START = "2025-05-19_18:00:00"


def frames(tmp_path: Path, hours) -> list[Path]:
    out = []
    for hour in hours:
        path = tmp_path / f"history.2025-05-19_{18 + hour:02d}.00.00.nc"
        path.write_bytes(b"")
        out.append(path)
    return out


def test_a_cold_start_opens_on_content_and_reflectivity_joins_at_its_first_echo(tmp_path, monkeypatch):
    peaks = {"history.2025-05-19_18.00.00.nc": -35.0, "history.2025-05-19_19.00.00.nc": 6.0,
             "history.2025-05-19_20.00.00.nc": 41.5, "history.2025-05-19_21.00.00.nc": 60.0}
    read = []
    monkeypatch.setattr(pd, "frame_echo_dbz", lambda path: read.append(Path(path).name) or peaks[Path(path).name])
    record = pd.first_maps(frames(tmp_path, range(4)), START, (*pd.FIRST_MAPS, pd.ECHO_MAP, "total_qpf"))
    assert record["maps"] == ["2m_temperature_10m_winds", "10m_wind_speed_and_direction", "sbcape",
                              "composite_reflectivity"]
    assert (record["reflectivity_from"], record["reflectivity_peak_dbz"]) == ("f002", 41.5)
    assert read == list(peaks)[:3]  # the scan stops at the first echo


def test_a_first_look_of_an_echo_free_f000_leads_with_temperature_wind_and_cape(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "frame_echo_dbz", lambda path: None)
    record = pd.first_maps(frames(tmp_path, [0]), START, (*pd.FIRST_MAPS, pd.ECHO_MAP))
    assert record["maps"] == list(pd.FIRST_MAPS)
    assert record["reflectivity_from"] is None


def test_storms_at_the_start_keep_reflectivity_from_f000(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "frame_echo_dbz", lambda path: 55.0)
    record = pd.first_maps(frames(tmp_path, [0, 1]), START, (*pd.FIRST_MAPS, pd.ECHO_MAP))
    assert record["reflectivity_from"] == "f000" and record["maps"][-1] == pd.ECHO_MAP


def test_an_unreadable_frame_never_fails_the_products(tmp_path, monkeypatch):
    def broken(path):
        raise OSError("truncated")
    monkeypatch.setattr(pd, "frame_echo_dbz", broken)
    record = pd.first_maps(frames(tmp_path, [0]), START, (*pd.FIRST_MAPS, pd.ECHO_MAP))
    assert record["maps"] == list(pd.FIRST_MAPS) and "truncated" in record["note"]


def test_the_index_lists_and_opens_on_the_opening_maps(tmp_path):
    case = tmp_path / "case"
    for product in ("10m_wind_speed_and_direction", "250mb_height_winds", "2m_temperature_10m_winds",
                    "composite_reflectivity", "sbcape"):
        for lead in ("f000", "f001", "f002"):
            png = case / "global" / product / "2025-05-19" / f"hex_global_{product}_{lead}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(b"png")
    first = {"maps": [*pd.FIRST_MAPS, pd.ECHO_MAP], "echo_map": pd.ECHO_MAP, "reflectivity_from": "f002"}
    page = pd.write_index(case, "case", START, first).read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (.*);\n", page).group(1))
    assert list(data["global"]) == ["2m_temperature_10m_winds", "10m_wind_speed_and_direction", "sbcape",
                                    "composite_reflectivity", "250mb_height_winds"]
    opening = json.loads(re.search(r"const FIRST = (.*);\n", page).group(1))
    assert opening == {"echo_map": "composite_reflectivity", "reflectivity_from": "f002"}
    # no echoes yet: reflectivity still listed right after the opening maps
    page = pd.write_index(case, "case", START, {**first, "maps": list(pd.FIRST_MAPS), "reflectivity_from": None})
    data = json.loads(re.search(r"const DATA = (.*);\n", page.read_text(encoding="utf-8")).group(1))
    assert list(data["global"])[:4] == [*pd.FIRST_MAPS, pd.ECHO_MAP]


def test_the_echo_read_is_the_column_maximum_in_cell_blocks(tmp_path, monkeypatch):
    """The real Rust reader on a real (Time, nCells, nVertLevels) frame, read whole and in cell blocks."""
    bridge = pytest.importorskip("gpuwm.netcdf_bridge")
    if bridge.find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not installed")
    netCDF4 = pytest.importorskip("netCDF4")
    import numpy as np

    path = tmp_path / "history.2025-05-19_19.00.00.nc"
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("nCells", 1000)
        nc.createDimension("nVertLevels", 7)
        refl = nc.createVariable("refl10cm", "f4", ("Time", "nCells", "nVertLevels"))
        values = np.full((1, 1000, 7), -35.0, dtype="f4")
        values[0, 977, 5] = 47.25
        refl[:] = values
    assert pd.frame_echo_dbz(path) == pytest.approx(47.25)
    monkeypatch.setattr(pd, "ECHO_BLOCK_VALUES", 700)  # 100-cell blocks
    assert pd.frame_echo_dbz(path) == pytest.approx(47.25)
    empty = tmp_path / "history.2025-05-19_18.00.00.nc"
    with netCDF4.Dataset(empty, "w") as nc:
        nc.createDimension("nCells", 4)
        nc.createVariable("t2m", "f4", ("nCells",))[:] = np.zeros(4, dtype="f4")
    assert pd.frame_echo_dbz(empty) is None
