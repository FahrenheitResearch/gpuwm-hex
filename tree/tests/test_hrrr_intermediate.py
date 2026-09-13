"""The projected-source intermediate: the writer, the target, the window, the table.

Engine-free where it can be: the writer is proved against this tree's own
reader; the decoder and the interpolation operator are the engine's and are
exercised on hardware by the lane's receipt, not here.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore import hrrr_intermediate as hi  # noqa: E402
from hexcore.cli import build_parser  # noqa: E402
from hexcore.hrrr_intermediate import IntermediateRefusal, LatLonTarget  # noqa: E402
from hexcore.wps_intermediate import WpsIntermediateReader, inventory  # noqa: E402


# ---------------------------------------------------------------------------
# the writer is the inverse of the tree's reader
# ---------------------------------------------------------------------------
def test_a_written_record_reads_back_through_the_frozen_reader(tmp_path: Path) -> None:
    target = LatLonTarget(south=37.5, west=-96.0, dlat=0.025, dlon=0.025, nx=5, ny=3)
    values = np.arange(15, dtype=np.float32).reshape(3, 5) + 280.0
    path = tmp_path / "MET:2026-09-13_15"
    path.write_bytes(hi.wps_record_bytes(
        hdate="2026-09-13_15:00:00", xfcst=0.0, map_source="NCEP HRRR", field_name="TT",
        units="K", description="Temperature", level=200100.0, target=target, values=values,
    ))
    with WpsIntermediateReader(path) as reader:
        fields = list(reader.iter_fields())
    assert len(fields) == 1
    got = fields[0]
    assert got.version == 5 and got.field == "TT" and got.level == 200100.0
    assert got.valid_time == "2026-09-13_15:00:00" and got.map_source == "NCEP HRRR"
    assert (got.nx, got.ny) == (5, 3)
    assert got.projection.code == 0 and got.projection.name == "latlon"
    assert got.projection.start_location == "SWCORNER"
    assert got.projection.start_latitude == pytest.approx(37.5)
    assert got.projection.start_longitude == pytest.approx(-96.0)
    assert got.projection.delta_latitude == pytest.approx(0.025)
    assert got.projection.delta_longitude == pytest.approx(0.025)
    assert got.projection.earth_radius_km == pytest.approx(6371.229)
    assert got.is_wind_grid_relative is False
    # The reader keeps Fortran (nx, ny); our slab was (ny, nx): the same numbers.
    assert np.array_equal(got.values, values.T)
    # Big-endian Fortran sequential markers, as ungrib writes them.
    assert struct.unpack(">i", path.read_bytes()[:4])[0] == 4


def test_write_intermediate_refuses_a_slab_of_the_wrong_shape_or_with_a_nan(tmp_path: Path) -> None:
    target = LatLonTarget(south=0.0, west=0.0, dlat=0.1, dlon=0.1, nx=2, ny=2)
    with pytest.raises(IntermediateRefusal):
        hi.wps_record_bytes(hdate="x", xfcst=0, map_source="s", field_name="TT", units="K",
                            description="d", level=1.0, target=target, values=np.zeros((3, 2)))
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.wps_record_bytes(hdate="x", xfcst=0, map_source="s", field_name="TT", units="K",
                            description="d", level=1.0, target=target,
                            values=np.array([[1.0, np.nan], [0.0, 0.0]]))
    assert "non-finite" in str(refusal.value)


def test_write_intermediate_inventories_every_record_before_signing(tmp_path: Path) -> None:
    target = LatLonTarget(south=30.0, west=-100.0, dlat=0.5, dlon=0.5, nx=4, ny=3)
    records = [
        {"field": "PRESSURE", "units": "Pa", "description": "P", "level": 1.0,
         "values": np.full((3, 4), 90000.0, dtype=np.float32)},
        {"field": "TT", "units": "K", "description": "T", "level": 200100.0,
         "values": np.full((3, 4), 290.0, dtype=np.float32)},
    ]
    written = hi.write_intermediate(
        tmp_path / "MET:2026-09-13_16", records, valid_time=datetime(2026, 9, 13, 16),
        forecast_hour=1.0, map_source="test", target=target,
    )
    assert written["field_records"] == 2 and written["fields"] == {"PRESSURE": 1, "TT": 1}
    assert written["valid_time"] == "2026-09-13_16:00:00" and written["record_endian"] == "big"
    seen = inventory(tmp_path / "MET:2026-09-13_16", include_statistics=True)
    assert seen["records"][0]["data"]["min"] == 90000.0


# ---------------------------------------------------------------------------
# the target box and the helpers
# ---------------------------------------------------------------------------
def test_the_target_covers_the_cap_plus_the_margin_on_the_stated_spacing() -> None:
    target = hi.target_for_cap((39.1, -94.58), 135.0, margin_km=30.0, spacing_deg=0.025)
    assert target.dlat == 0.025 and target.dlon == 0.025
    assert target.south < 39.1 - 165.0 / hi.KM_PER_DEG <= target.south + target.dlat
    assert target.north > 39.1 + 165.0 / hi.KM_PER_DEG - target.dlat
    assert target.west < -94.58 - 165.0 / (hi.KM_PER_DEG * np.cos(np.radians(39.1)))
    lat, lon = target.mesh()
    assert lat.shape == (target.ny, target.nx) == lon.shape
    assert lat[0, 0] == pytest.approx(target.south) and lon[0, 0] == pytest.approx(target.west)
    assert lat[1, 0] - lat[0, 0] == pytest.approx(0.025) and lon[0, 1] - lon[0, 0] == pytest.approx(0.025)
    with pytest.raises(IntermediateRefusal):
        hi.target_for_cap((89.0, 0.0), 500.0)


def test_the_target_from_a_plan_reads_the_cull_region_and_its_halo(tmp_path: Path) -> None:
    plan = {
        "schema": "gpuwm-hex.point-generate/v1",
        "plan": {"cull_region": {"kind": "cap", "center_deg": [39.1, -94.58], "radius_km": 135.0},
                 "cull": {"halo_km": 20.0}},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    target, basis = hi.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)
    assert basis == {"centre_deg": [39.1, -94.58], "cut_radius_km": 135.0, "halo_km": 20.0}
    reach = (135.0 + 20.0 + 30.0) / hi.KM_PER_DEG
    assert target.north - target.south >= 2.0 * reach - 0.05
    path.write_text('{"plan": {}}', encoding="utf-8")
    with pytest.raises(IntermediateRefusal):
        hi.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)


def test_cycle_and_hour_parsing_refuse_gaps_and_nonsense() -> None:
    assert hi.parse_cycle("2026-09-13T15") == datetime(2026, 9, 13, 15)
    assert hi.parse_cycle("2026-09-13_15:00:00") == datetime(2026, 9, 13, 15)
    assert hi.parse_hours("0-3") == (0, 1, 2, 3)
    assert hi.parse_hours("2,0,1") == (0, 1, 2)
    with pytest.raises(IntermediateRefusal):
        hi.parse_hours("0,2")
    with pytest.raises(IntermediateRefusal):
        hi.parse_cycle("yesterday")


def test_the_source_table_is_the_only_place_a_source_lives() -> None:
    row = hi.source_row("HRRR")
    assert row.decoder_key == "hrrr" and row.levels == 50
    assert row.grib_names[0].format(cycle=datetime(2026, 9, 13, 15), hour=2) == "hrrr.t15z.wrfnatf02.grib2"
    assert row.grib_names[1].format(cycle=datetime(2026, 9, 13, 15), hour=0) == "hrrr.t15z.soilf00.grib2"
    assert row.soil_layer_names[0] == ("ST000010", "SM000010")
    # The map source lands in a 32-character WPS field; a longer one refused
    # the first real regrid on 2026-09-13 after the decode had already run.
    for each in hi.SOURCE_ROWS.values():
        assert len(each.map_source.encode("ascii")) <= 32, each.map_source
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.source_row("rap")
    assert "SOURCE_ROWS" in str(refusal.value)
    # Nothing outside the table names the source.
    text = (ROOT / "src" / "hexcore" / "hrrr_intermediate.py").read_text(encoding="utf-8")
    body = text.split("SOURCE_ROWS: Mapping[str, SourceRow] = {", 1)[1].split("\ndef source_row", 1)[1]
    assert '== "hrrr"' not in body and "if row.name" not in body


def test_the_doors_are_subcommands_of_the_console_script() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert "intermediate" in commands and "lbc" in commands
    arguments = parser.parse_args([
        "intermediate", "--grib-dir", "d", "--cycle", "2026-09-13T15", "--out-dir", "o",
        "--point", "39.1,-94.58", "--radius-km", "155",
    ])
    assert arguments.hours == "0-3" and arguments.spacing_deg == hi.DEFAULT_SPACING_DEG
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.request_from_arguments(parser.parse_args([
            "intermediate", "--grib-dir", "d", "--cycle", "2026-09-13T15", "--out-dir", "o",
        ]))
    assert "neither --from-plan nor --point" in str(refusal.value)


def test_intermediate_valid_times_read_the_header_not_the_name(tmp_path: Path) -> None:
    target = LatLonTarget(south=30.0, west=-100.0, dlat=0.5, dlon=0.5, nx=2, ny=2)
    for hour in (1, 0):
        hi.write_intermediate(
            tmp_path / f"anything-{hour}", [{"field": "TT", "units": "K", "description": "T",
                                             "level": 200100.0, "values": np.zeros((2, 2), np.float32)}],
            valid_time=datetime(2026, 9, 13, 15 + hour), forecast_hour=float(hour),
            map_source="t", target=target,
        )
    (tmp_path / "notes.txt").write_text("not a met file", encoding="utf-8")
    times = hi.intermediate_valid_times(tmp_path)
    assert [t for t, _ in times] == [datetime(2026, 9, 13, 15), datetime(2026, 9, 13, 16)]
    assert times[0][1].name == "anything-0"


# ---------------------------------------------------------------------------
# the boundary producer resolves through the one ladder
# ---------------------------------------------------------------------------
def _no_lbc_anywhere(monkeypatch) -> None:
    from hexcore import engines

    for name in ("GPUWM_HEX_RW_MPAS_LBC", "RW_MPAS_LBC", "GPUWM_RW_MPAS_LBC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())


def test_a_staged_lbc_engine_resolves_with_no_environment_variable(monkeypatch, tmp_path: Path) -> None:
    """``gpuwm fetch-bridges`` alone opens the boundary leg.

    The chain's own resolver read one variable and PATH, so a box where the
    init leg had just resolved ``rw_mpas_init`` out of ``~/.gpuwm/bridges``
    refused the lbc leg of the same chain for a file sitting beside it.
    """

    from hexcore import engines
    from hexcore.cycle.chain import resolve_lbc_engine

    staged = tmp_path / engines.executable_name("rw_mpas_lbc")
    staged.write_bytes(b"")
    staged.chmod(0o755)
    for name in ("GPUWM_HEX_RW_MPAS_LBC", "RW_MPAS_LBC", "GPUWM_RW_MPAS_LBC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))

    assert resolve_lbc_engine(None) == staged.resolve()


def test_lbc_engine_absent_refusal_names_the_ladder_and_the_breakage(monkeypatch) -> None:
    from hexcore.cycle.chain import CycleRefusal, resolve_lbc_engine

    _no_lbc_anywhere(monkeypatch)
    with pytest.raises(CycleRefusal) as caught:
        resolve_lbc_engine(None)
    message = str(caught.value)
    assert "rw_mpas_lbc not found" in message
    assert "--lbc-exe" in message
    assert "RW_MPAS_LBC" in message
    assert "gpuwm fetch-bridges" in message
    assert "seven boundary rings" in message


def test_lbc_engine_named_but_missing_is_refused_not_skipped(tmp_path: Path) -> None:
    from hexcore.cycle.chain import CycleRefusal, resolve_lbc_engine

    with pytest.raises(CycleRefusal) as caught:
        resolve_lbc_engine(tmp_path / "no-such-exe")
    assert "which is not a file" in str(caught.value)


def test_the_lbc_engine_is_a_row_of_the_ladder_the_doctor_reads() -> None:
    from hexcore import engines

    assert engines.LBC in engines.ENGINES
    assert engines.LBC.name == "rw_mpas_lbc"
