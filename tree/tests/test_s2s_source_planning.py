from argparse import Namespace
from datetime import timedelta
import re

import pytest

from hexcore import init_sources as src
from hexcore.ensemble_prepare import prepare_plan
from hexcore.lagged_members import expand_lags, parse_lags
from hexcore.global_ensemble import member_plan


def test_lag_expansion_keeps_valid_time_across_month_and_year():
    row = src.resolve_source(src.load_table(), "gefs")
    records = expand_lags(row, src.Cycle.parse("2024-01-01T00"), [1, 2], parse_lags("0,6,12,18"))
    assert len(records) == 8
    assert len({(r["source_cycle"], r["source_member"]) for r in records}) == 8
    for record in records:
        cycle = src.Cycle.parse(record["source_cycle"])
        assert src.Cycle(cycle.when + timedelta(hours=record["lead_hours"])).start_time == "2024-01-01_00:00:00"
        products = src.plan_products(row, cycle, record["source_member"], record["lead_hours"])
        assert products[0].url.endswith(f".f{record['lead_hours']:03d}")
    assert records[-1]["source_cycle"] == "2023-12-31_06:00:00"
    for bad in ("0,0", "-6", "0,foo", ""):
        with pytest.raises(src.InitSourceRefusal):
            parse_lags(bad)


def test_cfs_row_field_map_and_archive_templates():
    row = src.resolve_source(src.load_table(), "cfs")
    assert row["first_guess"] == {"nfglevels": 38, "nfgsoillevels": 4, "use_spechumd": "no"}
    cycle = src.Cycle.parse("2023-03-31T00")
    plans = src.plan_products(row, cycle, None)
    assert plans[0].url.endswith("6-hourly-by-pressure/2023/202303/20230331/cdas1.t00z.pgrbh00.grib2")
    assert plans[0].index_url.endswith("cdas1.t00z.pgrbh00.inv")
    assert row["decode"]["map_source"] == "ncep-cdas-cfsv2"
    inventory = "\n".join(f"{i}:{i*100}:d=2023033100:{variable}:{level}:anl:"
                           for i,(variable,level) in enumerate([
                               ("RH","2 m above ground"),
                               *[(v,l+" m below ground") for v in ("TMP","SOILW")
                                 for l in ("0-0.1","0.1-0.4","0.4-1","1-2")]],1))
    assert len(src.select_ncep_idx_records(inventory, plans[0].index_select)) == 9
    fields = src.resolve_vtable(row).read_text()
    for name in ("TT", "UU", "VV", "RH", "PSFC", "SKINTEMP", "SEAICE", "LANDSEA", "SM000010", "SM010040", "SM040100", "SM100200", "ST000010", "SNOWH"):
        assert re.search(rf"\|\s*{name}\s*\|", fields), name
    assert re.search(r"SNOWH\s*\|.*\|\s*0\s*\|\s*1\s*\|\s*11\s*\|\s*1\s*\|", fields)
    forecasts = src.plan_products(row, cycle, None, 12)
    assert forecasts[0].url.endswith("2023033100/pgbf2023033112.01.2023033100.grb2")
    assert forecasts[1].url.endswith("2023033100/flxf2023033112.01.2023033100.grb2")


def test_total_twenty_members_balances_native_members_across_lags():
    records = prepare_plan(Namespace(source="gefs", source_cycle="2024010100", source_hour=0,
                                   members=20, lagged="0,6,12,18", table=None))
    assert len(records) == 20
    assert [r["source_member"] for r in records[:5]] == [1,2,3,4,5]
    assert [r["lag_hours"] for r in records[::5]] == [0,6,12,18]


def test_member_remainder_keeps_every_selected_cycle():
    records = prepare_plan(Namespace(source="gefs", source_cycle="2024010100", source_hour=0,
                                   members=5, lagged="0,6,12,18", table=None))
    assert [sum(r["lag_hours"] == lag for r in records) for lag in (0,6,12,18)] == [2,1,1,1]


def test_global_four_deterministic_lags_build_distinct_inputs(tmp_path):
    value = {"paths":{"work":str(tmp_path)}, "cycle":"2024-01-01T00", "source":"gfs",
             "commands":{"init":["init-source","--source","gfs","--cycle","2024-01-01T00","--out-dir","unused"]}}
    records = member_plan(value,[0],[0,6,12,18])
    assert len({r["init"] for r in records}) == 4
    assert all("--member" not in r["init_argv"] for r in records)
    assert [r["provenance"]["lead_hours"] for r in records] == [0,6,12,18]
    assert records[-1]["init_argv"][-2:] == ["--lead-hours","18"]


def test_shared_grib_message_selection_preserves_first_submessage():
    inventory = "1.1:0:d=2023033100:UGRD:500 mb:anl:\n1.2:0:d=2023033100:VGRD:500 mb:anl:\n2:100:d=2023033100:TMP:surface:anl:"
    selected = src.select_ncep_idx_records(inventory, {"records":[["UGRD","500 mb"]]})
    assert selected == [{"_offset":0,"_length":100,"record":"UGRD:500 mb"}]
