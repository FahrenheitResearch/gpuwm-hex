"""Consume a paired day with Rust-decoded forecasts and GFS analyses."""
import argparse
import json
from pathlib import Path
import time

from hexcore.stochastic_day import score_day


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--reader", type=Path, help="engine rw_netcdf; resolved by the engine if omitted")
    parser.add_argument("--grib-bridge", type=Path, help="engine Rust CPU bridge for raw GFS; resolved if omitted")
    parser.add_argument("--work", required=True, type=Path, help="disposable GFS/decoded-field workspace")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    reader = args.reader
    audit, started = [], time.perf_counter()
    try:
        if reader is None:
            from gpuwm.netcdf_bridge import resolve_netcdf_bin
            reader = resolve_netcdf_bin()
        result = score_day(json.loads(args.request.read_text()), reader=reader, work=args.work, audit=audit,
                           grib_bridge=args.grib_bridge)
    except Exception as error:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(error=str(error), input_reads=audit,
            consumer_wall_seconds=time.perf_counter()-started, qualification_complete=False),
            indent=2, allow_nan=False)+"\n")
        raise
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(receipt=str(args.out), members=len(result["members"]),
                          lead_hours=result["lead_hours"], qualification_complete=False)))


if __name__ == "__main__":
    main()
