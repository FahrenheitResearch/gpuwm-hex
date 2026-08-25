#!/usr/bin/env python3
"""Choose the referee's control cases from the observation archive, not by taste.

The production manifest carried its independent and weak-convection controls as
`selection_status: pending` with null dates, on the rule that code does not
invent weather cases. Selecting them by eye would have been the same failure
wearing a person's hat, so they were selected by a screen with a stated rule
that anyone can re-run.

* **Pool.** One day per month across a year -- the same day of month, so the
  pool has no seasonal preference baked into it -- read as the Stage-IV 24 h
  accumulation ending 12Z, which is the only hour the archive publishes a 24 h
  object at.
* **Coverage guard.** A day whose object covers far fewer cells than the pool
  is a River Forecast Centre outage, not a dry day: its wet fractions are taken
  over a different denominator, so comparing them selects on the outage. Days
  below `--coverage-floor` times the pool median are dropped and named.
* **Independent convective control.** Argmax heavy-rain coverage -- the
  fraction of valid cells at or above `--heavy-mm`. Coverage of heavy rain is
  the convective signature; total coverage is not, because a wide winter shield
  wets more cells than a squall line does.
* **Weak-convection control.** Argmin any-rain coverage, at or above
  `--wet-mm`. That is the null: a day the model should not make rain on.

The forecast for a selected day is the 24 h run *ending* at the screened
window, so its cycle is the previous day at 12Z. The screen prints that cycle.

```sh
python verification/producers/select_control_cases.py \
    --year 2025 --day-of-month 15 --cache /abs/cache --out screen.json
```
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np

from gpuwm.obs.frontdoor import STAGE4
from gpuwm.obs.obspack import read_grid_pack


def screen_day(exe: pathlib.Path, day: dt.date, work: pathlib.Path,
               cache: pathlib.Path) -> dict:
    out = work / day.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    when = f"{day.isoformat()}T12:00:00Z"
    fetch = subprocess.run(
        [str(exe), "fetch", "--start", when, "--end", when,
         "--accumulation", "24h", "--out", str(out), "--cache", str(cache)],
        capture_output=True, text=True)
    if fetch.returncode != 0:
        return {"day": day.isoformat(), "status": "FETCH_FAILED",
                "detail": fetch.stderr.strip()[-200:]}
    name = f"ST4.{day:%Y%m%d}12.24h.grib"
    matches = sorted(cache.rglob(name)) or sorted(out.rglob(name))
    if not matches:
        return {"day": day.isoformat(), "status": "NO_OBJECT"}
    pack = out / "acc24.obspack"
    decode = subprocess.run([str(exe), "decode", "--file", str(matches[0]),
                             "--out", str(pack)], capture_output=True, text=True)
    if decode.returncode != 0:
        return {"day": day.isoformat(), "status": "DECODE_FAILED",
                "detail": decode.stderr.strip()[-200:]}
    grid = read_grid_pack(pack)
    values = np.asarray(grid.arrays["values"], dtype=np.float64).ravel()
    valid = np.asarray(grid.arrays["valid"], dtype=np.uint8).ravel().astype(bool)
    observed = values[valid]
    return {"day": day.isoformat(), "status": "OK",
            "valid_cells": int(observed.size),
            "frac_wet": float(np.mean(observed >= 1.0)),
            "frac_heavy": float(np.mean(observed >= 25.0)),
            "max_mm": float(np.max(observed)),
            "mean_mm": float(np.mean(observed))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--day-of-month", type=int, default=15)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--work", default=None,
                        help="scratch for per-day packs (default: beside --out)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage-floor", type=float, default=0.9)
    parser.add_argument("--wet-mm", type=float, default=1.0)
    parser.add_argument("--heavy-mm", type=float, default=25.0)
    args = parser.parse_args(argv)

    exe = STAGE4.find()
    if exe is None:
        raise SystemExit(f"rw_stage4 is not resolvable.\n{STAGE4.remedy()}")
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work) if args.work else out.parent / "screen-work"
    work.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for month in range(1, 13):
        row = screen_day(exe, dt.date(args.year, month, args.day_of_month),
                         work, cache)
        rows.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)

    ok = [row for row in rows if row["status"] == "OK"]
    median_cells = statistics.median(row["valid_cells"] for row in ok) if ok else 0
    covered = [row for row in ok
               if row["valid_cells"] >= args.coverage_floor * median_cells]
    dropped = [row["day"] for row in ok if row not in covered]

    def cycle_of(day: str) -> str:
        selected = dt.date.fromisoformat(day) - dt.timedelta(days=1)
        return f"{selected.isoformat()}T12:00:00Z"

    convective = max(covered, key=lambda row: row["frac_heavy"]) if covered else None
    quiescent = min(covered, key=lambda row: row["frac_wet"]) if covered else None
    result = {
        "pool_rule": (f"day {args.day_of_month} of every month of {args.year}, "
                      "Stage-IV 24 h accumulation ending 12Z"),
        "coverage_guard": (f"valid_cells >= {args.coverage_floor} * median "
                           "valid_cells across the pool"),
        "median_valid_cells": median_cells,
        "dropped_for_coverage": dropped,
        "wet_threshold_mm": args.wet_mm, "heavy_threshold_mm": args.heavy_mm,
        "independent_control_rule": "argmax frac_heavy among covered days",
        "weak_convection_control_rule": "argmin frac_wet among covered days",
        "independent_control": convective,
        "weak_convection_control": quiescent,
        "independent_control_cycle": cycle_of(convective["day"]) if convective else None,
        "weak_convection_control_cycle": cycle_of(quiescent["day"]) if quiescent else None,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    json.dump({key: result[key] for key in (
        "dropped_for_coverage", "independent_control_cycle",
        "weak_convection_control_cycle")}, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
