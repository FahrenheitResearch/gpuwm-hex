#!/usr/bin/env python3
"""Place a tracer release line upwind of a chosen target, in a leg's own wind.

WHY THIS FILE EXISTS.  A paired run (``gpuwm-hex pair`` with
``--source-table``) runs one forecast twice, a control leg and a treatment
leg that is the same request plus a point-source table, and answers "what
did the model do differently when this tracer was released?".  A release the
model's own wind never carries over the region the question is about gives a
treatment leg that differs from its control nowhere that matters, and the
pair teaches nothing.  This tool reads the control leg's own wind at a
target the user names and places the release line upwind of it, so the
plume arrives when and where the question is asked.  How the target is
chosen is the user's decision and is not encoded here.

WHAT IT READS.  The history frames of a finished leg (``cuda-history.*.nc``,
the forecast door's own output: ``u_zonal``, ``v_meridional``), the grid
(``latCell``, ``lonCell``) and the init (``zgrid``, the converter's only
source of heights).

TWO COMMANDS.

``place``
    For ``--target LAT LON`` at ``--alt-m`` metres: the wind at the nearest
    cell and level, averaged over the named frames; the line is the target
    shifted UPWIND by ``--lead-min`` of that wind, with the target centred
    on it north to south.  The placed line is then read against every frame
    at that altitude: the wind along it, and how far that wind carries a
    release in the lead time.

``table``
    Writes the table for one explicit line in a fixed pass structure: three
    north-south passes along one longitude, five waypoints 180 s apart
    (0.1-degree steps, about 44 km), the last waypoint of each pass
    ``on=0``, passes 21 minutes apart, ``--rate`` particles per second, 54
    minutes from ``--release``.  Epoch seconds count from 2017-01-01T00:00Z,
    the engine's own point-source epoch.

Everything ``place`` reports is a MODEL RESULT of the control leg: what this
model's wind was in these frames.  It is not a measurement of the
atmosphere.  The table this tool writes is a SYNTHETIC release line authored
for a paired experiment; no real programme, operator or operation is
represented, and the file says so on its first lines.

Usage::

    python tools/point_source_placement.py place --leg-out OUT --grid G --init I \\
        --target LAT LON --alt-m Z --frames STAMP[,STAMP...] --json F [--lead-min 30]
    python tools/point_source_placement.py table --lat0 L --lon LON --alt-m Z \\
        --release YYYY-MM-DD_HH:MM:SS --out TABLE.txt [--note TEXT ...]

numpy and netCDF4 are imported inside ``place`` only, so ``table`` and
``--help`` work on a box with no scientific stack.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import sys
from typing import Any, Sequence

__all__ = [
    "EPOCH",
    "LINE_LENGTH_DEG",
    "PASS_GAP_SECONDS",
    "RATE_DEFAULT",
    "WAYPOINT_GAP_SECONDS",
    "build_parser",
    "epoch_seconds",
    "main",
    "table_rows",
    "table_text",
    "upwind_shift",
]

#: The point-source epoch: the engine counts release seconds from here.
EPOCH = _dt.datetime(2017, 1, 1)

#: The pass structure the ``table`` command writes.
LINE_LENGTH_DEG = 0.4
WAYPOINTS = 5
WAYPOINT_GAP_SECONDS = 180.0
PASS_GAP_SECONDS = 21 * 60.0
PASSES = 3
RATE_DEFAULT = 8.5e13

EARTH_RADIUS_KM = 6371.229
METRES_PER_DEGREE = 111_000.0

HISTORY_PREFIX = "cuda-history."


# --------------------------------------------------------------------------
# the table (stdlib only, so it is testable and usable anywhere)
# --------------------------------------------------------------------------

def epoch_seconds(when: _dt.datetime) -> float:
    """Seconds from the engine's point-source epoch to ``when`` (naive UTC)."""
    return (when - EPOCH).total_seconds()


def table_rows(lat0: float, lon: float, alt_m: float, release: _dt.datetime,
               rate: float = RATE_DEFAULT) -> list[tuple[float, float, float, float, int, float, int]]:
    """The fifteen waypoints of the pass structure, in release order.

    Pass 1 runs south to north from ``lat0``, pass 2 north to south, pass 3
    south to north again; the fifth waypoint of every pass carries ``on=0``
    (the release stops there), so each pass injects for four 180 s legs.
    """
    rows = []
    for pass_index in range(PASSES):
        start = release + _dt.timedelta(seconds=PASS_GAP_SECONDS * pass_index)
        lats = [round(lat0 + 0.1 * k, 3) for k in range(WAYPOINTS)]
        if pass_index % 2 == 1:
            lats = lats[::-1]
        for k, lat in enumerate(lats):
            when = start + _dt.timedelta(seconds=WAYPOINT_GAP_SECONDS * k)
            on = 0 if k == WAYPOINTS - 1 else 1
            rows.append((epoch_seconds(when), lat, lon, float(alt_m), on, rate, pass_index + 1))
    return rows


def table_text(lat0: float, lon: float, alt_m: float, release: _dt.datetime, *,
               rate: float = RATE_DEFAULT, case: str = "", notes: Sequence[str] = ()) -> str:
    """The table file: the disclaimer, the structure, the notes, the rows."""
    end = release + _dt.timedelta(seconds=PASS_GAP_SECONDS * (PASSES - 1)
                                  + WAYPOINT_GAP_SECONDS * (WAYPOINTS - 1))
    lines = [
        "# Point-source table%s." % ((", " + case) if case else ""),
        "# SYNTHETIC release line authored for a paired experiment; no real "
        "programme, operator or operation is represented.",
        "# Three N-S passes along one longitude, five waypoints 180 s apart "
        "(0.1 deg steps, ~44 km), the last waypoint of each pass on=0,",
        "# passes 21 min apart, %.1e particles per second, released %s-%sZ.  "
        "Epoch seconds count from 2017-01-01T00:00Z." % (
            rate, release.strftime("%H:%M"), end.strftime("%H:%M")),
    ]
    lines += ["# " + note for note in notes]
    lines.append("#   epoch_sec  lat_deg  lon_deg  alt_m_MSL  on  rate_part_per_s  src_id")
    for row in table_rows(lat0, lon, alt_m, release, rate):
        lines.append("%.1f  %.3f  %.2f  %.0f  %d  %.1e  %d" % row)
    return "\n".join(lines) + "\n"


def upwind_shift(lat: float, u: float, v: float, lead_seconds: float) -> tuple[float, float]:
    """(dlat, dlon) in degrees that moves a point UPWIND by ``lead_seconds`` of (u, v)."""
    dlat = -v * lead_seconds / METRES_PER_DEGREE
    dlon = -u * lead_seconds / (METRES_PER_DEGREE * math.cos(math.radians(lat)))
    return dlat, dlon


# --------------------------------------------------------------------------
# reading a leg (numpy + netCDF4, imported here)
# --------------------------------------------------------------------------

def _cell_major(variable):
    import numpy as np
    array = np.array(variable[:])
    array = array.reshape(array.shape[-2:]) if array.ndim >= 2 else array
    return array if array.shape[0] > array.shape[1] else array.T


def _haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def _frame_label(path: str) -> str:
    name = os.path.basename(path)
    return name[len(HISTORY_PREFIX):-3] if name.startswith(HISTORY_PREFIX) else name[:-3]


class _Leg:
    """One leg's geometry and the frames it wrote."""

    def __init__(self, leg_out: str, grid: str, init: str) -> None:
        import netCDF4
        import numpy as np
        with netCDF4.Dataset(grid) as g:
            lat = np.degrees(np.array(g.variables["latCell"][:]))
            lon = np.degrees(np.array(g.variables["lonCell"][:]))
        self.lat = lat
        self.lon = np.where(lon > 180, lon - 360, lon)
        with netCDF4.Dataset(init) as ini:
            zgrid = _cell_major(ini.variables["zgrid"])
        self.zmid = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
        self.frames = sorted(glob.glob(os.path.join(leg_out, HISTORY_PREFIX + "*.nc")))
        if not self.frames:
            raise SystemExit("no %s*.nc under %s" % (HISTORY_PREFIX, leg_out))

    def wind(self, path: str):
        """u and v as (nCells, nVertLevels) arrays."""
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            missing = [n for n in ("u_zonal", "v_meridional") if n not in ds.variables]
            if missing:
                raise SystemExit("%s carries no %s; the placement reads the leg's own wind"
                                 % (path, ", ".join(missing)))
            return _cell_major(ds.variables["u_zonal"]), _cell_major(ds.variables["v_meridional"])

    def nearest(self, lat: float, lon: float, alt_m: float) -> tuple[int, int]:
        import numpy as np
        cell = int(np.argmin(_haversine_km(self.lat, self.lon, lat, lon)))
        level = int(np.argmin(np.abs(self.zmid[cell] - alt_m)))
        return cell, level


# --------------------------------------------------------------------------
# place
# --------------------------------------------------------------------------

def _check_line(leg: _Leg, winds: dict, lat_a: float, lat_b: float, lon_c: float,
                alt_m: float, lead_seconds: float) -> dict[str, dict[str, float]]:
    """The wind at 21 points along the line, per frame, and how far it carries a release."""
    import numpy as np
    points = np.linspace(lat_a, lat_b, 21)
    picks = [leg.nearest(float(la), lon_c, alt_m) for la in points]
    cells = np.array([c for c, _ in picks])
    levels = np.array([k for _, k in picks])
    out = {}
    for label in sorted(winds):
        u, v = winds[label]
        uu = u[cells, levels].astype(np.float64)
        vv = v[cells, levels].astype(np.float64)
        speed = np.hypot(uu, vv)
        out[label] = {"u": float(uu.mean()), "v": float(vv.mean()),
                      "speed_min": float(speed.min()), "speed_max": float(speed.max()),
                      "carry_km": float(speed.mean() * lead_seconds / 1000.0),
                      "z_level_m": float(leg.zmid[cells, levels].mean())}
    return out


def run_place(args: argparse.Namespace) -> int:
    import numpy as np
    leg = _Leg(args.leg_out, args.grid, args.init)
    wanted = [s for s in args.frames.split(",") if s]
    by_label = {_frame_label(path): path for path in leg.frames}
    missing = [s for s in wanted if s not in by_label]
    if missing:
        raise SystemExit("frames not under %s: %s (have %s)" % (
            args.leg_out, ", ".join(missing), ", ".join(sorted(by_label))))
    winds = {label: leg.wind(path) for label, path in by_label.items()}
    lat_t, lon_t = args.target
    cell, level = leg.nearest(lat_t, lon_t, args.alt_m)
    u_t = float(np.mean([winds[s][0][cell, level] for s in wanted]))
    v_t = float(np.mean([winds[s][1][cell, level] for s in wanted]))
    lead = args.lead_min * 60.0
    dlat, dlon = upwind_shift(lat_t, u_t, v_t, lead)
    line_lon = round(lon_t + dlon, 2)
    line_lat0 = round(lat_t + dlat - LINE_LENGTH_DEG / 2, 3)
    line_lat1 = round(line_lat0 + LINE_LENGTH_DEG, 3)
    result = {
        "placement_frames": wanted,
        "target": {"lat": lat_t, "lon": lon_t, "alt_m": float(args.alt_m),
                   "cell": cell, "level": level, "z_level_m": float(leg.zmid[cell, level]),
                   "u": u_t, "v": v_t},
        "lead_min": args.lead_min,
        "line": {"lat0": line_lat0, "lat1": line_lat1, "lon": line_lon, "alt_m": float(args.alt_m)},
        "check_line": _check_line(leg, winds, line_lat0, line_lat1, line_lon, args.alt_m, lead),
        "result_sentence": "a model result of the control leg, not evidence about the atmosphere",
    }
    with open(args.json, "w") as handle:
        json.dump(result, handle, indent=1)
    print("target %.3f %.3f at %.0f m (cell %d, level %d): wind u %.1f v %.1f over %s"
          % (lat_t, lon_t, args.alt_m, cell, level, u_t, v_t, ",".join(wanted)))
    print("line (%.0f min upwind): %.3f-%.3f N at %.2f E, alt %.0f m" % (
        args.lead_min, line_lat0, line_lat1, line_lon, args.alt_m))
    for label, row in result["check_line"].items():
        print("  %s u %.1f v %.1f speed %.1f..%.1f carries %.0f km in %.0f min z %.0f" % (
            label, row["u"], row["v"], row["speed_min"], row["speed_max"],
            row["carry_km"], args.lead_min, row["z_level_m"]))
    return 0


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------

def run_table(args: argparse.Namespace) -> int:
    release = _dt.datetime.strptime(args.release, "%Y-%m-%d_%H:%M:%S")
    text = table_text(args.lat0, args.lon, args.alt_m, release, rate=args.rate,
                      case=args.case, notes=args.note)
    with open(args.out, "w", newline="\n") as handle:
        handle.write(text)
    record: dict[str, Any] = {
        "lat0": args.lat0, "lat1": round(args.lat0 + LINE_LENGTH_DEG, 3), "lon": args.lon,
        "alt_m": args.alt_m, "release": args.release, "rate": args.rate, "notes": list(args.note),
    }
    if args.placement:
        with open(args.placement) as handle:
            record["placement"] = json.load(handle)
    with open(args.out + ".placement.json", "w") as handle:
        json.dump(record, handle, indent=1)
    sys.stdout.write(text)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    place = sub.add_parser("place", help="place a release line upwind of a target, in the leg's own wind")
    place.add_argument("--leg-out", required=True, metavar="DIR",
                       help="the leg output tree holding cuda-history.*.nc")
    place.add_argument("--grid", required=True, metavar="GRID.nc")
    place.add_argument("--init", required=True, metavar="INIT.nc")
    place.add_argument("--target", nargs=2, type=float, required=True, metavar=("LAT", "LON"),
                       help="where the plume should arrive")
    place.add_argument("--alt-m", type=float, required=True,
                       help="release altitude (m MSL); the wind is read at the nearest level")
    place.add_argument("--frames", required=True, metavar="STAMP,STAMP",
                       help="frame stamps (as in cuda-history.<STAMP>.nc) whose wind is averaged")
    place.add_argument("--lead-min", type=float, default=30.0,
                       help="minutes of the target wind the line is shifted upwind by")
    place.add_argument("--json", required=True, metavar="FILE")
    place.set_defaults(run=run_place)

    table = sub.add_parser("table", help="write the point-source table for one explicit line")
    table.add_argument("--lat0", type=float, required=True, help="southern end of the line (deg N)")
    table.add_argument("--lon", type=float, required=True, help="the line longitude (deg E)")
    table.add_argument("--alt-m", type=float, required=True, help="release altitude (m MSL)")
    table.add_argument("--release", required=True, metavar="YYYY-MM-DD_HH:MM:SS",
                       help="start of the first pass (UTC)")
    table.add_argument("--rate", type=float, default=RATE_DEFAULT, help="particles per second")
    table.add_argument("--case", default="", help="case label for the header line")
    table.add_argument("--note", action="append", default=[],
                       help="a header comment line; repeatable")
    table.add_argument("--placement", default=None, metavar="FILE",
                       help="a `place` JSON to embed in the table placement record")
    table.add_argument("--out", required=True, metavar="TABLE.txt")
    table.set_defaults(run=run_table)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
