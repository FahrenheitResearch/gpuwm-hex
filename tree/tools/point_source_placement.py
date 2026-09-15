#!/usr/bin/env python3
"""Place a point-source table inside a control leg's OWN supercooled liquid.

WHY THIS FILE EXISTS.  A seeded pair (``gpuwm-hex pair`` with ``--source-table``)
answers "what did the model do differently when this table was applied?", and
the answer is exactly zero whenever the table's release line sits in air the
model holds no supercooled liquid in.  That happened twice on 2026-09-02: the
first 3.75 km placement went to a reflectivity cluster the model kept dry
(every weather-field difference 0.0), and the 937.5 m global run released into
warm cumulus because the core held no supercooled liquid in the window.  Both
pairs ran, closed their ledgers and taught nothing about the response.  The
placement that DID respond was authored from the control leg's own cloud
census -- inside the supercooled water and upwind of it, so the wind carried
the plume through.  This tool is that census and that placement as one
program, so the next pair does not re-derive it in a shell loop.

WHAT IT READS.  The history frames of a finished leg (``cuda-history.*.nc``,
the forecast door's own output: ``qc``, ``theta``, ``pressure``, ``u_zonal``,
``v_meridional``), the grid (``latCell``, ``lonCell``) and the init
(``zgrid``, the converter's only source of heights).  A cell is SUPERCOOLED
LIQUID when it holds cloud water above ``QC_MIN`` at a temperature below 0 C.

THREE COMMANDS.

``census``
    Per frame, inside ``--radius-km`` of the core centre: how many cells and
    columns hold supercooled liquid, the temperature and height bands, the
    qc-weighted centroid and the qc-weighted wind there.  The screening
    answer: is there anything to seed, and when.

``place``
    Bins the supercooled cells of the named frames (inside ``--alt-band``)
    into 0.05-degree columns and scores every north-south segment of
    ``LINE_LENGTH_DEG`` by the cells it and its two neighbouring columns hold;
    the densest segment within ``--max-km`` of the centre is the TARGET.  The
    line altitude is the qc-weighted height of the supercooled water in that
    segment (or ``--alt-m``, when the per-level profile the command prints
    argues for a colder level), the wind is the qc-weighted wind there, and
    the line is the target shifted UPWIND by ``--lead-min`` of that wind.  The
    placed line and the target are then CHECKED against every frame at that
    altitude: supercooled fraction along the line, temperature range, wind.

``table``
    Writes the table for one explicit line in the proven pass structure:
    three north-south passes along one longitude, five waypoints 180 s apart
    (0.1-degree steps, about 44 km), the last waypoint of each pass ``on=0``,
    passes 21 minutes apart, ``--rate`` particles per second, 54 minutes from
    ``--release``.  Epoch seconds count from 2017-01-01T00:00Z, the engine's
    own point-source epoch.

Everything ``census`` and ``place`` report is a MODEL RESULT of the control
leg: what this model held in these frames.  It is not a measurement of the
atmosphere.  The table this tool writes is a SYNTHETIC release line authored
for a paired experiment; no real programme, operator or operation is
represented, and the file says so on its first lines.

Usage::

    python tools/point_source_placement.py census --leg-out OUT --grid G --init I [--json F]
    python tools/point_source_placement.py place  --leg-out OUT --grid G --init I \\
        --frames STAMP,STAMP[,...] --json F [--alt-m Z] [--lead-min 30]
    python tools/point_source_placement.py table  --lat0 L --lon LON --alt-m Z \\
        --release YYYY-MM-DD_HH:MM:SS --out TABLE.txt [--note TEXT ...]

numpy and netCDF4 are imported inside ``census`` and ``place`` only, so
``table`` and ``--help`` work on a box with no scientific stack.
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
    "QC_MIN",
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

#: Cloud water above this is cloud; below it is the scheme's own residue.
QC_MIN = 1.0e-6

#: The proven pass structure (the 2026-09-02 3.75 km pair, run 2).
LINE_LENGTH_DEG = 0.4
WAYPOINTS = 5
WAYPOINT_GAP_SECONDS = 180.0
PASS_GAP_SECONDS = 21 * 60.0
PASSES = 3
RATE_DEFAULT = 8.5e13

RD, CP, P0 = 287.0, 1004.5, 1.0e5
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
    """The fifteen waypoints of the proven pass structure, in release order.

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

    def read(self, path: str):
        """qc, T [K], u, v as (nCells, nVertLevels) arrays; u and v None when absent."""
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            qc = _cell_major(ds.variables["qc"])
            theta = _cell_major(ds.variables["theta"])
            pressure = _cell_major(ds.variables["pressure"])
            u = _cell_major(ds.variables["u_zonal"]) if "u_zonal" in ds.variables else None
            v = _cell_major(ds.variables["v_meridional"]) if "v_meridional" in ds.variables else None
        temperature = theta * (pressure / P0) ** (RD / CP)
        return qc, temperature, u, v


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------

def run_census(args: argparse.Namespace) -> int:
    import numpy as np
    leg = _Leg(args.leg_out, args.grid, args.init)
    clat, clon = args.center
    core = _haversine_km(leg.lat, leg.lon, clat, clon) <= args.radius_km
    print("core cells within %.0f km of %.3f %.3f: %d of %d"
          % (args.radius_km, clat, clon, int(core.sum()), core.size))
    print("frame | slw cells | slw cols | qc_max slw | T range C | z 10-90% m | "
          "slw-wtd lat lon z | u v at wtd | tot qc cols")
    rows = []
    for path in leg.frames:
        qc, T, u, v = leg.read(path)
        label = _frame_label(path)
        slw = (qc > QC_MIN) & (T < 273.15) & core[:, None]
        anyqc = (qc > QC_MIN) & core[:, None]
        row: dict[str, Any] = {
            "frame": label, "slw_cells": int(slw.sum()),
            "slw_cols": int(slw.any(axis=1).sum()), "qc_cols": int(anyqc.any(axis=1).sum()),
        }
        if slw.any():
            # the frames store qc as float32; the weights are summed in float64 so
            # the centroid does not carry the float32 accumulation's 1e-7
            w = qc[slw].astype(np.float64)
            z = leg.zmid[slw]
            t = T[slw] - 273.15
            ci, _ = np.nonzero(slw)
            row.update(
                qc_max=float(w.max()), T_min=float(t.min()), T_max=float(t.max()),
                z10=float(np.percentile(z, 10)), z90=float(np.percentile(z, 90)),
                wlat=float(np.sum(w * leg.lat[ci]) / w.sum()),
                wlon=float(np.sum(w * leg.lon[ci]) / w.sum()),
                wz=float(np.sum(w * z) / w.sum()),
            )
            if u is not None and v is not None:
                row.update(u=float(np.sum(w * u[slw]) / w.sum()),
                           v=float(np.sum(w * v[slw]) / w.sum()))
            print("%s | %9d | %8d | %.2e | %6.1f..%5.1f | %6.0f-%6.0f | %8.3f %9.3f %6.0f | "
                  "%5.1f %5.1f | %d" % (
                      label, row["slw_cells"], row["slw_cols"], row["qc_max"], row["T_min"],
                      row["T_max"], row["z10"], row["z90"], row["wlat"], row["wlon"], row["wz"],
                      row.get("u", float("nan")), row.get("v", float("nan")), row["qc_cols"]))
        else:
            print("%s | %9d | %8d |    none  |            |             |"
                  "                        |            | %d" % (label, 0, 0, row["qc_cols"]))
        rows.append(row)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1)
    return 0


# --------------------------------------------------------------------------
# place
# --------------------------------------------------------------------------

def _check_line(leg: _Leg, data: dict, lat_a: float, lat_b: float, lon_c: float,
                alt_m: float) -> dict[str, dict[str, float]]:
    """The nearest cell to each of 21 points along the line, at the level nearest alt_m."""
    import numpy as np
    points = np.linspace(lat_a, lat_b, 21)
    cells = np.array([int(np.argmin(_haversine_km(leg.lat, leg.lon, la, lon_c))) for la in points])
    levels = np.array([int(np.argmin(np.abs(leg.zmid[c] - alt_m))) for c in cells])
    out = {}
    for label in sorted(data):
        qc, T, u, v = data[label]
        q = qc[cells, levels]
        t = T[cells, levels] - 273.15
        slw = (q > QC_MIN) & (t < 0)
        row = {"slw_frac": float(slw.mean()), "cloud_frac": float((q > QC_MIN).mean()),
               "T_min": float(t.min()), "T_max": float(t.max()), "qc_max": float(q.max()),
               "z_level_m": float(leg.zmid[cells, levels].mean())}
        if u is not None and v is not None:
            row["u"] = float(u[cells, levels].mean())
            row["v"] = float(v[cells, levels].mean())
        out[label] = row
    return out


def run_place(args: argparse.Namespace) -> int:
    import numpy as np
    leg = _Leg(args.leg_out, args.grid, args.init)
    clat, clon = args.center
    wanted = [s for s in args.frames.split(",") if s]
    data = {}
    for path in leg.frames:
        data[_frame_label(path)] = leg.read(path)
    missing = [s for s in wanted if s not in data]
    if missing:
        raise SystemExit("frames not under %s: %s (have %s)" % (
            args.leg_out, ", ".join(missing), ", ".join(sorted(data))))
    zlo, zhi = args.alt_band
    band = (leg.zmid >= zlo) & (leg.zmid <= zhi)

    # (1) score every candidate segment on the placement frames
    bins_lat = np.arange(clat - 1.5, clat + 1.5 + 1e-9, 0.05)
    bins_lon = np.arange(clon - 1.5, clon + 1.5 + 1e-9, 0.05)
    hist = np.zeros((len(bins_lat) - 1, len(bins_lon) - 1))
    for label in wanted:
        qc, T, _, _ = data[label]
        ci, _ = np.nonzero((qc > QC_MIN) & (T < 273.15) & band)
        h, _, _ = np.histogram2d(leg.lat[ci], leg.lon[ci], bins=[bins_lat, bins_lon])
        hist += h
    nlat = int(round(LINE_LENGTH_DEG / 0.05))
    best = None
    for j in range(1, hist.shape[1] - 1):
        column = hist[:, j - 1:j + 2].sum(axis=1)
        for i in range(0, hist.shape[0] - nlat):
            lat0 = float(bins_lat[i])
            lonc = float(bins_lon[j] + 0.025)
            if _haversine_km(lat0 + LINE_LENGTH_DEG / 2, lonc, clat, clon) > args.max_km:
                continue
            score = float(column[i:i + nlat + 1].sum())
            if best is None or score > best[0]:
                best = (score, round(lat0, 3), round(lonc, 3))
    if best is None or best[0] <= 0:
        raise SystemExit("no supercooled liquid in %s within %.0f km of %.3f %.3f between "
                         "%.0f and %.0f m: nothing to place" % (
                             args.frames, args.max_km, clat, clon, zlo, zhi))
    score, lat0, lonc = best
    lat1 = round(lat0 + LINE_LENGTH_DEG, 3)

    # (2) altitude, wind and temperature of the supercooled water in the target
    segment = ((leg.lat >= lat0 - 0.05) & (leg.lat <= lat1 + 0.05)
               & (np.abs(leg.lon - lonc) <= 0.075))
    wsum = zs = us = vs = ts = 0.0
    profile = []
    for label in wanted:
        qc, T, u, v = data[label]
        slw = (qc > QC_MIN) & (T < 273.15) & band & segment[:, None]
        w = qc[slw].astype(np.float64)
        wsum += float(w.sum())
        zs += float((w * leg.zmid[slw]).sum())
        ts += float((w * (T[slw] - 273.15)).sum())
        if u is not None and v is not None:
            us += float((w * u[slw]).sum())
            vs += float((w * v[slw]).sum())
        per_level = ((qc > QC_MIN) & (T < 273.15) & segment[:, None]).sum(axis=0)
        profile.append({"frame": label, "slw_cells_per_level": [int(n) for n in per_level],
                        "z_mean_m": [float(z) for z in leg.zmid[segment].mean(axis=0)],
                        "T_mean_C": [float(t) for t in (T[segment].mean(axis=0) - 273.15)]})
    weighted_alt = float(round(zs / wsum / 100.0) * 100.0)
    alt = float(args.alt_m) if args.alt_m is not None else weighted_alt
    uu, vv, tw = us / wsum, vs / wsum, ts / wsum
    lead = args.lead_min * 60.0
    dlat, dlon = upwind_shift(lat0 + LINE_LENGTH_DEG / 2, uu, vv, lead)
    line_lon = round(lonc + dlon, 2)
    line_lat0 = round(lat0 + dlat, 3)
    line_lat1 = round(line_lat0 + LINE_LENGTH_DEG, 3)

    result = {
        "placement_frames": wanted, "alt_band_m": [zlo, zhi],
        "segment_score_slw_cells": score,
        "target_segment": {"lat0": lat0, "lat1": lat1, "lon": lonc, "alt_m_weighted": weighted_alt,
                           "T_wtd_C": tw, "u": uu, "v": vv},
        "target_profile": profile,
        "lead_min": args.lead_min,
        "line": {"lat0": line_lat0, "lat1": line_lat1, "lon": line_lon, "alt_m": alt,
                 "alt_source": "explicit --alt-m" if args.alt_m is not None else "qc-weighted"},
        "check_line": _check_line(leg, data, line_lat0, line_lat1, line_lon, alt),
        "check_target": _check_line(leg, data, lat0, lat1, lonc, alt),
        "result_sentence": "a model result of the control leg, not evidence about the atmosphere",
    }
    with open(args.json, "w") as handle:
        json.dump(result, handle, indent=1)
    print("target segment: %.3f-%.3f N at %.3f E, qc-weighted alt %.0f m, T_wtd %.1f C, "
          "wind u %.1f v %.1f (score %.0f SLW cells)" % (lat0, lat1, lonc, weighted_alt, tw, uu, vv, score))
    print("target profile (level: z_mean T_mean slw cells per placement frame)")
    n_levels = len(profile[0]["z_mean_m"])
    for k in range(n_levels):
        if not (zlo - 500 <= profile[0]["z_mean_m"][k] <= zhi + 500):
            continue
        print("  k%02d %6.0f m %6.1f C  %s" % (
            k, profile[0]["z_mean_m"][k], profile[0]["T_mean_C"][k],
            "  ".join("%4d" % p["slw_cells_per_level"][k] for p in profile)))
    print("line (%.0f min upwind): %.3f-%.3f N at %.2f E, alt %.0f m (%s)" % (
        args.lead_min, line_lat0, line_lat1, line_lon, alt, result["line"]["alt_source"]))
    for name in ("check_line", "check_target"):
        print(name)
        for label, row in result[name].items():
            print("  %s slw %.2f cloud %.2f T %5.1f..%5.1f qc_max %.2e u %.1f v %.1f z %.0f" % (
                label, row["slw_frac"], row["cloud_frac"], row["T_min"], row["T_max"],
                row["qc_max"], row.get("u", float("nan")), row.get("v", float("nan")),
                row["z_level_m"]))
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

def _leg_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--leg-out", required=True, metavar="DIR",
                        help="the leg output tree holding cuda-history.*.nc")
    parser.add_argument("--grid", required=True, metavar="GRID.nc")
    parser.add_argument("--init", required=True, metavar="INIT.nc")
    parser.add_argument("--center", nargs=2, type=float, default=(35.0, -97.0),
                        metavar=("LAT", "LON"), help="the fine core centre")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census", help="supercooled-liquid census of a leg, per frame")
    _leg_arguments(census)
    census.add_argument("--radius-km", type=float, default=100.0)
    census.add_argument("--json", default=None, metavar="FILE")
    census.set_defaults(run=run_census)

    place = sub.add_parser("place", help="find the densest supercooled segment and shift it upwind")
    _leg_arguments(place)
    place.add_argument("--frames", required=True, metavar="STAMP,STAMP",
                       help="frame stamps (as in cuda-history.<STAMP>.nc) the plume should arrive in")
    place.add_argument("--max-km", type=float, default=60.0,
                       help="the target segment centre stays within this of the core centre")
    place.add_argument("--alt-band", nargs=2, type=float, default=(4500.0, 8000.0),
                       metavar=("LO", "HI"), help="heights (m) the scan counts supercooled water in")
    place.add_argument("--lead-min", type=float, default=30.0,
                       help="minutes of the target wind the line is shifted upwind by")
    place.add_argument("--alt-m", type=float, default=None,
                       help="release altitude (m MSL); default the qc-weighted height of the target")
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
