#!/usr/bin/env python3
"""Materialize one case's canonical observation bundles from the shipped doors.

The canonical-bundle contract says a rustwx producer is responsible for all raw
format decode, quality control, unit conversion and accumulation-window
semantics. Until this file existed the only thing that had ever written a
canonical bundle was the synthetic fixture, which a production manifest is
forbidden to accept — so the contract named a producer nobody could run.

This is that producer. It parses no archive format. `rw_stage4`, `rw_mrms` and
`rw_asos` own every GRIB2 and METAR byte; this script drives them through
gpuwm's own front-door resolver, reads the packs they write with gpuwm's own
pack reader, and repacks the arrays into the referee's containers. It changes
no value and derives no quantity.

Each receipt records the door that produced the bytes, the door binary's
SHA-256, and the archive object digest behind every frame, so a bundle names
its whole chain instead of asserting it.

```sh
python verification/producers/observation_bundles.py \
    --case CASE_ID --window-start 2025-07-14T12:00:00Z --hours 24 \
    --out-root /abs/case-root --stations /abs/asos-stations.json
```

Freeze the station table once, with the shipped door, before the first case:

```sh
rw_asos stations --networks "$(python -c '...')" --bbox W,S,E,N --out stations.json
```

`gpuwm.obs.surface_networks.networks_for_bbox` names the networks a box needs.
The table is frozen and hashed rather than read per fetch because coordinates
that arrive with the observations can move between the fetch that registers a
case and the fetch that scores it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from gpuwm.obs.frontdoor import ASOS, MRMS, STAGE4
from gpuwm.obs.obspack import read_grid_pack, read_pack

from mpas_port.obs_referee.bundle import write_grid_bundle, write_station_bundle
from mpas_port.obs_referee.canonical import write_json

PRODUCER = "gpuwm-hex"

#: What the reflectivity decode is trimmed to. The full CONUS composite is
#: 7000x3500 and 196 MB of float64 per frame; a cell outside the model window
#: can never be scored, so carrying it only makes an artifact that has to be
#: hashed and read bigger. Override for a different verification domain.
DEFAULT_BBOX = "-126,21,-66,53"

#: The station door's spellings, and the canonical ones. Only fields the door
#: actually emits appear. It reports MSLP -- a sea-level reduction -- and not
#: station pressure, so `surface_pressure_pa` is deliberately absent from every
#: bundle this writes: the two are not the same quantity, and mapping one onto
#: the other would fabricate the reduction.
STATION_FIELDS = {
    "temperature_2m": "temperature_k",
    "dewpoint_2m": "dewpoint_k",
    "wind_speed_10m": "wind_speed_ms",
}
STATION_PRESSURE_ABSENCE = (
    "the shipped ASOS door reports MSLP, a sea-level reduction, not station "
    "pressure; the two are not the same quantity and mapping one onto the other "
    "would fabricate the reduction"
)


def digest(path: pathlib.Path) -> str:
    running = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            running.update(chunk)
    return running.hexdigest()


def door(front) -> pathlib.Path:
    found = front.find()
    if found is None:
        raise SystemExit(
            f"{front.name} is not resolvable, so no {front.subject} can be "
            f"decoded and no bundle can be written.\n{front.remedy()}")
    return found


def run(argv: list[str]) -> dict:
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"door failed ({completed.returncode}): {' '.join(argv)}\n"
                         f"{completed.stderr.strip()[-1500:]}")
    text = completed.stdout.strip()
    return json.loads(text) if text.startswith("{") else {}


def stamp(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def amend_receipt(receipt_path: pathlib.Path, source_artifacts: list[dict]) -> None:
    """The bundle writer emits an empty provenance list; fill it in.

    A receipt saying only "gpuwm-hex made this" is not provenance. The chain
    that matters is which door binary read which archive object.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_artifacts"] = source_artifacts
    write_json(receipt_path, receipt)


def build_stage4(case_dir: pathlib.Path, valid_times: list[dt.datetime],
                 cache: pathlib.Path) -> dict:
    """Hourly multi-sensor QPE, mm, on the 4.7625 km HRAP grid."""
    exe = door(STAGE4)
    work = case_dir / ".stage4"
    work.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    sources: list[dict] = []
    latitude = longitude = None
    for when in valid_times:
        run([str(exe), "fetch", "--start", stamp(when), "--end", stamp(when),
             "--accumulation", "01h", "--out", str(work), "--cache", str(cache)])
        name = f"ST4.{when:%Y%m%d%H}.01h.grib"
        matches = sorted(cache.rglob(name)) or sorted(work.rglob(name))
        if not matches:
            raise SystemExit(f"Stage-IV object {name} did not arrive")
        archived = matches[0]
        pack = work / f"{name}.obspack"
        run([str(exe), "decode", "--file", str(archived), "--out", str(pack)])
        grid = read_grid_pack(pack)
        values = np.asarray(grid.arrays["values"], dtype=np.float64)
        valid = np.asarray(grid.arrays["valid"], dtype=np.uint8).astype(bool)
        frames.append(np.where(valid, values, np.nan))
        if latitude is None:
            geometry_pack = work / "geometry.obspack"
            run([str(exe), "grid", "--file", str(archived), "--out", str(geometry_pack)])
            geometry = read_pack(geometry_pack)
            latitude = np.asarray(geometry.arrays["latitude"], dtype=np.float64)
            longitude = np.asarray(geometry.arrays["longitude"], dtype=np.float64)
        sources.append({
            "door": exe.name, "door_sha256": digest(exe),
            "archive_object": archived.name, "archive_sha256": digest(archived),
            "pack_sha256": digest(pack), "valid_time": grid.meta["valid_time"],
            "quantity": grid.meta["quantity"], "units": grid.meta["units"],
        })
        pack.unlink(missing_ok=True)
    artifact, receipt = write_grid_bundle(
        case_dir / "stage4.npz",
        time_unix_s=np.asarray([int(w.timestamp()) for w in valid_times], dtype=np.int64),
        latitude_deg=latitude, longitude_deg=longitude,
        fields={"precip_1h_mm": np.stack(frames, axis=0)},
        producer=PRODUCER,
        producer_version=f"observation_bundles.py via {exe.name}",
        metadata={
            "instrument": "NCEP/EMC Stage-IV multi-sensor QPE, HRAP 4.7625 km",
            "accumulation_hours": 1,
            "raw_parser_boundary": "rustwx",
            "substitution": (
                "the declared referee names MRMS one-hour precipitation; the "
                "shipped MRMS door decodes composite reflectivity only, so the "
                "nearest runnable equivalent behind a shipped door is Stage-IV"),
        })
    amend_receipt(receipt, sources)
    shutil.rmtree(work, ignore_errors=True)
    return {"artifact": str(artifact), "sha256": digest(artifact),
            "times": len(valid_times), "shape": list(frames[0].shape)}


def build_mrms(case_dir: pathlib.Path, valid_times: list[dt.datetime],
               cache: pathlib.Path, bbox: str) -> dict:
    """QC'd column-max composite reflectivity, dBZ, on the 0.01 degree grid.

    The archive stamps carry off-cadence seconds, so a URL cannot be built for
    a valid time: `nearest` lists the day and picks the closest object, and the
    offset it reports goes in the receipt.
    """
    exe = door(MRMS)
    work = case_dir / ".mrms"
    work.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    sources: list[dict] = []
    latitude = longitude = None
    for when in valid_times:
        near = run([str(exe), "nearest", "--valid-time", stamp(when),
                    "--window-seconds", "240"])
        frame = near["frame"]
        exact = dt.datetime.strptime(frame["valid_time"], "%Y-%m-%dT%H:%M:%S")
        exact = exact.replace(tzinfo=dt.timezone.utc)
        run([str(exe), "fetch", "--start", stamp(exact), "--end", stamp(exact),
             "--out", str(work), "--cache", str(cache)])
        matches = (sorted(cache.rglob(frame["filename"]))
                   or sorted(work.rglob(frame["filename"])))
        if not matches:
            raise SystemExit(f"MRMS object {frame['filename']} did not arrive")
        archived = matches[0]
        pack = work / f"{frame['filename']}.obspack"
        run([str(exe), "decode", "--file", str(archived), "--out", str(pack),
             "--bbox", bbox])
        grid = read_grid_pack(pack)
        values = np.asarray(grid.arrays["values"], dtype=np.float64)
        valid = np.asarray(grid.arrays["valid"], dtype=np.uint8).astype(bool)
        frames.append(np.where(valid, values, np.nan))
        if latitude is None:
            geometry_pack = work / "geometry.obspack"
            run([str(exe), "grid", "--file", str(archived), "--out",
                 str(geometry_pack), "--bbox", bbox])
            geometry = read_pack(geometry_pack)
            latitude = np.asarray(geometry.arrays["latitude"], dtype=np.float64)
            longitude = np.asarray(geometry.arrays["longitude"], dtype=np.float64)
        sources.append({
            "door": exe.name, "door_sha256": digest(exe),
            "archive_object": archived.name, "archive_sha256": digest(archived),
            "pack_sha256": digest(pack), "requested_valid_time": stamp(when),
            "frame_valid_time": frame["valid_time"],
            "offset_seconds": near["offset_seconds"],
            "quantity": grid.meta["quantity"], "units": grid.meta["units"],
        })
        pack.unlink(missing_ok=True)
    artifact, receipt = write_grid_bundle(
        case_dir / "mrms.npz",
        time_unix_s=np.asarray([int(w.timestamp()) for w in valid_times], dtype=np.int64),
        latitude_deg=latitude, longitude_deg=longitude,
        fields={"reflectivity_dbz": np.stack(frames, axis=0)},
        producer=PRODUCER,
        producer_version=f"observation_bundles.py via {exe.name}",
        metadata={
            "instrument": "MRMS MergedReflectivityQCComposite_00.50, 0.01 degree",
            "no_echo_dbz": -35.0,
            "no_echo_note": (
                "a no-echo cell is an observation that the column was below "
                "detection and stays valid at the floor value; only no-coverage "
                "cells are masked"),
            "raw_parser_boundary": "rustwx", "bbox": bbox,
        })
    amend_receipt(receipt, sources)
    shutil.rmtree(work, ignore_errors=True)
    return {"artifact": str(artifact), "sha256": digest(artifact),
            "times": len(valid_times), "shape": list(frames[0].shape)}


def build_asos(case_dir: pathlib.Path, window_start: dt.datetime,
               valid_times: list[dt.datetime], stations: pathlib.Path) -> dict:
    """Screened, hourly-matched ASOS/AWOS surface reports in seam units."""
    exe = door(ASOS)
    work = case_dir / ".asos"
    work.mkdir(parents=True, exist_ok=True)
    csv = work / "obs.csv"
    fetched = run([str(exe), "fetch", "--stations", str(stations),
                   "--start", stamp(window_start), "--end", stamp(valid_times[-1]),
                   "--out", str(csv)])
    decoded = work / "surface.json"
    run([str(exe), "decode", "--obs", str(csv), "--stations", str(stations),
         "--start", stamp(valid_times[0]), "--end", stamp(valid_times[-1]),
         "--step-hours", "1", "--out", str(decoded)])
    record = json.loads(decoded.read_text(encoding="utf-8"))
    index = {station["station_id"]: station for station in record["stations"]}

    rows: list[dict] = []
    for report in record["reports"]:
        site = index[report["station_id"]]
        reported = report.get("values", {})
        fields = {canonical: float(reported[name])
                  for name, canonical in STATION_FIELDS.items()
                  if reported.get(name) is not None}
        if not fields:
            continue
        when = dt.datetime.strptime(report["valid_time"], "%Y-%m-%dT%H:%M:%S")
        rows.append({
            "station_id": report["station_id"],
            "time_unix_s": int(when.replace(tzinfo=dt.timezone.utc).timestamp()),
            "latitude_deg": float(site["latitude"]),
            "longitude_deg": float(site["longitude"]),
            "fields": fields,
        })
    if not rows:
        raise SystemExit(
            "the ASOS door returned no scoreable report; a station bundle with "
            "no record would read as an observed absence of weather rather than "
            "as a failed fetch")

    table_digest = json.loads(stations.read_text(encoding="utf-8"))["content_sha256"]
    artifact, receipt = write_station_bundle(
        case_dir / "asos.jsonl", records=rows, producer=PRODUCER,
        producer_version=f"observation_bundles.py via {exe.name}",
        metadata={
            "instrument": "ASOS/AWOS routine and special METARs, IEM archive",
            "raw_parser_boundary": "rustwx",
            "station_table_sha256": table_digest,
            "fields_absent": {"surface_pressure_pa": STATION_PRESSURE_ABSENCE},
        })
    amend_receipt(receipt, [{
        "door": exe.name, "door_sha256": digest(exe),
        "csv_sha256": fetched["sha256"], "csv_rows": fetched["rows"],
        "requests": fetched["requests"],
        "stations_requested": fetched["stations_requested"],
        "stations_kept": len(record["stations"]),
        "reports": len(record["reports"]),
        "station_table_sha256": table_digest,
    }])
    shutil.rmtree(work, ignore_errors=True)
    return {"artifact": str(artifact), "sha256": digest(artifact),
            "records": len(rows), "stations": len(record["stations"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--window-start", required=True,
                        help="YYYY-MM-DDTHH:MM:SSZ, the forecast's own start")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--out-root", required=True,
                        help="the manifest's OBS_REFEREE_DATA_ROOT")
    parser.add_argument("--stations", required=True,
                        help="a frozen table from `rw_asos stations`")
    parser.add_argument("--cache", required=True,
                        help="archive object cache, shared across cases")
    parser.add_argument("--bbox", default=DEFAULT_BBOX,
                        help=f"reflectivity decode box W,S,E,N (default {DEFAULT_BBOX})")
    parser.add_argument("--only", action="append", default=[],
                        choices=["stage4", "mrms", "asos"])
    args = parser.parse_args(argv)

    start = dt.datetime.strptime(args.window_start, "%Y-%m-%dT%H:%M:%SZ")
    start = start.replace(tzinfo=dt.timezone.utc)
    # The first history frame is the initial condition, which no hourly
    # accumulation can be differenced out of, so the scored times run t+1..t+N.
    valid_times = [start + dt.timedelta(hours=hour) for hour in range(1, args.hours + 1)]
    case_dir = pathlib.Path(args.out_root) / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    wanted = args.only or ["stage4", "asos", "mrms"]
    summary = {"case": args.case, "window_start": args.window_start,
               "hours": args.hours, "sources": wanted,
               "valid_times": [stamp(when) for when in valid_times]}
    if "stage4" in wanted:
        summary["stage4"] = build_stage4(case_dir, valid_times, cache)
    if "asos" in wanted:
        summary["asos"] = build_asos(case_dir, start, valid_times,
                                     pathlib.Path(args.stations))
    if "mrms" in wanted:
        summary["mrms"] = build_mrms(case_dir, valid_times, cache, args.bbox)
    write_json(case_dir / f"observation-build-{'-'.join(sorted(wanted))}.json", summary)
    json.dump({key: summary[key] for key in wanted if key in summary},
              sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
