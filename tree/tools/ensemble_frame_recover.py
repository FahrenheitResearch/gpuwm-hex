"""Publish the frame products of ensemble members whose in-door consumer did not.

One process, one frame at a time (netCDF is not thread-safe). A durable frame
still on disk ``--min-age`` seconds after its marker was written was not
consumed by the door; this tool writes the same MPAS-JEDI member file and
verification extract the door would have, then deletes the frame
(``--discard``). ``--watch`` keeps going until every member directory under
``--member-root`` has a finished forecast receipt and no frame left.

Origin: the first 30 km run of the frame-mode door (2026-10-12) ran its
consumers as threads of the free-threaded door process; concurrent netCDF use
killed the first member's consumer at its first frame.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from netCDF4 import Dataset

from hexcore.ensemble import (digest, export_jedi_member, extract_pressure_levels,
                              latlon_regrid_weights, write_json)
from hexcore.history_writer import durable_marker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--member-root", type=Path, required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD_HH:MM:SS")
    p.add_argument("--export-hours", type=lambda s: [float(x) for x in s.split(",") if x], default=[])
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--extract-root", type=Path, required=True)
    p.add_argument("--reference-init", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--extract-grid-deg", type=float, default=None)
    p.add_argument("--min-age", type=float, default=180.0)
    p.add_argument("--discard", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d_%H:%M:%S")
    with Dataset(a.reference_init) as ds:
        zg = np.asarray(ds["zgrid"][0] if ds["zgrid"].ndim == 3 else ds["zgrid"][:], np.float64)
    zmid = 0.5 * (zg[:, :-1] + zg[:, 1:])
    regrid = None
    if a.extract_grid_deg:
        with Dataset(a.grid) as g:
            regrid = latlon_regrid_weights(np.asarray(g["latCell"][:]), np.asarray(g["lonCell"][:]), a.extract_grid_deg)
    rows = json.loads(a.receipt.read_text())["frames"] if a.receipt.exists() else []
    while True:
        worked = False
        pending = False
        for member_dir in sorted(d for d in a.member_root.iterdir() if d.is_dir()):
            history = member_dir / "history"
            if not history.is_dir():
                continue
            if not (history / "forecast-receipt.json").exists():
                pending = True
            for frame in sorted(history.glob("cuda-history.*.nc")):
                marker = durable_marker(frame)
                if not marker.is_file() or time.time() - marker.stat().st_mtime < a.min_age:
                    pending = True
                    continue
                member = member_dir.name
                stamp = frame.name[len("cuda-history."):-len(".nc")]
                valid = datetime.strptime(stamp, "%Y-%m-%d_%H.%M.%S")
                hour = (valid - start).total_seconds() / 3600
                row = {"member": member, "forecast_hour": hour, "frame": str(frame),
                       "frame_bytes": frame.stat().st_size, "frame_sha256": digest(frame), "recovered": True}
                target = a.export_dir / f"mpas.{member}.{stamp}.nc"
                if any(abs(hour - h) < 1e-6 for h in a.export_hours) and not target.exists():
                    row["jedi"] = export_jedi_member(frame, a.template, target, valid.strftime("%Y-%m-%d_%H:%M:%S"))
                out = a.extract_root / member / f"{stamp}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    row["extract"] = extract_pressure_levels(frame, zmid, out, regrid)
                if a.discard:
                    frame.unlink()
                    row["frame_deleted"] = True
                rows.append(row)
                write_json(a.receipt, {"frames": rows})
                print(time.strftime("%H:%M:%SZ", time.gmtime()), member, f"f{hour:03.0f}", sorted(set(row) - {"member", "frame"}), flush=True)
                worked = True
        if not a.watch or (not pending and not worked):
            break
        if not worked:
            time.sleep(20)


if __name__ == "__main__":
    main()
