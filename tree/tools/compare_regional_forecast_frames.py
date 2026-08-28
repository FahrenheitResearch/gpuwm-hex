#!/usr/bin/env python3
"""Two regional forecast frame sets, compared bitwise and by boundary ring.

Written for the NVRTC reciprocal-rewrite A/B: the same regional configuration
run twice, once with the pre-fix CUDA translation units and once with the
fixed ones, so the question "what was the defect doing to a real forecast"
gets a number instead of an argument.

The ring split is the point.  The specified zone (rings 6-7) is written from
the lateral boundary every step, so it should be unmoved by anything that
changes interior arithmetic; the interior is where a flux-kernel change has
to show up.  A result that moved the specified zone would mean the change
reached somewhere it has no business reaching.

Usage:

    python tools/compare_regional_forecast_frames.py \\
        --left <dir> --right <dir> --grid <conus.grid.nc> --out delta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from netCDF4 import Dataset  # noqa: E402

FIELDS = ("u", "w", "theta", "rho", "qv", "pressure")


def read_frame(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name in FIELDS:
            if name not in dataset.variables:
                continue
            slab = np.array(dataset.variables[name][0][:], dtype=np.float32)
            fields[name] = np.ascontiguousarray(slab.T)
    return fields


def masks(grid: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(grid) as dataset:
        dataset.set_auto_maskandscale(False)
        cell = np.asarray(dataset.variables["bdyMaskCell"][:], dtype=np.int64)
        edge = np.asarray(dataset.variables["bdyMaskEdge"][:], dtype=np.int64)
    return cell, edge


def compare(
    name: str, left: np.ndarray, right: np.ndarray, mask: np.ndarray
) -> dict:
    left = np.ascontiguousarray(np.asarray(left, dtype=np.float32))
    right = np.ascontiguousarray(np.asarray(right, dtype=np.float32))
    same = left.view(np.uint32) == right.view(np.uint32)
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    difference[~np.isfinite(difference)] = np.inf
    changed = ~same
    row: dict = {
        "field": name,
        "values": int(left.size),
        "bitwise_equal": bool(same.all()),
        "changed": int(changed.sum()),
        "max_abs_delta": float(difference[changed].max()) if changed.any() else 0.0,
    }
    if changed.any():
        magnitude = np.abs(left.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(magnitude > 0.0, difference / magnitude, 0.0)
        row["max_rel_delta"] = float(relative[changed].max())
    by_ring = {}
    for ring in range(8):
        selected = mask == ring
        if not bool(selected.any()):
            continue
        block = changed[:, selected]
        by_ring[str(ring)] = {
            "elements": int(selected.sum()),
            "values": int(block.size),
            "changed": int(block.sum()),
            "bitwise_equal": bool(not block.any()),
        }
    row["by_ring"] = by_ring
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left_frames = sorted(Path(args.left).glob("history.*.nc"))
    right_frames = sorted(Path(args.right).glob("history.*.nc"))
    if not left_frames or len(left_frames) != len(right_frames):
        raise SystemExit("the two frame sets do not line up")
    bdy_cell, bdy_edge = masks(Path(args.grid))

    report = {
        "instrument": "compare_regional_forecast_frames",
        "schema": "mpas-port.regional-forecast-frame-delta/v1",
        "left": {"label": args.left_label, "dir": str(args.left)},
        "right": {"label": args.right_label, "dir": str(args.right)},
        "frames": [],
    }
    for left_path, right_path in zip(left_frames, right_frames):
        assert left_path.name == right_path.name, (left_path, right_path)
        left = read_frame(left_path)
        right = read_frame(right_path)
        rows = [
            compare(
                name,
                left[name],
                right[name],
                bdy_edge if name == "u" else bdy_cell,
            )
            for name in FIELDS
            if name in left and name in right
        ]
        report["frames"].append({"frame": left_path.name, "fields": rows})
        summary = ", ".join(
            f"{row['field']}={'same' if row['bitwise_equal'] else str(row['changed'])}"
            for row in rows
        )
        print(f"{left_path.name}: {summary}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"delta: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
