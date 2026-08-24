#!/usr/bin/env python3
"""Compare stock tracer-amplitude runs bitwise with the frozen dry control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np


CONTROL_FIELDS = (
    "rho_zz",
    "theta_m",
    "ru",
    "rw",
    "u",
    "w",
    "rho_base",
    "theta_base",
    "rtheta_base",
    "rho_p",
    "rtheta_p",
    "exner",
    "exner_base",
    "pressure_p",
    "pressure_base",
)


def _read(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name in CONTROL_FIELDS + ("qv",):
            variable = dataset.variables[name]
            values = np.asarray(variable[...])
            if variable.dimensions[0] == "Time":
                values = values[0]
            result[name] = np.ascontiguousarray(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    dry = {time: _read(directory / f"dry-{time}.nc") for time in ("t0", "t1")}
    report: dict[str, object] = {}
    for tag in args.tags:
        candidate = {
            time: _read(directory / f"{tag}-{time}.nc") for time in ("t0", "t1")
        }
        comparisons: dict[str, object] = {}
        all_equal = True
        for time in ("t0", "t1"):
            comparisons[time] = {}
            for field in CONTROL_FIELDS:
                left = dry[time][field]
                right = candidate[time][field]
                if left.shape != right.shape or left.dtype != right.dtype:
                    raise ValueError(f"incompatible {tag}/{time}/{field}")
                difference = right.astype(np.float64) - left.astype(np.float64)
                equal = bool(np.array_equal(left, right))
                all_equal &= equal
                comparisons[time][field] = {
                    "bitwise_equal": equal,
                    "changed_element_count": int(np.count_nonzero(left != right)),
                    "max_abs": float(np.max(np.abs(difference), initial=0.0)),
                }
        qv0 = candidate["t0"]["qv"]
        qv1 = candidate["t1"]["qv"]
        qv_difference = qv1.astype(np.float64) - qv0.astype(np.float64)
        max_initial = float(np.max(np.abs(qv0)))
        report[tag] = {
            "all_non_qv_control_fields_bitwise_equal": all_equal,
            "control_fields": comparisons,
            "tracer": {
                "t0_min": float(qv0.min()),
                "t0_max": float(qv0.max()),
                "t1_min": float(qv1.min()),
                "t1_max": float(qv1.max()),
                "changed_element_count": int(np.count_nonzero(qv0 != qv1)),
                "max_abs_change": float(np.max(np.abs(qv_difference), initial=0.0)),
                "max_change_relative_to_initial_max": (
                    float(np.max(np.abs(qv_difference), initial=0.0)) / max_initial
                ),
                "mean_abs_change": float(np.mean(np.abs(qv_difference))),
                "horizontal_ptp_min_across_levels": float(np.ptp(qv0, axis=0).min()),
                "vertical_ptp_min_across_cells": float(np.ptp(qv0, axis=1).min()),
            },
        }
        if args.summary:
            mismatches = {
                f"{time}/{field}": details
                for time, fields in comparisons.items()
                for field, details in fields.items()
                if not details["bitwise_equal"]
            }
            report[tag] = {
                "all_non_qv_control_fields_bitwise_equal": all_equal,
                "control_mismatches": mismatches,
                "tracer": report[tag]["tracer"],
            }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
