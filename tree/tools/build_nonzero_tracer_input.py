#!/usr/bin/env python3
"""Clone the frozen JW initial condition and inject only a 3-D qv tracer.

The source NetCDF is treated as immutable.  Every non-qv variable is hashed
before and after the copy so this tool fails closed if anything except qv is
changed.  The tracer is smooth, bounded, strictly positive, horizontally and
vertically non-uniform, and is rounded once to the model's binary32 precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from netCDF4 import Dataset
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variable_inventory(path: Path, *, omit: str | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name in sorted(dataset.variables):
            if name == omit:
                continue
            variable = dataset.variables[name]
            values = np.ascontiguousarray(variable[...])
            digest = hashlib.sha256()
            digest.update(name.encode("utf-8"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes(order="C"))
            result[name] = digest.hexdigest()
    return result


def _tracer(dataset: Dataset, *, amplitude_scale: float) -> np.ndarray:
    qv_variable = dataset.variables["qv"]
    if qv_variable.dimensions != ("Time", "nCells", "nVertLevels"):
        raise ValueError(f"unexpected qv dimensions {qv_variable.dimensions}")
    if np.dtype(qv_variable.dtype) != np.dtype("float32"):
        raise ValueError(f"qv must be binary32, got {qv_variable.dtype}")
    if qv_variable.shape[0] != 1:
        raise ValueError(f"expected one input time, got {qv_variable.shape[0]}")

    original = np.asarray(qv_variable[0], dtype=np.float32)
    if np.count_nonzero(original) != 0:
        raise ValueError("frozen JW source qv is not the expected all-zero control")

    latitude = np.asarray(dataset.variables["latCell"][:], dtype=np.float64)
    longitude = np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)
    zgrid = np.asarray(dataset.variables["zgrid"][:], dtype=np.float64)
    if zgrid.shape != (qv_variable.shape[1], qv_variable.shape[2] + 1):
        raise ValueError(f"unexpected zgrid shape {zgrid.shape}")

    # A global wave plus a vertically localized plume.  The additive floor
    # makes positivity independent of libm rounding at the extrema.
    horizontal = (
        0.50
        + 0.25 * np.sin(longitude) * np.cos(latitude)
        + 0.15 * np.cos(2.0 * longitude) * np.cos(latitude) ** 2
    )
    midpoint = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
    normalized_height = midpoint / zgrid[:, -1, None]
    vertical = 0.25 + 0.75 * np.exp(-(((normalized_height - 0.35) / 0.28) ** 2))
    values = np.asarray(
        amplitude_scale * (2.5e-4 + 7.5e-3 * horizontal[:, None] * vertical),
        dtype=np.float32,
        order="C",
    )
    if not np.all(np.isfinite(values)) or float(values.min()) <= 0.0:
        raise ValueError("constructed tracer is not finite and strictly positive")
    if np.ptp(values, axis=0).min() <= 0.0:
        raise ValueError("constructed tracer lacks horizontal structure")
    if np.ptp(values, axis=1).min() <= 0.0:
        raise ValueError("constructed tracer lacks vertical structure")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if source == output:
        raise ValueError(
            "source and output must differ; the authority input is immutable"
        )
    if not np.isfinite(args.amplitude_scale) or args.amplitude_scale <= 0.0:
        raise ValueError("--amplitude-scale must be finite and positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)

    source_non_qv = _variable_inventory(source, omit="qv")
    source_sha256 = _sha256(source)
    shutil.copyfile(source, output)
    with Dataset(output, "r+") as dataset:
        dataset.set_auto_maskandscale(False)
        values = _tracer(dataset, amplitude_scale=args.amplitude_scale)
        dataset.variables["qv"][0, :, :] = values
        dataset.sync()

    output_non_qv = _variable_inventory(output, omit="qv")
    if output_non_qv != source_non_qv:
        changed = sorted(
            name
            for name in set(source_non_qv) | set(output_non_qv)
            if source_non_qv.get(name) != output_non_qv.get(name)
        )
        raise RuntimeError(f"non-qv payload mutation detected: {changed}")

    with Dataset(output) as dataset:
        dataset.set_auto_maskandscale(False)
        written = np.asarray(dataset.variables["qv"][0], dtype=np.float32)
    if not np.array_equal(written, values):
        raise RuntimeError("qv readback is not bitwise equal to the requested tracer")

    qv_bytes = np.ascontiguousarray(written, dtype="<f4").tobytes(order="C")
    record = {
        "schema": "mpas-port.nonzero-tracer-input.v1",
        "operation": "copy immutable source; replace qv payload only",
        "amplitude_scale": args.amplitude_scale,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "mutation_control": {
            "qv_only": True,
            "non_qv_variable_count": len(source_non_qv),
            "all_non_qv_payload_hashes_equal": True,
        },
        "qv": {
            "dtype": "<f4",
            "shape": list(written.shape),
            "formula": (
                f"{args.amplitude_scale:.17g} * (2.5e-4 + 7.5e-3 * "
                "(0.50 + 0.25*sin(lonCell)*cos(latCell) + "
                "0.15*cos(2*lonCell)*cos(latCell)^2) * "
                "(0.25 + 0.75*exp(-((zmid/ztop-0.35)/0.28)^2)))"
            ),
            "sha256_logical_c_order": hashlib.sha256(qv_bytes).hexdigest(),
            "min": float(written.min()),
            "max": float(written.max()),
            "mean_float64": float(written.mean(dtype=np.float64)),
            "positive_count": int(np.count_nonzero(written > 0.0)),
            "count": int(written.size),
            "horizontal_ptp_min_across_levels": float(np.ptp(written, axis=0).min()),
            "vertical_ptp_min_across_cells": float(np.ptp(written, axis=1).min()),
        },
    }
    receipt.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
