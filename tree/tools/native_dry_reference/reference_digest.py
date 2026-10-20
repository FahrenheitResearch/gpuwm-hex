#!/usr/bin/env python3
"""Digest the native reference outputs, or check them against the pinned digests.

A native output file cannot be pinned by its own SHA-256: MPAS writes a random
``file_id`` attribute into every file.  The digest here covers the data only:
for each ``tracer.<time>.nc`` in time order, each floating-point variable in
name order, its name, shape and raw bytes.

usage: reference_digest.py JW_ROOT                 print the digests as JSON
       reference_digest.py JW_ROOT --check PINS    exit 1 unless every pinned arm matches
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def arm_digest(directory: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = sorted(directory.glob("tracer.*.nc"))
    for path in files:
        with Dataset(path) as dataset:
            for name in sorted(dataset.variables):
                variable = dataset.variables[name]
                if variable.dtype.kind != "f":
                    continue
                data = np.ascontiguousarray(np.asarray(variable[:]))
                digest.update(f"{path.name}:{name}:{data.shape}:{data.dtype.str}".encode())
                digest.update(data.tobytes())
    return {"outputs": len(files), "data_sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", type=Path, default=None)
    args = parser.parse_args()
    found = {f"{build.name}/{arm.name}": arm_digest(arm)
             for build in sorted(args.root.iterdir()) if build.is_dir() and build.name != "init"
             for arm in sorted(build.iterdir()) if arm.is_dir()}
    if args.check is None:
        print(json.dumps(found, indent=1))
        return 0
    pins = json.loads(args.check.read_text(encoding="utf-8"))["arms"]
    wrong = [name for name, pin in pins.items()
             if name in found and found[name]["data_sha256"] != pin["data_sha256"]]
    missing = [name for name in pins if name not in found]
    print(json.dumps({"matching": len(pins) - len(wrong) - len(missing),
                      "different": wrong, "not_built": missing}, indent=1))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
