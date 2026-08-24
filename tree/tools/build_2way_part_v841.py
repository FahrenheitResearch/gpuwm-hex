#!/usr/bin/env python
"""Merge the 12-way METIS part file into the exact-optimal 2-way split.

Design source: MPAS-MULTIGPUDESIGN probe 2 -- all C(12,4)=495 four-subsets of
the mkpart 12-way decomposition were enumerated; the minimum-cut split with
balance within 6% of ideal is small side = parts {0, 4, 8, 9}.  A union of
METIS parts is a valid MPAS block decomposition (mkpart's own doc), so the
merge is a pure relabel: big side -> partition 0 (rank 0, the 5090), small
side -> partition 1 (rank 1, the 5070 Ti).

    python tools/build_2way_part_v841.py \
        --part12 .../x4.163842.graph.info.part.12 \
        --out    .../x4.163842.graph.info.part.2

Then feed the written part file to tools/build_partition_assets_v841.py with
--halo-rings 2 (the deep-halo K=60 default is dead at P=2 by measurement).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_SMALL_PARTS = (0, 4, 8, 9)
EXPECTED_12WAY_PARTS = 12


def merge_to_two_way(
    part12_path: Path,
    out_path: Path,
    *,
    small_parts: Sequence[int] = DEFAULT_SMALL_PARTS,
) -> dict[str, Any]:
    part = np.loadtxt(part12_path, dtype=np.int64, ndmin=1)
    if part.ndim != 1 or part.size == 0:
        raise ValueError(f"{part12_path}: expected one partition id per line")
    if part.min() < 0:
        raise ValueError(f"{part12_path}: negative partition id")
    small = np.isin(part, np.asarray(sorted(set(int(p) for p in small_parts))))
    merged = np.where(small, 1, 0).astype(np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(int(v)) for v in merged) + "\n")
    receipt = {
        "schema": "mpas-port.two-way-part-merge/v1",
        "source": str(part12_path),
        "source_sha256": hashlib.sha256(part12_path.read_bytes()).hexdigest(),
        "source_partitions": int(part.max()) + 1,
        "small_parts": sorted(int(p) for p in set(small_parts)),
        "out": str(out_path),
        "out_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "n_cells": int(part.size),
        "big_side_cells": int(np.count_nonzero(~small)),
        "small_side_cells": int(np.count_nonzero(small)),
    }
    receipt["big_to_small_ratio"] = (
        receipt["big_side_cells"] / receipt["small_side_cells"]
        if receipt["small_side_cells"]
        else None
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part12", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--small-parts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SMALL_PARTS),
        help="12-way part ids merged into the small side (default: the "
        "probe's exact-optimal {0,4,8,9})",
    )
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--expect-parts", type=int, default=EXPECTED_12WAY_PARTS,
        help="refuse unless the source file carries exactly this many parts",
    )
    args = parser.parse_args(argv)

    receipt = merge_to_two_way(args.part12, args.out, small_parts=args.small_parts)
    if receipt["source_partitions"] != args.expect_parts:
        raise SystemExit(
            f"{args.part12} carries {receipt['source_partitions']} parts, "
            f"expected {args.expect_parts}"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
