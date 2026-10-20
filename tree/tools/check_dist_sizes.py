#!/usr/bin/env python3
"""Check distribution sizes against the 100,000,000 byte upload cap."""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import zipfile

LIMIT = 100_000_000
WARNING = 95_000_000


def largest_members(path: Path) -> list[tuple[int, str]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = [(row.file_size, row.filename) for row in archive.infolist() if not row.is_dir()]
    else:
        with tarfile.open(path, "r:*") as archive:
            members = [(row.size, row.name) for row in archive if row.isfile()]
    return sorted(members, key=lambda row: (-row[0], row[1]))[:10]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="Built wheels and sdists")
    args = parser.parse_args(argv)
    failed = False
    for path in args.files:
        try:
            size = path.stat().st_size
            status = "FAIL" if size >= LIMIT else "WARN" if size >= WARNING else "PASS"
            print(f"{status} {path}: {size} B; margin {LIMIT - size} B (cap {LIMIT} B)")
            if size >= WARNING:
                print(f"WARNING: {path} is at or above {WARNING} B")
            failed |= size >= LIMIT
            print("  10 largest members (uncompressed bytes):")
            for member_size, name in largest_members(path):
                print(f"  {member_size:12d}  {name}")
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
            print(f"FAIL {path}: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
