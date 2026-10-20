"""Record every untaken gcov branch with its pinned Fortran source location."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def collect(path: Path):
    line_number = 0
    source = ""
    routine = "module scope"
    branches = []
    routines = {}
    for line in path.read_text().splitlines():
        entry = re.match(r"\s*([^:]+):\s*(\d+):(.*)", line)
        if entry:
            line_number, source = int(entry[2]), entry[3].strip()
            begin = re.match(r"(?:subroutine|function)\s+(\w+)", source, re.I)
            if begin:
                routine = begin[1].lower()
            continue
        branch = re.match(r"branch\s+(\d+)\s+(.*)", line)
        if branch:
            untouched = "never executed" in branch[2] or bool(re.match(r"taken 0(?:%|\s|$)", branch[2]))
            row = routines.setdefault(routine, {"branches": 0, "untaken": 0})
            row["branches"] += 1
            row["untaken"] += untouched
            if untouched:
                branches.append({"routine": routine, "line": line_number,
                                 "source": source, "branch": int(branch[1]),
                                 "gcov": branch[2]})
    return {"gcov_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "routines": routines, "untaken_branches": branches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "native standalone synthetic decks; no CUDA or forecast coverage",
              "files": {p.name.removesuffix(".gcov"): collect(p)
                        for p in sorted(args.receipts.glob("*.gcov"))}}
    (args.receipts / "native-coverage.json").write_text(json.dumps(report, indent=2) + "\n")
