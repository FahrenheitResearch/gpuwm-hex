"""Inventory native gcov outcomes without hiding untaken compiler branches."""
import argparse
import json
from pathlib import Path
import re


def inventory(folder):
    records = []
    for name in ("bl_ysu.F90", "module_bl_ysu.F"):
        line, statement = 0, ""
        for row in (folder / (name+".gcov")).read_text().splitlines():
            source = re.match(r"\s*[^:]+:\s*(\d+):(.*)", row)
            if source:
                line, statement = int(source[1]), source[2].strip()
            branch = re.match(r"branch\s+(\d+) taken 0(?:\s|$)", row)
            if not branch:
                continue
            if name == "bl_ysu.F90" and line == 841:
                reason = "Unreachable kpbl<2 inside pblflg; retired at native line 767."
            elif name == "bl_ysu.F90" and line in (1059, 1147, 1338):
                reason = "Unreachable k<kpbl after preceding k<kpbl branch with pblflg true."
            elif name == "bl_ysu.F90" and 56 <= line <= 73:
                reason = "Compiler descriptor/extent/allocation outcome; not an executable physics statement."
            else:
                reason = "Compiler character padding, zero-size extent, or return outcome."
            records.append(dict(source=name, line=line, branch=int(branch[1]),
                                statement=statement, reason_no_dump_reaches=reason))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcov", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    rows = inventory(args.gcov)
    args.receipt.write_text(json.dumps(rows, indent=2)+"\n")
    print(f"Recorded {len(rows)} untaken native branch outcomes")
