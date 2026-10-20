"""Compare complete MYNN Fortran routines before permitting kernel reuse.

Comment/whitespace/case normalization is reported separately from byte identity.
This tool does not rewrite physics or silently normalize kind/constant changes.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re


def routines(path: Path):
    found = {}
    name = None
    start = None
    lines = path.read_text().splitlines(keepends=True)
    for lineno, line in enumerate(lines, 1):
        begin = re.match(r"\s*(subroutine|function)\s+(\w+)", line, re.I)
        end = re.match(r"\s*end\s*(subroutine|function)\b", line, re.I)
        if begin:
            assert name is None, (path, name, lineno)
            name, start = begin[2].lower(), lineno
        elif end and name is not None:
            found[name] = (start, lineno, "".join(lines[start - 1:lineno]))
            name = None
    assert name is None, (path, name)
    return found


def canonical(source: str):
    out = []
    for line in source.splitlines():
        # Strip comments while preserving quoted character values.
        quote = None
        for index, char in enumerate(line):
            if char in "'\"":
                quote = None if quote == char else char if quote is None else quote
            if char == "!" and quote is None:
                line = line[:index]
                break
        if line.strip():
            out.append(re.sub(r"\s+", "", line).lower())
    return "\n".join(out)


def compare(mpas: list[Path], wrf: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    wrf_routines = routines(wrf)
    report = {"wrf_path": str(wrf), "wrf_sha256": hashlib.sha256(wrf.read_bytes()).hexdigest(),
              "normalization": "comments, whitespace and case only; types and constants preserved",
              "routines": [], "reused_kernels": []}
    diffs = []
    for src in mpas:
        for name, (start, end, text) in routines(src).items():
            row = {"name": name, "mpas_file": src.name, "mpas_lines": [start, end]}
            if name not in wrf_routines:
                row["comparison"] = "absent from comparison WRF module"
            else:
                wstart, wend, wtext = wrf_routines[name]
                row["wrf_lines"] = [wstart, wend]
                row["byte_identical"] = text == wtext
                row["normalized_identical"] = canonical(text) == canonical(wtext)
                row["comparison"] = "identical" if text == wtext else "differs"
                diffs.extend(difflib.unified_diff(wtext.splitlines(True), text.splitlines(True),
                    fromfile=f"WRF/{wrf.name}:{wstart}-{wend}:{name}",
                    tofile=f"MPAS/{src.name}:{start}-{end}:{name}"))
            report["routines"].append(row)
    (out / "mynn-source-diff.patch").write_text("".join(diffs))
    report["diff_sha256"] = hashlib.sha256((out / "mynn-source-diff.patch").read_bytes()).hexdigest()
    (out / "mynn-source-diff.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpas", type=Path, action="append", required=True)
    parser.add_argument("--wrf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.mpas, args.wrf, args.out)
