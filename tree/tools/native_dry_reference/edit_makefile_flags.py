#!/usr/bin/env python3
"""Replace one substring of FFLAGS_OPT inside one build target of the MPAS Makefile.

usage: edit_makefile_flags.py MAKEFILE TARGET OLD NEW

The edit is refused unless OLD occurs in that target's FFLAGS_OPT line, so a
changed upstream Makefile cannot be built with flags nobody asked for.
"""
import sys


def main() -> int:
    path, target, old, new = sys.argv[1:5]
    text = open(path, encoding="utf-8").read()
    marker = f"\n{target}:"
    if text.count(marker) != 1:
        raise SystemExit(f"target {target!r} is not declared exactly once")
    head, tail = text.split(marker, 1)
    lines = tail.split("\n")
    for index, line in enumerate(lines):
        if '"FFLAGS_OPT = ' in line:
            if old not in line:
                raise SystemExit(f"{old!r} is not in {line.strip()!r}")
            lines[index] = line.replace(old, new, 1)
            break
        if index > 0 and line and not line.startswith(("\t", " ")):
            raise SystemExit(f"target {target!r} sets no FFLAGS_OPT")
    else:
        raise SystemExit(f"target {target!r} sets no FFLAGS_OPT")
    open(path, "w", encoding="utf-8").write(head + marker + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
