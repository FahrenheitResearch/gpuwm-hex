#!/usr/bin/env python3
"""Observe real verify-mpas branch arcs; never substitute the CLI or comparator.

Set MPAS_VERIFY_PYTHON to this executable and MPAS_VERIFY_TRACE to an owned
JSON path when running rw-mpas's verify_mpas_cli integration test. Records
observed line transitions and branch anchors, without claiming full coverage.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

source = Path(__file__).resolve().parents[1] / "src/hexcore/verify_mpas.py"
destination = Path(os.environ["MPAS_VERIFY_TRACE"])
arcs = set()
previous = {}


def observe(frame, event, arg):
    if Path(frame.f_code.co_filename) != source:
        return None
    key = id(frame)
    if event == "call":
        previous[key] = -frame.f_code.co_firstlineno
    elif event in ("line", "return"):
        next_line = frame.f_lineno if event == "line" else -frame.f_code.co_firstlineno
        if key in previous:
            arcs.add((previous[key], next_line))
        if event == "return":
            previous.pop(key, None)
        else:
            previous[key] = next_line
    return observe


if sys.argv[1:3] != ["-m", "hexcore.cli"]:
    raise SystemExit("coverage wrapper requires the real hexcore.cli module")
sys.argv = sys.argv[2:]
sys.settrace(observe)
try:
    runpy.run_module("hexcore.cli", run_name="__main__", alter_sys=True)
finally:
    sys.settrace(None)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["source_sha256"] != sha:
            raise RuntimeError("cannot combine branch observations from different source bytes")
        arcs.update(tuple(pair) for pair in old["observed_arcs"])
    anchors = []
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.IfExp)):
            anchors.append({"line": node.lineno, "kind": type(node).__name__,
                            "observed_successors": sorted(b for a, b in arcs if a == node.lineno)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema": "mpas-observed-branch-arcs-v1",
        "source": str(source), "source_sha256": sha, "observed_arcs": sorted(arcs),
        "branch_anchors": sorted(anchors, key=lambda row: row["line"]),
        "scope": "real CLI synthetic tapes; gate/GPU branches are unmeasured"}, indent=2) + "\n")
