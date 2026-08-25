#!/usr/bin/env python3
"""Enumerate the frozen-Arwen physics seam's resident device arrays.

Read-only instrument.  It runs the real forecast tool, and at each committed
step boundary walks the physics backend's object graph and reports every CuPy
array reachable from it, keyed by the device allocation that owns the bytes so
views are never double counted.

The question it answers: how much of the measured per-cell physics residency is
reachable-and-reboundable from the adapter, given that the Arwen seam source is
frozen and hash-verified and cannot be edited.

No behaviour is changed; nothing is freed; nothing is transferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


_SKIP_TYPES = (str, bytes, bytearray, int, float, complex, bool, type(None))


def walk_device_arrays(cp: Any, root: Any, *, max_depth: int = 12) -> dict[str, Any]:
    """Collect every CuPy array reachable from ``root``.

    Returns owners keyed by the base allocation pointer, plus the holder paths
    that reference each one.  A reshape/slice view is recorded against the
    allocation it borrows from, never as separate bytes.
    """

    seen_objects: set[int] = set()
    owners: dict[int, dict[str, Any]] = {}
    paths: list[tuple[str, int, int, bool]] = []

    def record(path: str, arr: Any) -> None:
        base = arr.base if arr.base is not None else arr
        key = int(base.data.ptr)
        entry = owners.get(key)
        if entry is None:
            owners[key] = {
                "ptr": key,
                "owner_nbytes": int(base.nbytes),
                "owner_shape": list(base.shape),
                "owner_dtype": str(base.dtype),
                "holders": [],
            }
            entry = owners[key]
        entry["holders"].append(path)
        paths.append((path, int(arr.nbytes), key, arr.base is not None))

    def visit(path: str, obj: Any, depth: int) -> None:
        if depth > max_depth or isinstance(obj, _SKIP_TYPES):
            return
        ident = id(obj)
        if ident in seen_objects:
            return
        seen_objects.add(ident)
        if isinstance(obj, cp.ndarray):
            record(path, obj)
            return
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                visit(f"{path}[{key!r}]", value, depth + 1)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for index, value in enumerate(list(obj)):
                visit(f"{path}[{index}]", value, depth + 1)
            return
        if isinstance(obj, (type, ModuleType, FunctionType, MethodType, BuiltinFunctionType)):
            return
        slots: list[str] = []
        for klass in type(obj).__mro__:
            slots.extend(getattr(klass, "__slots__", ()) or ())
        state = getattr(obj, "__dict__", None)
        if state is None and not slots:
            return
        if state is not None:
            for key, value in list(state.items()):
                visit(f"{path}.{key}", value, depth + 1)
        for key in slots:
            visit(f"{path}.{key}", getattr(obj, key, None), depth + 1)

    visit("seam", root, 0)
    total = sum(entry["owner_nbytes"] for entry in owners.values())
    return {
        "owner_count": len(owners),
        "owner_bytes": total,
        "reference_count": len(paths),
        "owners": sorted(owners.values(), key=lambda e: -e["owner_nbytes"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps-to-sample", type=int, default=2)
    parser.add_argument("--mesh", required=True)
    args, rest = parser.parse_known_args(argv)

    import mpas_mesh_binding as binding
    import run_cuda_v841_forecast as forecast

    proof = forecast.proof
    grid = rest[rest.index("--grid") + 1]
    static = rest[rest.index("--static") + 1]
    bind_receipt = binding.bind_mesh(
        proof, args.mesh, grid=Path(grid), static=Path(static), forecast=forecast
    )

    report: dict[str, Any] = {
        "mesh": args.mesh,
        "n_cells": binding.MESH_BINDINGS[args.mesh].n_cells,
        "n_edges": binding.MESH_BINDINGS[args.mesh].n_edges,
        "n_levels": binding.MESH_BINDINGS[args.mesh].n_levels,
        "bind_rebound": bind_receipt.get("rebound"),
        "samples": {},
    }
    holder: dict[str, Any] = {}

    original_construct = proof._construct_device_stack
    original_step = proof.execute_composite_step
    counter = {"n": 0}

    def wrapped_construct(*a: Any, **k: Any) -> Any:
        stack = original_construct(*a, **k)
        holder["stack"] = stack
        return stack

    def wrapped_step(*a: Any, **k: Any) -> Any:
        out = original_step(*a, **k)
        counter["n"] += 1
        if counter["n"] <= args.steps_to_sample:
            import cupy as cp

            stack = holder.get("stack")
            backend = stack["backend"]
            pool = cp.get_default_memory_pool()
            seam_walk = walk_device_arrays(cp, backend._seam)
            adapter_walk = walk_device_arrays(cp, backend)
            report["samples"][f"after_step_{counter['n']}"] = {
                "pool_used_bytes": int(pool.used_bytes()),
                "pool_total_bytes": int(pool.total_bytes()),
                "seam_reachable": seam_walk,
                "adapter_reachable_owner_bytes": adapter_walk["owner_bytes"],
                "adapter_reachable_owner_count": adapter_walk["owner_count"],
                "phase": backend._phase,
            }
        return out

    proof._construct_device_stack = wrapped_construct
    proof.execute_composite_step = wrapped_step
    try:
        rc = forecast.main(rest)
    finally:
        proof._construct_device_stack = original_construct
        proof.execute_composite_step = original_step
        report["rc"] = locals().get("rc")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
