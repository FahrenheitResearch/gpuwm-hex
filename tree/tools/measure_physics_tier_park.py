#!/usr/bin/env python3
"""Measure where a step's device high-water mark sits, and what parking costs.

Runs the real forecast tool with a CuPy allocator hook that tracks live device
bytes and attributes the running maximum to the part of the step it happened
in: phase-1 physics, the dynamics half, phase-2 physics, or the commit.  With
``--park-physics-tier`` forwarded through, the same instrument reports what the
park moved.

The payload digest the forecast tool prints is the bit-identity comparator: two
arms that agree byte for byte are the only evidence that a park changed nothing
numerical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


class LiveBytesHook:
    """Track live device bytes and the maximum reached inside each label."""

    def __init__(self) -> None:
        self.label = "startup"
        self.live = 0
        self.peak = 0
        self.peak_label = "startup"
        self.by_label: dict[str, int] = {}
        self.sizes: dict[int, int] = {}

    def malloc_postprocess(self, **kwargs: Any) -> None:
        size = int(kwargs.get("mem_size") or 0)
        pointer = int(kwargs.get("mem_ptr") or 0)
        if pointer:
            self.sizes[pointer] = size
        self.live += size
        if self.live > self.by_label.get(self.label, 0):
            self.by_label[self.label] = self.live
        if self.live > self.peak:
            self.peak = self.live
            self.peak_label = self.label

    def free_postprocess(self, **kwargs: Any) -> None:
        pointer = int(kwargs.get("mem_ptr") or 0)
        size = int(self.sizes.pop(pointer, kwargs.get("mem_size") or 0))
        self.live -= size

    # unused hook surface
    def alloc_preprocess(self, **kwargs: Any) -> None: ...
    def alloc_postprocess(self, **kwargs: Any) -> None: ...
    def malloc_preprocess(self, **kwargs: Any) -> None: ...
    def free_preprocess(self, **kwargs: Any) -> None: ...

    def __enter__(self) -> "LiveBytesHook":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def install_labels(hook: LiveBytesHook) -> None:
    """Label the four parts of a composite step, changing nothing else."""

    from mpas_port.cuda_arwen_physics_v841 import (
        PersistentTwoPhaseCudaPhysicsBackendV841 as Backend,
    )
    from mpas_port.cuda_driver import CudaDryDycoreDriver

    def wrap(owner: Any, name: str, entered: str, left: str) -> None:
        original = getattr(owner, name)

        def wrapped(*a: Any, **k: Any) -> Any:
            hook.label = entered
            try:
                return original(*a, **k)
            finally:
                hook.label = left

        setattr(owner, name, wrapped)

    wrap(Backend, "begin_step", "phase1_physics", "dynamics")
    wrap(CudaDryDycoreDriver, "step_device_with_physics", "dynamics", "dynamics")
    wrap(Backend, "finish_step", "phase2_physics", "commit")
    wrap(Backend, "commit_step", "commit", "between_steps")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
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

    import cupy as cp
    from cupy.cuda import memory_hook as memory_hook_module

    hook = LiveBytesHook()
    install_labels(hook)

    report: dict[str, Any] = {
        "mesh": args.mesh,
        "n_cells": binding.MESH_BINDINGS[args.mesh].n_cells,
        "n_edges": binding.MESH_BINDINGS[args.mesh].n_edges,
        "n_levels": binding.MESH_BINDINGS[args.mesh].n_levels,
        "bind_rebound": bind_receipt.get("rebound"),
        "parked": "--park-physics-tier" in rest,
        "argv": list(rest),
    }

    class _Hook(memory_hook_module.MemoryHook):
        name = "LiveBytesHook"

        def malloc_postprocess(self, **kwargs: Any) -> None:
            hook.malloc_postprocess(**kwargs)

        def free_postprocess(self, **kwargs: Any) -> None:
            hook.free_postprocess(**kwargs)

    started = time.perf_counter()
    with _Hook():
        rc = forecast.main(rest)
    wall = time.perf_counter() - started

    pool = cp.get_default_memory_pool()
    report.update(
        {
            "rc": int(rc),
            "wall_seconds": wall,
            "peak_live_bytes": hook.peak,
            "peak_label": hook.peak_label,
            "peak_by_label_bytes": dict(hook.by_label),
            "pool_total_bytes_at_exit": int(pool.total_bytes()),
            "pool_used_bytes_at_exit": int(pool.used_bytes()),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        "[park-measure] peak %.1f MiB in %s; pool_total %.1f MiB; wall %.1f s"
        % (
            hook.peak / (1024 * 1024),
            hook.peak_label,
            pool.total_bytes() / (1024 * 1024),
            wall,
        )
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
