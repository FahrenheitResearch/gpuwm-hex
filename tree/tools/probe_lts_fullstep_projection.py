"""Measure whether FULL-STEP local time stepping could pay on this card.

The decision this serves
------------------------
The shipped ``--local-timestep`` option varies only the acoustic sub-step
count, so its whole-step ceiling is the acoustic share of a model step --
measured at 23.5% on the published x4.163842 mesh, giving the option a 1.057x
ceiling there and a measured 0.988x.  The stronger form -- every dycore kernel
launched per rate class, whole RK steps at ``rate * dt`` -- has an arithmetic
cell-step prize of 1.4x-2.7x on real variable-resolution meshes.  That prize
assumes a kernel launched over a class costs proportionally less than a launch
over the whole mesh.  Below the card's occupancy knee it does not: a launch
over a few thousand cells pays nearly the same latency floor as a launch over
forty thousand.

Without this measurement the program either opens a full-step LTS construction
lane on an arithmetic estimate the launch floor may already have destroyed, or
declines the prize without ever measuring whether the card can collect it.
This instrument produces the go/no-go number: the projected whole-step speedup
composed from MEASURED per-class launch costs of the port's own kernels over
the real mesh's own class index lists.

What is and is not measured
---------------------------
* The kernels are the port's own: the pinned ``acoustic_*_v841`` text for the
  global arm and the landed index-list (``*_lts``) derivations for the
  per-class arm, compiled through the port's own ``KernelCache``.
* The connectivity, class populations and index lists come from a real grid
  file, classed by ``hexcore.lts_v841.classify_from_grid_file`` -- the same
  code the shipped option runs.
* The field VALUES are synthetic (uniform in [0.5, 1.5]); nothing physical is
  claimed.  Launch cost at a given entity count over the real gather pattern
  is the only measurement, and it is the only input the projection needs.
* Interface bookkeeping (time interpolation, reflux) is NOT included, so the
  projection is an upper bound on the full-step form.  The shipped acoustic
  form measured its own bookkeeping at about 7% of a step on x4.163842
  (1.057x ceiling, 0.988x delivered); a full-step build pays a cost of the
  same character.

The instrument validates itself in both directions before reporting: the
composition arithmetic against hand-computed cases before any GPU work, and
the timing loop across the measured set afterwards -- it must have resolved
work-bound scaling somewhere AND seen the small-count launch floor somewhere,
or it refuses to emit the record.  A kernel sitting under its launch floor at
one particular mesh size is reported as a finding, not treated as a timer
fault: that regime is precisely the effect that erodes the arithmetic prize.

Usage (on a CUDA node, from the repo root):

    PYTHONPATH=src python tools/probe_lts_fullstep_projection.py \
        --grid /path/to/a.grid.nc --label mesh-a \
        --grid /path/to/b.grid.nc --label mesh-b \
        --json fullstep-projection.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from typing import Any

import numpy as np

NLEV = 55
WARM = 5
REPS = 30


# --------------------------------------------------------------------------
# Composition arithmetic (host, self-tested before any GPU work)
# --------------------------------------------------------------------------

def macro_step(rates: tuple[int, ...]) -> int:
    return math.lcm(*(int(rate) for rate in rates))


def compose(
    rates: tuple[int, ...],
    class_costs: dict[int, float],
    global_cost: float,
) -> dict[str, float]:
    """Per-macro-step cost of per-class launches vs whole-mesh launches.

    ``class_costs[rate]`` is the measured per-launch cost of the class that
    steps at ``rate * dt``.  Over one macro step of ``lcm(rates)`` fine steps,
    that class launches ``macro // rate`` times; the global arm launches
    ``macro`` times over the whole mesh.
    """

    macro = macro_step(rates)
    lts = sum((macro // int(rate)) * float(class_costs[int(rate)]) for rate in rates)
    total = macro * float(global_cost)
    return {
        "macro": float(macro),
        "global_cost": total,
        "lts_cost": lts,
        "projected_speedup": total / lts if lts > 0.0 else float("inf"),
    }


def arithmetic_cell_step_speedup(
    rates: tuple[int, ...], cells_per_rate: dict[int, int]
) -> float:
    """Launch-free bound: cell-steps under the ladder vs under the global dt."""

    total = sum(cells_per_rate.values())
    macro = macro_step(rates)
    steps = sum((macro // int(rate)) * count for rate, count in cells_per_rate.items())
    return total * macro / steps if steps else 1.0


def self_test() -> None:
    checks: list[tuple[str, bool]] = []
    # Two classes, half the cells at rate 2: 2g vs 2a+b.
    r = compose((1, 2), {1: 1.0, 2: 1.0}, 2.0)
    checks.append(("macro lcm", r["macro"] == 2.0))
    checks.append(("compose 1,2", abs(r["projected_speedup"] - 4.0 / 3.0) < 1e-12))
    # Single class is exactly the global arm when the costs match.
    r = compose((1,), {1: 3.0}, 3.0)
    checks.append(("single class", abs(r["projected_speedup"] - 1.0) < 1e-12))
    # Three classes, hand-computed: macro 4, 4a + 2b + c vs 4g.
    r = compose((1, 2, 4), {1: 0.25, 2: 0.5, 4: 2.0}, 1.0)
    checks.append(("compose 1,2,4", abs(r["lts_cost"] - 4.0) < 1e-12))
    checks.append(("compose 1,2,4 spd", abs(r["projected_speedup"] - 1.0) < 1e-12))
    # Arithmetic bound, hand case: 4 cells, half at rate 4 -> 8/5.
    spd = arithmetic_cell_step_speedup((1, 4), {1: 2, 4: 2})
    checks.append(("arith 1,4", abs(spd - 1.6) < 1e-12))
    # Uniform: no saving, by construction.
    checks.append(("arith uniform", arithmetic_cell_step_speedup((1,), {1: 7}) == 1.0))
    failed = [name for name, passed in checks if not passed]
    if failed:
        raise SystemExit(f"SELF-TEST FAILED: {failed}")
    print(f"self-test {len(checks)}/{len(checks)} PASS")


# --------------------------------------------------------------------------
# GPU arms
# --------------------------------------------------------------------------

class MeshArrays:
    """Device arrays shaped by a real mesh's connectivity, values synthetic."""

    def __init__(self, cp: Any, grid: dict[str, np.ndarray]) -> None:
        rng = cp.random.default_rng(20260824)
        n_cells = int(grid["n_edges_on_cell"].size)
        n_edges = int(grid["dc_edge"].size)
        max_edges = int(grid["edges_on_cell"].shape[1])
        self.n_cells = n_cells
        self.n_edges = n_edges
        self.max_edges = max_edges

        def field(count: int) -> Any:
            return (0.5 + rng.random(count, dtype=cp.float32)).astype(cp.float32)

        big_c = (NLEV + 2) * n_cells
        big_e = (NLEV + 2) * n_edges
        eoc = grid["edges_on_cell"].astype(np.int64) - 1
        eoc = np.maximum(eoc, 0)  # padding slots gather edge 0, never OOB
        coe = grid["cells_on_edge"].astype(np.int64) - 1
        if coe.min() < 0 or coe.max() >= n_cells:
            raise ValueError("cellsOnEdge references a cell outside the mesh")
        sign = np.where(
            (np.arange(n_cells * max_edges) % 2) == 0, 1.0, -1.0
        ).astype(np.float32)
        self.n_edges_on_cell = cp.asarray(
            grid["n_edges_on_cell"].astype(np.int32)
        )
        self.edges_on_cell = cp.asarray(eoc.ravel().astype(np.int32))
        self.cells_on_edge = cp.asarray(coe.ravel().astype(np.int32))
        self.edge_sign_on_cell = cp.asarray(sign)
        self.dv_edge = cp.asarray(grid["dv_edge"].astype(np.float32))
        self.inv_dc_edge = cp.asarray(
            (1.0 / grid["dc_edge"]).astype(np.float32)
        )
        self.inv_area_cell = field(n_cells)
        self.cell = [field(big_c) for _ in range(21)]
        self.edge = [field(big_e) for _ in range(6)]
        self.vert = [field(NLEV + 2) for _ in range(6)]


def kernel_args(arrays: MeshArrays, name: str) -> tuple[Any, ...]:
    """Argument tuple for one pinned kernel, mesh dims from the real grid."""

    i32 = np.int32
    f32 = np.float32
    c = arrays.cell
    e = arrays.edge
    v = arrays.vert
    if name == "acoustic_rs_ts":
        return (
            i32(NLEV), i32(arrays.n_cells), i32(arrays.n_edges),
            i32(arrays.max_edges), f32(1.0),
            arrays.n_edges_on_cell, arrays.edges_on_cell, arrays.cells_on_edge,
            arrays.edge_sign_on_cell, arrays.dv_edge, arrays.inv_area_cell,
            # theta_m, rdzw, cofrz, coftz, ewm
            c[0], v[0], v[1], c[1], v[2],
            # ru_p, rw_p, rho_pp, rtheta_pp, tend_rho, tend_rt, rs, ts
            e[0], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
        )
    if name == "acoustic_ru":
        return (
            i32(NLEV), i32(arrays.n_edges), i32(arrays.n_cells), i32(2),
            f32(1.0), f32(9.80616), f32(287.0), f32(1004.5),
            arrays.cells_on_edge, arrays.inv_dc_edge,
            # zz, exner, cqu, zxu, tend_ru, rho_pp, rtheta_pp, ru_p, ru_avg
            c[0], c[1], e[1], e[2], e[3], c[2], c[3], e[4], e[5],
        )
    if name == "acoustic_column_solve":
        return (
            i32(NLEV), i32(arrays.n_cells), f32(1.0),
            # zz, rho_zz, fzm, fzp, rdzw, dss, w, rw, rw_save, tend_rw, rs, ts
            c[0], c[1], v[0], v[1], v[2], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
            # cofwr, cofwz, coftz, cofwt, cofrz
            c[9], c[10], c[11], c[12], v[3],
            # a_tri, alpha_tri, gamma_tri
            c[13], c[14], c[15],
            # etp, etm, ewp, ewm
            v[4], v[5], v[3], v[3],
            # rw_p, rho_pp, rtheta_pp, ww_avg
            c[16], c[17], c[18], c[19],
        )
    raise ValueError(f"unknown kernel family {name!r}")


KERNELS = (
    ("acoustic_rs_ts", "cell"),
    ("acoustic_ru", "edge"),
    ("acoustic_column_solve", "cell"),
)
_THREADS = 128


def time_launch(cp: Any, kern: Any, count: int, args: tuple[Any, ...]) -> float:
    """Milliseconds per launch over ``count`` threads, event-timed."""

    blocks = (int(count) + _THREADS - 1) // _THREADS
    for _ in range(WARM):
        kern((blocks,), (_THREADS,), args)
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for _ in range(REPS):
        kern((blocks,), (_THREADS,), args)
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop)) / REPS


def load_grid(path: str) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(path, "r") as ds:
        return {
            "dc_edge": np.asarray(ds.variables["dcEdge"][:], dtype=np.float64),
            "dv_edge": np.asarray(ds.variables["dvEdge"][:], dtype=np.float64),
            "edges_on_cell": np.asarray(ds.variables["edgesOnCell"][:]),
            "n_edges_on_cell": np.asarray(ds.variables["nEdgesOnCell"][:]),
            "cells_on_edge": np.asarray(ds.variables["cellsOnEdge"][:]),
        }


def measure_mesh(
    cp: Any,
    cache: Any,
    label: str,
    path: str,
    ladders: list[tuple[int, ...]],
    buffer_rings: int,
) -> dict[str, Any]:
    from hexcore import cuda_acoustic_lts, cuda_acoustic_v841
    from hexcore.lts_v841 import cell_min_spacing, classify_from_grid_file

    grid = load_grid(path)
    arrays = MeshArrays(cp, grid)
    h_cell = cell_min_spacing(
        grid["dc_edge"], grid["edges_on_cell"], grid["n_edges_on_cell"]
    )
    h_min = float(h_cell.min())
    record: dict[str, Any] = {
        "label": label,
        "path": path,
        "n_cells": arrays.n_cells,
        "n_edges": arrays.n_edges,
        "cell_h_max_over_min": float(h_cell.max() / h_min),
        "perfect_continuous_speedup": float(
            arrays.n_cells / np.sum(h_min / h_cell)
        ),
        "buffer_rings": buffer_rings,
    }
    print(f"\n===== {label} =====")
    print(f"  {path}")
    print(
        "  nCells %d  nEdges %d  h max/min %.3f  perfect continuous %.3fx"
        % (
            arrays.n_cells,
            arrays.n_edges,
            record["cell_h_max_over_min"],
            record["perfect_continuous_speedup"],
        )
    )

    def orig(name: str) -> Any:
        return cuda_acoustic_v841._kernel(f"{name}_v841", cache)

    def gathered(name: str) -> Any:
        return cuda_acoustic_lts.kernel(f"{name}_lts", cache)

    def gathered_args(name: str, active: Any, count: int) -> tuple[Any, ...]:
        return (*kernel_args(arrays, name), active, np.int32(count))

    # ---- instrument checks, both directions, on this mesh's own arrays ----
    checks: dict[str, Any] = {}
    for name, entity in KERNELS:
        full = arrays.n_cells if entity == "cell" else arrays.n_edges
        kern = gathered(name)
        cost: dict[int, float] = {}
        for count in (128, 2048, full // 2, full):
            active = cp.arange(int(count), dtype=cp.int32)
            cost[count] = time_launch(
                cp, kern, count, gathered_args(name, active, count)
            )
        work_ratio = cost[full] / cost[full // 2]
        floor_ratio = cost[2048] / cost[128]
        checks[name] = {
            "ms_at_128": cost[128],
            "ms_at_2048": cost[2048],
            "ms_at_half": cost[full // 2],
            "ms_at_full": cost[full],
            "full_over_half": work_ratio,
            "x16_cells_cost_ratio": floor_ratio,
        }
        print(
            "  SCALE %-22s full/half %.2f (2.0 = work-bound, 1.0 = under the "
            "launch floor)  2048/128 %.2f"
            % (name, work_ratio, floor_ratio)
        )
    record["instrument_checks"] = checks

    # ---- the arms ----
    full_cost: dict[str, float] = {}
    gather_full_cost: dict[str, float] = {}
    for name, entity in KERNELS:
        full = arrays.n_cells if entity == "cell" else arrays.n_edges
        full_cost[name] = time_launch(cp, orig(name), full, kernel_args(arrays, name))
        active = cp.arange(full, dtype=cp.int32)
        gather_full_cost[name] = time_launch(
            cp, gathered(name), full, gathered_args(name, active, full)
        )
        print(
            "  FULL  %-22s pinned %.4f ms  index-list-over-arange %.4f ms "
            "(gather overhead %.1f%%)"
            % (
                name,
                full_cost[name],
                gather_full_cost[name],
                100.0 * (gather_full_cost[name] / full_cost[name] - 1.0),
            )
        )
    record["full_mesh_ms"] = dict(full_cost)
    record["full_mesh_index_list_ms"] = dict(gather_full_cost)

    ladder_records: list[dict[str, Any]] = []
    for ladder in ladders:
        classing = classify_from_grid_file(
            path, rates=ladder, buffer_rings=buffer_rings
        )
        summary = classing.summary()
        rates = tuple(int(rate) for rate in classing.rates)
        per_kernel: dict[str, Any] = {}
        trio_global = 0.0
        trio_lts = 0.0
        for name, entity in KERNELS:
            lists = classing.cell_lists if entity == "cell" else classing.edge_lists
            kern = gathered(name)
            class_costs: dict[int, float] = {}
            for rate, host_list in zip(rates, lists):
                count = int(host_list.size)
                if count == 0:
                    class_costs[rate] = 0.0
                    continue
                active = cp.asarray(host_list.astype(np.int32))
                class_costs[rate] = time_launch(
                    cp, kern, count, gathered_args(name, active, count)
                )
            composed = compose(rates, class_costs, full_cost[name])
            composed["class_ms"] = {str(k): v for k, v in class_costs.items()}
            per_kernel[name] = composed
            trio_global += composed["global_cost"]
            trio_lts += composed["lts_cost"]
        arith = arithmetic_cell_step_speedup(
            rates, {int(k): int(v) for k, v in summary["cells_per_rate"].items()}
        )
        trio = trio_global / trio_lts if trio_lts > 0.0 else float("inf")
        ladder_records.append(
            {
                "ladder_requested": list(ladder),
                "rates_present": list(rates),
                "cells_per_rate": summary["cells_per_rate"],
                "edges_per_rate": summary["edges_per_rate"],
                "interface_edges": summary["interface_edges"],
                "arithmetic_cell_step_speedup": arith,
                "per_kernel": per_kernel,
                "trio_projected_speedup": trio,
            }
        )
        pops = " ".join(
            f"{rate}:{summary['cells_per_rate'][rate]}"
            for rate in summary["cells_per_rate"]
        )
        print(
            "  LADDER %-12s cells{%s}  arithmetic %.3fx  PROJECTED %.3fx  (%s)"
            % (
                str(list(ladder)),
                pops,
                arith,
                trio,
                "  ".join(
                    f"{name} {per_kernel[name]['projected_speedup']:.3f}x"
                    for name, _ in KERNELS
                ),
            )
        )
    record["ladders"] = ladder_records

    del arrays
    cp.get_default_memory_pool().free_all_blocks()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="append", required=True, help="MPAS grid file")
    parser.add_argument(
        "--label", action="append", required=True, help="one label per --grid"
    )
    parser.add_argument(
        "--ladder",
        action="append",
        default=None,
        help="comma-separated whole-step rate ladder; repeatable "
        "(default: 1,2 and 1,2,4 and 1,2,4,8)",
    )
    parser.add_argument("--buffer-rings", type=int, default=1)
    parser.add_argument("--json", default=None, help="write the record here")
    args = parser.parse_args(argv)
    if len(args.grid) != len(args.label):
        parser.error("--grid and --label must be paired")
    ladders = [
        tuple(int(piece) for piece in spec.split(","))
        for spec in (args.ladder or ["1,2", "1,2,4", "1,2,4,8"])
    ]

    self_test()

    import cupy as cp

    from hexcore.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(min_compute=(12, 0))
    cache = KernelCache(capability=capability)
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"device: {device}")

    record = {
        "instrument": "probe_lts_fullstep_projection",
        "device": device,
        "sm": capability.sm,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "nlev": NLEV,
        "warm": WARM,
        "reps": REPS,
        "buffer_rings": int(args.buffer_rings),
        "what_this_is": (
            "projected whole-step speedup of FULL-STEP local time stepping, "
            "composed from measured per-class launch costs of the port's own "
            "kernels over the real mesh's class index lists; synthetic field "
            "values; interface bookkeeping excluded, so an upper bound"
        ),
        "meshes": [
            measure_mesh(cp, cache, label, path, ladders, int(args.buffer_rings))
            for path, label in zip(args.grid, args.label)
        ],
    }

    # Instrument gates, both directions, judged across every mesh measured.
    # A kernel sitting under its launch floor at one mesh's size is a finding,
    # not a timer fault, so the refusal keys on the timer's own validity:
    # it must have RESOLVED work-bound scaling somewhere, and it must have
    # SEEN the launch floor somewhere, or the projection is not evidence.
    all_checks = [
        (mesh["label"], name, check)
        for mesh in record["meshes"]
        for name, check in mesh["instrument_checks"].items()
    ]
    if not any(check["full_over_half"] >= 1.5 for _, _, check in all_checks):
        raise SystemExit(
            "INSTRUMENT: no kernel on any mesh scaled >=1.5x for 2x entities; "
            "the timer never resolved a work-bound launch and the projection "
            "would be meaningless"
        )
    if not any(check["x16_cells_cost_ratio"] < 8.0 for _, _, check in all_checks):
        raise SystemExit(
            "INSTRUMENT: 16x the threads always cost >=8x; the launch floor "
            "this probe exists to measure is invisible to the timer"
        )
    print("\nINSTRUMENT both directions PASS across the measured set")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as sink:
            json.dump(record, sink, indent=1, sort_keys=False)
            sink.write("\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
