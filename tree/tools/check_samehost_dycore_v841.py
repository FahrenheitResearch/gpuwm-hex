#!/usr/bin/env python3
"""Compare every dry-dycore boundary between one card and resident N ranks.

This instrument calls the existing native-init loaders and the existing JW
v8.4.1 dry configuration. It moves bytes for partitioning and reconstruction;
it adds no model arithmetic. Every prognostic and saved-diagnostic array is
hashed in full at the initial boundary and after every complete model step.

The receipt proves implementation identity on the supplied initialized mesh.
It does not qualify full column physics, regional boundaries, native-MPAS
identity, or scaling across distinct cards. Repeated device ordinals exercise
concurrent ranks on one card. External Nsight capture can profile this door.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.cuda_dualrun import (  # noqa: E402
    PreparedCudaInputs,
    SIDECAR_FIELDS,
    STATE_FIELDS,
    fingerprint_atmosphere,
    sha256_file,
)
from hexcore.driver import load_mpas_initial_state, load_mpas_vertical_grid  # noqa: E402
from hexcore.dynamics_v841 import load_v841_reference_wind_profiles  # noqa: E402
from hexcore.integration import RKSchedule  # noqa: E402
from hexcore.mesh import load_precision_preserving_mesh_pair  # noqa: E402
from hexcore.config_v841 import V841DryDycoreConfig  # noqa: E402
from run_cuda_v841_jw_dualrun import _config  # noqa: E402


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _ensure_deformation(mesh: Any) -> None:
    """Retain initialized geometry or use the existing inactive declaration."""
    present = {name for name in ("defc_a", "defc_b") if name in mesh.arrays}
    if present and present != {"defc_a", "defc_b"}:
        raise ValueError("initialized deformation geometry requires both defc_a and defc_b")
    if not present:
        from run_cuda_v841_full_physics_x4 import attach_inactive_zero_deformation
        attach_inactive_zero_deformation(mesh)


def prepare(grid: Path, init: Path, config: Any) -> PreparedCudaInputs:
    # Initialized transport rows live in the native init, not the bare grid.
    # Use the same precision-preserving merge as the ordinary forecast door.
    mesh, _output_mesh, _mesh_receipt = load_precision_preserving_mesh_pair(grid, init)
    _ensure_deformation(mesh)
    declarations = dict(getattr(mesh, "variable_dimensions", {}) or {})
    for name in ("defc_a", "defc_b"):
        declarations[name] = ("nCells", "maxEdges")
    mesh.variable_dimensions = declarations
    native = load_mpas_vertical_grid(
        init, mesh, config_coef_3rd_order=config.config_coef_3rd_order,
    )
    state, reference, saved = load_mpas_initial_state(
        init, mesh, native.vertical_grid, scalar_names=("qv",),
        reference_path=init, terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    profiles = load_v841_reference_wind_profiles(
        init, n_vert_levels=native.vertical_grid.n_vert_levels,
    )
    return PreparedCudaInputs.validated(
        config=config, profile="samehost-v841-dry-native-init",
        target="full-boundary N-rank implementation identity",
        preparation_method="existing native-init loaders with no field edits",
        mesh=mesh, state=state, vertical=native.vertical_grid,
        reference=reference, saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        reference_wind_profiles=profiles,
        input_bytes={"grid": _file_record(grid), "init": _file_record(init)},
    )


def _write_partition_inputs(mesh: Any, nranks: int, out: Path,
                            part_source: Path | None = None) -> tuple[Path, Path]:
    """Emit topology and copy supplied labels or balanced contiguous labels."""
    counts = np.asarray(mesh.nEdgesOnCell)
    neighbors = np.asarray(mesh.cellsOnCell)
    ncells = int(mesh.dimensions["nCells"])
    rows = [neighbors[cell, :int(counts[cell])] for cell in range(ncells)]
    if any(np.any(row < 0) for row in rows):
        raise ValueError("this dry identity instrument requires a closed global mesh")
    graph = out / "graph.info"
    part = out / f"graph.info.part.{nranks}"
    with graph.open("x", encoding="ascii") as stream:
        stream.write(f"{ncells} {sum(len(row) for row in rows) // 2}\n")
        for row in rows:
            stream.write(" ".join(str(int(index) + 1) for index in row) + "\n")
    if part_source is not None:
        shutil.copyfile(part_source.resolve(strict=True), part)
        return graph, part
    labels = np.empty(ncells, dtype=np.int32)
    for rank, indexes in enumerate(np.array_split(np.arange(ncells), nranks)):
        labels[indexes] = rank
    with part.open("x", encoding="ascii") as stream:
        stream.write("\n".join(str(int(label)) for label in labels) + "\n")
    return graph, part


def _driver(prepared: Any, config: Any, cache: Any) -> Any:
    from hexcore.cuda_driver import CudaDryDycoreDriver

    driver = CudaDryDycoreDriver.from_host(
        prepared.mesh, prepared.state.copy(), prepared.vertical,
        prepared.reference, config, saved_diagnostics=prepared.saved_diagnostics,
        terrain_metrics=prepared.terrain_metrics,
        advection_coefficients=prepared.advection_coefficients,
        reference_wind_profiles=prepared.reference_wind_profiles,
        kernel_cache=cache,
    )
    _assert_device(driver, int(cache.capability.device_id))
    return driver


def _assert_device(driver: Any, expected_device: int) -> None:
    current = int(driver.cp.cuda.runtime.getDevice())
    if current != expected_device:
        raise RuntimeError(f"driver changed its current card from {expected_device} to {current}")
    for group, names in ((driver.atmosphere.state, STATE_FIELDS),
                         (driver.atmosphere.saved, SIDECAR_FIELDS)):
        for name in names:
            actual = int(getattr(group, name).device.id)
            if actual != expected_device:
                raise RuntimeError(f"driver array {name} belongs to card {actual}, expected {expected_device}")


def step_and_publish(driver: Any) -> Any:
    """Chain the public dry-step result before clock checks or halo publication."""
    result = driver.step_device()
    driver.atmosphere = result.atmosphere
    return result


def _download(atmosphere: Any, cp: Any) -> dict[str, Any]:
    return {
        "time_seconds": float(atmosphere.state.time_seconds),
        "state": {name: cp.asnumpy(getattr(atmosphere.state, name))
                  for name in STATE_FIELDS},
        "saved": {name: cp.asnumpy(getattr(atmosphere.saved, name))
                  for name in SIDECAR_FIELDS},
    }


def _reconstruct(blocks: list[dict[str, Any]], layouts: list[Any]) -> Any:
    from hexcore.partition_executor_v841 import owned_block, scatter_owned_axis

    clocks = [item["time_seconds"] for item in blocks]
    if len(set(clocks)) != 1:
        raise RuntimeError(f"rank clocks differ at the hash boundary: {clocks}")
    state = SimpleNamespace(time_seconds=clocks[0])
    saved = SimpleNamespace()
    for group, names, target in (("state", STATE_FIELDS, state),
                                 ("saved", SIDECAR_FIELDS, saved)):
        for name in names:
            edge = name in ("rho_u", "normal_velocity")
            plural = "edges" if edge else "cells"
            kind = "edge" if edge else "cell"
            shape = list(blocks[0][group][name].shape)
            shape[-1] = int(getattr(layouts[0], f"_global_{plural}"))
            union = np.empty(shape, dtype=blocks[0][group][name].dtype)
            for block, layout in zip(blocks, layouts):
                n_owned = int(getattr(layout, f"n_owned_{plural}"))
                scatter_owned_axis(
                    union, owned_block(block[group][name], union.ndim - 1, n_owned),
                    getattr(layout, f"{kind}_l2g"), n_owned, union.ndim - 1,
                )
            setattr(target, name, union)
    return SimpleNamespace(state=state, saved=saved)


def _differences(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    differences = []
    if reference["model_time_seconds"] != candidate["model_time_seconds"]:
        differences.append("model_time_seconds")
    for group in ("state", "saved_diagnostics"):
        for name, expected in reference[group]["fields"].items():
            if expected != candidate[group]["fields"].get(name):
                differences.append(f"{group}/{name}")
    return differences


def run(args: argparse.Namespace) -> int:
    from hexcore.cuda_backend import KernelCache, require_cuda
    from hexcore.partition_assets_v841 import build_partition_layouts
    from hexcore.partition_executor_v841 import expected_round_sequence
    from hexcore.partition_local_mesh_v841 import slice_prepared_host
    from hexcore.partition_samehost_tables_v841 import build_all
    from hexcore.partition_samehost_transport_v841 import SameHostTransport
    from hexcore.partition_samehost_v841 import SameHostHaloExchanger, SameHostRanks
    import cupy as cp

    devices = tuple(int(value) for value in args.devices.split(","))
    if len(devices) < 2 or min(devices) < 0:
        raise ValueError("--devices requires at least two non-negative ordinals")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The older JW helper selected zero damping while leaving the descriptive
    # switch enabled. Use the current native Registry default consistently
    # in both reference and partitioned arms, without relaxing validation.
    config = replace(_config(args.dt, args.acoustic_substeps),
                     config_smdiv=V841DryDycoreConfig().config_smdiv)
    config.validate()
    prepared = prepare(args.grid.resolve(strict=True), args.init.resolve(strict=True), config)
    graph, part = _write_partition_inputs(prepared.mesh, len(devices), output, args.part)
    layouts = build_partition_layouts(
        graph, part, halo_rings=args.halo_rings,
        cells_on_edge=np.asarray(prepared.mesh.cellsOnEdge),
        cells_on_vertex=np.asarray(prepared.mesh.cellsOnVertex),
    )
    tables = build_all(layouts)
    schedule = RKSchedule.from_mpas(
        config.config_dt, order=config.config_time_integration_order,
        acoustic_substeps=config.config_number_of_sub_steps,
        dynamics_splits=config.config_dynamics_split_steps,
    )
    expected = expected_round_sequence(
        dynamics_splits=config.config_dynamics_split_steps,
        acoustic_steps=tuple(stage.acoustic_steps for stage in schedule.stages),
        monotonic=config.config_monotonic or config.config_positive_definite,
    )
    # Close both global and partition-local host contracts before opening a
    # card. Plain views avoid the legacy init=False dataclass slice fallback.
    slice_source = SimpleNamespace(**{
        name: getattr(prepared, name) for name in (
            "mesh", "state", "vertical", "reference", "saved_diagnostics",
            "terrain_metrics", "reference_wind_profiles", "advection_coefficients",
        )
    })
    local_prepared = [slice_prepared_host(slice_source, layout) for layout in layouts]
    from hexcore.cuda_driver import _validate_v841_host_execution
    for item in (prepared, *local_prepared):
        _validate_v841_host_execution(item.mesh, item.state, item.vertical,
                                      item.reference, item.saved_diagnostics, item.terrain_metrics)
    baseline = []
    reference_walls = []
    capability_records = {}
    with cp.cuda.Device(devices[0]):
        cache_dir = args.cache_root / "reference"
        capability = require_cuda(device_id=devices[0], cache_dir=cache_dir)
        capability_records["reference"] = capability.as_dict()
        cache = KernelCache(capability=capability, cache_dir=cache_dir)
        driver = _driver(prepared, config, cache)
        baseline.append(fingerprint_atmosphere(driver.atmosphere))
        for step in range(1, args.steps + 1):
            cp.cuda.nvtx.RangePush(f"samehost-reference-step-{step}")
            started = time.perf_counter()
            step_and_publish(driver)
            cp.cuda.get_current_stream().synchronize()
            reference_walls.append(time.perf_counter() - started)
            cp.cuda.nvtx.RangePop()
            baseline.append(fingerprint_atmosphere(driver.atmosphere))
        reference_manifest = cache.compile_manifest()
        del driver, cache
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    transport = SameHostTransport(tables, devices, xp=cp,
                                  copy_policy=args.copy_policy, timeout=args.timeout)
    candidates = []
    rank_walls = []
    mismatches = []
    with SameHostRanks(devices, transport, cp=cp) as ranks:
        def construct(rank: int) -> Any:
            cache_dir = args.cache_root / f"rank-{rank}"
            capability = require_cuda(device_id=devices[rank], cache_dir=cache_dir)
            capability_records[str(rank)] = capability.as_dict()
            cache = KernelCache(capability=capability, cache_dir=cache_dir)
            driver = _driver(local_prepared[rank], config, cache)
            driver.halo_exchanger_v841 = SameHostHaloExchanger(
                tables[rank], transport, expected_sequence=expected,
            )
            return driver

        drivers = ranks.call(construct)

        def capture() -> dict[str, Any]:
            blocks = ranks.call(lambda rank: _download(drivers[rank].atmosphere, cp))
            return fingerprint_atmosphere(_reconstruct(blocks, layouts))

        candidates.append(capture())
        for step in range(1, args.steps + 1):
            def advance(rank: int) -> Any:
                driver = drivers[rank]
                exchanger = driver.halo_exchanger_v841
                exchanger.begin_step(step)
                cp.cuda.nvtx.RangePush(f"samehost-rank-{rank}-step-{step}")
                try:
                    result = step_and_publish(driver)
                    exchanger.round_step_boundary(driver.atmosphere.state,
                                                  driver.atmosphere.saved)
                    exchanger.end_step()
                    _assert_device(driver, devices[rank])
                    return result.receipt
                finally:
                    cp.cuda.nvtx.RangePop()

            started = time.perf_counter()
            ranks.step(advance, lambda rank: drivers[rank].atmosphere.state.time_seconds,
                       dt=config.config_dt)
            rank_walls.append(time.perf_counter() - started)
            candidates.append(capture())
            print(f"step {step}: identical={candidates[-1] == baseline[step]}", flush=True)
        rank_manifests = [driver.cache.compile_manifest() for driver in drivers]
        exchanger_receipts = [driver.halo_exchanger_v841.receipt() for driver in drivers]

    for step, (reference, candidate) in enumerate(zip(baseline, candidates)):
        for path in _differences(reference, candidate):
            mismatches.append({"step": step, "path": path})
    profile = {
        "reference_step_seconds": reference_walls,
        "nrank_step_seconds": rank_walls,
        "first_step_includes_lazy_compilation": True,
        "timing_excludes_boundary_downloads_and_hashing": True,
        "n_cells": int(prepared.mesh.dimensions["nCells"]),
        "n_levels": int(prepared.state.rho.shape[0]),
        "expected_rounds_per_step": len(expected),
        "transport": transport.receipt(),
    }
    points = profile["n_cells"] * profile["n_levels"]
    profile["reference_ns_per_point_step"] = [value * 1e9 / points for value in reference_walls]
    profile["nrank_ns_per_point_step"] = [value * 1e9 / points for value in rank_walls]
    receipt = {
        "schema": "mpas-port.cuda-v841-samehost-dry-identity/v1",
        "tip": args.tip, "devices": devices, "same_card_ranks": len(set(devices)) == 1,
        "configuration": asdict(config), "input_files": prepared.input_bytes,
        "instrument": _file_record(Path(__file__).resolve()),
        "configuration_authority": _file_record(ROOT / "tools/run_cuda_v841_jw_dualrun.py"),
        "graph": _file_record(graph), "partition": _file_record(part),
        "halo_rings": args.halo_rings, "tables": [table.receipt() for table in tables],
        "steps": args.steps, "reference_boundaries": baseline,
        "nrank_boundaries": candidates, "mismatches": mismatches,
        "identical": not mismatches, "final_state_identical": baseline[-1] == candidates[-1],
        "clock_trace": [item["model_time_seconds"] for item in candidates],
        "capabilities": capability_records,
        "reference_compile_manifest": reference_manifest,
        "rank_compile_manifests": rank_manifests,
        "exchangers": exchanger_receipts, "profile": profile,
        "nonclaims": ["full column physics", "regional boundaries", "unbounded native MPAS agreement",
                      "forecast skill", "distinct-card scaling when devices repeat"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"identical": receipt["identical"], "mismatches": len(mismatches),
                      "receipt": str(output / "receipt.json")}), flush=True)
    return 0 if not mismatches else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--devices", default="0,0")
    parser.add_argument("--part", type=Path,
                        help="existing METIS labels; validated against the rank count and mesh")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--acoustic-substeps", type=int, default=6)
    parser.add_argument("--halo-rings", type=int, default=3)
    parser.add_argument("--copy-policy", choices=("auto", "peer", "staged"), default="auto")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tip", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
