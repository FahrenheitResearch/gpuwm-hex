#!/usr/bin/env python3
"""Per-kernel A/B of the element-parallel dycore kernels on a live forecast.

The instrument runs the forecast DOOR (``gpuwm-hex forecast``) in this
process with every resolved CUDA kernel wrapped in a recorder; the door is
the route that binds a registered mesh row, so a point cull runs here the
way it runs for a user, where the engineering driver on its own is pinned
to the x4 authority and refuses any other grid.  Once the forecast is past
its warm-up steps, the first launch of every kernel this tree re-mapped is
captured: the launch geometry and a device copy of every argument before and
after the launch, taken from the running forecast's own state.  The run is
then stopped, and each record is replayed three ways on the same card in the
same process:

1. this tree's kernel, at the recorded geometry, from the recorded inputs --
   the output must be byte-identical to the recorded output (the replay is
   deterministic), and its device time is event-timed over ``--repeats``
   launches with the inputs restored between them;
2. the BASELINE translation unit's kernel -- the source string of the
   untouched tree, compiled under the same NVRTC options -- at the baseline
   geometry (one thread per owner), from the same inputs; its output is
   compared to this tree's bit for bit and its device time is taken the same
   way;
3. the baseline kernel at this tree's element geometry, as a control: a
   baseline kernel guards ``owner >= count`` and exits, so the extra threads
   change nothing but the launch shape.

The receipt records, per kernel, the two geometries, the two device times,
the bitwise verdict and the bytes compared.  A kernel whose baseline output
differs from this tree's is reported, never hidden; the forecast frames and
the contract deck remain the proofs of record and this instrument is the
per-kernel dissection beside them.

Nothing here runs on the default path.  Usage on the card, from this tree's
venv, with the baseline tree's sources extracted by the baseline venv first
(``--baseline-sources`` is a JSON file mapping module key -> source string,
written by ``--dump-sources``)::

    python tools/kernel_element_ab.py --dump-sources baseline.json
    python tools/kernel_element_ab.py --baseline-sources baseline.json \\
        --out receipt.json --repeats 20 --capture-after-steps 2 -- \\
        <gpuwm-hex forecast arguments: --mesh ROW --grid ... --out ...>

The door's ``--out`` and ``--scratch`` must be absent, as for any forecast,
and the run's ``--hours`` / ``--history-every-minutes`` must pass the door's
schedule (a whole number of steps, the cadence dividing the run); the
capture stops the run after its first few steps regardless of the length.

WHAT THE FIRST RUN OF THIS INSTRUMENT GOT WRONG, 2026-09-14, on the point
cull, 49 kernels compared, five reported as differing and none of them
was the arithmetic:

* ``recover_pressure_f32``, ``recover_edge_velocity_f32``,
  ``recover_terrain_w_f32`` and ``transport_standard_finish_regional_v841``
  differed in exactly 55 values each, all of them the garbage column
  (column ``n_cells_solve`` of a padded cell array) of an INPUT the kernel
  never writes: the RK stage's ``rho`` for the three recovery kernels and
  ``rho_zz_old`` for the transport finish.  This tree's own replay did not
  reproduce its own recorded output either.  The cause is the replay
  environment, not the kernel.  Every dycore launch resolves through the
  one ``KernelCache`` the regional runtime arms with
  ``RegionalGarbageDiscipline.scrub`` as ``post_launch``; the scrub
  rewrites the garbage column of every padded float32 argument after every
  launch (the four physics units and the solve-region unit manage their
  own and are skipped), writing 1.0 into an array bound to the unit pool
  for the step and 0.0 into any other.  The replay runs after the forecast
  has finished, when ``release_unit_pool`` has dropped every step-bound
  image, so the same scrub wrote 0.0 where the live step wrote 1.0.  The
  recorder now asks the discipline, at the launch, which arguments are
  bound (``bound_to_unit_pool``) and the replay re-binds them before every
  launch of either translation unit and releases after, so the replayed
  scrub writes what the live scrub wrote and every byte is compared.  The
  receipt also classifies each remaining difference by whether it lies in
  a garbage column, since a kernel's own value there is dead by
  construction -- the scrub overwrites it before any consumer launches --
  but the verdict of record stays the whole-array comparison.
* ``acoustic_ru_regional_v841`` differed in 7.2 million values because its
  baseline was launched at one owner: the owner count of a kernel with no
  entry in ``OWNER_COUNT_ARG`` is the trailing dimension of its LAST device
  argument, and that kernel's last device argument is the (1,) acoustic
  invalid flag.  The control launch of the baseline at the element
  geometry was byte-identical, which is the arithmetic's verdict; the
  rule now names the edge count, and ``tests/test_kernel_element_ab.py``
  parses every target's signature so a flag or index array can never be
  the owner axis again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def import_unit(module_name: str) -> Any:
    """Import a translation unit named by its identity, ``hexcore.<module>``.

    The identity keeps its bytes wherever the package goes (it is framed into
    digests); the import uses the package's own import name, which a tree
    that folds the package in may change.
    """

    import importlib

    import hexcore

    return importlib.import_module(hexcore.__name__ + module_name[len("hexcore"):])


#: module key -> (module path, attribute holding the translation unit)
SOURCE_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "hexcore.cuda_regional_v841": ("hexcore.cuda_regional_v841", "CUDA_REGIONAL_SOURCE"),
    "hexcore.cuda_dynamics_v841": ("hexcore.cuda_dynamics_v841", "_CUDA_SOURCE"),
    "hexcore.cuda_driver": ("hexcore.cuda_driver", "_CUDA_SOURCE"),
    "hexcore.cuda_acoustic": ("hexcore.cuda_acoustic", "_CUDA_SOURCE"),
    "hexcore.cuda_horizontal": ("hexcore.cuda_horizontal", "_CUDA_SOURCE"),
    "hexcore.cuda_horizontal_v841": ("hexcore.cuda_horizontal_v841", "_CUDA_SOURCE"),
    "hexcore.cuda_transport": ("hexcore.cuda_transport", "_CUDA_SOURCE"),
    "hexcore.cuda_backend.recovery": ("hexcore.cuda_backend.recovery", "RECOVERY_CUDA_SOURCE"),
}

#: The kernels this tree re-mapped to element geometry, by module key, with
#: the baseline block size of their translation unit.
TARGETS: dict[str, tuple[int, tuple[str, ...]]] = {
    "hexcore.cuda_regional_v841": (128, (
        "regional_lbc_rho_edge_v841",
        "regional_speczone_tend_cell_v841",
        "regional_speczone_tend_edge_v841",
        "regional_relaxzone_rayleigh_cell_v841",
        "regional_relaxzone_rayleigh_edge_v841",
        "regional_relaxzone_filter_cell_v841",
        "regional_relaxzone_filter_edge_v841",
        "regional_speczone_u_ru_v841",
        "regional_zero_speczone_w_v841",
        "regional_reset_speczone_values_v841",
        "regional_bdy_adjust_scalars_compute_v841",
        "regional_bdy_adjust_scalars_copyback_v841",
        "regional_bdy_set_scalars_v841",
        "acoustic_ru_regional_v841",
        "acoustic_rs_ts_regional_v841",
        "transport_edge_values_regional_v841",
        "transport_standard_finish_regional_v841",
    )),
    "hexcore.cuda_dynamics_v841": (128, (
        "vector_momentum_v841_f32",
        "theta_finish_v841_f32",
        "w_finish_v841_f32",
        "split_flux_first_v841_f32",
        "split_flux_add_v841_f32",
        "split_flux_finish_v841_f32",
    )),
    "hexcore.cuda_driver": (128, (
        "euler_w_f32",
        "vertical_u_flux_f32",
        "vertical_u_finish_f32",
        "theta_edge_flux_f32",
        "theta_vertical_flux_f32",
        "w_edge_flux_f32",
        "w_vertical_flux_f32",
        "add_inplace_f32",
        "scale_f32",
        "recover_cells_f32",
        "recover_edges_f32",
        "recover_interfaces_f32",
    )),
    "hexcore.cuda_acoustic": (128, ("tendency_w_to_omega",)),
    "hexcore.cuda_horizontal": (256, (
        "tangential_velocity_f32",
        "laplacian_divergence_f32",
        "theta_filter_lap2_f32",
        "theta_filter_lap4_f32",
        "w_filter_lap2_f32",
        "w_filter_lap4_f32",
    )),
    "hexcore.cuda_horizontal_v841": (128, (
        "vertex_diagnostics_v841_f32",
        "cell_diagnostics_v841_f32",
        "smagorinsky_v841_f32",
        "pv_cell_v841_f32",
        "pv_apvm_v841_f32",
        "mass_flux_divergence_v841_f32",
    )),
    "hexcore.cuda_transport": (128, ("transport_vertical_flux",)),
    "hexcore.cuda_backend.recovery": (256, (
        "recover_pressure_f32",
        "recover_edge_velocity_f32",
        "recover_terrain_w_f32",
        "recover_flat_w_f32",
    )),
}

#: Where the baseline's per-owner launch count sits in the argument tuple,
#: for the kernels whose owners are a zone slot list rather than the last
#: array's trailing dimension.
OWNER_COUNT_ARG: dict[str, Any] = {
    "regional_speczone_tend_cell_v841": 2,
    "regional_speczone_tend_edge_v841": 2,
    "regional_relaxzone_rayleigh_cell_v841": 2,
    "regional_relaxzone_rayleigh_edge_v841": 2,
    "regional_relaxzone_filter_cell_v841": 4,
    "regional_relaxzone_filter_edge_v841": 6,
    "regional_speczone_u_ru_v841": 2,
    "regional_zero_speczone_w_v841": 2,
    "regional_reset_speczone_values_v841": 2,
    "regional_bdy_adjust_scalars_compute_v841": (5, 6),
    "regional_bdy_adjust_scalars_copyback_v841": 3,
    "regional_bdy_set_scalars_v841": 3,
    # Its last device argument is the (1,) acoustic invalid flag, not an
    # edge array: the owner axis is the second scalar, ``nedges``.
    "acoustic_ru_regional_v841": 1,
}

#: The step counter: one launch of this kernel per forecast step.
STEP_MARK = "regional_reset_speczone_values_v841"

#: Recorded on its first launch whatever the step: it runs once per run, and
#: the momentum baseline needs its angle_edge / u_init / v_init.
ALWAYS_CAPTURE = {("hexcore.cuda_dynamics_v841", "reference_wind_edge_v841_f32")}

#: Targets a configuration may never launch (a terrain run recovers w with
#: the terrain kernel), so their absence does not hold the capture open.
OPTIONAL = {("hexcore.cuda_backend.recovery", "recover_flat_w_f32")}

NVRTC_OPTIONS = ("--std=c++17", "--fmad=false")


class CaptureDone(BaseException):
    """Raised inside the forecast once every target has been recorded.

    A ``BaseException`` on purpose: the door wraps any ``Exception`` the
    driver raises into a refusal with a traceback beside the run, and the
    capture ending is not a failure the door should report."""


def _cupy() -> Any:
    import cupy as cp

    return cp


def _is_device_array(value: Any) -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False
    return isinstance(value, cp.ndarray)


def _copy_args(cp: Any, args: tuple[Any, ...]) -> list[Any]:
    return [cp.array(a, copy=True) if _is_device_array(a) else a for a in args]


def _restore(cp: Any, live: list[Any], saved: list[Any]) -> None:
    for target, source in zip(live, saved):
        if _is_device_array(target):
            cp.copyto(target, source)


def _digest(cp: Any, args: list[Any]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for value in args:
        if _is_device_array(value):
            host = cp.asnumpy(value)
            digest.update(np.ascontiguousarray(host).tobytes())
            total += int(host.nbytes)
    return digest.hexdigest(), total


def _compare_host(
    position: int, ha: np.ndarray, hb: np.ndarray, extents: dict[int, int] | None = None
) -> dict[str, Any] | None:
    """One argument's difference report, or None when the bytes agree.

    ``extents`` maps a padded trailing dimension to its solve count (the
    discipline's own table); an array whose trailing dimension is padded
    has its differing values split into the garbage column and the rest.
    """

    if ha.tobytes() == hb.tobytes():
        return None
    if ha.dtype == np.float32:
        differs = ha.view(np.uint32) != hb.view(np.uint32)
    else:
        differs = ha != hb
    report: dict[str, Any] = {
        "argument": position,
        "shape": list(ha.shape),
        "differing_values": int(np.count_nonzero(differs)),
    }
    if ha.dtype.kind == "f":
        delta = np.abs(ha.astype(np.float64) - hb.astype(np.float64))
        where = int(np.nanargmax(delta))
        report.update({
            "max_abs_difference": float(np.nanmax(delta)),
            "at_flat_index": where,
            "mine": float(ha.flat[where]),
            "baseline": float(hb.flat[where]),
        })
    solve = (extents or {}).get(int(ha.shape[-1])) if ha.ndim else None
    if solve is not None:
        garbage = differs[..., solve]
        report["garbage_column"] = solve
        report["differing_values_in_garbage_column"] = int(np.count_nonzero(garbage))
        report["differing_values_outside_garbage_column"] = (
            report["differing_values"] - report["differing_values_in_garbage_column"]
        )
    return report


def _mismatch(
    cp: Any, mine: list[Any], other: list[Any], extents: dict[int, int] | None = None
) -> list[dict[str, Any]]:
    report = []
    for position, (a, b) in enumerate(zip(mine, other)):
        if not _is_device_array(a):
            continue
        entry = _compare_host(position, cp.asnumpy(a), cp.asnumpy(b), extents)
        if entry is not None:
            report.append(entry)
    return report


def _outside_garbage_columns(mismatches: list[dict[str, Any]]) -> int:
    """Differing values that are not in a garbage column, over a report."""

    return sum(
        int(m.get("differing_values_outside_garbage_column", m["differing_values"]))
        for m in mismatches
    )


def _disciplines(caches: list[Any]) -> list[Any]:
    """The garbage disciplines armed on the recorded caches, if any."""

    found = []
    for cache in caches:
        owner = getattr(getattr(cache, "post_launch", None), "__self__", None)
        if owner is not None and hasattr(owner, "bound_to_unit_pool") and owner not in found:
            found.append(owner)
    return found


def _unit_pool_positions(
    args: tuple[Any, ...], disciplines: list[Any], is_device: Any = _is_device_array
) -> list[int]:
    """Argument positions the disciplines hold in the unit pool right now."""

    return [
        position
        for position, value in enumerate(args)
        if is_device(value) and any(d.bound_to_unit_pool(value) for d in disciplines)
    ]


def _extents(disciplines: list[Any]) -> dict[int, int]:
    table: dict[int, int] = {}
    for d in disciplines:
        for solve in (d.n_cells_solve, d.n_edges_solve, d.n_vertices_solve):
            table[int(solve) + 1] = int(solve)
    return table


def _rebind(disciplines: list[Any], positions: list[int], live: list[Any]) -> None:
    """Put the disciplines in the state the recorded launch saw."""

    for d in disciplines:
        d.release_unit_pool()
        for position in positions:
            d.bind_unit_pool(live[position])


def _release(disciplines: list[Any]) -> None:
    for d in disciplines:
        d.release_unit_pool()


class Recorder:
    """A resolved kernel that records one launch of itself."""

    def __init__(self, kernel: Any, name: str, module_key: str, session: "Session") -> None:
        self._kernel = kernel
        self._name = name
        self._module_key = module_key
        self._session = session

    def __call__(self, grid: Any, block: Any, args: Any, **kwargs: Any) -> Any:
        session = self._session
        if self._name == STEP_MARK and self._module_key == session.step_mark_module:
            session.steps_seen += 1
        key = (self._module_key, self._name)
        capture = (
            key in session.targets
            and key not in session.records
            and (session.steps_seen >= session.capture_after_steps
                 or key in ALWAYS_CAPTURE)
        )
        if not capture:
            result = self._kernel(grid, block, args, **kwargs)
            if session.finished():
                raise CaptureDone()
            return result
        cp = _cupy()
        cp.cuda.get_current_stream().synchronize()
        before = _copy_args(cp, tuple(args))
        # Which arguments the discipline will scrub to the unit pool for
        # THIS launch: asked now, because the registrations are released at
        # the step boundary and the replay runs after the run.
        unit_pool = _unit_pool_positions(tuple(args), _disciplines(session.caches))
        result = self._kernel(grid, block, args, **kwargs)
        cp.cuda.get_current_stream().synchronize()
        after = _copy_args(cp, tuple(args))
        session.records[key] = {
            "grid": tuple(int(g) for g in grid),
            "block": tuple(int(b) for b in block),
            "kwargs": dict(kwargs),
            "before": before,
            "after": after,
            "live": list(args),
            "kernel": self._kernel,
            "unit_pool_args": unit_pool,
        }
        print(f"[capture] {self._module_key}:{self._name} grid={grid} block={block} "
              f"({len(session.records)}/{len(session.targets)})", flush=True)
        if session.finished():
            raise CaptureDone()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


class Session:
    def __init__(self, capture_after_steps: int) -> None:
        self.targets: set[tuple[str, str]] = {
            (module_key, name)
            for module_key, (_, names) in TARGETS.items()
            for name in names
        } | set(ALWAYS_CAPTURE)
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        self.steps_seen = 0
        self.capture_after_steps = int(capture_after_steps)
        self.step_mark_module = "hexcore.cuda_regional_v841"
        self.caches: list[Any] = []

    def finished(self) -> bool:
        # Every target that this run launches at all; a kernel the
        # configuration never launches (the flat-w recovery on a terrain
        # run, say) cannot be waited for, so the run also ends on its own.
        return all(key in self.records for key in self.targets - OPTIONAL)


def install(session: Session) -> None:
    from hexcore.cuda_backend import runtime

    original = runtime.KernelCache.raw_kernel

    def raw_kernel(self: Any, name: str, source: str, *, module_key: str, options: Any = ()) -> Any:
        kernel = original(self, name, source, module_key=module_key, options=options)
        if self not in session.caches:
            session.caches.append(self)
        if (module_key, name) in session.targets or name == STEP_MARK:
            return Recorder(kernel, name, module_key, session)
        return kernel

    runtime.KernelCache.raw_kernel = raw_kernel  # type: ignore[method-assign]


def dump_sources(path: Path) -> None:
    import importlib

    import hexcore

    payload = {"__hexcore__": str(Path(hexcore.__file__).resolve())}
    for module_key, (module_name, attribute) in SOURCE_ATTRIBUTES.items():
        module = import_unit(module_name)
        payload[module_key] = getattr(module, attribute)
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(payload) - 1} translation units from {payload['__hexcore__']} to {path}")


def _baseline_args(name: str, session: Session, record: dict[str, Any], before: list[Any]) -> list[Any]:
    """The baseline kernel's argument tuple from this tree's recorded one."""

    if name != "vector_momentum_v841_f32":
        return list(before)
    # (nlev, ncells, nedges, max_edges2, u, rho_edge, pv_edge, kinetic,
    #  mass_div, cells_on_edge, edges_on_edge, n_edges_on_edge,
    #  weights_on_edge, inv_dc_edge, reference_u, f_edge, out)  ->
    # (..., inv_dc_edge, angle_edge, f_edge, u_init, v_init, out)
    reference = session.records.get(("hexcore.cuda_dynamics_v841", "reference_wind_edge_v841_f32"))
    if reference is None:
        raise RuntimeError("the reference wind kernel was not recorded; the momentum baseline needs its angle_edge/u_init/v_init")
    ref_before = reference["before"]
    angle_edge, u_init, v_init = ref_before[2], ref_before[3], ref_before[4]
    return list(before[:14]) + [angle_edge, before[15], u_init, v_init, before[16]]


def _owner_count(name: str, before: list[Any]) -> int:
    rule = OWNER_COUNT_ARG.get(name)
    if rule is None:
        last = [a for a in before if _is_device_array(a)][-1]
        return int(last.shape[-1])
    if isinstance(rule, tuple):
        return int(sum(int(before[i]) for i in rule))
    return int(before[rule])


def _time_launch(cp: Any, launch: Any, restore: Any, repeats: int) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        restore()
        cp.cuda.get_current_stream().synchronize()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        launch()
        end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, end)) * 1000.0)
    arr = np.asarray(times)
    return {
        "min_us": float(arr.min()),
        "median_us": float(np.median(arr)),
        "mean_us": float(arr.mean()),
        "repeats": int(repeats),
    }


def replay(session: Session, baseline_sources: dict[str, str], repeats: int) -> dict[str, Any]:
    cp = _cupy()
    modules: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    disciplines = _disciplines(session.caches)
    extents = _extents(disciplines)
    for (module_key, name), record in sorted(session.records.items()):
        if name not in TARGETS.get(module_key, (0, ()))[1]:
            continue
        before, after, live = record["before"], record["after"], record["live"]
        grid, block = record["grid"], record["block"]
        kernel = record["kernel"]
        kwargs = record["kwargs"]
        unit_pool = list(record.get("unit_pool_args", ()))

        def restore() -> None:
            # The inputs, then the discipline as the live launch had it, so
            # the post-launch scrub writes the live launch's pool values.
            _restore(cp, live, before)
            _rebind(disciplines, unit_pool, live)

        # 1. this tree, replayed at the recorded geometry
        restore()
        kernel(grid, block, tuple(live), **kwargs)
        cp.cuda.get_current_stream().synchronize()
        replay_mismatch = _mismatch(cp, live, after, extents)
        mine_time = _time_launch(
            cp,
            lambda: kernel(grid, block, tuple(live), **kwargs),
            restore,
            repeats,
        )

        # 2. the baseline translation unit at the baseline geometry
        entry: dict[str, Any] = {
            "module_key": module_key,
            "kernel": name,
            "element_grid": list(grid),
            "element_block": list(block),
            "unit_pool_arguments": unit_pool,
            "replay_reproduces_recorded_output": not replay_mismatch,
            "replay_mismatches": replay_mismatch,
            "mine": mine_time,
        }
        source = baseline_sources.get(module_key)
        if source is None:
            _release(disciplines)
            entry["baseline"] = "no baseline source for this module key"
            results.append(entry)
            continue
        module = modules.get(module_key)
        if module is None:
            module = cp.RawModule(code=source, options=NVRTC_OPTIONS, backend="nvrtc")
            modules[module_key] = module
        try:
            base_kernel = module.get_function(name)
        except Exception as error:  # pragma: no cover - a renamed kernel
            _release(disciplines)
            entry["baseline"] = f"baseline has no kernel {name}: {error}"
            results.append(entry)
            continue
        base_block = (TARGETS[module_key][0],)
        owners = _owner_count(name, before)
        base_grid = ((owners + base_block[0] - 1) // base_block[0],)
        base_args = _baseline_args(name, session, record, live)
        hooks = [cache.post_launch for cache in session.caches if getattr(cache, "post_launch", None) is not None]

        def launch_baseline(grid_=base_grid, block_=base_block) -> None:
            base_kernel(grid_, block_, tuple(base_args))
            for hook in hooks:
                hook(name, base_args, module_key=module_key)

        restore()
        launch_baseline()
        cp.cuda.get_current_stream().synchronize()
        mismatch = _mismatch(cp, live, after, extents)
        base_time = _time_launch(cp, launch_baseline, restore, repeats)
        # 3. the baseline kernel at the element geometry, as a control
        restore()
        launch_baseline(grid, block)
        cp.cuda.get_current_stream().synchronize()
        control_mismatch = _mismatch(cp, live, after, extents)
        _release(disciplines)
        digest, nbytes = _digest(cp, after)
        entry.update({
            "baseline_grid": list(base_grid),
            "baseline_block": list(base_block),
            "baseline_owner_count": owners,
            "baseline": base_time,
            "bitwise_identical": not mismatch,
            "bitwise_identical_outside_garbage_columns": _outside_garbage_columns(mismatch) == 0,
            "mismatches": mismatch,
            "baseline_at_element_geometry_identical": not control_mismatch,
            "output_sha256": digest,
            "bytes_compared": nbytes,
            "speedup": (base_time["median_us"] / mine_time["median_us"]) if mine_time["median_us"] > 0 else None,
        })
        results.append(entry)
        print(f"[replay] {name:44s} base {base_time['median_us']:9.1f} us  mine {mine_time['median_us']:9.1f} us  "
              f"x{entry['speedup']:.2f}  {'BITWISE' if not mismatch else 'DIFFERS'}", flush=True)
    return {"kernels": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-sources", type=Path, default=None)
    parser.add_argument("--baseline-sources", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--capture-after-steps", type=int, default=2)
    parser.add_argument("instrument_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.dump_sources is not None:
        dump_sources(args.dump_sources)
        return 0
    if args.baseline_sources is None or args.out is None:
        parser.error("--baseline-sources and --out are required to replay")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    baseline_sources = json.loads(args.baseline_sources.read_text(encoding="utf-8"))
    baseline_origin = baseline_sources.pop("__hexcore__", "unknown")
    instrument_args = list(args.instrument_args)
    if instrument_args and instrument_args[0] == "--":
        instrument_args = instrument_args[1:]
    session = Session(args.capture_after_steps)
    install(session)
    started = time.perf_counter()
    from hexcore.cli import main as door_main

    ended_by = "capture-complete"
    try:
        rc = door_main(["forecast", *instrument_args])
        ended_by = f"forecast-finished-rc-{rc}"
    except CaptureDone:
        pass
    except SystemExit as exit_code:
        ended_by = f"door-exit-{exit_code.code}"
    capture_seconds = time.perf_counter() - started
    print(f"[capture] {len(session.records)} records after {session.steps_seen} steps ({ended_by}, {capture_seconds:.1f} s)", flush=True)
    missing = sorted(f"{m}:{n}" for (m, n) in session.targets - set(session.records))
    started = time.perf_counter()
    report = replay(session, baseline_sources, int(args.repeats))
    report.update({
        "schema": "gpuwm-hex.kernel-element-ab/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card": str(_cupy().cuda.runtime.getDeviceProperties(0)["name"].decode()),
        "nvrtc_options": list(NVRTC_OPTIONS),
        "baseline_sources_from": baseline_origin,
        "captured_after_steps": session.steps_seen,
        "capture_ended_by": ended_by,
        "capture_seconds": capture_seconds,
        "replay_seconds": time.perf_counter() - started,
        "repeats": int(args.repeats),
        "targets_not_launched_by_this_run": missing,
        "all_bitwise": all(k.get("bitwise_identical") is True for k in report["kernels"] if "baseline_grid" in k),
        "all_bitwise_outside_garbage_columns": all(
            k.get("bitwise_identical_outside_garbage_columns") is True
            for k in report["kernels"] if "baseline_grid" in k
        ),
        "all_replays_reproduce_recorded_output": all(
            k.get("replay_reproduces_recorded_output") is True for k in report["kernels"]
        ),
        "kernels_compared": sum(1 for k in report["kernels"] if "baseline_grid" in k),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[done] {report['kernels_compared']} kernels compared, all_bitwise={report['all_bitwise']}, receipt {args.out}")
    return 0 if report["all_bitwise"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
