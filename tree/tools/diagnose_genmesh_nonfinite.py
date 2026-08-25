#!/usr/bin/env python
"""Name the FIRST kernel and array that goes non-finite inside a v8.4.1 step.

The production refusal is a single four-byte flag read once per outer step
(``cuda_driver._step_device_v841``), so a run that dies at composite step 0
reports only "the validation flag refused the outer step before publish".
That names nothing: several dozen kernels write into that one flag, and a
generated mesh that dies there leaves no way to tell which array, which cell
and which geometry term produced the first bad value.

This instrument closes that gap without editing a single SHA-256-pinned
execution source.  It wraps ``KernelCache.raw_kernel`` at run time, so every
resolved kernel returns a proxy that, after each launch, synchronizes and
scans every float argument for non-finite values.  The FIRST array whose
non-finite population grows is reported with the kernel that grew it, the
argument slot, the count, and the first flat indices - which decode to
(level, cell) or (level, edge) against the mesh dimensions.

The scan is post-launch only and the proxy forwards everything else, so the
executed numerics are the production numerics; the run is slower, never
different.  Usage mirrors ``run_cuda_v841_forecast_mesh.py``:

    python tools/diagnose_genmesh_nonfinite.py \
        --repo <tree> --mesh v15.150.38857 --report <json> -- \
        --grid ... --static ... --init ... --hours 0.05 ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any


class FirstNonFinite(RuntimeError):
    """Raised to stop the run at the first array that goes non-finite."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(
            f"{record['kernel']} arg[{record['arg']}] {record['shape']}: "
            f"{record['nonfinite']} non-finite values appeared"
        )
        self.record = record


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_probe(
    runtime_module: Any,
    cp: Any,
    events: list[dict[str, Any]],
    *,
    stop_on_first: bool,
    max_indices: int,
    trace_path: Path | None = None,
) -> None:
    """Wrap every resolved kernel with a post-launch non-finite scan.

    With ``trace_path`` the probe also records the largest magnitude every
    float argument carries after each launch, with the flat index that
    carries it.  A blow-up is visible in that trace several kernels BEFORE
    the first non-finite value: the amplitude climbs through the acoustic
    substeps and only becomes NaN when a negative mass-weighted potential
    temperature reaches ``powf``.
    """

    kernel_cache = runtime_module.KernelCache
    original = kernel_cache.raw_kernel
    # Keyed by device pointer, not by object identity: CuPy reuses ndarray
    # wrappers, and a freed-then-reallocated buffer must not inherit a
    # previous array's baseline.
    baseline: dict[int, int] = {}
    counter = {"launches": 0}
    trace = trace_path.open("w", encoding="utf-8") if trace_path else None

    def _scan(kernel_name: str, args: tuple[Any, ...]) -> None:
        counter["launches"] += 1
        for slot, value in enumerate(args):
            if not isinstance(value, cp.ndarray):
                continue
            if value.dtype.kind != "f" or value.size == 0:
                continue
            if trace is not None:
                flat = value.reshape(-1)
                magnitude = cp.abs(cp.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0))
                where = int(cp.argmax(magnitude))
                trace.write(
                    json.dumps(
                        {
                            "n": counter["launches"],
                            "k": kernel_name,
                            "a": slot,
                            "shape": list(value.shape),
                            "max_abs": float(magnitude[where]),
                            "at": where,
                        }
                    )
                    + "\n"
                )
            bad = int(cp.count_nonzero(~cp.isfinite(value)))
            key = int(value.data.ptr)
            was = baseline.get(key, 0)
            baseline[key] = bad
            if bad <= was:
                continue
            flat = cp.asnumpy(
                cp.argwhere(~cp.isfinite(value.reshape(-1)))[:max_indices]
            ).reshape(-1)
            record = {
                "kernel": kernel_name,
                "arg": slot,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nonfinite": bad,
                "nonfinite_before": was,
                "launch_ordinal": counter["launches"],
                "first_flat_indices": [int(x) for x in flat],
            }
            events.append(record)
            if stop_on_first:
                if trace is not None:
                    trace.flush()
                raise FirstNonFinite(record)

    class _Probe:
        __slots__ = ("_kernel", "_name")

        def __init__(self, kernel: Any, name: str) -> None:
            self._kernel = kernel
            self._name = name

        def __call__(self, grid, block, args, **kwargs):
            result = self._kernel(grid, block, args, **kwargs)
            cp.cuda.get_current_stream().synchronize()
            _scan(self._name, tuple(args))
            return result

        def __getattr__(self, item):
            return getattr(self._kernel, item)

    def patched(self, name, source, *, module_key, options=()):
        kernel = original(
            self, name, source, module_key=module_key, options=options
        )
        if isinstance(kernel, _Probe):
            return kernel
        return _Probe(kernel, str(name))

    kernel_cache.raw_kernel = patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-indices", type=int, default=32)
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="JSONL of per-launch max-abs per float argument, with its index",
    )
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="keep running past the first event and record every growth",
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))

    import cupy as cp  # noqa: PLC0415  - device import belongs after sys.path

    from mpas_port.cuda_backend import runtime as runtime_module  # noqa: PLC0415

    events: list[dict[str, Any]] = []
    install_probe(
        runtime_module,
        cp,
        events,
        stop_on_first=not args.all_events,
        max_indices=int(args.max_indices),
        trace_path=args.trace,
    )

    binding_mod = _load(
        "mpas_mesh_binding", repo / "tools" / "mpas_mesh_binding.py"
    )
    forecast = _load(
        "v841_forecast", repo / "tools" / "run_cuda_v841_forecast.py"
    )

    def _flag(name: str) -> str | None:
        for index, token in enumerate(rest):
            if token == name and index + 1 < len(rest):
                return rest[index + 1]
        return None

    grid = _flag("--grid")
    static = _flag("--static")
    if grid is None or static is None:
        raise SystemExit("refusing: --grid and --static are required after '--'")

    bind = binding_mod.bind_mesh(
        forecast.proof, args.mesh, grid=Path(grid), static=Path(static),
        forecast=forecast,
    )

    report: dict[str, Any] = {
        "mesh": args.mesh,
        "grid": grid,
        "static": static,
        "bind_fingerprint_after": bind.get("constants_fingerprint_after"),
        "events": events,
    }
    rc = 1
    try:
        rc = forecast.main(rest)
        report["outcome"] = "completed"
    except FirstNonFinite as error:
        report["outcome"] = "first_nonfinite"
        report["first_event"] = error.record
        report["traceback"] = traceback.format_exc().splitlines()[-25:]
    except BaseException as error:  # noqa: BLE001 - the report is the product
        report["outcome"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc().splitlines()[-25:]
    report["rc"] = rc
    report["events"] = events
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(f"[probe] outcome={report['outcome']} events={len(events)}")
    for event in events[:5]:
        print(
            f"[probe] {event['kernel']} arg[{event['arg']}] "
            f"shape={event['shape']} nonfinite={event['nonfinite']} "
            f"first={event['first_flat_indices'][:8]}"
        )
    return 0 if report["outcome"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
