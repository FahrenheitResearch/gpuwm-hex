#!/usr/bin/env python3
"""Run the port's ledger probe with the kernel-reservation tracer installed.

The pool ledger (hex_ledger_probe.py) attributes everything the CuPy pool
allocates.  It cannot attribute the non-pool residue, which on this port is
the larger half of the fixed term.  This driver adds the missing half: it
patches cupy.RawModule / cupy.RawKernel before the port compiles anything,
so every kernel's FIRST launch is bracketed with cudaMemGetInfo and the
local-memory backing store lands on the kernel that reserved it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kernel_reservation import KernelReservationTracer, nvsmi_process_mib  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kernel-json", type=Path, required=True)
    p.add_argument("--probe", type=Path,
                   default=HERE / "hex_ledger_probe.py")
    p.add_argument("--traced-launches", type=int, default=2)
    args, rest = p.parse_known_args(argv)

    import cupy as cp

    roots = []
    for i, a in enumerate(rest):
        if a in ("--repo", "--arwen-checkout") and i + 1 < len(rest):
            roots.append(str(Path(rest[i + 1]).resolve()))
    tracer = KernelReservationTracer(cp, roots=tuple(roots),
                                     max_traced_launches=args.traced_launches)
    tracer.install()

    probe = _load("hex_ledger_probe", args.probe)
    rc = 1
    try:
        rc = probe.main(rest)
    finally:
        try:
            snap = tracer.snapshot()
        except Exception as exc:  # pragma: no cover
            snap = {"snapshot_error": repr(exc)}
        snap["rc"] = rc
        snap["pid"] = os.getpid()
        snap["nvsmi_process_mib"] = nvsmi_process_mib(os.getpid())
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            snap["device_used_bytes_at_exit"] = int(total_b - free_b)
            snap["pool_total_bytes_at_exit"] = int(
                cp.get_default_memory_pool().total_bytes())
        except Exception:
            pass
        args.kernel_json.parent.mkdir(parents=True, exist_ok=True)
        args.kernel_json.write_text(json.dumps(snap, indent=1))
        print("kernel ledger ->", args.kernel_json)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
