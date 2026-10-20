#!/usr/bin/env python3
"""Bake the hex forecast start-up caches into a directory an image can carry.

WHAT IT IS FOR.  A freshly rented box starts a hex forecast with empty
caches: it rebuilds every mesh table from the mesh bytes (host preparation,
``hexcore.prep_cache``) and compiles about 200 CUDA modules (CuPy's kernel
cache, replayed in the background by ``hexcore.kernel_prewarm``).  Measured
2026-10-05 at tip 6fcb5db: conus12km on an RTX 5090 reached the end of its
first step in 61.5 s with both caches warm, 90.7 s with only the prep cache,
232.8 s with neither; the published 30 km mesh on an RTX PRO 6000 in 135 s
against 580 s.  This is the hex counterpart of the regional image's
``gpuwm warm-kernels`` step: the image build runs it once per production
mesh, and the rented box starts warm.

WHAT IT DOES.  It runs the production forecast command you give it twice,
with the caches pointed into ``--cache-dir``:

1. fill: one short forecast; every host table and every compiled module
   lands in the cache directory;
2. verify: the same forecast again, which must add NOTHING to either cache
   (every host table served, zero modules compiled) and must write history
   files byte-identical to the fill run's.

Then it writes ``BAKE-RECORD.json`` beside the caches: what was baked, under
which interpreter, NumPy, CuPy, NVRTC and card, and the verify verdict.

WHY THE RESULT IS SAFE TO SHIP TO ANY BOX.  Both caches are content-keyed.
A prep-cache entry is found only under the SHA-256 of every input array and
scalar, every hexcore source file, the reachable code's bindings, the NumPy
build and CPU dispatch, the interpreter and the C library; mesh paths are
not part of the key, so a box that stores its meshes elsewhere is still
served.  A CuPy cache entry is found only under the hash of the kernel
source, options, architecture and compiler.  A box that differs in any of
these misses and computes, exactly as it would without a baked cache.

RUNTIME ON THE RENTED BOX.  Point the job at the baked directory and keep it
writable by the job user (a miss stores a new entry):

    GPUWM_HEX_PREP_CACHE=<dir>/prep
    CUPY_CACHE_DIR=<dir>/cupy

USAGE (the forecast arguments are exactly the production ones; this tool
appends ``--out`` and ``--scratch``)::

    python tools/bake_startup_caches.py --cache-dir /opt/hex-cache \\
        -- -m hexcore forecast --mesh conus12km --grid G --static S --init I \\
           --lbc-dir L --start-time T --hours 0.25 --history-every-minutes 15 ...

Exit status: 0 baked and verified; 1 the verify pass compiled or recomputed
something, or the two runs' history files differ; 2 a forecast failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

SCHEMA = "gpuwm-hex.startup-cache-bake/v1"


def _files(root: Path) -> dict[str, int]:
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }


def _prep_entries(root: Path) -> list[str]:
    return sorted(path.parent.name for path in root.glob("??/*/manifest.json"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _find(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _receipt(out: Path) -> dict[str, Any]:
    for name in ("cuda-v841-forecast-receipt.json", "forecast-receipt.json"):
        candidate = out / name
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if _find(data, "host_preparation_cache") is not None:
                return data
    return {}


def _run(label: str, command: list[str], work: Path, env: dict[str, str]) -> tuple[int, Path, float]:
    out = work / label
    argv = [*command, "--out", str(out), "--scratch", str(work / f"{label}-scratch")]
    log = work / f"{label}.log"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        rc = subprocess.call(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return rc, out, time.perf_counter() - started


def _environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {"python": sys.version, "executable": sys.executable}
    try:
        from hexcore import prep_cache

        record["prep_producer_digest"] = prep_cache.producer_digest()
        record["numpy_cpu_dispatch"] = json.loads(prep_cache.numpy_cpu_dispatch())
    except Exception as error:  # noqa: BLE001 - recorded, the verify verdict stands alone
        record["prep_cache_error"] = repr(error)
    try:
        import numpy

        record["numpy"] = numpy.__version__
        import cupy

        record["cupy"] = cupy.__version__
        record["nvrtc"] = list(cupy.cuda.nvrtc.getVersion())
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        record["card"] = name.decode() if isinstance(name, bytes) else str(name)
        record["compute_capability"] = f"sm_{properties['major']}{properties['minor']}"
    except Exception as error:  # noqa: BLE001
        record["cuda_error"] = repr(error)
    return record


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: bake_startup_caches.py --cache-dir DIR [--work DIR] [--keep-out] -- <forecast argv>",
              file=sys.stderr)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=None,
                        help="where the two short forecasts write (default: a temporary directory)")
    parser.add_argument("--keep-out", action="store_true", help="keep the two forecasts' outputs")
    args = parser.parse_args(argv[:split])
    forecast = argv[split + 1:]
    if not forecast or any(flag in forecast for flag in ("--out", "--scratch")):
        print("give the forecast arguments without --out/--scratch; the tool sets them", file=sys.stderr)
        return 2

    cache = args.cache_dir.resolve()
    prep, cupy_dir, driver = cache / "prep", cache / "cupy", cache / "cuda-driver"
    for path in (prep, cupy_dir, driver):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(GPUWM_HEX_PREP_CACHE=str(prep), CUPY_CACHE_DIR=str(cupy_dir), CUDA_CACHE_PATH=str(driver))
    env.pop("GPUWM_HEX_KERNEL_PREWARM", None)
    if args.work is not None:
        args.work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hex-bake-", dir=args.work))
    command = [sys.executable, *forecast]

    before_cupy, before_prep = _files(cupy_dir), _prep_entries(prep)
    rc, fill_out, fill_seconds = _run("fill", command, work, env)
    if rc != 0:
        print(f"fill forecast failed rc={rc}; log {work / 'fill.log'}", file=sys.stderr)
        return 2
    mid_cupy, mid_prep = _files(cupy_dir), _prep_entries(prep)
    rc, verify_out, verify_seconds = _run("verify", command, work, env)
    if rc != 0:
        print(f"verify forecast failed rc={rc}; log {work / 'verify.log'}", file=sys.stderr)
        return 2
    after_cupy, after_prep = _files(cupy_dir), _prep_entries(prep)

    events = [row.get("event") for row in
              (_find(_receipt(verify_out), "host_preparation_cache") or {}).get("calls", [])]
    histories = {}
    for path in sorted(fill_out.glob("cuda-history.*.nc")):
        other = verify_out / path.name
        histories[path.name] = {"fill": _sha256(path),
                                "verify": _sha256(other) if other.is_file() else None}
    new_modules = sorted(set(after_cupy) - set(mid_cupy))
    new_entries = sorted(set(after_prep) - set(mid_prep))
    verdict = {
        "verify_new_cupy_modules": len(new_modules),
        "verify_new_prep_entries": len(new_entries),
        "verify_prep_events": {event: events.count(event) for event in sorted(set(map(str, events)))},
        "histories_identical": bool(histories) and all(row["fill"] == row["verify"] for row in histories.values()),
    }
    verdict["baked"] = (not new_modules and not new_entries and bool(events)
                        and "miss" not in events and verdict["histories_identical"])
    names = sorted(after_cupy)
    record = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": forecast,
        "cache_dir": str(cache),
        "runtime_environment": {"GPUWM_HEX_PREP_CACHE": str(prep), "CUPY_CACHE_DIR": str(cupy_dir)},
        "environment": _environment_record(),
        "prep_cache": {"entries": len(after_prep), "added_by_fill": len(set(mid_prep) - set(before_prep)),
                       "bytes": sum(_files(prep).values())},
        "cupy_cache": {"files": len(names), "added_by_fill": len(set(mid_cupy) - set(before_cupy)),
                       "bytes": sum(after_cupy.values()),
                       "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()},
        "seconds": {"fill": round(fill_seconds, 1), "verify": round(verify_seconds, 1)},
        "histories": histories,
        "verdict": verdict,
        "work": str(work) if args.keep_out else None,
    }
    (cache / "BAKE-RECORD.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"baked": verdict["baked"], "record": str(cache / "BAKE-RECORD.json"), **verdict}))
    if not args.keep_out:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if verdict["baked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
