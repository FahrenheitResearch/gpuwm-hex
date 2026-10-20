"""Box-only whole-MET validation, using real decoded windows and the predecessor.

This is a receipt tool, not a runtime Python regrid route. It writes small
receipts before removing its disposable MET and decoded window files.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from hexcore import hrrr_intermediate as hi
from projected_reference import regrid_hour as reference_regrid, reference_window


def evict(paths):
    """Prove that the input pages are absent without reading weather values."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rows = []
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if not size:
                continue
            pages = (size + os.sysconf("SC_PAGE_SIZE") - 1) // os.sysconf("SC_PAGE_SIZE")
            residency = (ctypes.c_ubyte * pages)()
            address = libc.mmap(None, size, 0, 1, fd, 0)  # PROT_NONE, MAP_SHARED
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), f"mmap residency probe: {path}")
            try:
                for attempt in range(3):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    if libc.mincore(address, size, residency):
                        raise OSError(ctypes.get_errno(), f"mincore: {path}")
                    resident = sum(value & 1 for value in residency)
                    if not resident:
                        break
                    time.sleep(0.1)
                rows.append({"path": str(path), "bytes": size, "pages": pages,
                             "resident_pages": resident, "attempts": attempt + 1})
                if resident:
                    raise RuntimeError(f"COLD input still has {resident}/{pages} resident pages: {path}")
            finally:
                libc.munmap(address, size)
        finally:
            os.close(fd)
    return rows


def reference_build(request, workers, log):
    """The predecessor whole-series decode then Python regrid/write schedule."""
    from gpuwm.ingest import hrrr, soil
    out=request.out_dir
    out.mkdir()
    pieces=hi._engine_pieces(request.source)
    pieces.update(load=hrrr.load_hrrr_pipeline_ready_window,
                  plan=hrrr._ProjectedCpuPlan, rotation=hrrr._source_window_rotation,
                  ranges=hrrr._require_source_physical_ranges, interp_nodes=soil._interp_nodes)
    ledger=out / "timeline.jsonl"
    hi._ledger_emit(ledger,"prep.intermediate.decode","begin",{"hours":len(request.hours),"workers":workers})
    decoded=hi.decode_window(request.source,decoder=hi.resolve_decoder(request.source),
                             grib_dir=request.grib_dir,cycle=request.cycle,hours=request.hours,
                             window=reference_window(request.target,request.source),
                             out_dir=out,workers=workers,log=log)
    hi._ledger_emit(ledger,"prep.intermediate.decode","end",{"seconds":decoded["seconds"]})
    backend = pieces["resolve_backend"]("cpu")
    files=[]
    for hour in request.hours:
        hi._ledger_emit(ledger,"prep.intermediate.hour","begin",{"hour":hour,"decode_s":0.0,"regrid_s":0.0})
        started=time.perf_counter()
        snap=pieces["load"](Path(decoded["native"]),hour)
        records,measured=reference_regrid(request.source,pieces,snap,request.target, backend=backend)
        path=out / f"MET:{snap.valid_time:%Y-%m-%d_%H}"
        written=hi.write_intermediate(path,records,valid_time=snap.valid_time,
                                      forecast_hour=hour,map_source=request.source.map_source,target=request.target)
        elapsed=time.perf_counter()-started
        written["regrid_seconds"]=elapsed
        written["operator"]=measured["operator"]
        files.append(written)
        hi._ledger_emit(ledger,"prep.intermediate.hour","end",{"hour":hour,"decode_s":0.0,"regrid_s":elapsed})
        log(f"REFERENCE f{hour:02d}: {elapsed:.6f} s, {written['sha256']}")
        del snap,records
    return {"files":files,"decode":{"hours":[decoded]*len(request.hours)},"target":request.target.as_dict(),
            "reference_operator":files[0]["operator"], "reference_backend_workers":backend.workers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grib-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int, required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--point", required=True, help="latitude,longitude from the approved pin")
    parser.add_argument("--radius-km", type=float, required=True)
    parser.add_argument("--margin-km", type=float, default=hi.DEFAULT_MARGIN_KM)
    parser.add_argument("--spacing-deg", type=float, default=hi.DEFAULT_SPACING_DEG)
    parser.add_argument("--target-json", type=Path, help="exact target from the sealed case preparation receipt")
    parser.add_argument("--pin-json", type=Path)
    parser.add_argument("--pin-sha256")
    parser.add_argument("--reserved-cpus", help="comma-separated CPU IDs reserved for this run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--reuse-receipt", type=Path)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    point = tuple(float(value) for value in args.point.split(","))
    if len(point) != 2:
        parser.error("--point needs latitude,longitude")
    pin = None
    if args.pin_json:
        if not args.pin_sha256 or hi.sha256_file(args.pin_json) != args.pin_sha256:
            parser.error("the input pin does not match its approved SHA-256")
        if not (args.pin_json.parent / "SEALED").is_file():
            parser.error("the input pack has no SEALED marker")
        pin = json.loads(args.pin_json.read_text())
        for item in pin["files"]:
            path = args.pin_json.parent / item["path"]
            if path.stat().st_size != item["bytes"] or hi.sha256_file(path) != item["sha256"]:
                parser.error(f"sealed input changed: {item['path']}")
    reserved = sorted(int(value) for value in args.reserved_cpus.split(",")) if args.reserved_cpus else []
    if reserved and not set(os.sched_getaffinity(0)).issubset(reserved):
        parser.error("CPU affinity escapes the declared reservation")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["GPUWM_TIME_LEDGER"]=str(args.out / "timeline.jsonl")
    os.environ["GPUWM_RUN_ID"]=args.out.name
    if args.target_json:
        metadata = json.loads(args.target_json.read_text())
        spec = metadata.get("target", metadata)
        target = hi.LatLonTarget(**{name: spec[name] for name in ("south", "west", "dlat", "dlon", "nx", "ny")})
    else:
        target = hi.target_for_cap(point, args.radius_km, margin_km=args.margin_km, spacing_deg=args.spacing_deg)
    request = hi.IntermediateRequest(hi.source_row("hrrr"), args.grib_dir, hi.parse_cycle(args.cycle),
                                     tuple(range(args.hours+1)), target, args.out / "native", workers=args.workers)
    if args.cold:
        eviction = evict([p for p in args.grib_dir.rglob("*.grib2") if p.is_file()])
        (args.out / "input-eviction.json").write_text(json.dumps(eviction, indent=2)+"\n")
    if args.baseline:
        hi._ledger_emit(args.out / "timeline.jsonl","worker.claim","mark",{"basis":"prep-only proxy"})
        started=time.perf_counter()
        receipt=reference_build(request,args.workers,print)
        native_wall=time.perf_counter()-started
        hi._ledger_emit(args.out / "timeline.jsonl","finish","mark",{"basis":"prep-only proxy"})
    elif args.reuse_receipt:
        receipt = json.loads(args.reuse_receipt.read_text())
        native_wall = None
    else:
        hi._ledger_emit(args.out / "timeline.jsonl","worker.claim","mark",{"basis":"prep-only proxy"})
        started = time.perf_counter()
        receipt = hi.build_intermediates(request)
        native_wall = time.perf_counter()-started
        hi._ledger_emit(args.out / "timeline.jsonl","finish","mark",{"basis":"prep-only proxy"})
    from gpuwm.ingest import hrrr, soil
    pieces = hi._engine_pieces(request.source)
    pieces.update(load=hrrr.load_hrrr_pipeline_ready_window,
                  ranges=hrrr._require_source_physical_ranges, interp_nodes=soil._interp_nodes,
                  plan=hrrr._ProjectedCpuPlan, rotation=hrrr._source_window_rotation)
    # Independent NumPy reference, even when the installed CPU bridge has a
    # Rust donor entry. It is test code, never a production fallback.
    backend = SimpleNamespace(indexed_donor_interp=False)
    rows = []
    reference_wall = 0.0
    for hour, decoded, written in zip(request.hours, receipt["decode"]["hours"], receipt["files"]):
        snap = pieces["load"](Path(decoded["native"]), hour)
        path = args.out / f"reference-f{hour:02d}.met"
        records=None
        if args.baseline:
            reference=written
            seconds=written["regrid_seconds"]
        else:
            begin = time.perf_counter()
            records, measured = reference_regrid(request.source, pieces, snap, target, backend=backend)
            reference = hi.write_intermediate(path, records, valid_time=request.cycle+timedelta(hours=hour),
                                              forecast_hour=hour, map_source=request.source.map_source, target=target)
            seconds = time.perf_counter()-begin
        reference_wall += seconds
        rows.append({"hour":hour, "reference_s":seconds, "native_regrid_s":written["regrid_seconds"],
                     "reference_sha256":reference["sha256"],"native_sha256":written["sha256"],
                     "bytes":written["bytes"], "equal":reference["sha256"]==written["sha256"]})
        # Same length plus SHA-256 binds every MET byte and every field in it.
        if reference["bytes"] != written["bytes"] or not rows[-1]["equal"]:
            raise AssertionError(f"MET identity failed for f{hour:02d}: {rows[-1]}")
        (args.out / "identity-progress.json").write_text(json.dumps(rows,indent=2)+"\n")
        del records, snap
        payloads = [p for role in ("atmosphere", "soil")
                    for p in (Path(decoded["native"]) / f"{role}-f{hour:02d}").rglob("*")]
        with (args.out / "deleted-files.jsonl").open("a") as log:
            for generated in [path, Path(written["path"]), *sorted(payloads)]:
                if generated.is_file() and not generated.is_symlink():
                    log.write(json.dumps({"path":str(generated),"bytes":generated.stat().st_size})+"\n")
                    generated.unlink()
        # Remove empty generated directories after recording every file size.
        for role in ("atmosphere", "soil"):
            directory=Path(decoded["native"]) / f"{role}-f{hour:02d}"
            directory.rmdir()
    output={"schema":"projected-intermediate.validation/v1", "cycle":args.cycle, "forecast_hours":args.hours,
            "workers":args.workers, "affinity":sorted(os.sched_getaffinity(0)),
            "cache_arm":"COLD" if args.cold else "WARM", "target":target.as_dict(),
            "arm":"predecessor-python" if args.baseline else "native", "prep_wall_s":native_wall,
            "native_wall_s":native_wall, "reference_regrid_write_s":reference_wall,
            "native_regrid_write_s":sum(r["native_regrid_s"] for r in rows),
            "identity":"REFERENCE_ONLY" if args.baseline else "PASS", "hours":rows,
            "input_pin_sha256":args.pin_sha256, "reserved_cpus":reserved,
            "reference_operator":receipt.get("reference_operator", "independent NumPy oracle"),
            "reference_backend_workers":receipt.get("reference_backend_workers"),
            "qualification":"sealed input and reserved affinity" if pin and reserved else
                            "component timing; C3 input or CPU reservation incomplete"}
    (args.out / "validation.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)


if __name__ == "__main__":
    main()
