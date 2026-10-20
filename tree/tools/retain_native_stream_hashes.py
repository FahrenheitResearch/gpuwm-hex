"""Retain raw stored-stream fingerprints before retiring a completed forecast.

Usage: python retain_native_stream_hashes.py RUN RECEIPT

CPU/disk only. This creates a reference inventory, not a comparison proof.
It does not inspect native internal state, rewrite histories, or retain arrays.
The right_sha256 field is compatible with retained-reference comparison tools.
"""
import datetime, hashlib, json, sys, time
from pathlib import Path
import netCDF4
import numpy as np
run, destination = map(Path, sys.argv[1:])
start = datetime.datetime.now(datetime.timezone.utc).isoformat(); tick = time.monotonic()
native = json.loads((run / "native-physics-receipt.json").read_text())
assert native["completed"] and native["steps_completed"] == native["steps_requested"]
rows = []
for path in sorted(run.glob("*.nc")):
    if not path.name.startswith(("output.", "diagnostics.", "surface.", "cuda-history.")):
        continue
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        for name, var in ds.variables.items():
            a = np.asarray(var[:])
            if a.dtype.hasobject:
                raise TypeError(f"Object-valued variable cannot be a raw-word reference: {path.name}/{name}")
            rows.append(dict(file=path.name, variable=name, shape=list(a.shape), dtype=str(a.dtype),
                             words=int(a.size), bytes=int(a.nbytes), right_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(a))).hexdigest()))
assert rows
r = dict(started_utc=start, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         wall_seconds=time.monotonic()-tick, parallelism="CPU1 plus disk, no GPU", run=str(run),
         scope="Stored output/diagnostics/surface variables; reference fingerprints, not an identity proof or all-native-state claim",
         completed=True, steps=native["steps_completed"], arrays=len(rows),
         words=sum(x["words"] for x in rows), bytes=sum(x["bytes"] for x in rows), variables=rows)
destination.write_text(json.dumps(r, indent=2)+"\n")
print(json.dumps({k:v for k,v in r.items() if k != "variables"}))
