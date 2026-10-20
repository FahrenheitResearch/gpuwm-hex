"""Run on an admitted GPU against retained native files; never recreate weather.

Example under the the reference GPU host OWNER wrapper:
python tree/tools/audit_s2s_surface.py --static STATIC --grid GRID \
 --series SERIES --source-receipt RECEIPT --history HISTORY --out n1out/ice-audit
Reads use the shipping Rust transport/record reader. Reductions require CUDA.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import time


def native_array(ds, name, *, word_audit=False):
    """Restore stored types after the Rust bridge's float64 numeric transport."""
    import numpy as np
    variable = ds.variables[name]
    dtype = np.dtype(variable.dtype)
    if word_audit:
        if dtype != np.dtype('float32'):
            raise ValueError(f"surface word audit requires stored float32 {name}, got {dtype}")
        # Compare original finite float words, without CF scale/mask transforms.
        variable.set_auto_maskandscale(False)
    values = variable[0] if variable.dimensions[0] == "Time" else variable[:]
    return np.asarray(values, dtype=dtype).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("static", "grid", "series", "source-receipt", "history", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from gpuwm.netcdf_bridge import open_dataset
    from hexcore.surface_series_reader import covering_reader
    from hexcore import s2s_surface_audit as audit
    started = time.perf_counter()

    with open_dataset(args.static) as ds:
        area = cp.asarray(native_array(ds, "areaCell"), dtype=cp.float64)
        ocean = cp.asarray(native_array(ds, "landmask")) == 0
    with open_dataset(args.grid) as ds:
        latitude = cp.asarray(native_array(ds, "latCell"), dtype=cp.float64)
        ids = cp.asarray(native_array(ds, "indexToCellID"))
    if area.shape != ids.shape or int(cp.unique(ids).size) != len(ids):
        raise ValueError("static/grid extent or unique cell IDs differ")
    files = sorted(args.history.glob("cuda-history.*.nc"))
    times = [datetime.strptime(p.name, "cuda-history.%Y-%m-%d_%H.%M.%S.nc") for p in files]
    if not times or any((b-a).total_seconds() != 86400 for a,b in zip(times[:-1], times[1:])):
        raise ValueError("audit requires the complete daily history schedule")
    source = json.loads(args.source_receipt.read_text())
    if source["source"] != "cfs" or source["n_cells"] != len(ids) or source["status"] != "complete":
        raise ValueError("audit requires the matched native CFS producer receipt")
    if (times[0] != datetime.fromisoformat(source["cycle"].replace("_", "T")) or
            times[-1] != datetime.fromisoformat(source["coverage_end"].replace("_", "T"))):
        raise ValueError("daily history does not span the complete producer/run window")
    provenance = {r["valid_time"].replace("_", "T"): r for r in source["records"]}
    reader = covering_reader(args.series, n_cells=len(ids), start=times[0], end=times[-1])

    def read_history(path):
        with open_dataset(path) as ds:
            required = {name: native_array(ds, name, word_audit=name in ("sst", "xice"))
                        for name in ("sst", "xice", "t2", "indexToCellID")}
            for name in ("xland", "ivgtyp", "isltyp", "albedo"):
                if name in ds.variables:
                    required[name] = native_array(ds, name)
            return required

    rows = []
    # Bound the queue to four frames. Do not accumulate raw histories.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {n: pool.submit(read_history, p) for n,p in enumerate(files[:4])}
        for at, (valid, path) in enumerate(zip(times, files)):
            data = pending.pop(at).result()
            if at + 4 < len(files):
                pending[at + 4] = pool.submit(read_history, files[at + 4])
            history_ids = cp.asarray(data.pop("indexToCellID"))
            order = cp.argsort(history_ids)
            if not bool(cp.array_equal(history_ids[order], cp.sort(ids))):
                raise ValueError("history cell IDs differ from the registered grid")
            gather = order[cp.searchsorted(history_ids[order], ids)]
            applied = {k: cp.asarray(v)[gather] for k,v in data.items()}
            reference = {name: cp.empty(len(ids), dtype=cp.float32) for name in ("sst", "xice")}
            reader(valid.isoformat(), reference)
            labels, device_values = audit.metrics(cp, area, latitude, ocean,
                applied.pop("sst"), applied.pop("xice"), reference["sst"], reference["xice"],
                applied.pop("t2"), **applied)
            values = cp.asnumpy(device_values)  # One compact scalar-summary copy per day.
            record = provenance[reader.times[reader.last_record].isoformat()]
            row = dict(day=(valid-times[0]).days, valid_time=valid.isoformat(),
                source_cycle=record["source_cycle"], source_lead_hours=record["lead_hours"],
                **{k: float(v) if math.isfinite(v) else None for k,v in zip(labels,values)})
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "daily.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(rows=rows, wall_seconds=time.perf_counter()-started,
        parallelism=dict(history_decode_threads=4, reductions="GPU", cuda_graphs=False),
        source_receipt_sha256=hashlib.sha256(args.source_receipt.read_bytes()).hexdigest(),
        definitions=dict(poles="ocean cells north of 60N or south of 60S",
            means="ocean cell-area weighted; model T2 uses ice-fraction times cell area",
            source="remapped CFS producer values; not an independent GRIB decode",
            series_order="producer native order of registered static; no count-only identity claim"),
        classification_available=all("water_ice_classification_mismatches" in r for r in rows),
        albedo_available=all("ice_albedo_outside_0_1" in r for r in rows),
        reader=reader.reads)
    (args.out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
