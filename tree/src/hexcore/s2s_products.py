"""Bounded-memory S2S reductions on CUDA, Rust NetCDF transport and maps.

Endpoint trapezoids describe the available daily/6-hourly diagnostic samples,
not hidden subdaily physics averages. Rain uses differences of model cumulative
rainc + rainnc. A missing time or member refuses the product.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import urllib.request

from .s2s_door import S2SError, cores, write_json
from .s2s_terrain import LAPSE_RATE_K_PER_M, normal_orography, temperature_on_model_terrain

FIELDS = {"temperature2": "K", "precipitation_rate": "mm day-1", "z500": "m", "mslp": "hPa"}
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def windows(days):
    result = [(f"week-{week}", (week - 1) * 7, week * 7) for week in range(1, days // 7 + 1)]
    result += [(name, low, high) for name, low, high in
               (("weeks-1-2", 0, 14), ("weeks-3-4", 14, 28), ("weeks-5-6", 28, 42)) if high <= days]
    return result


def history_index(member, attempts):
    """Newest attempt wins at restart overlap; ignore unfinished writer files."""
    result = {}
    for attempt in attempts:
        if member["run_id"] not in attempt["members"]:
            continue
        root = Path(attempt["out"]) / member["run_id"]
        for path in root.glob("cuda-history.*.nc"):
            try:
                valid = datetime.strptime(path.name, "cuda-history.%Y-%m-%d_%H.%M.%S.nc")
            except ValueError:
                continue
            # The history writer atomically installs the final .nc, then a
            # durable sidecar. A final name is never the in-flight .partial.
            if path.stat().st_size:
                result[valid] = path
    return result


def validate_history(member, attempts, start, days, cadence):
    index = history_index(member, attempts)
    expected = [start + timedelta(hours=h) for h in range(0, days * 24 + 1, cadence)]
    missing = [time.strftime(TIME_FORMAT) for time in expected if time not in index]
    if missing:
        raise S2SError(f"{member['run_id']} missing {len(missing)} scheduled histories: {', '.join(missing[:4])}")
    return [(time, index[time]) for time in expected]


def interp_log_pressure(xp, pressure, field, target=50000):
    """Column-major bottom-up interpolation; below-ground columns stay NaN."""
    above = pressure < target
    upper = xp.argmax(above, axis=1)
    valid = above.any(axis=1) & (upper > 0)
    upper = xp.where(valid, upper, 1)
    lower = upper - 1
    rows = xp.arange(pressure.shape[0])
    p0, p1 = pressure[rows, lower], pressure[rows, upper]
    weight = (xp.log(xp.asarray(target, dtype=pressure.dtype)) - xp.log(p0)) / (xp.log(p1) - xp.log(p0))
    return xp.where(valid, field[rows, lower] + weight * (field[rows, upper] - field[rows, lower]), xp.nan)


def ensemble_statistics(xp, members, baseline=None):
    """A fixed roster denominator; mask any point with a missing member."""
    complete = xp.isfinite(members).all(axis=0)
    mean = members.mean(axis=0)
    baseline = mean if baseline is None else xp.asarray(baseline)
    comparable = complete & xp.isfinite(baseline)
    anomaly = members - baseline
    return {"mean": xp.where(complete, mean, xp.nan),
            "spread": xp.where(complete, members.std(axis=0, ddof=0), xp.nan),
            "anomaly": xp.where(comparable, anomaly.mean(axis=0), xp.nan),
            "anomaly_spread": xp.where(comparable, members.std(axis=0, ddof=0), xp.nan),
            "anomaly_finite_count": xp.where(xp.isfinite(baseline), xp.isfinite(members).sum(axis=0), 0),
            "probability": xp.where(comparable, (anomaly > 0).sum(axis=0) / members.shape[0], xp.nan),
            "finite_count": xp.isfinite(members).sum(axis=0)}


def integrate_window(xp, samples, cadence_hours, days):
    """Trapezoidal means and mean daily precipitation from exact endpoints."""
    if len(samples) != int(days * 24 / cadence_hours) + 1:
        raise S2SError("window samples do not cover its exact endpoints")
    result = {}
    for field in FIELDS:
        if field == "precipitation_rate":
            increments = xp.stack([b[field] - a[field] for a, b in zip(samples[:-1], samples[1:])])
            if bool((increments < -1e-3).any()):
                raise S2SError("cumulative precipitation decreased; restart/reset cannot be treated as rain")
            result[field] = xp.maximum(increments, 0).sum(axis=0) / days
        else:
            # Each interval has the same length; endpoints carry half weight.
            total = samples[0][field] * 0.5 + samples[-1][field] * 0.5
            for sample in samples[1:-1]:
                total = total + sample[field]
            result[field] = total / (len(samples) - 1)
    return result


def read_native(path):
    from gpuwm.netcdf_bridge import open_dataset
    names = ("t2", "rainc", "rainnc", "pressure", "theta", "qv")
    with open_dataset(path) as ds:
        return {name: ds.variables[name][0] for name in names}


def native_diagnostics(cp, data, heights):
    from .mpasdiag.isobaric import compute_slp
    pressure = cp.ascontiguousarray(cp.asarray(data["pressure"], cp.float32))
    temperature = cp.asarray(data["theta"], cp.float32) * (pressure / 100000) ** (287.0 / 1004.5)
    middle = (heights[:, 1:] + heights[:, :-1]) * 0.5
    return {"temperature2": cp.asarray(data["t2"], cp.float32),
            "precipitation_rate": cp.asarray(data["rainc"], cp.float32) + cp.asarray(data["rainnc"], cp.float32),
            "z500": interp_log_pressure(cp, pressure, middle),
            "mslp": compute_slp(cp.ascontiguousarray(pressure / 100), cp.ascontiguousarray(temperature),
                                 heights, cp.ascontiguousarray(cp.asarray(data["qv"], cp.float32)))}


def product_grid(cp, latitude, longitude, step):
    """One GPU nearest-cell remap, reused for every reduced plane.

    Chunk the dot products so even a 60 km globe never builds a multi-GB
    target-by-source matrix. Native fields are reduced before this remap.
    """
    lat = cp.asarray(latitude, cp.float32)
    lon = cp.asarray(longitude, cp.float32)
    source = cp.stack((cp.cos(lat) * cp.cos(lon), cp.cos(lat) * cp.sin(lon), cp.sin(lat)), axis=0)
    ylats = cp.linspace(-90, 90, round(180 / step) + 1, dtype=cp.float32)
    xlons = cp.arange(round(360 / step), dtype=cp.float32) * step - 180
    yy, xx = cp.meshgrid(ylats, xlons, indexing="ij")
    a, b = cp.deg2rad(yy.ravel()), cp.deg2rad(xx.ravel())
    target = cp.stack((cp.cos(a) * cp.cos(b), cp.cos(a) * cp.sin(b), cp.sin(a)), axis=1)
    index = cp.empty(target.shape[0], cp.int64)
    for at in range(0, target.shape[0], 256):
        index[at:at + 256] = cp.argmax(target[at:at + 256] @ source, axis=1)
    return yy, xx, index


def climatology_path(value, args, env, latitude=None, longitude=None):
    if args.climatology:
        return args.climatology, "ERA5 supplied window climatology"
    url = os.environ.get("GPUWM_HEX_S2S_ERA5_CLIMATOLOGY_URL")
    if url:
        path = Path(value["paths"]["work"]) / "era5-window-climatology.nc"
        if not path.is_file():
            temporary = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1 << 20):
                    stream.write(block)
            temporary.replace(path)
        return path, "ERA5 fetched window climatology"
    from .s2s_climatology import fetch_normals
    try:
        return fetch_normals(value, args, latitude, longitude)
    except (OSError, ValueError, KeyError, ImportError) as error:
        write_json(Path(value["paths"]["work"]) / "era5-climatology-unavailable.json",
                   {"error": type(error).__name__ + ": " + str(error),
                    "fallback": "ensemble-relative baseline on model terrain; not a historical normal"})
        return None, "ensemble-relative baseline; public ERA5 climatology unavailable; not a historical normal"


def write_aggregate(path, cp, latitude, longitude, fields, count, valid, window, baseline_label,
                    member_fields=None, member_ids=None, terrain_fields=None):
    import numpy as np
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    schema = ClassicSchema("cdf5")
    y = schema.def_dim("south_north", latitude.shape[0])
    x = schema.def_dim("west_east", latitude.shape[1])
    payloads = []
    def variable(name, units, data, dims=(y, x)):
        number = schema.def_var(name, "f4", dims)
        schema.put_var_attr(number, "units", units)
        payloads.append((number, np.ascontiguousarray(cp.asnumpy(data), dtype=np.float32)))
        return number
    schema.put_global_attr("ensemble_contract", "gpuwm-resident-ensemble-products.v1")
    schema.put_global_attr("probability_scale", "fraction")
    schema.put_global_attr("ensemble_members", count, dtype="i4")
    schema.put_global_attr("valid_time", valid)
    schema.put_global_attr("s2s_window", window)
    schema.put_global_attr("s2s_baseline", baseline_label)
    if member_ids is not None:
        schema.put_global_attr("s2s_member_ids", json.dumps(member_ids))
    schema.put_global_attr("s2s_sampling", "endpoint trapezoids; cumulative precipitation differences in mm/day")
    variable("XLAT", "degrees_north", latitude)
    variable("XLONG", "degrees_east", longitude)
    if terrain_fields is not None:
        schema.put_global_attr("s2s_temperature2_terrain_correction",
                               "normal - 0.0065 K/m * (model_orography - climatology_orography)"
                               if "climatology_orography" in terrain_fields else
                               "none; ensemble-relative baseline already on model terrain")
        for name, data in terrain_fields.items():
            variable(name, "K" if name == "temperature2_baseline" else "m", data)
    for field, stats in fields.items():
        units = FIELDS[field]
        if member_fields is not None:
            member_dim = schema.def_dim(field + "_member", count)
            variable(field + "_members", units, member_fields[field], (member_dim, y, x))
        for kind in ("mean", "spread", "finite_count"):
            variable(f"{field}_{kind}", units if kind != "finite_count" else "1", stats[kind])
        anomaly = field + "_anomaly"
        variable(anomaly + "_mean", units, stats["anomaly"])
        variable(anomaly + "_spread", units, stats["anomaly_spread"])
        variable(anomaly + "_finite_count", "1", stats["anomaly_finite_count"])
        threshold_dim = schema.def_dim(anomaly + "_threshold", 1)
        variable(anomaly + "_thresholds", units, cp.zeros(1), (threshold_dim,))
        number = variable(anomaly + "_probability", "1", stats["probability"][None], (threshold_dim, y, x))
        schema.put_var_attr(number, "comparison", "gt")
    with schema.create(path) as writer:
        for number, payload in payloads:
            writer.write_var(number, payload)


def build_products(value, state, args, env):
    import cupy as cp
    from gpuwm.netcdf_bridge import open_dataset
    from .engines import RENDERER, resolve
    start = datetime.strptime(value["start_time"], TIME_FORMAT)
    cadence = args.history_hours
    output = Path(value["paths"]["work"]) / "s2s-products"
    output.mkdir(parents=True, exist_ok=True)
    product_windows = windows(args.days)
    # Initialize immutable geometry once, then bound live diagnostics to the
    # longest requested window (14 days), independently of forecast length.
    with open_dataset(value["members"][0]["init"]) as ds:
        height_array = ds.variables["zgrid"][:]
    if height_array.ndim == 3:
        height_array = height_array[0]
    heights = cp.ascontiguousarray(cp.asarray(height_array, cp.float32))
    with open_dataset(value["paths"]["grid"]) as ds:
        latitude, longitude = ds.variables["latCell"][:], ds.variables["lonCell"][:]
    lat, lon, index = product_grid(cp, latitude, longitude, 1 if args.row == "global-120km" else 0.5)
    shape = lat.shape
    model_orography = heights[:, 0][index].reshape(shape)
    climatology, baseline_label = climatology_path(value, args, env, lat, lon)
    if climatology:
        baseline_label += "; T2 normal adjusted to native model terrain at 6.5 K/km"
    reduced = {name: {field: [] for field in FIELDS} for name, _, _ in product_windows}
    coverage = []
    for member in value["members"]:
        with open_dataset(member["init"]) as ds:
            member_grid = ds.variables["zgrid"][:]
        if member_grid.ndim == 3:
            member_grid = member_grid[0]
        if (member_grid.shape != height_array.shape or not bool(cp.array_equal(
                cp.asarray(member_grid[:, 0], cp.float32), heights[:, 0]))):
            raise S2SError("ensemble members must share the model terrain used for T2 anomalies")
        schedule = validate_history(member, state["attempts"], start, args.days, cadence)
        cache = {}
        # Rust decoding runs ahead on several host cores; GPU reductions run
        # on one card. No CPU fallback is admitted for meteorological arrays.
        with ThreadPoolExecutor(max_workers=min(cores(), 4)) as pool:
            futures = {}
            for at in range(min(4, len(schedule))):
                futures[at] = pool.submit(read_native, schedule[at][1])
            for at, (valid, path) in enumerate(schedule):
                data = futures.pop(at).result()
                if at + 4 < len(schedule):
                    futures[at + 4] = pool.submit(read_native, schedule[at + 4][1])
                cache[at] = native_diagnostics(cp, data, heights)
                day = (valid - start).total_seconds() / 86400
                for name, low, high in product_windows:
                    if day == high:
                        first = int(low * 24 / cadence)
                        samples = [cache[n] for n in range(first, at + 1)]
                        products = integrate_window(cp, samples, cadence, high - low)
                        for field, plane in products.items():
                            reduced[name][field].append(plane[index].reshape(shape))
                # Keep only endpoints needed by unfinished windows.
                needed = [int(low * 24 / cadence) for _, low, high in product_windows if high > day]
                floor = min(needed, default=at + 1)
                cache = {n: data for n, data in cache.items() if n >= floor}
        coverage.append({"member": member["run_id"], "frames": len(schedule),
                         "first": schedule[0][0].isoformat(), "last": schedule[-1][0].isoformat()})
    renderer = resolve(RENDERER)
    abi = subprocess.run([str(renderer), "--ensemble-products-abi"], capture_output=True, text=True, check=True).stdout
    missing = [token for token in ("s2s-window", "global-ensemble-projection") if token not in abi]
    if missing:
        raise S2SError("rw_wrfbatch lacks " + ", ".join(missing)
                       + "; install both accompanying engine patches before rendering")
    products_receipt = []
    for name, low, high in product_windows:
        fields = {}
        member_fields = {}
        baseline_fields = {}
        terrain_fields = None
        if climatology:
            with open_dataset(climatology) as ds:
                if str(ds.getncattr("s2s_climatology_source")).upper() != "ERA5":
                    raise S2SError("climatology must declare s2s_climatology_source=ERA5")
                if ds.getncattr("s2s_climatology_start_mmdd") != start.strftime("%m-%d"):
                    raise S2SError("ERA5 climatology start month/day differs from the forecast window")
                for coordinate, expected in (("XLAT", lat), ("XLONG", lon)):
                    values = cp.asarray(ds.variables[coordinate][:])
                    if values.shape != shape or not bool(cp.allclose(values, expected, atol=1e-5, rtol=0)):
                        raise S2SError("ERA5 climatology coordinates differ from the product grid")
                try:
                    source_orography = normal_orography(cp, ds, shape)
                except ValueError as error:
                    raise S2SError(str(error)) from error
                for field in FIELDS:
                    key = name.replace("-", "_") + "_" + field
                    variable = ds.variables[key]
                    if variable.getncattr("units") != FIELDS[field]:
                        raise S2SError(f"ERA5 climatology {key} units must be {FIELDS[field]}")
                    array = variable[:]
                    if array.shape != shape:
                        raise S2SError(f"ERA5 climatology {key} grid shape differs from product grid")
                    baseline_fields[field] = cp.asarray(array)
                try:
                    baseline_fields["temperature2"] = temperature_on_model_terrain(cp,
                        baseline_fields["temperature2"], model_orography, source_orography)
                except ValueError as error:
                    raise S2SError(str(error)) from error
                terrain_fields = dict(model_orography=model_orography,
                    climatology_orography=source_orography,
                    temperature2_baseline=baseline_fields["temperature2"])
        for field in FIELDS:
            member_fields[field] = cp.stack(reduced[name][field])
            fields[field] = ensemble_statistics(cp, member_fields[field], baseline_fields.get(field))
        if terrain_fields is None:
            terrain_fields = dict(model_orography=model_orography,
                temperature2_baseline=fields["temperature2"]["mean"])
        path = output / f"{name}.nc"
        write_aggregate(path, cp, lat, lon, fields, len(value["members"]),
                        (start + timedelta(days=high)).strftime(TIME_FORMAT), name, baseline_label,
                        member_fields, [member["run_id"] for member in value["members"]], terrain_fields)
        images = output / name
        argv = [str(renderer), "--ensemble-products", str(path), "--out-dir", str(images),
                "--products", "mean,spread,prob", "--domain", "global",
                "--source-label", f"S2S {name} days {low}-{high}; {baseline_label}"]
        log = output / f"{name}.render.log"
        with log.open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, **env})
        if result.returncode:
            raise S2SError(f"Rust renderer exited {result.returncode}; {log}")
        metrics = {}
        for field, stats in fields.items():
            metrics[field] = {}
            for kind in ("mean", "spread", "anomaly", "probability"):
                plane = stats[kind]
                finite = cp.isfinite(plane)
                count = int(finite.sum())
                metrics[field][kind] = {"finite": count, "cells": int(plane.size),
                    "min": float(cp.min(plane[finite])) if count else None,
                    "max": float(cp.max(plane[finite])) if count else None,
                    "units": "1" if kind == "probability" else FIELDS[field]}
        products_receipt.append({"window": name, "days": [low, high], "file": str(path),
                                 "bytes": path.stat().st_size, "field_summary": metrics,
                                 "renderer_argv": argv, "log": str(log), "images": len(list(images.rglob('*.png')))})
    result = {"temperature2_terrain_correction": {"applied": bool(climatology),
              "lapse_rate_K_per_m": LAPSE_RATE_K_PER_M if climatology else 0.,
              "model_terrain": "first native zgrid interface, same nearest cell as T2",
              "source_terrain": "ERA5 orography variable on product grid" if climatology else "same model terrain",
              "convention": "fixed lapse adjustment of normal; local inversions are not diagnosed"},
              "windows": products_receipt, "baseline": baseline_label, "coverage": coverage,
              "parallelism": {"backend": "GPU reductions / Rust renderer", "reduction_gpus": 1,
                              "history_reader_threads": min(cores(), 4), "renderer_threads": cores()},
              "unavailable_windows": [name for name, _, end in (("weeks-5-6", 28, 42),) if end > args.days],
              "reduction_device": int(cp.cuda.Device().id), "sampling": "endpoint trapezoids",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
              "spread": "population standard deviation of complete member window means",
              "regrid": "nearest native cell on the sphere, GPU, after native-grid temporal reduction"}
    write_json(output / "products-receipt.json", result)
    return result


def verify_first_ten_days(value, state, args, env):
    """Run the existing observation scorer with explicit daily coverage limits."""
    from .global_ensemble import _cli
    work = Path(value["paths"]["work"])
    config = {**value["score_config"], "verification_end_time":
              (datetime.strptime(value["start_time"], TIME_FORMAT) + timedelta(days=10)).strftime(TIME_FORMAT)}
    workers = min(cores(), len(value["members"]))
    threads = max(1, cores() // workers)
    def score(member):
        complete = [a for a in state["attempts"] if a["state"] == "complete" and member["run_id"] in a["members"]]
        if not complete:
            raise S2SError(f"no completed forecast for verification: {member['run_id']}")
        # A resumed attempt does not contain the pre-checkpoint history. Build
        # a small link view over every attempt, with the same overlap policy as
        # the products. Explicit provenance keeps lagged members on the common
        # verification clock rather than silently shifting the selected leads.
        root = work / "members" / member["run_id"]
        out = root / "verification-history"
        out.mkdir(parents=True, exist_ok=True)
        start = datetime.strptime(value["start_time"], TIME_FORMAT)
        end = start + timedelta(days=10)
        selected = {valid: source for valid, source in history_index(member, state["attempts"]).items()
                    if start <= valid <= end}
        for source in selected.values():
            target = out / source.name
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            temporary = target.with_suffix(".link-part")
            # A stop between symlink creation and atomic replacement leaves
            # this private link behind, including when its source is gone.
            # Remove only a symlink; never consume an unexpected regular file.
            if temporary.is_symlink():
                temporary.unlink()
            temporary.symlink_to(source.absolute())
            temporary.replace(target)
        path = root / "verification.config.json"
        write_json(path, {**config, "start": value["start_time"], "init": member["init"],
                          "grid": value["paths"]["grid"]})
        log = root / "verification.log"
        rc = _cli(["score", str(out), "--config", str(path), "--leads", ",".join(str(h) for h in range(24, 241, 24))], log,
                  {**env, "RAYON_NUM_THREADS": str(threads), "OMP_NUM_THREADS": str(threads)})
        if rc not in (0, 3):
            raise S2SError(f"verification exited {rc}; {log}")
        return {"member": member["run_id"], "returncode": rc, "log": str(log), "config": str(path),
                "history": str(out), "frames": len(selected),
                "coverage": "inspect scorecard; exit 3 means incomplete analysis coverage"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(score, value["members"]))
    from .s2s_analysis import verify_analyses
    analysis = verify_analyses(value, state, args, env)
    return {"members": result, "requested_days": 10,
            "parallelism": {"score_processes": workers, "threads_per_score_process": threads,
                            "analysis": analysis.get("parallelism", {})},
            "analysis_verification": analysis,
            "era5_coverage": "prepared analyses only; missing dates recorded; no credential access"}
