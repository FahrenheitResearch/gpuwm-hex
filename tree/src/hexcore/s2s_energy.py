"""S2S energy orchestration. Rust owns all meteorological data processing.

Rows are illustrative conversion proxies. Historical anomalies require a dated,
matching energy climatology; absence stays unavailable, never ensemble-relative.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from .s2s_door import S2SError, cores, write_json

TABLE = Path(__file__).parent / "data" / "s2s-energy.v1.json"
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def energy_config(args):
    """Offline, content-addressed configuration, also part of restart identity."""
    path = Path(getattr(args, "energy_table", None) or TABLE)
    raw = path.read_bytes()
    table = json.loads(raw)
    if table.get("schema") != "gpuwm-hex.s2s-energy-rows/v1":
        raise S2SError("energy table must use gpuwm-hex.s2s-energy-rows/v1")
    rows = table.get("rows", [])
    if not rows or len({r.get("id") for r in rows}) != len(rows):
        raise S2SError("energy rows need unique ids and at least one row")
    for row in rows:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", row.get("id", "")):
            raise S2SError("energy row ids must be NetCDF-safe lowercase identifiers")
        if row.get("kind") not in ("wind", "ghi", "turbine", "pv"):
            raise S2SError("energy kind must be wind, ghi, turbine or pv")
        def finite(key, low=None, high=None):
            number = row.get(key)
            if (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
                    or (low is not None and number <= low) or (high is not None and number > high)):
                raise S2SError(f"energy row {row['id']} invalid {key}")
        finite("display_max", 0)
        if row.get("units") != {"wind": "m s-1", "ghi": "W m-2", "turbine": "MW", "pv": "MW"}[row["kind"]]:
            raise S2SError(f"energy row {row['id']} has incorrect units")
        if row["kind"] in ("turbine", "pv"):
            finite("rated_mw", 0)
        if row["kind"] == "turbine":
            finite("cut_out_ms", 0)
            curve = row.get("curve", [])
            if (len(curve) < 2 or any(not isinstance(p, list) or len(p) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
                    or p[0] < 0 or not 0 <= p[1] <= 1 for p in curve)
                    or any(b[0] <= a[0] for a, b in zip(curve, curve[1:]))
                    or curve[0] != [0, 0] or curve[-1][0] >= row["cut_out_ms"]):
                raise S2SError(f"energy row {row['id']} needs an ordered fractional power curve below cut-out")
        if row["kind"] == "pv":
            for key in ("dc_ac_ratio", "inverter_efficiency"):
                finite(key, 0, 1 if key == "inverter_efficiency" else None)
            finite("loss_fraction", -1e-12, 1)
            finite("noct_c", 20, 100)
            finite("temperature_coefficient_per_k", -0.02, 0)
    heights = getattr(args, "hub_heights", "80,100")
    try:
        heights = [float(v) for v in heights.split(",")]
    except (ValueError, AttributeError):
        raise S2SError("--hub-heights needs comma-separated heights in metres AGL") from None
    if not heights or any(not math.isfinite(v) or v <= 10 or v > 1000 for v in heights) or len(set(heights)) != len(heights):
        raise S2SError("--hub-heights must contain unique finite heights in (10,1000] metres AGL")
    if args.history_hours > 6:
        raise S2SError("S2S energy requires --history-hours 6; daily samples alias the solar cycle")
    normal = getattr(args, "energy_climatology", None)
    return {"schema": table["schema"], "rows": rows, "table_sha256": hashlib.sha256(raw).hexdigest(),
            "heights_m_agl": heights, "climatology": str(normal) if normal else None,
            "climatology_sha256": hashlib.sha256(Path(normal).read_bytes()).hexdigest() if normal else None,
            "sampling": "6-hour endpoint trapezoids of instantaneous fields; sampled proxy, not an hourly yield",
            "conversion": "per sample before temporal reduction; no density correction or plant calibration"}


def forcing_objects(surface):
    """Expand every dated forcing object from source metadata, without I/O."""
    from .init_sources import load_table
    row = next(r for r in load_table()["sources"] if r["id"] == surface["source"])
    cycle = datetime.fromisoformat(surface["forecast_cycle"]) if surface.get("forecast_cycle") else None
    series = row["surface_series"]["forecast"] if cycle else row["surface_series"]
    start, end = (datetime.strptime(surface[k], TIME_FORMAT) for k in ("start", "end"))
    cadence = series["interval_hours"]
    last = math.ceil((end - start).total_seconds() / (cadence * 3600))
    objects = []
    for at in range(last + 1):
        valid = start + timedelta(hours=at * cadence)
        source = cycle or valid
        lead = int((valid - source).total_seconds() / 3600)
        if source.hour % 6 or source.minute or source.second or lead < 0 or (cycle and lead > series["max_hours"]):
            raise S2SError("dated surface forecast does not cover the requested cycle/window")
        context = {"lead": lead}
        for prefix, date in (("", source), ("valid_", valid)):
            for key, fmt in (("yyyy", "%Y"), ("mm", "%m"), ("dd", "%d"), ("hh", "%H"),
                             ("yyyymmdd", "%Y%m%d"), ("yyyymmddhh", "%Y%m%d%H")):
                context[prefix + key] = date.strftime(fmt)
        for product in series["products"]:
            urls = product["url"]
            urls = [urls] if isinstance(urls, str) else urls
            objects.append({"valid": valid.strftime(TIME_FORMAT), "source_cycle": source.strftime(TIME_FORMAT),
                            "lead_hours": lead, "urls": [url.format(**context) for url in urls]})
    return objects


def probe_object(obj):
    attempts = []
    for url in obj["urls"]:
        try:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-15"})
            with urllib.request.urlopen(request, timeout=30) as response:
                prefix = response.read(16)
                match = re.fullmatch(r"bytes 0-15/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match or int(match[1]) < 20 or len(prefix) != 16 or prefix[:4] != b"GRIB":
                    raise ValueError("expected a 206 GRIB prefix and exact object Content-Range")
                return {**obj, "status": "available", "url": url, "bytes": int(match[1]),
                        "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "attempts": attempts}
        except (OSError, ValueError) as error:
            attempts.append({"url": url, "error": str(error)})
    return {**obj, "status": "missing", "attempts": attempts}


def check_forcing(value):
    started = time.monotonic()
    objects = forcing_objects(value["surface"])
    workers = min(cores(), 16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(probe_object, objects))
    missing = [r for r in rows if r["status"] != "available"]
    path = Path(value["paths"]["work"]) / "s2s-forcing-coverage.json"
    write_json(path, {"schema": "gpuwm-hex.s2s-forcing-coverage/v1", "status": "missing" if missing else "complete",
                      "objects": rows, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                      "seconds": time.monotonic() - started, "workers": workers,
                      "contract": "every scheduled object probed; GRIB prefix, not full payload validation"})
    if missing:
        raise S2SError(f"dated forcing missing {len(missing)}/{len(rows)} objects; first valid {missing[0]['valid']}; {path}")
    return {"receipt": str(path), "objects": len(rows)}


def build_energy_products(value, state, args, env):
    from .s2s_products import validate_history, windows
    from .engines import RENDERER, resolve
    from gpuwm.rustwx import run_ensemble_product_renderer
    config = value.get("energy") or energy_config(args)
    output = Path(value["paths"]["work"]) / "s2s-energy"
    output.mkdir(parents=True, exist_ok=True)
    binary = os.environ.get("GPUWM_HEX_RW_S2S_ENERGY") or shutil.which("rw_s2s_energy")
    if not binary:
        candidate = Path(__file__).parent / "libexec" / "rw_s2s_energy"
        if candidate.is_file():
            binary = str(candidate)
    if not binary:
        raise S2SError("rw_s2s_energy is missing; build tools/build_s2s_energy.sh ENGINE OUT and set GPUWM_HEX_RW_S2S_ENERGY")
    start = datetime.strptime(value["start_time"], TIME_FORMAT)
    manifest = {"config": config, "start": value["start_time"], "cadence_hours": args.history_hours,
                "days": args.days, "grid": value["paths"]["grid"], "output": str(output),
                "step_degrees": 1.0 if args.row == "global-120km" else 0.5,
                "windows": [{"name": name, "low": lo, "high": hi} for name, lo, hi in windows(args.days)],
                "members": [{"id": m["run_id"], "init": m["init"],
                    "histories": [{"valid": stamp.strftime(TIME_FORMAT), "path": str(path)}
                        for stamp, path in validate_history(m, state["attempts"], start, args.days, args.history_hours)]}
                    for m in value["members"]]}
    request = output / "request.json"
    write_json(request, manifest)
    started = time.monotonic()
    command = [binary, str(request)]
    log = output / "reduce.log"
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, **env})
    if result.returncode:
        raise S2SError(f"Rust energy reducer exited {result.returncode}; {log}")
    receipt = json.loads((output / "reduction-receipt.json").read_text())
    renderer = resolve(RENDERER)
    abi = subprocess.run([str(renderer), "--ensemble-products-abi"], capture_output=True, text=True, check=True).stdout
    if "s2s-energy-scales" not in abi:
        raise S2SError("renderer lacks s2s-energy-scales; apply docs/s2s/engine-patches/0002-s2s-energy-scales.patch and rebuild both renderers")
    rendered = []
    map_fields = [name for name in receipt["windows"][0]["field_summary"] if not name.endswith("_energy")]
    if config["climatology"]:
        map_fields += [name + "_anomaly" for name in list(map_fields)]
    for window in receipt["windows"]:
        frames, stdout = run_ensemble_product_renderer(renderer, Path(window["file"]),
            out_dir=output / window["name"], domain="global", fields=map_fields, products=("mean",),
            source_label=f"S2S {window['name']}; proxy; " +
                ("dated reference supplied" if config["climatology"] else "no historical normal"))
        (output / (window["name"] + ".render.log")).write_text(stdout)
        rendered.extend(map(str, frames))
    receipt.update(command=command, log=str(log), images=rendered, map_fields=map_fields, seconds=time.monotonic() - started,
                   parallelism={"backend": "Rust Rayon / gpuwm.rustwx renderer", "threads": cores()})
    write_json(output / "energy-receipt.json", receipt)
    return receipt
