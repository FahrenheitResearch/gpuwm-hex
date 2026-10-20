"""Render hex history frames as they land: native mesh cells, Rust renderer only.

usage: render_native_mesh.py RUN_DIR INIT_NC GRID_NC OUT_ROOT --label NAME [--start YYYY-mm-dd_HH]
                     [--once] [--workers N] [--region-every H]

Each cuda-history.<valid>.nc in RUN_DIR/fc is turned into one small frame of
2-D diagnostics on the SAME cells (no regrid): 500 hPa height, 250 hPa wind
speed, 2 m temperature, sea-level pressure, 10 m wind speed and 6 h
precipitation.  Every map is drawn by rw_wrfbatch's mesh lane, resolved and
environment-configured by gpuwm.rustwx (the render law).  The draw-grid seam
fix, region subsets and clipped basemaps are the HEX-GLOBAL-PLOTS lane's
workarounds for two renderer defects (antimeridian rings; unclipped basemap
polylines), applied here unchanged in method.

Diagnostics (numpy, host side):
- height_500hPa / wind_speed_250hPa: linear in ln p between the two model
  levels that bracket the surface, heights at layer midpoints from the init's
  zgrid (the forecast's own height coordinate);
- sea_level_pressure: lowest model level reduced with a 6.5 K/km lapse rate
  (T from theta and pressure at that level);
- precipitation_6h: rainc + rainnc difference from the previous history frame.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from netCDF4 import Dataset

if os.environ.get('E'):  # an engine source checkout to import gpuwm from, else the installed engine
    sys.path.insert(0, os.environ['E'])
from gpuwm import rustwx  # noqa: E402

G0, RD, CP, LAPSE = 9.80665, 287.04, 1004.5, 0.0065
KT = 1.943844

PRODUCTS = {
    "z500": ("mesh:height_500hPa@474..609", "500 hPa height (dam): troughs cool colours, ridges warm"),
    "jet250": ("mesh:wind_speed_250hPa@40..184", "250 hPa wind speed (kt): the jet stream, below 40 kt left white"),
    "t2m": ("mesh:temperature_2m@-50..50", "2 m air temperature (C)"),
    "mslp": ("mesh:sea_level_pressure@96..104.4", "Sea-level pressure (kPa; 100 kPa = 1000 mb)"),
    "wind10m": ("mesh:wind_speed_10m@5..80", "10 m wind speed (kt), below 5 kt left white"),
    "precip6h": ("mesh:precipitation_6h@0.25..75", "Precipitation in the 6 h ending at valid time (mm), below 0.25 mm left white"),
}
PALETTES = {
    "z500": ["#2b1a5a", "#3b4cc0", "#4f8fd6", "#72c6d9", "#bfe8c5", "#f4e59a", "#f6b25c", "#e3632f", "#a8211f"],
    "jet250": ["#9fd3f5", "#4f9fe0", "#3456c7", "#6b3fb8", "#a53fa8", "#d6408a", "#ef5b52", "#f79a3b", "#fbd85a"],
    "t2m": ["#5b2a86", "#3a4fc4", "#4c8fe0", "#9fd0f2", "#f2f2f2", "#f6d58a", "#f19a4a", "#d8452d", "#8c1a1a"],
    "mslp": ["#3b1f6e", "#3d58c7", "#56a3e3", "#a9dcef", "#f0f0e8", "#f7d79a", "#f2a35c", "#d9582f", "#9a2a1c"],
    "wind10m": ["#d8eef8", "#9fd3f5", "#4f9fe0", "#2f9e8f", "#7cc24a", "#f2d43a", "#f2953a", "#e0482a", "#a8202a"],
    "precip6h": ["#dff2d8", "#a8dca0", "#4fb05a", "#1f7a3a", "#f2e35a", "#f2a33a", "#e0482a", "#a8202a", "#8a3ab8"],
}
UNITS = {"height_500hPa": "dam", "wind_speed_250hPa": "kt", "temperature_2m": "degC",
         "sea_level_pressure": "kPa", "wind_speed_10m": "kt", "precipitation_6h": "mm"}
LONG = {"height_500hPa": "height of the 500 hPa surface", "wind_speed_250hPa": "wind speed at 250 hPa",
        "temperature_2m": "2 m air temperature", "sea_level_pressure": "sea-level pressure",
        "wind_speed_10m": "10 m wind speed", "precipitation_6h": "6 h accumulated precipitation"}
REGIONS = {  # W, E, S, N
    "north-america": (-170.0, -50.0, 10.0, 75.0),
    "europe": (-25.0, 45.0, 30.0, 72.0),
    "east-asia": (70.0, 150.0, 5.0, 60.0),
}


def slug(token):
    body = token.split(":", 1)[1]
    field, _, rng = body.partition("@")
    s = "mesh_" + re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")
    if rng:
        lo, _, hi = rng.partition("..")
        safe = lambda v: re.sub(r"[^a-z0-9]+", "_", v.lower()).strip("_")
        s += "_from" + safe(lo)
        if hi:
            s += "_to" + safe(hi)
    return s


def write_theme(path, source_label, global_frame):
    theme = {
        "name": "hex-ops",
        "linework": {"county_visible": False, "coast": "#15191e", "international": "#3a4048",
                     "state": "#9aa2ac00" if global_frame else "#9aa2ac66", "lake": "#3a4048"},
        "mesh": {"edges": False, "empty": "#ffffff"},
        "colormaps": {"products": {slug(PRODUCTS[k][0]): PALETTES[k] for k in PRODUCTS}},
        "text": {"source_label": source_label},
        "footer": {"height": 58, "background": "#f4f6f8", "rule": "#c9ced6", "title": "{product_title}",
                   "caption": "valid {valid_time} | {note} | {mesh_or_grid}",
                   "title_ink": "#15191e", "ink": "#4a525c"},
    }
    Path(path).write_text(json.dumps(theme, indent=1))


# ---------------------------------------------------------------- geometry (once)
def seamfix(src, out):
    """HEX-GLOBAL-PLOTS seamfix_grid.py, unchanged in method: clip rings at 180 E."""
    LIM = 179.9999
    with Dataset(src) as ds:
        latC = np.asarray(ds["latCell"][:]); lonC = np.asarray(ds["lonCell"][:])
        latV = np.degrees(np.asarray(ds["latVertex"][:])); lonV = np.degrees(np.asarray(ds["lonVertex"][:]))
        voc = np.asarray(ds["verticesOnCell"][:]); ne = np.asarray(ds["nEdgesOnCell"][:])
        area = np.asarray(ds["areaCell"][:])
    n, me = voc.shape
    c = ((np.degrees(lonC) + 180) % 360) - 180
    mask = np.arange(me)[None, :] < ne[:, None]
    idx = np.where(mask, voc - 1, 0)
    d = ((lonV[idx] - c[:, None] + 180) % 360) - 180
    ul = np.where(mask, c[:, None] + d, np.nan)
    span = np.nanmax(ul, 1) - np.nanmin(ul, 1)
    cross = (np.nanmax(ul, 1) > LIM) | (np.nanmin(ul, 1) < -LIM)
    polar = span > 180.0
    new_lat = list(latV); new_lon = list(lonV)
    ME = me + 2
    voc2 = np.zeros((n, ME), np.int32); voc2[:, :me] = voc
    ne2 = ne.copy()
    for i in np.where(cross | polar)[0]:
        k = ne[i]
        lats = latV[voc[i, :k] - 1]
        if polar[i]:
            lons = ((lonV[voc[i, :k] - 1] + 180) % 360) - 180
            o = np.argsort(lons)
            lons = np.clip(lons[o], -LIM, LIM); lats = lats[o]
            pole = 90.0 if np.degrees(latC[i]) > 0 else -90.0
            ring_lat = list(lats) + [pole, pole]; ring_lon = list(lons) + [LIM, -LIM]
        else:
            ring_lat = list(lats); ring_lon = list(np.clip(ul[i, :k], -LIM, LIM))
        start = len(new_lat)
        new_lat.extend(ring_lat); new_lon.extend(ring_lon)
        m = len(ring_lat)
        voc2[i, :] = 0; voc2[i, :m] = np.arange(start + 1, start + m + 1); ne2[i] = m
    lonC = lonC.copy(); lonC[polar] = 0.0
    with Dataset(out, "w", format="NETCDF4") as ds:
        ds.createDimension("nCells", n); ds.createDimension("nVertices", len(new_lat))
        ds.createDimension("maxEdges", ME)
        ds.sphere_radius = 1.0
        ds.drawing_variant = "antimeridian-clipped rings for a global geographic frame; values unaffected"
        for vname, data, dim in (("latCell", latC, "nCells"), ("lonCell", lonC, "nCells"),
                                 ("areaCell", area, "nCells"),
                                 ("latVertex", np.radians(np.array(new_lat)), "nVertices"),
                                 ("lonVertex", np.radians(np.array(new_lon)), "nVertices")):
            v = ds.createVariable(vname, "f8", (dim,), zlib=True, complevel=1); v[:] = data
        v = ds.createVariable("nEdgesOnCell", "i4", ("nCells",)); v[:] = ne2
        v = ds.createVariable("verticesOnCell", "i4", ("nCells", "maxEdges")); v[:] = voc2
    return int(cross.sum()), int(polar.sum())


def subset_grid(src, out, box):
    """HEX-GLOBAL-PLOTS subset_mesh.py: the same cells whose centres fall in the box."""
    W, E, S, N = box
    with Dataset(src) as ds:
        latC = np.asarray(ds["latCell"][:]); lonC = np.asarray(ds["lonCell"][:])
        latV = np.asarray(ds["latVertex"][:]); lonV = np.asarray(ds["lonVertex"][:])
        voc = np.asarray(ds["verticesOnCell"][:]); ne = np.asarray(ds["nEdgesOnCell"][:])
        area = np.asarray(ds["areaCell"][:])
    lat = np.degrees(latC); lon = ((np.degrees(lonC) + 180) % 360) - 180
    sel = (lat >= S) & (lat <= N) & (((lon >= W) & (lon <= E)) if W <= E else ((lon >= W) | (lon <= E)))
    keep = np.where(sel)[0]
    sub_voc = voc[keep]
    mask = np.arange(voc.shape[1])[None, :] < ne[keep][:, None]
    used = np.unique(sub_voc[mask])
    remap = np.zeros(latV.size + 1, np.int32); remap[used] = np.arange(1, used.size + 1)
    new_voc = np.where(mask, remap[sub_voc], 0).astype(np.int32)
    with Dataset(out, "w", format="NETCDF4") as ds:
        ds.createDimension("nCells", keep.size); ds.createDimension("nVertices", used.size)
        ds.createDimension("maxEdges", voc.shape[1])
        ds.sphere_radius = 1.0
        for vname, data, dim in (("latCell", latC[keep], "nCells"), ("lonCell", lonC[keep], "nCells"),
                                 ("areaCell", area[keep], "nCells"), ("latVertex", latV[used - 1], "nVertices"),
                                 ("lonVertex", lonV[used - 1], "nVertices")):
            v = ds.createVariable(vname, "f8", (dim,), zlib=True, complevel=1); v[:] = data
        v = ds.createVariable("nEdgesOnCell", "i4", ("nCells",)); v[:] = ne[keep]
        v = ds.createVariable("verticesOnCell", "i4", ("nCells", "maxEdges")); v[:] = new_voc
    return keep


def clip_basemap(src, out, box):
    """HEX-GLOBAL-PLOTS clip_basemap.py: the renderer's own Natural Earth lines cut to the box."""
    import shapefile
    W, E, S, N = box

    def inside(pts):
        lon = pts[:, 0]; lat = pts[:, 1]
        return (lat >= S) & (lat <= N) & (((lon >= W) & (lon <= E)) if W <= E else ((lon >= W) | (lon <= E)))

    for sub in os.listdir(src):
        sdir = os.path.join(src, sub)
        if not os.path.isdir(sdir):
            if os.path.isfile(sdir):
                shutil.copy(sdir, os.path.join(out, sub))
            continue
        odir = os.path.join(out, sub); os.makedirs(odir, exist_ok=True)
        for f in os.listdir(sdir):
            stem, ext = os.path.splitext(f)
            if ext.lower() != ".shp":
                if ext.lower() in (".prj", ".cpg", ".txt", ".md"):
                    shutil.copy(os.path.join(sdir, f), os.path.join(odir, f))
                continue
            pin, pout = os.path.join(sdir, stem), os.path.join(odir, stem)
            if "lakes" in stem:
                r = shapefile.Reader(pin); w = shapefile.Writer(pout, shapeType=r.shapeType); w.field("id", "N"); k = 0
                for shp in r.iterShapes():
                    if shp.points and inside(np.asarray(shp.points)).all():
                        w.poly([shp.points[a:b] for a, b in zip(list(shp.parts), list(shp.parts[1:]) + [len(shp.points)])])
                        w.record(k); k += 1
                w.close()
            elif any(t in stem for t in ("coastline", "boundary_lines", "provinces_lines")):
                r = shapefile.Reader(pin); w = shapefile.Writer(pout, shapeType=shapefile.POLYLINE); w.field("id", "N"); k = 0
                for shp in r.iterShapes():
                    if not shp.points:
                        continue
                    pts = np.asarray(shp.points); parts = list(shp.parts) + [len(pts)]
                    for a, b in zip(parts[:-1], parts[1:]):
                        seg = pts[a:b]; ok = inside(seg); run = []
                        for p_, good in zip(seg, ok):
                            if good:
                                run.append(p_.tolist())
                            elif run:
                                if len(run) > 1:
                                    w.line([run]); w.record(k); k += 1
                                run = []
                        if len(run) > 1:
                            w.line([run]); w.record(k); k += 1
                w.close()
            else:
                for e in (".shp", ".shx", ".dbf"):
                    if os.path.exists(pin + e):
                        shutil.copy(pin + e, pout + e)


# ---------------------------------------------------------------- diagnostics
def interp_lnp(p, field, target):
    """Linear in ln p between the bracketing levels; NaN where target is outside the column."""
    above = p < target  # pressure falls with height
    k1 = np.argmax(above, axis=1)  # first level above target
    ok = above.any(axis=1) & (k1 > 0)
    k1 = np.where(ok, k1, 1)
    k0 = k1 - 1
    rows = np.arange(p.shape[0])
    p0, p1 = p[rows, k0], p[rows, k1]
    w = (np.log(target) - np.log(p0)) / (np.log(p1) - np.log(p0))
    out = field[rows, k0] + w * (field[rows, k1] - field[rows, k0])
    return np.where(ok, out, np.nan)


def diagnose(hist, zmid, prev_total):
    with Dataset(hist) as ds:
        ds.set_auto_mask(False)
        p = np.asarray(ds["pressure"][0], np.float64)
        u = np.asarray(ds["u_zonal"][0], np.float64)
        v = np.asarray(ds["v_meridional"][0], np.float64)
        th0 = np.asarray(ds["theta"][0, :, 0], np.float64)
        t2 = np.asarray(ds["t2"][0], np.float64)
        u10 = np.asarray(ds["u10"][0], np.float64); v10 = np.asarray(ds["v10"][0], np.float64)
        total = np.asarray(ds["rainc"][0], np.float64) + np.asarray(ds["rainnc"][0], np.float64)
    z500 = interp_lnp(p, zmid, 50000.0)
    u250 = interp_lnp(p, u, 25000.0); v250 = interp_lnp(p, v, 25000.0)
    t1 = th0 * (p[:, 0] / 1.0e5) ** (RD / CP)
    slp = p[:, 0] * (1.0 + LAPSE * zmid[:, 0] / t1) ** (G0 / (RD * LAPSE))
    fields = {
        "height_500hPa": z500 * 0.1,
        "wind_speed_250hPa": np.hypot(u250, v250) * KT,
        "temperature_2m": t2 - 273.15,
        "sea_level_pressure": slp / 1000.0,
        "wind_speed_10m": np.hypot(u10, v10) * KT,
    }
    if prev_total is not None:
        fields["precipitation_6h"] = np.maximum(total - prev_total, 0.0)
    return fields, total


def write_frame(path, stamp, fields, keep=None, note=""):
    tmp = str(path) + ".tmp"
    n = next(iter(fields.values())).size if keep is None else keep.size
    with Dataset(tmp, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", 1); ds.createDimension("nCells", n); ds.createDimension("StrLen", 64)
        ds.note = note
        xt = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
        xt[0, :] = np.array(list(stamp.ljust(64)), dtype="S1")
        for name, val in fields.items():
            var = ds.createVariable(name, "f4", ("Time", "nCells"), zlib=True, complevel=1)
            var.units = UNITS[name]; var.long_name = LONG[name]
            var[0, :] = np.asarray(val if keep is None else val[keep], np.float32)
    os.replace(tmp, path)


def render_one(renderer, env, inp, token, grid, out_png, theme, size, footer, basemap=None, scratch=None):
    tmp = Path(scratch) / out_png.stem
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    cmd = [str(renderer), "--store-root", str(tmp / "_store"), "--out-dir", str(tmp),
           "--products", token, "--width", str(size[0]), "--height", str(size[1]),
           "--mesh-grid", str(grid), "--theme", str(theme)]
    for k, v in footer.items():
        cmd += [f"--footer-{k}", v]
    cmd.append(str(inp))
    e = dict(env)
    if basemap:
        e["RUSTWX_BASEMAP_DIR"] = str(basemap)
    t0 = time.monotonic()
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=e)
    written = [Path(l.split(" ", 2)[2]) for l in res.stdout.splitlines() if l.startswith("RENDERED ")]
    if res.returncode != 0 or len(written) != 1:
        tail = [l for l in (res.stderr or "").splitlines() if l.strip()][-3:]
        return f"FAIL {out_png.name} rc={res.returncode} {tail}"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(written[0]), str(out_png))
    shutil.rmtree(tmp, ignore_errors=True)
    return f"OK {out_png.name} {time.monotonic() - t0:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("init"); ap.add_argument("grid"); ap.add_argument("out")
    ap.add_argument("--start", default=None, help="YYYY-mm-dd_HH; default: the init file's xtime")
    ap.add_argument("--label", required=True, help="file-name stem: model, mesh and spacing, e.g. hex-30km_x1.655362")
    ap.add_argument("--mesh-text", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--region-every", type=int, default=12)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--hours", type=int, default=120)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    work = out / "_work"; work.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / (out.name + ".scratch"); scratch.mkdir(parents=True, exist_ok=True)
    log = open(out / "render-log.jsonl", "a")

    def note(**kw):
        kw["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        log.write(json.dumps(kw) + "\n"); log.flush()
        print(json.dumps(kw), flush=True)

    renderer = rustwx.find_renderer()
    ok, why = rustwx.probe_renderer(renderer)
    if not ok:
        raise SystemExit(f"renderer refused: {why}")
    env = rustwx.renderer_env()
    with Dataset(a.init) as ds:
        xt = b"".join(np.asarray(ds["xtime"][0]).tolist()).decode().strip() if "xtime" in ds.variables else None
    with Dataset(a.grid) as ds:
        km = float(np.asarray(ds["nominalMinDc"][:]).item()) / 1000.0 if "nominalMinDc" in ds.variables else None
    start = dt.datetime.strptime(a.start or xt[:13], "%Y-%m-%d_%H")
    kmt = f"{km:g} km " if km else ""
    mesh_text = a.mesh_text or f"native {kmt}hexagon cells, no regrid"
    src_label = f"hex {kmt}global forecast | start {start:%Y-%m-%d %H}Z | Rust renderer rw_wrfbatch"
    theme_g, theme_r = work / "theme_global.json", work / "theme_region.json"
    write_theme(theme_g, src_label, True); write_theme(theme_r, src_label, False)
    gdraw = work / "global-draw.grid.nc"
    t0 = time.monotonic()
    if not gdraw.exists():
        c, pl = seamfix(a.grid, gdraw)
        note(event="seamfix", straddling=c, polar=pl, seconds=round(time.monotonic() - t0, 1))
    keeps = {}
    for name, box in REGIONS.items():
        rg = work / f"{name}.grid.nc"; kp = work / f"{name}.keep.npy"
        if not kp.exists():
            np.save(kp, subset_grid(a.grid, rg, box))
            bm = work / f"basemap-{name}"; bm.mkdir(exist_ok=True)
            clip_basemap(env["RUSTWX_BASEMAP_DIR"], bm, box)
        keeps[name] = np.load(kp)
    with Dataset(a.init) as ds:
        zg = np.asarray(ds["zgrid"][0] if ds["zgrid"].ndim == 3 else ds["zgrid"][:], np.float64)
    zmid = 0.5 * (zg[:, :-1] + zg[:, 1:])
    note(event="setup", seconds=round(time.monotonic() - t0, 1), regions=list(REGIONS))

    done = set()
    prev_total = None
    expected = [start + dt.timedelta(hours=h) for h in range(0, a.hours + 1, 6)]
    pool = ThreadPoolExecutor(a.workers)
    while True:
        progressed = False
        for valid in expected:
            if valid in done:
                continue
            stamp = valid.strftime("%Y-%m-%d_%H.%M.%S")
            hist = run / "fc" / f"cuda-history.{stamp}.nc"
            if not hist.exists() or time.time() - hist.stat().st_mtime < 20:
                break  # frames land in order; wait for this one
            lead = int((valid - start).total_seconds() // 3600)
            t1 = time.monotonic()
            try:
                fields, total = diagnose(hist, zmid, prev_total)
            except Exception as exc:  # partially written file: retry next pass
                note(event="diagnose-retry", lead=lead, error=repr(exc)[:300])
                break
            prev_total = total
            xstamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
            gframe = work / f"frame_{stamp}.nc"
            write_frame(gframe, xstamp, fields, note=f"{a.label} F{lead:03d}")
            rframes = {}
            do_regions = lead % a.region_every == 0
            if do_regions:
                for name, keep in keeps.items():
                    rframes[name] = work / f"frame_{name}_{stamp}.nc"
                    write_frame(rframes[name], xstamp, fields, keep=keep, note=f"{a.label} F{lead:03d} {name}")
            tdiag = time.monotonic() - t1
            jobs = []
            for key, (token, title) in PRODUCTS.items():
                if key == "precip6h" and lead == 0:
                    continue
                foot = {"title": title, "note": f"hex forecast hour {lead} from {start:%Y-%m-%d %H}Z",
                        "mesh": mesh_text}
                jobs.append((gframe, token, gdraw, out / "global" / key / f"{a.label}_global_{key}_f{lead:03d}.png",
                             theme_g, (2000, 1010), foot, None))
                for name in rframes:
                    jobs.append((rframes[name], token, work / f"{name}.grid.nc",
                                 out / name / key / f"{a.label}_{name}_{key}_f{lead:03d}.png",
                                 theme_r, (1600, 1200), foot, work / f"basemap-{name}"))
            results = list(pool.map(lambda j: render_one(renderer, env, *j, scratch=scratch), jobs))
            fails = [r for r in results if r.startswith("FAIL")]
            stats = {k: [float(np.nanmin(v)), float(np.nanmax(v))] for k, v in fields.items()}
            note(event="frame", lead=lead, valid=xstamp, maps=len(results) - len(fails), fails=fails[:5],
                 diag_seconds=round(tdiag, 1), total_seconds=round(time.monotonic() - t1, 1), ranges=stats)
            done.add(valid)
            progressed = True
        if len(done) == len(expected):
            note(event="all-frames-done", frames=len(done))
            break
        if a.once and not progressed:
            break
        summary = run / "summary.json"
        if summary.exists() and not progressed:
            # forecast ended; render what exists and stop once nothing new arrives
            last = [v for v in expected if v not in done]
            stamp = last[0].strftime("%Y-%m-%d_%H.%M.%S")
            if not (run / "fc" / f"cuda-history.{stamp}.nc").exists():
                note(event="forecast-ended", frames=len(done), missing_from=stamp)
                break
        time.sleep(5)
    pool.shutdown()


if __name__ == "__main__":
    main()
