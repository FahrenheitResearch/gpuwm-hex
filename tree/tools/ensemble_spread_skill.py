"""Spread-skill of a gpuwm-hex global ensemble against GFS analyses and radiosondes, beside GEFS.

Reads the per-member 0.5 degree extracts written by `gpuwm-hex ensemble --extract-levels
--extract-grid-deg 0.5` (one compressed npz per member and valid time), fetches the GEFS members
with the same identities from the same cycle (0.5 degree pgrb2a, NOAA NODD, byte ranges from
the .idx files), GFS 0.25 degree analyses (f000 of each verifying cycle, every second point = the
same 0.5 degree grid) and IGRA2 radiosondes (NOAA NCEI year-to-date files), and scores both
ensembles identically:

  RMSE of the ensemble mean, ensemble spread sqrt(mean variance, ddof=1), their ratio, the
  mean member RMSE, the control member's RMSE, and the fair CRPS, area weighted (cos lat) over
  NH 20-90N, tropics 20S-20N, SH 20-90S; and at radiosonde stations the same mean error and
  spread, with spread also shown with a stated observation error added in quadrature.

A spread-skill-consistent ensemble has RMSE(mean) = sqrt((N+1)/N) * spread on average.

  python ensemble_spread_skill.py --extracts RUN/extracts [RUN2/extracts ...] --start 2026-10-03T12
        --hours 72 --igra DIR --out OUT [--members gep01,...,gec00]

Analysis charts only (matplotlib for score lines). No weather-field maps (render law).
"""
import argparse
import datetime as dt
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LEVELS = (850, 700, 500, 250)
FIELDS = [f"{k}{p}" for p in LEVELS for k in ("z", "t", "u", "v")] + ["mslp"]
GRIB_TAG = {"z": "HGT", "t": "TMP", "u": "UGRD", "v": "VGRD"}
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
NY, NX = 361, 720
LAT = np.linspace(90.0, -90.0, NY)
LON = np.arange(NX) * 0.5
REGIONS = {"NH": LAT >= 20.0, "TR": np.abs(LAT) <= 20.0, "SH": LAT <= -20.0}
SCORED = ("z500", "t850", "w250", "mslp", "t500", "w850", "z250")
# Radiosonde observation errors (stated, not tuned): roughly the GSI conventional table at
# these levels for report type 120/220.
OBS_ERROR = {"t850": 1.2, "t500": 0.9, "t250": 1.2, "z850": 8.0, "z500": 12.0, "z250": 20.0,
             "w850": 2.2, "w500": 2.6, "w250": 3.2}
UA = {"User-Agent": "hex-ensemble-verify/1.0"}


def log(*a):
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def http(url, rng=None, tries=6):
    headers = dict(UA)
    if rng:
        headers["Range"] = "bytes=%d-%d" % rng
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
        except Exception:
            pass
        time.sleep(min(30, 2 ** k))
    raise RuntimeError(f"fetch failed: {url}")


def field_tag(key):
    if key == "mslp":
        return ":PRMSL:mean sea level:"
    return ":%s:%d mb:" % (GRIB_TAG[key[0]], int(key[1:]))


def decode(data, lat_first_expected):
    import eccodes
    h = eccodes.codes_new_from_message(data)
    try:
        nx, ny = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        v = eccodes.codes_get_values(h).reshape(ny, nx).astype(np.float64)
    finally:
        eccodes.codes_release(h)
    if la0 < 0:
        v = v[::-1]
    step = 360.0 / nx
    shift = int(round((lo0 % 360.0) / step)) % nx
    if shift:
        v = np.roll(v, shift, axis=1)
    return v


def fetch_fields(url, keys, cache):
    """{key: (NY,NX) array on the 0.5 degree grid} from one GRIB2 file via its .idx."""
    out = {}
    missing = [k for k in keys if not os.path.exists(os.path.join(cache, k + ".npy"))]
    if missing:
        raw = http(url + ".idx")
        if raw is None:
            return None
        lines = raw.decode().splitlines()
        os.makedirs(cache, exist_ok=True)
        for key in missing:
            tag = field_tag(key)
            for i, line in enumerate(lines):
                if tag in line:
                    start = int(line.split(":")[1])
                    end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
                    data = http(url, (start, end) if end else (start, start + 50_000_000))
                    v = decode(data, 90.0)
                    if v.shape == (721, 1440):
                        v = v[::2, ::2]
                    assert v.shape == (NY, NX), (url, key, v.shape)
                    if key == "mslp":
                        v = v / 100.0
                    np.save(os.path.join(cache, key + ".npy"), v.astype(np.float32))
                    break
            else:
                raise RuntimeError(f"{key} not in {url}")
    for key in keys:
        out[key] = np.load(os.path.join(cache, key + ".npy")).astype(np.float64)
    return out


def with_wind(d):
    for p in LEVELS:
        if f"u{p}" in d:
            d[f"w{p}"] = (d[f"u{p}"], d[f"v{p}"])
    return d


def area_mean(x, rows):
    w = np.cos(np.radians(LAT))[:, None] * np.ones((1, NX))
    m = rows[:, None] & np.isfinite(x)
    return float((x[m] * w[m]).sum() / w[m].sum())


def fair_crps(members, obs):
    """Fair ensemble CRPS per grid point (Ferro 2014): mean|x-y| - sum|xi-xj| / (2 N (N-1))."""
    n = members.shape[0]
    a = np.abs(members - obs[None]).mean(0)
    s = np.sort(members, axis=0)
    # sum_{i,j}|xi-xj| = 2 * sum_k (2k - n - 1) x_(k), k = 1..n
    k = np.arange(1, n + 1)[:, None, None]
    pair = 2.0 * ((2 * k - n - 1) * s).sum(0)
    return a - pair / (2.0 * n * (n - 1))


def score_ensemble(stack, analysis, control=None):
    """stack: {var: (N,NY,NX) or tuple for winds}; analysis same keys (NY,NX)."""
    res = {}
    for var in SCORED:
        if var not in analysis or var not in stack:
            continue
        if var.startswith("w"):
            (u, v), (ua, va) = stack[var], analysis[var]
            n = u.shape[0]
            mu, mv = u.mean(0), v.mean(0)
            err2 = (mu - ua) ** 2 + (mv - va) ** 2
            var_ = u.var(0, ddof=1) + v.var(0, ddof=1)
            member_err2 = ((u - ua[None]) ** 2 + (v - va[None]) ** 2)
            crps = None
            ctrl = None if control is None else (control[var][0] - ua) ** 2 + (control[var][1] - va) ** 2
        else:
            x, y = stack[var], analysis[var]
            n = x.shape[0]
            err2 = (x.mean(0) - y) ** 2
            var_ = x.var(0, ddof=1)
            member_err2 = (x - y[None]) ** 2
            crps = fair_crps(x, y)
            ctrl = None if control is None else (control[var] - y) ** 2
        row = {}
        for region, rows in REGIONS.items():
            rmse = np.sqrt(area_mean(err2, rows))
            spread = np.sqrt(area_mean(var_, rows))
            r = {"rmse_mean": rmse, "spread": spread, "ratio": spread / rmse if rmse else None,
                 "consistency_ratio": np.sqrt((n + 1) / n) * spread / rmse if rmse else None,
                 "member_rmse": float(np.mean([np.sqrt(area_mean(e, rows)) for e in member_err2]))}
            if crps is not None:
                r["crps"] = area_mean(crps, rows)
            if ctrl is not None:
                r["control_rmse"] = np.sqrt(area_mean(ctrl, rows))
            row[region] = r
        res[var] = row
    return res


# ------------------------------------------------------------------------- radiosondes (IGRA2)
def igra_profiles(folder, times):
    """{valid datetime: list of (lat, lon, {level: (z, t, u, v)})} for mandatory levels."""
    wanted = {t.strftime("%Y%m%d%H"): t for t in times}
    out = {t: [] for t in times}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".zip"):
            continue
        try:
            with zipfile.ZipFile(os.path.join(folder, name)) as z:
                text = z.read(z.namelist()[0]).decode("ascii", "replace").splitlines()
        except zipfile.BadZipFile:
            continue
        i = 0
        while i < len(text):
            h = text[i]
            if not h.startswith("#"):
                i += 1
                continue
            nlev = int(h[32:36])
            key = h[13:17] + h[18:20] + h[21:23] + h[24:26]
            if key in wanted:
                lat, lon = int(h[55:62]) / 1e4, int(h[63:71]) / 1e4
                levels = {}
                for line in text[i + 1:i + 1 + nlev]:
                    press = int(line[9:15])
                    if press <= 0 or press % 100 or press // 100 not in LEVELS:
                        continue
                    gph, temp = int(line[16:21]), int(line[22:27])
                    wdir, wspd = int(line[40:45]), int(line[46:51])
                    z = gph if gph > -8888 else np.nan
                    t = temp / 10.0 + 273.15 if temp > -8888 else np.nan
                    if wdir > -8888 and wspd > -8888:
                        sp, rad = wspd / 10.0, np.radians(wdir)
                        u, v = -sp * np.sin(rad), -sp * np.cos(rad)
                    else:
                        u = v = np.nan
                    levels[press // 100] = (z, t, u, v)
                if levels:
                    out[wanted[key]].append((lat, lon, levels))
            i += 1 + nlev
    return out


def bilinear(field, lat, lon):
    """field (..., NY, NX) on the 0.5 degree grid, at arrays lat/lon (degrees)."""
    y = (90.0 - lat) / 0.5
    x = (lon % 360.0) / 0.5
    j0 = np.clip(np.floor(y).astype(int), 0, NY - 2)
    i0 = np.floor(x).astype(int) % NX
    i1 = (i0 + 1) % NX
    fy, fx = y - j0, x - np.floor(x)
    return ((1 - fy) * (1 - fx) * field[..., j0, i0] + (1 - fy) * fx * field[..., j0, i1] +
            fy * (1 - fx) * field[..., j0 + 1, i0] + fy * fx * field[..., j0 + 1, i1])


def score_sondes(stack, sondes):
    if not sondes:
        return {}
    lat = np.array([s[0] for s in sondes])
    lon = np.array([s[1] for s in sondes])
    res = {}
    for p in LEVELS:
        for kind, index in (("t", 1), ("z", 0), ("w", None)):
            var = f"{kind}{p}"
            if kind == "w":
                obs_u = np.array([s[2].get(p, (np.nan,) * 4)[2] for s in sondes])
                obs_v = np.array([s[2].get(p, (np.nan,) * 4)[3] for s in sondes])
                u, v = bilinear(stack[f"u{p}"], lat, lon), bilinear(stack[f"v{p}"], lat, lon)
                ok = np.isfinite(obs_u) & np.isfinite(obs_v) & np.isfinite(u).all(0) & np.isfinite(v).all(0)
                if ok.sum() < 20:
                    continue
                err2 = (u.mean(0) - obs_u) ** 2 + (v.mean(0) - obs_v) ** 2
                var_ = u.var(0, ddof=1) + v.var(0, ddof=1)
                sigma2 = 2 * OBS_ERROR.get(var, 0) ** 2  # per-component error, two components
            else:
                obs = np.array([s[2].get(p, (np.nan,) * 4)[index] for s in sondes])
                x = bilinear(stack[var], lat, lon)
                ok = np.isfinite(obs) & np.isfinite(x).all(0)
                if ok.sum() < 20:
                    continue
                err2 = (x.mean(0) - obs) ** 2
                var_ = x.var(0, ddof=1)
                sigma2 = OBS_ERROR.get(var, 0) ** 2
            rmse = float(np.sqrt(err2[ok].mean()))
            spread = float(np.sqrt(var_[ok].mean()))
            res[var] = {"stations": int(ok.sum()), "rmse_mean": rmse, "spread": spread,
                        "spread_with_obs_error": float(np.sqrt(var_[ok].mean() + sigma2)),
                        "obs_error": OBS_ERROR.get(var), "ratio": spread / rmse}
    return res


def load_hex(extract_dirs, members, stamp):
    stack = {k: [] for k in FIELDS}
    found = []
    for m in members:
        path = None
        for root in extract_dirs:
            p = os.path.join(root, m, stamp + ".npz")
            if os.path.exists(p):
                path = p
                break
        if path is None:
            continue
        d = np.load(path)
        for k in FIELDS:
            stack[k].append(d[k].astype(np.float64))
        found.append(m)
    if not found:
        return None, []
    return {k: np.stack(v) for k, v in stack.items()}, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracts", nargs="+", required=True)
    ap.add_argument("--start", default="2026-10-03T12")
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--members", default=",".join([f"gep{i:02d}" for i in range(1, 21)] + ["gec00"]))
    ap.add_argument("--control", default="gec00")
    ap.add_argument("--igra", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-members", type=int, default=2)
    a = ap.parse_args()
    start = dt.datetime.strptime(a.start, "%Y-%m-%dT%H")
    members = a.members.split(",")
    leads = list(range(0, a.hours + 1, a.step))
    os.makedirs(a.out, exist_ok=True)
    valid = {L: start + dt.timedelta(hours=L) for L in leads}
    sonde_times = [valid[L] for L in leads if valid[L].hour in (0, 12)]
    sondes = igra_profiles(a.igra, sonde_times) if a.igra else {}
    log("sondes", {t.strftime("%m%dT%H"): len(v) for t, v in sondes.items()})
    keys = FIELDS
    # analyses and GEFS, fetched in parallel
    def gfs_an(L):
        vt = valid[L]
        url = "%s/gfs.%s/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f000" % (GFS, vt.strftime("%Y%m%d"), vt.hour, vt.hour)
        return L, fetch_fields(url, keys, os.path.join(a.cache, "gfs-an", vt.strftime("%Y%m%d%H")))
    def gefs(args):
        m, L = args
        url = "%s/gefs.%s/%02d/atmos/pgrb2ap5/%s.t%02dz.pgrb2a.0p50.f%03d" % (
            GEFS, start.strftime("%Y%m%d"), start.hour, m, start.hour, L)
        return m, L, fetch_fields(url, keys, os.path.join(a.cache, "gefs", m, "f%03d" % L))
    with ThreadPoolExecutor(16) as ex:
        analyses = dict(ex.map(gfs_an, leads))
        gefs_fields = {}
        for m, L, d in ex.map(gefs, [(m, L) for m in members for L in leads]):
            gefs_fields[(m, L)] = d
    log("fetched analyses and GEFS")
    table = {"start": a.start, "members_requested": members, "leads": {}, "obs_error": OBS_ERROR,
             "method": __doc__.split("\n\n")[1].strip()}
    for L in leads:
        stamp = valid[L].strftime("%Y-%m-%d_%H.%M.%S")
        hx, found = load_hex(a.extracts, members, stamp)
        an = analyses[L]
        if an is None:
            log("skip lead", L, "no analysis")
            continue
        an = with_wind(dict(an))
        hex_ok = hx is not None and len(found) >= a.min_members
        # GEFS is scored on the members the hex ensemble has (same identities); with no hex
        # members yet, on every requested member (baseline only).
        pool = found if hex_ok else members
        gefs_members = [m for m in pool if gefs_fields.get((m, L)) is not None]
        gx = {k: np.stack([gefs_fields[(m, L)][k] for m in gefs_members]) for k in keys}
        row = {"valid": valid[L].isoformat(), "hex_members": found, "gefs_members": gefs_members}
        hctrl = gctrl = None
        if a.control in gefs_members:
            gctrl = with_wind({k: gx[k][gefs_members.index(a.control)] for k in keys})
        if hex_ok and a.control in found:
            hctrl = with_wind({k: hx[k][found.index(a.control)] for k in keys})
        if hex_ok:
            row["hex"] = score_ensemble(with_wind(dict(hx)), an, hctrl)
        row["gefs"] = score_ensemble(with_wind(dict(gx)), an, gctrl)
        if valid[L] in sondes:
            if hex_ok:
                row["sondes_hex"] = score_sondes(hx, sondes[valid[L]])
            row["sondes_gefs"] = score_sondes(gx, sondes[valid[L]])
            row["sonde_profiles"] = len(sondes[valid[L]])
        table["leads"][str(L)] = row
        g = row["gefs"]["z500"]["NH"]
        h = row.get("hex", {}).get("z500", {}).get("NH", {})
        log("lead", L, "hex members", len(found) if hex_ok else 0, "z500 NH hex rmse/spread",
            h.get("rmse_mean"), h.get("spread"), "gefs %.1f/%.1f" % (g["rmse_mean"], g["spread"]))
    with open(os.path.join(a.out, "spread-skill.json"), "w") as f:
        json.dump(table, f, indent=1, default=float)
    charts(table, a.out)
    log("wrote", a.out)


def charts(table, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leads = sorted(int(k) for k in table["leads"])
    if not leads:
        return
    units = {"z500": "m", "t850": "K", "w250": "m/s", "mslp": "hPa"}
    for var in ("z500", "t850", "w250", "mslp"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
        for ax, region in zip(axes, ("NH", "TR", "SH")):
            for system, color in (("hex", "#2a78d6"), ("gefs", "#eb6834")):
                have = [L for L in leads if system in table["leads"][str(L)]]
                if not have:
                    continue
                rm = [table["leads"][str(L)][system][var][region]["rmse_mean"] for L in have]
                sp = [table["leads"][str(L)][system][var][region]["spread"] for L in have]
                label = "hex 30 km" if system == "hex" else "GEFS"
                ax.plot(have, rm, "-", color=color, lw=2, label=f"{label} RMSE of mean")
                ax.plot(have, sp, "--", color=color, lw=2, label=f"{label} spread")
            ax.set_title(f"{var} {region}")
            ax.set_xlabel("lead (h)")
            ax.set_xticks(range(0, leads[-1] + 1, 12))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(units[var])
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Spread (dashed) vs error of the ensemble mean (solid), against GFS analyses: {var}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"spread-skill-{var}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
