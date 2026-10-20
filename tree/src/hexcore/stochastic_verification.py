"""Paired GPU field reductions and small-receipt S2S verification.

Fields must already be decoded and aligned by the existing Rust data path.
This module reads no weather files and provides no CPU field fallback.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import time

SCHEMA = "hex-stochastic-spread-skill-v1"
FIELDS = {"z500": "m", "t2m": "K"}


def utc_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("verification times require explicit UTC")
    return result.astimezone(timezone.utc)


def forecast_fields_gpu(pressure_pa, height_m, t2m_k):
    """Extract z500 on the GPU from aligned mass-level forecast columns.

    Pressure and geopotential height have shape (cell, level), bottom first;
    t2m has shape (cell,). Height must already be geopotential metres at
    those mass levels, not interface zgrid or geometric height. All inputs
    are device arrays. The Rust reader/caller owns dimension ordering,
    height conversion, units and cell alignment.

    Use linear interpolation in log pressure, as in ensemble._interp_lnp,
    with exact endpoint support and no extrapolation. Nonfinite heights,
    nonpositive/nonfinite pressures or nondecreasing pressure invalidate
    the entire column. Such missing forecast values remain visible to the
    paired scorer; this function does not manufacture a terrain mask.
    """
    import cupy as cp
    arrays = (pressure_pa, height_m, t2m_k)
    if any(not isinstance(a, cp.ndarray) or a.dtype.kind != "f" for a in arrays):
        raise TypeError("forecast fields must be floating-point GPU arrays")
    if (pressure_pa.ndim != 2 or not pressure_pa.shape[0]
            or pressure_pa.shape[1] < 2 or height_m.shape != pressure_pa.shape
            or t2m_k.shape != pressure_pa.shape[:1]):
        raise ValueError("forecast fields require matching (cell, level>=2) columns and (cell,) t2m")
    p = pressure_pa.astype(cp.float64, copy=False)
    h = height_m.astype(cp.float64, copy=False)
    valid = (cp.all(cp.isfinite(p) & (p > 0) & cp.isfinite(h), axis=1)
             & cp.all(p[:, :-1] > p[:, 1:], axis=1)
             & (p[:, 0] >= 50000.) & (p[:, -1] <= 50000.))
    # Count levels strictly below the target height. Clipping also permits
    # exact bottom/top endpoints without ever indexing outside a column.
    upper = cp.clip(cp.sum(p > 50000., axis=1), 1, p.shape[1]-1)
    cell = cp.arange(p.shape[0])
    p0, p1 = p[cell, upper-1], p[cell, upper]
    h0, h1 = h[cell, upper-1], h[cell, upper]
    # Safe placeholders keep invalid columns from producing log/divide
    # warnings; the final mask always returns NaN for those columns.
    p0, p1 = cp.where(valid, p0, 100000.), cp.where(valid, p1, 25000.)
    h0, h1 = cp.where(valid, h0, 0.), cp.where(valid, h1, 0.)
    weight = (math.log(50000.)-cp.log(p0))/(cp.log(p1)-cp.log(p0))
    return dict(z500=cp.where(valid, h0+weight*(h1-h0), cp.nan), t2m=t2m_k)


def score_pair_gpu(control, stochastic, analysis, area, *, lead_hours,
                   valid_time, analysis_id):
    """Score two member mappings against one GFS analysis on common cells.

    Each member maps z500 (metres) and t2m (Kelvin) to 1-D CuPy arrays.
    The caller supplies physical cell areas and one identically aligned GFS
    analysis. Member names align the arms, independent of mapping order.
    Below-ground z500 cells are excluded from BOTH arms, as are any other
    missing values. The receipt records the remaining area fraction.
    """
    import cupy as cp
    started = time.perf_counter()
    if not all(isinstance(member, str) and member for member in (*control, *stochastic)):
        raise ValueError("member identities must be nonempty strings")
    members = sorted(control)
    if len(members) < 2 or set(members) != set(stochastic):
        raise ValueError("paired scoring needs the same two or more member identities")
    if type(lead_hours) is not int or lead_hours < 0:
        raise ValueError("lead_hours must be a nonnegative integer")
    valid = utc_time(valid_time).isoformat()
    if not isinstance(analysis_id, str) or not analysis_id.strip():
        raise ValueError("analysis_id must identify the shared GFS analysis")
    if not isinstance(area, cp.ndarray) or area.ndim != 1 or not area.size:
        raise ValueError("area must be a nonempty 1-D GPU array")
    if bool(cp.any(~cp.isfinite(area) | (area <= 0))):
        raise ValueError("cell areas must be finite and positive")
    weights = area.astype(cp.float64, copy=False)
    total_area = cp.sum(weights)
    event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
    event_start.record()
    fields = {}
    for field, unit in FIELDS.items():
        arrays = [analysis[field]] + [arm[member][field]
                  for arm in (control, stochastic) for member in members]
        if any(not isinstance(a, cp.ndarray) or a.shape != area.shape for a in arrays):
            raise ValueError(f"{field}: every field must be a GPU array matching area")
        observed = arrays[0].astype(cp.float64, copy=False)
        stacks = [cp.stack([arm[m][field] for m in members]).astype(cp.float64)
                  for arm in (control, stochastic)]
        analysis_support = cp.isfinite(observed)
        common = analysis_support.copy()
        for stack in stacks:
            common &= cp.all(cp.isfinite(stack), axis=0)
        count = int(cp.count_nonzero(common))
        if not count:
            raise ValueError(f"{field}: paired verification has no common finite cells")
        supported_area = cp.sum(weights[common])
        w = weights[common] / supported_area
        y = observed[common]
        row = dict(unit=unit, valid_cells=count,
                   area_fraction=float(supported_area / total_area),
                   analysis_area_fraction=float(cp.sum(weights[analysis_support])/total_area),
                   forecast_missing_cells=int(cp.count_nonzero(analysis_support & ~common)))
        for name, stack in zip(("control", "stochastic"), stacks):
            values = stack[:, common]
            mean = cp.mean(values, axis=0)
            spread = float(cp.sqrt(cp.sum(cp.var(values, axis=0, ddof=1)*w)))
            rmse = float(cp.sqrt(cp.sum((mean-y)**2*w)))
            row[name] = dict(spread=spread, rmse_mean=rmse,
                             ratio=spread/rmse if rmse else None)
        fields[field] = row
    event_end.record()
    event_end.synchronize()
    return dict(schema=SCHEMA, lead_hours=lead_hours, valid_time=valid,
                members=members, analysis_source="GFS", analysis_id=analysis_id,
                method="area-weighted common cells; sample variance ddof=1",
                variance_ddof=1, fields=fields, parallelism="GPU CuPy reductions",
                gpu_elapsed_ms=cp.cuda.get_elapsed_time(event_start, event_end),
                wall_seconds=time.perf_counter()-started)


def qualify_daily(receipts, *, cycle):
    """Validate the requested ten-member, ten-day paired verification deck.

    This grades the daily spread/mean-error bar only. It cannot establish
    GPU repeatability, timing, landing gates or visual inspection.
    """
    start = utc_time(cycle)
    rows = sorted(receipts, key=lambda row: row["lead_hours"])
    if [row["lead_hours"] for row in rows] != list(range(24, 241, 24)):
        raise ValueError("qualification requires each daily lead 24..240 exactly once")
    members = rows[0]["members"]
    if (len(members) != 10 or len(set(members)) != 10
            or not all(isinstance(m, str) and m for m in members)):
        raise ValueError("qualification requires ten distinct member identities")
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("variance_ddof") != 1:
            raise ValueError("daily receipt schema or sample variance differs")
        if row["members"] != members:
            raise ValueError("member roster changes across daily verification")
        if row.get("analysis_source") != "GFS" or not row.get("analysis_id", "").strip():
            raise ValueError("daily receipt must identify the shared GFS analysis")
        if utc_time(row["valid_time"]) != start + timedelta(hours=row["lead_hours"]):
            raise ValueError("daily verification valid time differs from cycle plus lead")
        for field, unit in FIELDS.items():
            metric = row["fields"][field]
            if metric.get("unit") != unit:
                raise ValueError(f"{field}: verification units differ")
            if (type(metric["valid_cells"]) is not int or metric["valid_cells"] <= 0
                    or not 0 < metric["area_fraction"] <= 1):
                raise ValueError(f"{field}: invalid common verification support")
            if (not metric["area_fraction"] <= metric["analysis_area_fraction"] <= 1
                    or type(metric["forecast_missing_cells"]) is not int
                    or metric["forecast_missing_cells"] < 0):
                raise ValueError(f"{field}: invalid analysis/forecast support accounting")
            if metric["forecast_missing_cells"]:
                raise ValueError(f"{field}: forecast values missing on valid analysis cells; "
                                 "resolve or document a shared terrain mask before grading")
            for arm in ("control", "stochastic"):
                score = metric[arm]
                spread, rmse = score["spread"], score["rmse_mean"]
                if any(not math.isfinite(x) or x < 0 for x in (spread, rmse)):
                    raise ValueError(f"{field}: nonfinite or negative score")
                expected = spread/rmse if rmse else None
                if ((expected is None and score["ratio"] is not None)
                        or (expected is not None and (score["ratio"] is None
                            or not math.isclose(score["ratio"], expected, rel_tol=1e-12)))):
                    raise ValueError(f"{field}: spread/skill ratio differs from its scores")
    results = {}
    for field in FIELDS:
        daily = []
        for row in rows:
            metric = row["fields"][field]
            a, b = metric["control"], metric["stochastic"]
            daily.append(dict(lead_hours=row["lead_hours"],
                              spread_increased=b["spread"] > a["spread"],
                              mean_not_worse=b["rmse_mean"] <= a["rmse_mean"]))
        results[field] = dict(daily=daily,
                             passed=all(d["spread_increased"] and d["mean_not_worse"]
                                        for d in daily))
    return dict(schema=SCHEMA, cycle=start.isoformat(), members=members, days=rows,
                fields=results, spread_skill_passed=all(r["passed"] for r in results.values()),
                policy="strictly larger spread and no larger ensemble-mean RMSE on every day and field",
                qualification_complete=False,
                pending=["visual inspection", "GPU word repeatability", "pattern step cost <5%",
                         "basis/evaluation parallel timing", "landing gates"])


def plot_daily(result, output):
    """Analysis charts from compact JSON scores; no weather-field rendering."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for field, unit in FIELDS.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        days = [row["lead_hours"]/24 for row in result["days"]]
        for arm, label, color in (("control", "SPPT/SKEB off", "#315b9c"),
                                  ("stochastic", "SPPT/SKEB on", "#b74c27")):
            metrics = [row["fields"][field][arm] for row in result["days"]]
            for ax, key in zip(axes, ("spread", "rmse_mean", "ratio")):
                ax.plot(days, [m[key] for m in metrics], marker="o", color=color, label=label)
        for ax, title, ylabel in zip(axes, ("Ensemble spread", "RMSE of ensemble mean", "Spread / RMSE"),
                                     (unit, unit, "ratio")):
            ax.set(title=title, xlabel="Forecast day", ylabel=ylabel, xticks=days)
            ax.grid(alpha=.25)
        axes[2].axhline(1, color="gray", linestyle=":", linewidth=1)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{field}, 10 paired members vs GFS, {result['cycle']}")
        fig.tight_layout()
        path = output / f"spread-skill-{field}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths
