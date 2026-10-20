"""Summarize a moving run directory (CPU only): segments, moves, conservation and the A3 transient metric.

usage: summarize_run.py RUNDIR [--post-minutes 10] [--out SUMMARY.json]

A3 metric (stated here so every package measures the same thing): for each move at time t, the
maximum over macro boundaries in (t, t+post] of |dp_sfc/dt| (150 s native difference, Pa/s) and of
max|w|, divided by the 99th percentile of the same series over all macro boundaries of the run that
are NOT within post minutes after any move and not within the first post minutes of the forecast
(initialisation spin-up). Pass: both ratios <= 1.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--post-minutes", type=float, default=10.0)
    p.add_argument("--out", type=Path)
    p.add_argument("--control", type=Path, action="append", default=[],
                   help="matched no-move run: a `gpuwm-hex moving fixed` dir or a forecast dir holding "
                        "fc/fullstep-lts-receipt.json (repeatable; series are merged by model time)")
    a = p.parse_args()
    post = a.post_minutes * 60.0
    segments, shock, moves = [], [], []
    for segment in sorted(a.run.glob("segment-*")):
        result = load(segment / "result.json") or load(segment / "command.json") or {}
        lts = load(segment / "fc" / "fullstep-lts-receipt.json") or {}
        remap = lts.get("moving_remap") or load(segment / "fc" / "moving-remap.json")
        rows = lts.get("remap_shock_series") or []
        shock.extend(rows)
        begin = float(result.get("begin_seconds", 0.0))
        if result.get("remesh"):
            moves.append(begin)
        conservation = None
        if remap and remap.get("budgets"):
            # Frozen-far-field zone transfer (hexcore.moving_zone_state): global FP64 budgets.
            conservation = {k: v["relative"] for k, v in remap["budgets"].items()}
            conservation["zone_divergence_difference_rms_over_expected_rms"] = (
                remap["divergence"]["zone_difference_rms"] / max(remap["divergence"]["zone_expected_rms"], 1e-300))
            conservation["carried_cells"] = remap.get("carried_cells")
            conservation["zone_cells"] = remap.get("zone_cells")
        elif remap:
            totals = {"mass": remap.get("mass_total_kg"), "theta_mass": remap.get("theta_mass_total_kg_kelvin"),
                      **(remap.get("tracer_totals_kg") or {})}
            conservation = {k: (v["absolute_error"] / v["before"] if v and v.get("before") else None)
                            for k, v in totals.items()}
            conservation["divergence_difference_rms_over_expected_rms"] = (
                (remap.get("divergence") or {}).get("difference_rms", 0.0)
                / max((remap.get("divergence") or {}).get("expected_overlap_rms", 0.0), 1e-300))
        segments.append(dict(segment=segment.name, rc=result.get("rc"), begin_seconds=begin,
                             end_seconds=result.get("end_seconds"), remesh=result.get("remesh"),
                             center_deg=result.get("selected_center_deg"),
                             forecast_process_wall_seconds=result.get("forecast_process_wall_seconds"),
                             wall_including_lease_wait_seconds=result.get("wall_including_lease_wait_seconds"),
                             integration_seconds=sum(r.get("seconds", 0.0) for r in lts.get("macros", [])) or None,
                             construct_seconds=lts.get("construct_seconds"),
                             atmospheric_transfer_seconds=(remap or {}).get("total_atmospheric_transfer_seconds"),
                             overlap_preparation_seconds=result.get("overlap_preparation_seconds"),
                             conservation=conservation))
    t = np.array([r["model_seconds"] for r in shock], dtype=float)
    dp = np.array([np.nan if r.get("surface_pressure_tendency_abs_max_pa_s") is None
                   else r["surface_pressure_tendency_abs_max_pa_s"] for r in shock], dtype=float)
    w = np.array([np.nan if r.get("vertical_velocity_abs_max") is None else r["vertical_velocity_abs_max"]
                  for r in shock], dtype=float)
    quiet = np.ones(len(t), dtype=bool)
    quiet &= t > post
    for m in moves:
        quiet &= ~((t > m + 1e-6) & (t <= m + post))
    envelope = dict(dp=float(np.nanpercentile(dp[quiet], 99)) if np.any(quiet & np.isfinite(dp)) else None,
                    w=float(np.nanpercentile(w[quiet], 99)) if np.any(quiet & np.isfinite(w)) else None,
                    samples=int(np.sum(quiet)))
    rows = []
    for m in moves:
        window = (t > m + 1e-6) & (t <= m + post)   # macros after the move only
        peak_dp = float(np.nanmax(dp[window])) if np.any(window & np.isfinite(dp)) else None
        peak_w = float(np.nanmax(w[window])) if np.any(window & np.isfinite(w)) else None
        where = None
        if peak_dp is not None:
            index = np.where(window)[0][int(np.nanargmax(dp[window]))]
            where = {k: shock[index].get(k) for k in ("model_seconds", "surface_pressure_tendency_peak_latitude_deg",
                                                       "surface_pressure_tendency_peak_longitude_deg",
                                                       "surface_pressure_tendency_peak_signed_pa_s")}
        rows.append(dict(move_seconds=m, post_dp_max=peak_dp, post_w_max=peak_w, dp_peak_at=where,
                         dp_ratio=None if peak_dp is None or not envelope["dp"] else peak_dp / envelope["dp"],
                         w_ratio=None if peak_w is None or not envelope["w"] else peak_w / envelope["w"]))
    # Matched no-move control at the same model times, and the changed-zone split.
    control = {}
    for path in a.control:
        for receipt in [path / "fc" / "fullstep-lts-receipt.json", path / "fullstep-lts-receipt.json",
                        *sorted(path.glob("segment-*/fc/fullstep-lts-receipt.json"))]:
            data = load(receipt)
            for row in (data or {}).get("remap_shock_series", []):
                control[float(row["model_seconds"])] = row
    for row, m in zip(rows, moves):
        window = [r for r in shock if m + 1e-6 < r["model_seconds"] <= m + post]
        zone = [r.get("zone_surface_pressure_tendency_abs_max_pa_s") for r in window]
        zone = [v for v in zone if v is not None]
        carried = [r.get("carried_surface_pressure_tendency_abs_max_pa_s") for r in window]
        carried = [v for v in carried if v is not None]
        row["post_zone_dp_max"] = max(zone) if zone else None
        row["post_carried_dp_max"] = max(carried) if carried else None
        row["zone_dp_ratio"] = (None if not zone or not envelope["dp"] else max(zone) / envelope["dp"])
        matched = [control.get(float(r["model_seconds"])) for r in window]
        matched = [c for c in matched if c and c.get("surface_pressure_tendency_abs_max_pa_s") is not None]
        if matched and row["post_dp_max"] is not None:
            ref_dp = max(c["surface_pressure_tendency_abs_max_pa_s"] for c in matched)
            ref_w = max(c.get("vertical_velocity_abs_max") or 0.0 for c in matched)
            row["control_post_dp_max"] = ref_dp
            row["control_post_w_max"] = ref_w
            row["dp_ratio_vs_control"] = row["post_dp_max"] / ref_dp if ref_dp else None
            row["w_ratio_vs_control"] = (row["post_w_max"] / ref_w) if ref_w and row["post_w_max"] else None
    a3_control = None
    if control and rows:
        a3_control = all((r.get("dp_ratio_vs_control") or 0) <= 1.5 for r in rows)
    zone_pass = None
    if rows and any(r.get("zone_dp_ratio") is not None for r in rows):
        zone_pass = all((r.get("zone_dp_ratio") or 0) <= 1.5 for r in rows)
    summary = dict(run=str(a.run), segments=segments, moves=rows, quiet_envelope_p99=envelope,
                   a3_vs_control_pass=a3_control, a3_zone_pass=zone_pass,
                   conservation_worst_relative=max((abs(v) for seg in segments if seg["conservation"]
                                                    for k, v in seg["conservation"].items()
                                                    if k not in ("carried_cells", "zone_cells") and
                                                    not k.endswith("over_expected_rms") and v is not None),
                                                   default=None),
                   post_window_seconds=post, a3_pass=all((r["dp_ratio"] or 0) <= 1.5 and (r["w_ratio"] or 0) <= 1.5
                                                         for r in rows) if rows and envelope["samples"] else None)
    text = json.dumps(summary, indent=2)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
