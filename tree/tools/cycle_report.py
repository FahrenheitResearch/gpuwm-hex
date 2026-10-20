"""Cycle timeline summary: one run directory -> stage table, key latencies, 5-day projection.

Usage: python cycle_report.py RUN_DIR [--hours H] [--json OUT]

Reads RUN_DIR/cycle-timeline.jsonl (the cycle's own marks, epoch seconds), RUN_DIR/fc/timeline.jsonl
(the forecast's ledger: t_utc per event) and RUN_DIR/forecast.log ({"step","seconds"} rows).
T0 is the start of fetch (the moment an operational cycle starts once GFS f000 is available).
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def epoch(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def spans(rows, key="u"):
    opened, out = {}, []
    for r in rows:
        k = (r["stage"], json.dumps(r.get("step")), json.dumps((r.get("attrs") or {}).get("step")),
             r.get("card"), json.dumps((r.get("attrs") or {}).get("rank")))
        if r["phase"] == "begin":
            opened[k] = r
        elif r["phase"] in ("end", "failed") and k in opened:
            b = opened.pop(k)
            out.append({"stage": r["stage"], "begin": b[key], "end": r[key], "row": r, "attrs": r.get("attrs") or {},
                        "step": r.get("step", (r.get("attrs") or {}).get("step")), "card": r.get("card")})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--frame-minutes", type=int, default=180)
    p.add_argument("--frames-rendered", type=int, default=1, help="frames in a single after-the-forecast render stage")
    a = p.parse_args()
    cyc = [json.loads(l) for l in (a.run / "cycle-timeline.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    cyc = [r for r in cyc if "stage" in r]
    fc_rows = []
    tl = a.run / "fc" / "timeline.jsonl"
    if tl.exists():
        for line in tl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["u"] = epoch(r["t_utc"])
                fc_rows.append(r)
    t0 = next(r["u"] for r in cyc if r["stage"] == "fetch" and r["phase"] == "begin")
    # A shared-box card queue is not part of a cycle: when the forecast waited
    # for its grant after the front half, that wait is cut out of every later
    # stamp and reported on its own.
    queue_gap = 0.0
    grant = next((r["u"] for r in cyc if r["stage"] == "forecast.grant"), None)
    front = next((r["u"] for r in cyc if r["stage"] in ("front_half", "prep") and r["phase"] == "end"), None)
    if grant is not None and front is not None and grant > front:
        queue_gap = grant - front
        for r in cyc:
            if r["u"] >= grant:
                r["u"] -= queue_gap
        for r in fc_rows:
            if r["u"] >= grant:
                r["u"] -= queue_gap
    avail = next((r for r in cyc if r["stage"] == "cycle.available"), None)
    out = {"run": str(a.run), "t0_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
           "card_queue_wait_removed_seconds": round(queue_gap, 1),
           "gfs_available_utc": avail.get("available_utc") if avail else None}
    rows = []
    cyc_spans = spans(cyc)
    for s in cyc_spans:
        if s["stage"] == "render.frame":
            continue
        rows.append((s["stage"], s["begin"] - t0, s["end"] - t0))
    # forecast internals
    fsp = spans(fc_rows)
    def first(stage, which="begin"):
        vals = [s[which] for s in fsp if s["stage"] == stage]
        return (min(vals) if which == "begin" else max(vals)) if vals else None
    internals = {}
    for stage in ("prep.door", "prep.host", "prep.device_upload", "forecast.construct", "forecast.first_step"):
        b, e = first(stage, "begin"), first(stage, "end")
        if b is not None and e is not None:
            internals[stage] = {"begin": round(b - t0, 1), "end": round(e - t0, 1), "seconds": round(e - b, 1)}
            rows.append((f"  fc:{stage}", b - t0, e - t0))
    caps = [s for s in fsp if s["stage"] == "forecast.output.capture"]
    hist = [s for s in fsp if s["stage"] == "forecast.output.history"]
    internals["capture_on_path_seconds"] = [round(s["end"] - s["begin"], 2) for s in caps]
    internals["history_build_write_seconds"] = [round(s["end"] - s["begin"], 1) for s in hist]
    marks = [r for r in fc_rows if r["stage"] == "forecast.integrate" and r["phase"] == "mark"]
    steps = {}
    for line in (a.run / "forecast.log").read_text(errors="replace").splitlines():
        if line.startswith('{"step"'):
            try:
                d = json.loads(line)
                steps[int(d["step"])] = float(d["seconds"])
            except (ValueError, KeyError):
                pass
    if steps:
        ks = sorted(steps)
        internals["steps"] = len(ks)
        internals["step1_seconds"] = round(steps[ks[0]], 2)
        internals["median_step_seconds"] = round(statistics.median(steps[k] for k in ks[1:]), 3)
    fstep = first("forecast.first_step", "begin")
    last_mark = max((r["u"] for r in marks), default=None)
    hours = a.hours
    if hours is None and steps:
        cmd = json.loads((a.run / "command.json").read_text()) if (a.run / "command.json").exists() else {}
        hours = cmd.get("hours")
    fend = next((r["u"] for r in cyc if r["stage"] in ("forecast", "forecast.process") and r["phase"] == "end"), None)
    if fstep and last_mark and hours:
        loop = last_mark - fstep
        internals["loop_seconds"] = round(loop, 1)
        integration = sum(steps.values()) if steps else loop
        # Every capture after the initial frame holds the ranks; the last one falls after the last
        # integrate mark, so the rate is built from its parts rather than from the loop wall.
        in_loop_caps = sum(s["end"] - s["begin"] for s in caps if (s["step"] or 0) > 0)
        internals["integration_seconds_per_forecast_hour"] = round(integration / hours, 2)
        internals["on_path_output_seconds_per_forecast_hour"] = round(in_loop_caps / hours, 2)
        internals["all_in_seconds_per_forecast_hour"] = round((integration + in_loop_caps) / hours, 2)
    # renders
    renders = [s for s in cyc_spans if s["stage"] == "render.frame"]
    whole = [s for s in cyc_spans if s["stage"] == "render"]
    first_maps = {}
    if renders:
        by_step = sorted(renders, key=lambda s: s["end"])
        f000 = [s for s in by_step if int(s["row"].get("step", -1)) == 0]
        fcst = [s for s in by_step if int(s["row"].get("step", -1)) > 0]
        first_maps["f000_maps_out"] = round(f000[0]["end"] - t0, 1) if f000 else None
        first_maps["first_forecast_maps_out"] = round(fcst[0]["end"] - t0, 1) if fcst else None
        first_maps["all_maps_out"] = round(max(s["end"] for s in renders) - t0, 1)
        first_maps["render_frame_seconds"] = [round(s["end"] - s["begin"], 1) for s in sorted(renders, key=lambda s: int(s["row"]["step"]))]
        for s in sorted(renders, key=lambda s: int(s["row"]["step"])):
            rows.append((f"render.frame step {s['row']['step']}", s["begin"] - t0, s["end"] - t0))
    elif whole:
        first_maps["f000_maps_out"] = first_maps["first_forecast_maps_out"] = first_maps["all_maps_out"] = \
            round(whole[0]["end"] - t0, 1)
    out["first_maps"] = first_maps
    out["forecast_done"] = round(fend - t0, 1) if fend else None
    rate = internals.get("all_in_seconds_per_forecast_hour")
    if out["forecast_done"] is not None and rate and hours:
        out["projected_5day_done"] = round(out["forecast_done"] + (120 - hours) * rate, 1)
        frames_5day = 120 * 60 // a.frame_minutes + 1
        if renders:
            # Frames render as they land; the maps finish one frame-render after the forecast.
            out["projected_5day_maps_done"] = round(out["projected_5day_done"] + max(first_maps["render_frame_seconds"]), 1)
        elif whole:
            per_frame = (whole[0]["end"] - whole[0]["begin"]) / max(1, a.frames_rendered)
            out["projected_5day_maps_done"] = round(out["projected_5day_done"] + frames_5day * per_frame, 1)
        out["projection_note"] = (f"forecast done + {120 - hours:g} more forecast hours at the measured all-in rate "
                                  f"(steps + on-path output); maps: concurrent render adds one frame, a render after "
                                  f"the forecast adds {frames_5day} frames at the measured per-frame time")
    out["internals"] = internals
    out["stages"] = [{"stage": s, "begin": round(b, 1), "end": round(e, 1), "seconds": round(e - b, 1)}
                     for s, b, e in sorted(rows, key=lambda r: r[1])]
    print(f"{'stage':44s} {'begin':>8s} {'end':>8s} {'dur':>8s}")
    for r in out["stages"]:
        print(f"{r['stage']:44s} {r['begin']:8.1f} {r['end']:8.1f} {r['seconds']:8.1f}")
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
