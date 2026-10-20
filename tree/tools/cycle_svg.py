"""Draw before/after cycle timelines (from cycle_report.py --json outputs) as one SVG Gantt chart.

Usage: python cycle_svg.py OUT.svg LABEL=REPORT.json [LABEL=REPORT.json ...]
Each run is one band of bars; the x axis is minutes since GFS f000 was fetched (T0).
"""
import json
import sys
from html import escape

COLORS = {"fetch": "#4e79a7", "intermediate": "#59a14f", "prep": "#9c755f", "init": "#9c755f",
          "fc:prep.door": "#bab0ac", "fc:prep.host": "#f28e2b", "fc:prep.device_upload": "#e15759",
          "fc:forecast.construct": "#ff9da7", "step": "#76b7b2", "render": "#b07aa1", "maps": "#edc948"}


def color(stage):
    for key, value in COLORS.items():
        if stage.startswith(key) or stage.strip().startswith(key):
            return value
    return "#888888"


def main():
    out, *specs = sys.argv[1:]
    runs = []
    for spec in specs:
        label, path = spec.split("=", 1)
        runs.append((label, json.load(open(path))))
    span = max(max([s["end"] for s in r["stages"]] + [r.get("forecast_done") or 0,
                    (r.get("first_maps") or {}).get("all_maps_out") or 0]) for _, r in runs)
    width, left, row_h = 1100, 230, 18
    scale = (width - left - 30) / span
    rows = []
    for label, report in runs:
        rows.append(("title", label, report))
        for stage in report["stages"]:
            if stage["stage"] in ("cycle",):
                continue
            rows.append(("bar", stage, report))
        maps = report.get("first_maps") or {}
        for key in ("f000_maps_out", "first_forecast_maps_out", "all_maps_out"):
            if maps.get(key):
                rows.append(("mark", (key.replace("_", " "), maps[key]), report))
        if report.get("forecast_done"):
            rows.append(("mark", ("forecast done", report["forecast_done"]), report))
    height = 60 + row_h * len(rows) + 40
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="Inter,Helvetica,Arial,sans-serif" font-size="11">',
           f'<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="12" y="22" font-size="15" font-weight="600">15 km global hex cycle on 8 x RTX PRO 6000: '
           f'GFS f000 fetched -> maps</text>',
           f'<text x="12" y="40" fill="#555">minutes since T0 (start of the GFS fetch); card-queue waits removed</text>']
    for minute in range(0, int(span / 60) + 2, 5 if span < 3600 else 10):
        x = left + minute * 60 * scale
        svg.append(f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="{height - 30}" stroke="#eee"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" fill="#666">{minute}</text>')
    y = 60
    for kind, item, report in rows:
        if kind == "title":
            svg.append(f'<text x="12" y="{y + 13}" font-weight="700" font-size="12">{escape(item)}</text>')
        elif kind == "bar":
            x0, x1 = left + item["begin"] * scale, left + item["end"] * scale
            svg.append(f'<text x="{left - 6}" y="{y + 12}" text-anchor="end" fill="#333">{escape(item["stage"].strip())}</text>')
            svg.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(1.5, x1 - x0):.1f}" height="{row_h - 6}" '
                       f'fill="{color(item["stage"])}" rx="2"/>')
            svg.append(f'<text x="{x1 + 4:.1f}" y="{y + 12}" fill="#555">{item["seconds"]:.0f} s</text>')
        else:
            name, at = item
            x = left + at * scale
            svg.append(f'<text x="{left - 6}" y="{y + 12}" text-anchor="end" fill="#000" font-weight="600">{escape(name)}</text>')
            svg.append(f'<circle cx="{x:.1f}" cy="{y + 9}" r="5" fill="#000"/>')
            svg.append(f'<text x="{x + 8:.1f}" y="{y + 12}" font-weight="600">{at / 60:.1f} min</text>')
        y += row_h
    svg.append("</svg>")
    open(out, "w").write("\n".join(svg))


if __name__ == "__main__":
    main()
