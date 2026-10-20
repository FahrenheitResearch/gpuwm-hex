"""Analysis chart of storm tracks and their errors (matplotlib; no weather field is drawn).

usage: track_chart.py --score SCORE.json --best BEST.json --start-time T --title TEXT --out OUT.png
       [--region-log RUN/interior-check.jsonl]

Left: tracks in latitude/longitude (best track in ink, forecasts in fixed categorical order, 12 h markers,
legend). Right: position error against the best track at every output time.
With --region-log the fine-region centres in effect at each output of a moving run are drawn as rings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hexcore import storm_tracker as st  # noqa: E402

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#eda100']  # fixed categorical order
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'


def load_track(path):
    data = json.loads(Path(path).read_text())
    rows = data['track'] if 'track' in data else [dict(seconds=f['seconds'], fix=f) for f in data['fixes']]
    return [(r['seconds'] / 3600, r['fix']['lat'], r['fix']['lon']) for r in rows if r['fix'].get('ok')]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', type=Path, required=True)
    p.add_argument('--best', type=Path, required=True)
    p.add_argument('--start-time', required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--region-log', type=Path)
    p.add_argument('--hours', type=float, default=48.0)
    a = p.parse_args(argv)
    score = json.loads(a.score.read_text())
    start = datetime.strptime(a.start_time, '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
    best = []
    for r in json.loads(a.best.read_text())['rows']:
        t = (datetime.strptime(r['time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc) - start).total_seconds() / 3600
        if -0.01 <= t <= a.hours + 0.01:
            best.append((t, r['lat'], r['lon']))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for axis in (ax, ax2):
        axis.set_facecolor('white')
        axis.grid(color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            axis.spines[side].set_color(MUTED)
        axis.tick_params(colors=MUTED, labelsize=9)
    bt = np.array(best)
    ax.plot(bt[:, 2], bt[:, 1], color=INK, lw=2.4, label='Best track (IBTrACS)', zorder=5)
    m = np.isclose(bt[:, 0] % 12, 0)
    ax.plot(bt[m, 2], bt[m, 1], 'o', color=INK, ms=7, zorder=6)
    if a.region_log and a.region_log.is_file():
        rows = [json.loads(line) for line in a.region_log.read_text().splitlines() if line.strip()]
        centres = {tuple(r['region']) for r in rows}
        for c in centres:
            ax.plot(c[1], c[0], marker='+', color=MUTED, ms=12, mew=1.5, zorder=3)
            lat, lon = st.unproject(c[0], c[1], 396 * np.sin(np.linspace(0, 2 * np.pi, 120)),
                                    396 * np.cos(np.linspace(0, 2 * np.pi, 120)))
            ax.plot(lon, lat, color=MUTED, lw=1, ls='--', zorder=2)
    for k, (label, t) in enumerate(score['tracks'].items()):
        tr = np.array([x for x in load_track(t['path']) if x[0] <= a.hours + 0.01])
        color = SERIES[k % len(SERIES)]
        ax.plot(tr[:, 2], tr[:, 1], color=color, lw=2, label=label, zorder=4)
        m = np.isclose(tr[:, 0] % 12, 0)
        ax.plot(tr[m, 2], tr[m, 1], 'o', color=color, ms=8, mec='white', mew=2, zorder=5)
        errs = []
        for h, la, lo in tr:
            b = None
            for i in range(len(bt) - 1):
                if bt[i, 0] <= h <= bt[i + 1, 0]:
                    f = (h - bt[i, 0]) / (bt[i + 1, 0] - bt[i, 0])
                    b = (bt[i, 1] + f * (bt[i + 1, 1] - bt[i, 1]), bt[i, 2] + f * (bt[i + 1, 2] - bt[i, 2]))
            if b is not None:
                errs.append((h, float(st.distance_km(la, lo, *b))))
        e = np.array(errs)
        ax2.plot(e[:, 0], e[:, 1], color=color, lw=2, marker='o', ms=4, label=label)
    ax.set_xlabel('Longitude (deg E)', color=MUTED)
    ax.set_ylabel('Latitude (deg N)', color=MUTED)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(bt[:, 1]))), adjustable='datalim')
    ax.legend(frameon=False, fontsize=9, loc='best')
    ax.set_title('Storm centre tracks (dots every 12 h)', color=INK, fontsize=11, loc='left')
    ax2.set_xlabel('Forecast hour', color=MUTED)
    ax2.set_ylabel('Distance from best track (km)', color=MUTED)
    ax2.set_xlim(0, a.hours)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(np.arange(0, a.hours + 1, 12))
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title('Track error at every output', color=INK, fontsize=11, loc='left')
    fig.suptitle(a.title, x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == '__main__':
    main()
