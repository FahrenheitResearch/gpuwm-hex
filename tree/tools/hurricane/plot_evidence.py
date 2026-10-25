"""Plot track/intensity and mesh-centering charts from scored run evidence.

These are coordinate and time-series charts, not weather-field maps. Missing
observations or forecast values stay missing; no temporal interpolation is used.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored-case', type=Path, required=True)
    p.add_argument('--cycle', required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--centering', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-label', default='Best track (1-minute wind)',
                   help='Explicit source/averaging label, including provisional status when applicable')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(a.scored_case.read_text())
    initial = datetime.strptime(a.cycle, '%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    model_names = ['BEST', *data['case']['models']]
    for index, model in enumerate(model_names):
        rows = data['best'] if model == 'BEST' else [r for r in data['forecasts']
            if r['model'] == model and r['init'] == initial]
        rows = sorted([r for r in rows if 0 <= (r['valid']-initial)/3600 <= a.hours],
                      key=lambda r:r['valid'])
        color = 'black' if model == 'BEST' else plt.get_cmap('tab10')((index-1) % 10)
        for ax, xkey, ykey in zip(axes, ['lon', 'hour', 'hour'], ['lat', 'wind', 'pressure']):
            x = [(r['valid']-initial)/3600 if xkey == 'hour' else r[xkey] for r in rows]
            y = [np.nan if r[ykey] is None else r[ykey] for r in rows]
            ax.plot(x, y, 'o-', lw=1.6, ms=2.5, color=color,
                    label=a.reference_label if model == 'BEST' else model)
    for ax, title, xlabel, ylabel in zip(axes,
        ['Track coordinates', '10 m wind', 'Minimum pressure'],
        ['Longitude (degrees)', 'Forecast hour', 'Forecast hour'],
        ['Latitude (degrees)', 'Wind speed (kt)', 'hPa']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_xlim(0, a.hours)
    fig.suptitle(f"{data['case']['storm']} {a.cycle} UTC; hex wind is instantaneous\nReference: {a.reference_label}")
    fig.savefig(a.out/'track-intensity.png', dpi=140)
    plt.close(fig)
    with a.centering.open() as stream:
        rows = list(csv.DictReader(stream))
    x = [float(r['model_seconds'])/3600 for r in rows]
    distance = [float(r['center_distance_km']) if r['center_distance_km'] else np.nan for r in rows]
    fine = [float(r['fine_radius_km']) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    ax.plot(x, distance, 'o-', ms=2, label='Model storm to mesh centre')
    ax.plot(x, fine, '--', color='black', label='Guaranteed fine-region interior')
    for i in range(1, len(rows)):
        if rows[i]['member'] != rows[i-1]['member']:
            ax.axvline(x[i], color='gray', alpha=.25)
    ax.set(xlabel='Forecast hour', ylabel='Great-circle distance (km)', xlim=(0,a.hours),
           title='Storm containment; gray lines mark first recorded frame on a new member')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(a.out/'centering.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
