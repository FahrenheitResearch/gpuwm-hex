"""Transient time-series chart (analysis chart, not a weather field): moving run vs matched fixed control.

  plot_shock.py MOVING_SHOCK.json CONTROL_SHOCK.json OUT.png [TITLE]

Inputs are compare_shock.py outputs (shock-series.json). Move times are the moving run's
segment starts with a remap. Panels: max |dp_sfc/dt| (global), changed-zone vs carried split, max |w|.
"""
from __future__ import annotations

import json
import sys


def main(moving_path, control_path, out, title='Move transient: moving run vs fixed control'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    moving = json.load(open(moving_path))
    control = json.load(open(control_path))
    moves = sorted({r['model_seconds'] for r in moving if r.get('remap') and r.get(
        'surface_pressure_tendency_abs_max_pa_s') is None})
    if not moves:
        starts = {}
        for r in moving:
            starts.setdefault(r['segment'], r)
        moves = [r['model_seconds'] for r in starts.values() if r.get('remap')]

    def series(rows, key):
        pairs = [(r['model_seconds'] / 3600, r.get(key)) for r in rows if r.get(key) is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    ax.plot(*series(control, 'surface_pressure_tendency_abs_max_pa_s'), color='#888888', lw=1.5,
            label='fixed region (no moves)')
    ax.plot(*series(moving, 'surface_pressure_tendency_abs_max_pa_s'), color='#1f5fbf', lw=1.2,
            label='moving region')
    ax.set_ylabel('max |dp_sfc/dt| (Pa/s)')
    ax.legend(loc='upper right', frameon=False)
    ax = axes[1]
    ax.plot(*series(moving, 'zone_surface_pressure_tendency_abs_max_pa_s'), color='#d4541a', lw=1.2,
            label='changed zone (remapped cells)')
    ax.plot(*series(moving, 'carried_surface_pressure_tendency_abs_max_pa_s'), color='#1f5fbf', lw=1.0,
            label='carried cells (copied exactly)')
    ax.set_ylabel('max |dp_sfc/dt| (Pa/s)')
    ax.legend(loc='upper right', frameon=False)
    ax = axes[2]
    ax.plot(*series(control, 'vertical_velocity_abs_max'), color='#888888', lw=1.5, label='fixed region')
    ax.plot(*series(moving, 'vertical_velocity_abs_max'), color='#1f5fbf', lw=1.2, label='moving region')
    ax.set_ylabel('max |w| (m/s)')
    ax.set_xlabel('forecast hour')
    ax.legend(loc='upper right', frameon=False)
    for ax in axes:
        for m in moves:
            ax.axvline(m / 3600, color='#cccccc', lw=0.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title(title + '  (grey verticals = moves)')
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == '__main__':
    main(*sys.argv[1:])
