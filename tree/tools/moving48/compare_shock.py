"""Side-by-side move-transient series of runs (box-side, CPU only).

  compare_shock.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR is a run_moving_patch.py output. Prints, per 150 s macro boundary, the global
max |dp_sfc/dt| (Pa/s), its area RMS, the zone/carried split when recorded, max |w| and the
peak position, for every run, aligned on model time. Also writes RUN_DIR/shock-series.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def series(run):
    rows = []
    for segment in sorted(Path(run).glob('segment-*')):
        receipt = segment / 'fc' / 'fullstep-lts-receipt.json'
        if not receipt.is_file():
            receipt = segment / 'fullstep-lts-receipt.json'
        if not receipt.is_file():
            continue
        data = json.loads(receipt.read_text())
        for row in data.get('remap_shock_series', []):
            row = dict(row, segment=segment.name, remap=bool(data.get('moving_remap')))
            rows.append(row)
    return rows


def main(runs):
    table = {}
    for run in runs:
        rows = series(run)
        (Path(run) / 'shock-series.json').write_text(json.dumps(rows, indent=1) + '\n')
        for row in rows:
            table.setdefault(row['model_seconds'], {})[run] = row
    print('time_s  ' + '  |  '.join(Path(r).name for r in runs))
    for seconds in sorted(table):
        cells = []
        for run in runs:
            row = table[seconds].get(run)
            if row is None or row.get('surface_pressure_tendency_abs_max_pa_s') is None:
                cells.append('-')
                continue
            text = (f"dp {row['surface_pressure_tendency_abs_max_pa_s']:.3f} rms "
                    f"{row.get('surface_pressure_tendency_area_rms_pa_s', 0):.4f}")
            if 'zone_surface_pressure_tendency_abs_max_pa_s' in row:
                text += (f" zone {row['zone_surface_pressure_tendency_abs_max_pa_s']:.3f}/"
                         f"{row['zone_surface_pressure_tendency_area_rms_pa_s']:.4f}"
                         f" carried {row['carried_surface_pressure_tendency_abs_max_pa_s']:.3f}/"
                         f"{row['carried_surface_pressure_tendency_area_rms_pa_s']:.4f}")
            text += (f" w {row.get('vertical_velocity_abs_max', float('nan')):.2f} @"
                     f"({row.get('surface_pressure_tendency_peak_latitude_deg', 0):.1f},"
                     f"{row.get('surface_pressure_tendency_peak_longitude_deg', 0):.1f})")
            cells.append(text)
        print(f'{seconds:7.0f}  ' + '  |  '.join(cells))


if __name__ == '__main__':
    main(sys.argv[1:])
