"""Recover unsmoothed central pressure from hash-verified marine surface evidence.

A deliberately narrow referee adapter: every native cell within the requested
core radius must be at zero terrain, where sea-level pressure equals surface
pressure exactly. Land or missing fields produce no intensity value. This does
not reuse the 20 km-smoothed pressure that the tracker uses for localization.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from hexcore.storm_tracker import distance_km


def extract(folder, radius_km):
    record = json.loads((folder / 'receipt.json').read_text())
    fix = record['frame']['fix']
    result = dict(model_seconds=record['frame']['model_seconds'],
                  history_sha256=record['history_sha256'], radius_km=radius_km,
                  mslp_hpa=None, method='native surface pressure; all core cells at zero terrain')
    if not fix.get('ok'):
        return dict(result, reason='invalid tracker fix')
    with np.load(folder / 'surface.npz', allow_pickle=False) as source:
        arrays = {}
        for name in ('latCell', 'lonCell', 'ter', 'surface_pressure'):
            value = np.asarray(source[name])
            expected = record['fields'][name]
            if (list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
                raise ValueError(f'changed retained field {name}: {folder}')
            arrays[name] = value.reshape(-1)
    lat, lon = arrays['latCell'], arrays['lonCell']
    if np.max(np.abs(lat)) <= np.pi / 2 + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon)
    near = distance_km(fix['lat'], fix['lon'], lat, lon) <= radius_km
    terrain, pressure = arrays['ter'][near], arrays['surface_pressure'][near]
    if (not pressure.size or not np.isfinite(pressure).all()
            or not np.isfinite(terrain).all() or not np.all(terrain == 0)):
        return dict(result, reason='core is not entirely finite zero-terrain marine cells')
    i = int(np.argmin(pressure))
    return dict(result, mslp_hpa=float(pressure[i]) / 100,
                cells=int(pressure.size), minimum_location=[float(lat[near][i]), float(lon[near][i])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--radius-km', type=float, default=50)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not np.isfinite(a.radius_km) or a.radius_km <= 0:
        p.error('radius must be finite and positive')
    rows = [extract(f.parent, a.radius_km) for f in sorted(a.evidence.glob('s*/receipt.json'))]
    a.out.write_text(json.dumps(dict(records=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
