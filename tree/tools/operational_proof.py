#!/usr/bin/env python3
"""Summarize real forecast receipts without decoding or changing weather data.

The forecast driver's CUDA reductions supply field sanity. This tool verifies
their coverage and hashes the actual published files. Missing evidence fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(run, manifest):
    run = Path(run)
    errors = []
    hours = float(manifest['hours'])
    dt = float(manifest['dt_seconds'])
    if not math.isfinite(hours) or not math.isfinite(dt) or hours <= 0 or dt <= 0:
        raise ValueError('forecast hours and timestep must be finite and positive')
    steps_float = hours * 3600 / dt
    if not steps_float.is_integer():
        raise ValueError('forecast length must contain a whole number of steps')
    expected = int(steps_float)
    driver_path = run / 'fc' / 'cuda-v841-forecast-receipt.json'
    driver = json.loads(driver_path.read_text()) if driver_path.is_file() else {}
    forecast = driver.get('forecast') or {}
    samehost_path = run / 'fc' / 'samehost-receipt.json'
    if samehost_path.is_file():
        forecast = json.loads(samehost_path.read_text())
    if driver.get('status') != 'passed' or manifest.get('forecast_rc') != 0:
        errors.append('real CLI or driver did not complete successfully')
    if forecast.get('refusal') is not None:
        errors.append('forecast records a model refusal')
    walls = forecast.get('walls') or {}
    if walls.get('steps') != expected:
        errors.append('completed steps differ from requested forecast length')
    health = forecast.get('health', forecast.get('step_health', []))
    rows = [item for row in health for item in (row if isinstance(row, list) else [row])]
    by_step = {}
    for row in rows:
        by_step.setdefault(row.get('step'), []).append(row)
    cards = len(forecast.get('devices') or [0])
    if set(by_step) != set(range(1, expected + 1)) or any(len(v) != cards for v in by_step.values()):
        errors.append('per-step per-rank health coverage is incomplete')
    needed = ('rho_min', 'theta_m_min', 'theta_m_max', 'vertical_velocity_abs_max',
              'qv_min', 'qv_max', 'hydrometeor_min')
    for row in rows:
        if row.get('finite') is not True or any(
            isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float))
            or not math.isfinite(row[k]) for k in needed
        ):
            errors.append('health field is missing or non-finite')
            break
        if row['rho_min'] <= 0 or row['theta_m_min'] <= 0 or row['hydrometeor_min'] < 0:
            errors.append('health field violates the driver physical bounds')
            break
    frames = forecast.get('snapshot_files') or {}
    if not {'0', str(expected)} <= set(map(str, frames)):
        errors.append('initial or final history frame is absent')
    inventory = {}
    for step, descriptor in frames.items():
        name = Path(descriptor.get('path', '')).name
        path = run / 'fc' / name
        if not name or not path.is_file() or path.stat().st_size == 0:
            errors.append(f'published history at step {step} is absent or empty')
            continue
        if name in inventory:
            errors.append('history filename is declared more than once')
        inventory[name] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    elapsed = manifest.get('forecast_wall_seconds')
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
        errors.append('measured forecast wall clock is absent or invalid')
        elapsed = None
    rate = manifest.get('whole_box_usd_per_hour')
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0):
        raise ValueError('whole-box hourly price must be finite and positive')
    complete = not errors
    observed_w = [r['vertical_velocity_abs_max'] for r in rows
                  if isinstance(r.get('vertical_velocity_abs_max'), (int, float))
                  and math.isfinite(r['vertical_velocity_abs_max'])]
    return {
        'schema': 'hex-operational-proof/v1', 'status': 'PASS' if not errors else 'INCOMPLETE',
        'errors': errors, 'manifest': manifest, 'expected_steps': expected,
        'completed_steps': walls.get('steps'), 'health_rows': len(rows),
        'max_abs_w_m_s': max(observed_w, default=None),
        'history_files': inventory, 'driver_receipt_sha256': file_hash(driver_path) if driver_path.is_file() else None,
        'forecast_seconds_per_hour': elapsed / hours if complete and elapsed is not None else None,
        'forecast_card_hours': elapsed * cards / 3600 if complete and elapsed is not None else None,
        'forecast_whole_box_usd': elapsed * rate / 3600 if complete and elapsed is not None and rate is not None else None,
        'cost_scope': 'whole-box wall-time allocation; excludes other stages, queue and idle rental time',
        'nonclaims': ['observation-based forecast skill', 'native MPAS full-model identity',
                      'cross-card byte identity without a separate comparison receipt'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = summarize(args.run, json.loads((args.run / 'manifest.json').read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
