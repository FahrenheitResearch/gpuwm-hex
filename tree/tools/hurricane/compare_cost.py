"""Measure completed moving/fixed whole-case costs from existing receipts.

No timers or synchronizations are added to the forecast. Wall cost includes
member preparation waits, construction, transfer, integration and output.
Stage sums are descriptive: concurrent work must not be added to wall time.
"""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def measure(run, exit_path, hours):
    guard = read(exit_path)
    if guard['returncode'] or guard.get('reason'):
        raise ValueError(f'producer did not finish: {run}')
    paths = sorted(run.glob('segment-*/fc/fullstep-lts-receipt.json'))
    if not paths:
        paths = [run/'fc/fullstep-lts-receipt.json']
    segments = []
    expected_begin = 0.0
    for path in paths:
        r = read(path)
        macros = r['macros']
        steps = sum(m['base_steps'] for m in macros)
        if abs(r['clock_offset_seconds']-expected_begin) > 1e-5:
            raise ValueError(f'integration has a gap or repeated interval: {path}')
        expected_begin += steps*r['base_dt']
        remap = r.get('moving_remap') or {}
        segments.append(dict(receipt=str(path), fine_steps=steps,
            model_seconds=steps*r['base_dt'],
            integration_seconds=sum(m['seconds'] for m in macros),
            construct_seconds=r['construct_seconds'],
            atmospheric_transfer_seconds=remap.get('total_atmospheric_transfer_seconds'),
            geometry_seconds=remap.get('geometry_read_identify_seconds'),
            physical_area_seconds=remap.get('physical_area_reconciliation_seconds'),
            height_stencil_seconds=remap.get('physical_height_stencil_seconds'),
            atmosphere_remap_seconds=remap.get('atmosphere_remap_seconds'),
            height_execution_timings=remap.get('height_execution_timings')))
    duration = sum(s['model_seconds'] for s in segments)
    if abs(duration-hours*3600) > 1e-5:
        raise ValueError(f'integration receipts do not cover exactly {hours} hours: {duration/3600:g}')
    steps = sum(s['fine_steps'] for s in segments)
    wall = guard['wall_seconds']
    return dict(wall_seconds=wall, seconds_per_simulated_day=wall*24/hours,
        wall_ms_per_fine_step=wall*1000/steps, fine_steps=steps,
        integration_seconds=sum(s['integration_seconds'] for s in segments),
        construct_seconds=sum(s['construct_seconds'] for s in segments),
        peak_vram_mib=guard['peak_vram_mib'], peak_owned_bytes=guard['peak_bytes'],
        visible_devices=guard['visible_devices'], devices=guard['devices'],
        source_commit=guard.get('source_commit'), segments=segments)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--moving', type=Path, required=True)
    p.add_argument('--moving-exit', type=Path, required=True)
    p.add_argument('--fixed', type=Path, required=True)
    p.add_argument('--fixed-exit', type=Path, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--cards', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.hours <= 0 or a.cards <= 0:
        p.error('hours and cards must be positive')
    result = dict(status='unverified', duration_hours=a.hours, cards=a.cards)
    try:
        on = measure(a.moving,a.moving_exit,a.hours)
        off = measure(a.fixed,a.fixed_exit,a.hours)
        if (on['visible_devices'],on['devices']) != (off['visible_devices'],off['devices']):
            raise ValueError('ON and OFF device identities differ')
        if on['fine_steps'] != off['fine_steps']:
            raise ValueError('ON and OFF fine-step counts differ')
        for arm in (on,off):
            arm['card_seconds_per_simulated_day'] = arm['seconds_per_simulated_day']*a.cards
        result.update(status='measured', moving=on, fixed=off,
            added_wall_ms_per_fine_step=on['wall_ms_per_fine_step']-off['wall_ms_per_fine_step'],
            added_wall_percent=100*(on['wall_seconds']/off['wall_seconds']-1),
            added_integration_percent=100*(on['integration_seconds']/off['integration_seconds']-1),
            caveat='One whole-case pair. Wall includes overlap of preparation and forecast; do not sum concurrent stage times. Compiler and physics parity require the accompanying launch and engine identity receipts.')
    except (OSError,ValueError,KeyError) as error:
        result['reason'] = str(error)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='measured' else 1


if __name__ == '__main__':
    raise SystemExit(main())
