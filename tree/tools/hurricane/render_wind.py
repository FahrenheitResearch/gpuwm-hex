"""Redraw retained surface evidence on a fixed 0-200 kt Rust colour scale.

The source reduction is checked against the closed-history variable hashes.
Only the original surface variables are restored, without synthetic profiles.
The normal Rust converter supplies the lattice and gpuwm.rustwx draws it.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from netCDF4 import Dataset
from gpuwm import rustwx
from hexcore import render_door


def draw(frame_dir, member, case, window, output, keep_cf=False):
    began = time.monotonic()
    record = json.loads((frame_dir / 'receipt.json').read_text())
    valid = datetime.strptime(case['start_time'], '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
    valid += timedelta(seconds=record['frame']['model_seconds'])
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / 'scratch'
    scratch.mkdir(exist_ok=True)
    native = scratch / ('history.' + valid.strftime('%Y-%m-%d_%H.%M.%S') + '.nc')
    selected = {}
    with np.load(frame_dir / 'surface.npz', allow_pickle=False) as surface:
        for name in ('latCell', 'lonCell', 'ter', 'u10', 'v10', 't2', 'surface_pressure', 'rainc', 'rainnc'):
            value = np.asarray(surface[name])
            expected = record['fields'][name]
            if (list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
                raise ValueError(f'retained {name} differs from the closed history: {frame_dir}')
            selected[name] = value
    with Dataset(native, 'w', format='NETCDF3_64BIT_OFFSET') as nc:
        nc.createDimension('Time', 1)
        nc.createDimension('nCells', selected['latCell'].size)
        nc.createDimension('nVertLevels', int(case['n_levels']))
        nc.createDimension('StrLen', 64)
        nc.createVariable('xtime', 'S1', ('Time', 'StrLen'))[:] = np.array(
            list(valid.strftime('%Y-%m-%d_%H:%M:%S').ljust(64)), dtype='S1')[None, :]
        for name, value in selected.items():
            dims = ('nCells',) if value.ndim == 1 else ('Time', 'nCells')
            nc.createVariable(name, value.dtype, dims)[:] = value
    report, command = render_door.convert_frames(render_door.resolve_convert_exe(),
        histories=[native], mesh=Path(member['static']), scratch=scratch,
        window=window, field_set='surface', simulation_start=case['start_time'],
        nc_format='cdf2', model_label='hex retained surface', remap='product')
    converted = Path(report['frames'][0]['output'])
    cf = output / 'surface.cf.nc'
    with Dataset(converted) as source, Dataset(cf, 'w', format='NETCDF3_64BIT_OFFSET') as dest:
        lat, lon = np.asarray(source['XLAT'][0]), np.asarray(source['XLONG'][0])
        u, v = np.asarray(source['U10'][0]), np.asarray(source['V10'][0])
        speed = np.hypot(u, v)
        if not np.isfinite(speed).all() or float(speed.max()) * 1.9438444924406 > 200:
            raise ValueError('wind is nonfinite or exceeds the fixed 200 kt scale; choose an explicit wider scale')
        dest.source = 'hex forecast, exact retained surface; Rust product remap'
        dest.createDimension('time', 1)
        dest.createDimension('y', lat.shape[0])
        dest.createDimension('x', lat.shape[1])
        t = dest.createVariable('time', 'f8', ('time',))
        t.units = 'seconds since 1970-01-01 00:00:00 UTC'
        t.standard_name = 'time'
        t[:] = [valid.timestamp()]
        for name, value, units, standard in [('lat', lat, 'degrees_north', 'latitude'),
                ('lon', lon, 'degrees_east', 'longitude')]:
            var = dest.createVariable(name, 'f4', ('y', 'x'))
            var.units, var.standard_name = units, standard
            var[:] = value
        fields = [('wind10', speed, 'm s-1')]
        for name in ('RAINC', 'RAINNC'):
            if name in source.variables:
                fields.append((name.lower(), np.asarray(source[name][0]), 'mm'))
        for name, value, units in fields:
            var = dest.createVariable(name, 'f4', ('time', 'y', 'x'))
            var.units = units
            var[:] = value[None]
    table = output / 'products.json'
    table.write_text(json.dumps([dict(names=['wind10'], standard_names=[], product='wind10_tc',
        title='10 m wind speed, fixed TC scale', default_units='m s-1', display_units='kt',
        unit_factors={'m s-1': 1.9438444924406}, levels=[0,20,40,60,80,100,120,140,160,200])], indent=2)+'\n')
    rc = rustwx.run_cf_renderer(render_door.resolve_renderer_exe(), cf, variable='wind10',
        out_dir=output / 'map', table=table, valid=valid.isoformat(), size=(1000,800),
        receipt=output / 'render-receipt.json')
    if rc:
        raise RuntimeError(f'Rust CF renderer exited {rc}; retained scratch and source')
    summary = dict(original_history_sha256=record['history_sha256'], frame=record['frame'],
        converter=report, command=command, wind_scale_kt=[0,200],
        peak_render_wind_kt=float(speed.max())*1.9438444924406,
        wall_seconds=time.monotonic()-began)
    (output / 'receipt.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.rmtree(scratch)
    if not keep_cf:
        cf.unlink()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--window', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--keep-cf', action='store_true')
    a = p.parse_args()
    controller = a.run / 'controller.json'
    state = json.loads((controller if controller.exists() else a.run / 'command.json').read_text())
    case = json.loads(Path(state['case']).read_text())
    members = state.get('members') or {'fixed': state['member']}
    for receipt in sorted(a.evidence.glob('s*/receipt.json')):
        record = json.loads(receipt.read_text())
        if record['rendering'] is None:
            continue
        draw(receipt.parent, members[record['frame']['member']], case, a.window,
             a.out / receipt.parent.name, a.keep_cf)


if __name__ == '__main__':
    main()
