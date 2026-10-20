"""Whole radiation-support module replay on native-tap PRE inputs, MPI1.

MPAS mpas_atmphys_o3climatology.F:59-327; module_ra_rrtmg_vinterp.F:22-177;
mpas_atmphys_driver_radiation_lw.F:414-506,688-712,845-863.
Rust decodes, encodes and compares every output word. Pristine strict Fortran
generates expectations independently; CUDA receives only native PRE inputs.
This stated real-case replay is not an assertion about every forecast call.
"""
import argparse
import ctypes as ct
from datetime import date
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from hexcore.mpasphys import radshared_v842 as rad
from prepare_tapped_support import sha


P = ct.POINTER(ct.c_float)


def ptr(value):
    return value.ctypes.data_as(P)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'transport', 'words', 'oracle', 'tables', 'decoder', 'work', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--date', required=True, help='Stated calendar replay date; real monthly inputs unchanged')
    args = parser.parse_args()
    import cupy as cp
    args.work.mkdir(exist_ok=False)
    native = ct.CDLL(str(args.oracle.resolve()))
    for routine, integers, floats, pointers in (
        ('ozone_init', 2, 0, 3), ('ozone_time', 2, 1, 2),
        ('ozone_update', 5, 0, 2), ('ozone_vertical_pa', 3, 0, 4),
        ('radius_from', 4, 0, 4), ('radius_to', 4, 0, 4)):
        function = getattr(native, routine)
        function.argtypes = [ct.c_int] * integers + [ct.c_float] * floats + [P] * pointers
        function.restype = None
    native.radiation_controls.argtypes = [ct.c_int] * 6 + [ct.POINTER(ct.c_int32)]
    resources = dict(rust_decoder=dict(path=str(args.decoder.resolve()), sha256=sha(args.decoder)),
        table_directory=str(args.tables.resolve()), nOznLevels=59,
        table_sha256={n: sha(args.tables / n) for n in
                     ('OZONE_DAT.TBL', 'OZONE_LAT.TBL', 'OZONE_PLEV.TBL')})
    tables = rad.load_ozone_resources(resources, cp)
    rows = []
    calendar = rad.OzoneCalendarState()
    year, month, day = map(int, args.date.split('-'))
    julian = np.float32(date(year, month, day).timetuple().tm_yday - .5)
    for number, entry in enumerate(json.loads(args.index.read_text())):
        work = args.work / str(number)
        work.mkdir()
        def record(name):
            member = entry['prefix'] + '.pre.' + name + '.bin'
            path = Path(entry['directory']) / member
            expected = entry['files'][member]
            if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError('native PRE input differs: ' + member)
            return path
        def metadata(name):
            return json.loads(subprocess.check_output([str(args.transport), 'metadata', str(record(name)), '-']))
        def field(name):
            raw = work / (name + '.raw')
            header = json.loads(subprocess.check_output([str(args.transport), 'decode', str(record(name)), str(raw)]))
            if header['code'] != 1 or header['width'] != 4:
                raise ValueError('expected native RKIND single')
            value = np.frombuffer(raw.read_bytes(), dtype='<f4').reshape(header['shape'], order='F').copy(order='F')
            raw.unlink()
            return value
        outputs = {}
        def compare(name, expected, actual):
            expected = np.ascontiguousarray(expected)
            actual = np.ascontiguousarray(cp.asnumpy(actual))
            if expected.shape != actual.shape:
                raise ValueError('output shape differs: ' + name)
            left, right = work / (name + '.native'), work / (name + '.cuda')
            left.write_bytes(expected.tobytes()); right.write_bytes(actual.tobytes())
            result = json.loads(subprocess.check_output([str(args.words), 'compare-raw', str(left), str(right)]))
            result.update(native_sha256=sha(left), cuda_sha256=sha(right), shape=list(expected.shape))
            outputs[name] = result
            left.unlink(); right.unlink()
        nc, solve = metadata('block.dimensions.nCells'), metadata('block.dimensions.nCellsSolve')
        latitude, monthly, pin = (field(n) for n in ('mesh.latCell', 'atm_input.ozmixm', 'atm_input.pin'))
        nl = pin.size
        expected_monthly = np.empty((nc, nl, 12), dtype='f4')
        native_pin = np.empty_like(pin)
        old = Path.cwd()
        os.chdir(args.tables.resolve())
        try:
            native.ozone_init(nc, nl, ptr(np.ascontiguousarray(latitude[:nc])), ptr(native_pin), ptr(expected_monthly))
        finally:
            os.chdir(old)
        initialized = rad.init_o3climatology(latCell=cp.asarray(latitude[:nc]),
            lat_ozone=tables['lat_ozone'], ozmixin=tables['ozmixin'])
        compare('ozone_initialization', expected_monthly, initialized)
        compare('pressure_table', native_pin, tables['pin'])
        # The captured monthly data are real PRE inputs to both implementations.
        month_input = np.ascontiguousarray(monthly[:, :, :solve].transpose(2, 1, 0))
        month_device = cp.asarray(month_input)
        climate = np.empty((solve, nl), dtype='f4')
        native.ozone_time(solve, nl, julian, ptr(month_input), ptr(climate))
        compare('ozone_CAM_time', climate, rad.o3climatology_from_MPAS(julian, month_device))
        native.ozone_update(solve, nl, year, month, day, ptr(month_input), ptr(climate))
        compare('ozone_monthly_date', climate, rad.update_o3climatology(args.date, month_device, calendar_state=calendar))
        pressure = field('state.mpas_atmphys_vars.pres_hyd_p')
        nz = pressure.shape[1] - 1
        native_pressure = np.ascontiguousarray(pressure[:solve, :nz, 0])
        pressure = cp.asarray(native_pressure)
        # Both paths independently execute the native LW Pa-to-hPa conversion.
        # The strict Fortran driver slice receives unchanged native PRE words.
        captured_climate = np.ascontiguousarray(field('diag_physics.o3clim')[:, :solve].T)
        expected = np.empty((solve, nz), dtype='f4')
        native.ozone_vertical_pa(solve, nz, nl, ptr(native_pressure), ptr(pin), ptr(captured_climate), ptr(expected))
        compare('ozone_vertical', expected, rad.vinterp_ozn(pressure, cp.asarray(pin),
            cp.asarray(captured_climate), pressure_in_pa=True))
        config = {name: metadata('block.configs.' + name) for name in (
            'config_radt_cld_overlap', 'config_radt_cld_dcorrlen', 'config_o3climatology',
            'config_microp_re', 'config_microp_scheme', 'config_radt_lw_scheme', 'config_radt_sw_scheme')}
        for direction, prefix in (('from', 're_'), ('to', 'rre_')):
            source = [np.ascontiguousarray(field('diag_physics.' + prefix + name)[:, :solve].T)
                      for name in ('cloud', 'ice', 'snow')]
            device = [cp.asarray(value) for value in source]
            mode = rad.RADIUS_SCHEMES.index(config['config_microp_scheme'])
            enabled = config['config_microp_re']
            expected = np.empty((3, solve, nz), dtype='f4')
            getattr(native, 'radius_' + direction)(solve, nz, mode, enabled, *map(ptr, source), ptr(expected))
            names = ('re_cloud', 're_ice', 're_snow') if direction == 'from' else ('rre_cloud', 'rre_ice', 'rre_snow')
            actual = getattr(rad, 'radii_' + direction + '_MPAS')(
                microp_scheme=config['config_microp_scheme'], microp_re=enabled, **dict(zip(names, device)))
            for index, name in enumerate(names):
                compare('radius_' + direction + '_' + name, expected[index], actual[index])
        options = rad.radiation_options(config)
        controls = np.empty(6, dtype='i4')
        pair = (config['config_radt_lw_scheme'] == 'rrtmg_lw', config['config_radt_sw_scheme'] == 'rrtmg_sw')
        pair_number = {(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}[pair]
        native.radiation_controls(tuple(rad.OVERLAP).index(config['config_radt_cld_overlap']),
            ('constant', 'latitude_varying', 'unknown').index(config['config_radt_cld_dcorrlen']),
            config['config_o3climatology'], enabled, mode, pair_number,
            controls.ctypes.data_as(ct.POINTER(ct.c_int32)))
        actual = np.asarray(tuple(vars(options).values()), dtype='i4')
        # Rust raw comparison treats integer output words identically to real words.
        compare('radiation_controls', controls, cp.asarray(actual))
        rows.append(dict(prefix=entry['prefix'], original_dump_root_sha256=entry['original_dump_root_sha256'],
            native_process_count=1, replay_process_count=1, arrays=outputs))
        print(entry['prefix'], sum(x['different_words'] for x in outputs.values()), flush=True)
    arrays = [a for row in rows for a in row['arrays'].values()]
    receipt = dict(schema='p21-native-tap-support-module-replay/v1', process_count=1, backend='hex-cuda',
        calendar_date=args.date, calendar_scope='stated date on real captured monthly PRE data',
        captures=rows, words=sum(a['words'] for a in arrays),
        differing_words=sum(a['different_words'] for a in arrays),
        all_outputs_bitwise_equal=all(a['bitwise_equal'] for a in arrays),
        expected_source='strict pristine Fortran whole module and verbatim driver slices',
        artifacts={str(p.resolve()): sha(p) for p in (args.index, args.oracle, args.transport, args.words, args.decoder,
                   Path(rad.__file__), Path(__file__))}, table_sha256=resources['table_sha256'],
        captured_private_state_complete=False, forecast_proven=False)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return int(not receipt['all_outputs_bitwise_equal'])


if __name__ == '__main__':
    raise SystemExit(main())
