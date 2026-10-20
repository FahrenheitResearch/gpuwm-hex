"""Replay the complete ozone table/init/time adapters on genuine native pools.

Rust owns table decoding. The pristine full-module oracle supplies expected
words independently of CUDA inputs. All native halo and padding words are compared.
This is a synthetic column/pool proof, never a forecast or nested-init proof.
"""
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from hexcore.mpasphys import Backend, SlotCall, StepFlags
from hexcore.mpasphys import radshared_v842 as rad


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def ptr(array):
    return array.ctypes.data_as(ct.POINTER(ct.c_float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'provider', 'decoder', 'oracle', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    native_module = module('hexcore.mpasphys._p21_ozone_native', args.provider / 'native_backend.py')
    device_module = module('hexcore.mpasphys._p21_ozone_device', args.provider / 'device_pools_v842.py')
    tables_directory = args.reference / 'src/core_atmosphere/physics/physics_wrf/files'
    resources = dict(rust_decoder=dict(path=str(args.decoder), sha256=sha(args.decoder)),
        table_directory=str(tables_directory), nOznLevels=59,
        table_sha256={name: sha(tables_directory / name) for name in
                     ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL')})
    decoded = rad.load_ozone_resources(resources, np)
    tables = rad.load_ozone_resources(resources, cp)
    # Test fixtures may use NumPy. Production interpolation stays in CUDA.
    levels, solve, total = 59, 64, 66
    degrees = np.r_[decoded['lat_ozone'], decoded['lat_ozone'][0] - np.float32(1),
                    decoded['lat_ozone'][-1] + np.float32(1)].astype('f4')
    degrees[63] += np.float32(.5)  # avoid the separately witnessed exact-final-latitude native overrun
    latitude = (degrees * np.float32(np.float32(3.1415926) / np.float32(180))).astype('f4')
    oracle = ct.CDLL(str(args.oracle))
    oracle.ozone_init.argtypes = [ct.c_int, ct.c_int, *[ct.POINTER(ct.c_float)] * 3]
    oracle.ozone_time.argtypes = [ct.c_int, ct.c_int, ct.c_float, *[ct.POINTER(ct.c_float)] * 2]
    oracle.ozone_update.argtypes = [ct.c_int] * 5 + [ct.POINTER(ct.c_float)] * 2
    for routine in ('ozone_init', 'ozone_time', 'ozone_update'):
        getattr(oracle, routine).restype = None
    native_pin = np.empty(levels, dtype='f4')
    expected_monthly = np.empty((12, levels, total), dtype='f4', order='F')
    old = Path.cwd()
    os.chdir(tables_directory.resolve())
    try:
        oracle.ozone_init(total, levels, ptr(latitude), ptr(native_pin), ptr(expected_monthly))
        monthly_solve = np.empty((12, levels, solve), dtype='f4', order='F')
        oracle.ozone_init(solve, levels, ptr(latitude[:solve]), ptr(native_pin), ptr(monthly_solve))
    finally:
        os.chdir(old)
    np.testing.assert_array_equal(native_pin.view('u4'), decoded['pin'].view('u4'))
    registry = args.reference / 'src/core_atmosphere/Registry.xml'
    tree = ET.parse(registry).getroot()
    wanted = {'mesh': ('latCell',), 'atm_input': ('ozmixm', 'pin'), 'diag_physics': ('o3clim',)}
    rows = []
    for pool, names in wanted.items():
        struct = tree.find(f"./var_struct[@name='{pool}']")
        for name in names:
            row = next(v for v in struct if v.get('name_in_code', v.get('name')) == name)
            rows.append(dict(row.attrib, pool=pool, pool_name=name,
                dimensions=[dim for dim in row.get('dimensions', '').split() if dim != 'Time']))
    dims = dict(nCells=total, nCellsSolve=solve, nMonths=12, nOznLevels=levels)
    layout = SimpleNamespace(pools=set(wanted), fields=rows, dimensions=dims, configs={})
    calendar = dict(julian=np.float32(0), current_date='2000-01-01_12:00:00.0000')
    results = []
    def compare(name, actual, expected):
        actual = np.asarray(actual)
        np.testing.assert_array_equal(actual.view('u4'), expected.view('u4'), err_msg=name)
        results.append(dict(name=name, words=actual.size, differing_words=0))
    with native_module.NativePhysics(args.provider / 'libmpasphys_native.so', test_only=True) as native:
        original_monthly = np.full((12, levels, total + 1), np.float32(-91), order='F')
        original_monthly[:, :, -1] = np.float32(-0.)
        initial_ozone = np.full((levels, total + 1), np.float32(-37), order='F')
        initial_ozone[:, -1] = np.float32(-0.)
        mesh_latitude = np.r_[latitude, np.float32(-0.)]
        native.put_field('mesh', 'latCell', mesh_latitude)
        native.put_field('atm_input', 'pin', np.zeros(levels, dtype='f4'))
        native.put_field('atm_input', 'ozmixm', original_monthly)
        native.put_field('diag_physics', 'o3clim', initial_ozone)
        session = SimpleNamespace(native=native, layout=layout,
                                  arguments=SimpleNamespace(calendar=calendar))
        provider = device_module.NativeDevicePoolViews(session, cp)
        provider.radshared_v842_tables = tables
        call = SlotCall('ozone', 'o3climatology', Backend.HEX_CUDA, 0, 0., 10., StepFlags(), None, {})
        provider.begin(call, {})
        pools = rad.bind_native_ozone_initial_pools(provider, call)
        rad.init_ozone_slot(replace(call, pools=pools))
        shared = pools['radshared']
        provider.commit()
        complete_monthly = original_monthly.copy(order='F')
        complete_monthly[:, :, :total] = expected_monthly
        compare('all nCells monthly initialization and ABI padding', native.get_field('atm_input', 'ozmixm'), complete_monthly)
        compare('strict native pressure table', native.get_field('atm_input', 'pin'), native_pin)
        compare('initialization preserves all latitude words', native.get_field('mesh', 'latCell'), mesh_latitude)
        for julian in (0., 14.5, 44., 74., 104.875, 135., 165., 196., 227., 257., 288., 318., 349.75, 364.5, 365.25, 730.125):
            calendar['julian'] = np.float32(julian)
            provider.begin(call, {})
            pools = rad.bind_native_ozone_pools(provider, call)
            assert pools['radshared'] is shared
            rad.ozone_slot(replace(call, pools=pools))
            provider.commit()
            expected = initial_ozone.copy(order='F')
            active = np.empty((levels, solve), dtype='f4', order='F')
            oracle.ozone_time(solve, levels, ct.c_float(julian), ptr(monthly_solve), ptr(active))
            expected[:, :solve] = active
            compare('CAM calendar ' + str(julian), native.get_field('diag_physics', 'o3clim'), expected)
        for year, month, day in ((2000, 2, 29), (2001, 3, 1), (2100, 2, 28), (2100, 3, 1), (2000, 12, 31)):
            calendar['current_date'] = f'{year:04d}-{month:02d}-{day:02d}_12:00:00.0000'
            provider.begin(call, {})
            pools = rad.bind_native_ozone_pools(provider, call)
            rad.update_ozone_slot(replace(call, pools=pools))
            provider.commit()
            active = np.empty((levels, solve), dtype='f4', order='F')
            oracle.ozone_update(solve, levels, year, month, day, ptr(monthly_solve), ptr(active))
            expected = initial_ozone.copy(order='F')
            expected[:, :solve] = active
            compare('persistent monthly calendar ' + calendar['current_date'], native.get_field('diag_physics', 'o3clim'), expected)
        compare('every monthly input/halo/padding preserved by time updates', native.get_field('atm_input', 'ozmixm'), complete_monthly)
    receipt = dict(schema='p21-full-ozone-resource-native-pool-cuda-replay-v1',
        process_count=1, backend='hex-cuda', results=results,
        total_words=sum(r['words'] for r in results), differing_words=0,
        table_sha256=resources['table_sha256'],
        source_sha256={str(path): sha(path) for path in (Path(rad.__file__), Path(__file__), args.decoder,
            args.oracle, args.provider / 'libmpasphys_native.so', args.provider / 'native_backend.py',
            args.provider / 'device_pools_v842.py', registry)},
        full_nCells_init_and_nCellsSolve_time_updates=True, native_halo_and_padding_compared=True,
        native_expected_never_used_as_cuda_inputs=True, forecast_proven=False,
        nested_native_initialization_hook_executed=False,
        exact_final_native_latitude_defect_not_excluded_from_a_forecast_claim=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('backend', 'total_words', 'differing_words', 'forecast_proven')}))


if __name__ == '__main__':
    main()
