"""Five complete driver updates through f4's actual CUDA copy transaction.

The strict native KF driver is the oracle, including step1/step2 and halos.
This is a driver replay, never CLI or whole-forecast acceptance.
MPAS: mpas_atmphys_driver_convection.F:405-621,991-1102.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
from hexcore.mpasphys import kain_fritsch_v842 as kf
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from support import PARAMS, fixture, word_diff


def replay(library, loader, receipt):
    import cupy as cp
    spec = importlib.util.spec_from_file_location('kf_transaction_native', loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nz, nc, owned = 55, 3, 2
    columns = [fixture(nz, 8, .95, 300, 2), fixture(nz, 3, .3, 285, 0), fixture(nz)]
    fields = np.stack([x[0] for x in columns], axis=2)
    state = np.stack([x[1] for x in columns], axis=2)
    surface = np.stack([x[2] for x in columns], axis=1)
    state[:, :, owned:] = np.float32(99)
    surface[2:4, owned:] = np.float32(99)
    surface[4, owned:] = np.float32(300)
    surface[5:7, owned:] = np.float32(99)
    rows = []
    with module.NativePhysics(library, test_only=True) as native:
        for prefix, size, total in [('i', owned, nc), ('j', 1, 1), ('k', nz, nz + 1)]:
            for scope in ['d', 'm', 't']:
                native.put_variable(prefix + scope + 's', np.asarray(1, np.int32))
                native.put_variable(prefix + scope + 'e', np.asarray(
                    total if scope in ('d', 'm') else size, np.int32))
        for name, value in [('dt_dyn', 60.), ('dt_cu', 300.)]:
            native.put_variable(name, np.asarray(value, np.float32))
        native.put_variable('n_cu', np.asarray(5, np.int32))
        for name in ('f_qr', 'f_qi', 'f_qs', 'l_acrain'):
            native.put_variable(name, np.asarray(True))
        for name, value in [('config_convection_scheme', kf.SCHEME),
                            ('config_len_disp', 0.), ('config_do_restart', False),
                            ('config_gfconv_closure_deep', 3),
                            ('config_gfconv_closure_shallow', 3),
                            ('config_bucket_rainc', 1.)]:
            native.config(name, value)
        native.pool('sfc_input')
        native.allocate_convection(configs='configs')
        for i, name in enumerate(kf._PREPARED_FIELDS):
            native.put_variable(name, fields[i].T[:, :, None].copy())
        native.put_field('mesh', 'meshDensity', np.ones(nc, np.float32))
        native.put_field('mesh', 'areaCell', surface[1].copy())
        native.put_field('diag_physics', 'w0avg', state[0, :nz].copy())
        for i, name in enumerate(kf._SURFACE_POOL):
            if i >= 2:
                native.put_field('diag_physics', name, surface[i].copy())
        for i, name in enumerate(kf.STATE):
            if i >= 1:
                native.put_field('tend_physics', name, state[i, :nz].copy())
        rain = np.zeros(nc, np.float32)
        counters = np.zeros(nc, np.int32)
        rain[owned:], counters[owned:] = np.float32(99), np.int32(99)
        native.put_field('diag_physics', 'rainc', rain)
        native.put_field('diag_physics', 'i_rainc', counters)
        lut = native.lib.__module_cu_kfeta_MOD_kf_lutab
        lut.argtypes = [C.POINTER(C.c_float)] * 4
        constants = [C.c_float(x) for x in PARAMS[12:]]
        lut(*[C.byref(x) for x in constants])
        layout = SimpleNamespace(
            dimensions={'nCellsSolve': owned}, configs={'config_len_disp': 0.},
            pools=('mesh', 'diag_physics', 'tend_physics'),
            fields=[dict(pool=pool, name=name, pool_name=name, dimensions=['nCells'])
                    for pool, name in native._fields])
        provider = NativeDevicePoolViews(SimpleNamespace(native=native, layout=layout), cp)
        # F_QI=true makes the unassigned native warm_rain branch irrelevant.
        provider.kain_fritsch_native_arguments = kf.load_native_arguments(
            {'warm_rain_word': 0, 'witness_sha256': 'a' * 64}, cp)
        descriptor = {'module': 'kain_fritsch_v842', 'bind': 'bind_native_pools'}
        names = tuple(native._fields)
        update = dict(configs='configs', diag_physics='diag_physics', its=1, ite=owned)
        for step, displacement in enumerate((125000., 100000., 75000., 230000., 75000.), 1):
            native.config('config_len_disp', displacement)
            native.update_convection_step1(tend_physics='tend_physics', **update)
            before = {key: native.get_field(*key).copy() for key in names}
            prepared_before = {name: native.get_variable(name)[0].copy()
                               for name in kf._PREPARED_FIELDS}
            native.driver_convection(itimestep=step, mesh='mesh', sfc_input='sfc_input',
                                     tend_physics='tend_physics', **update)
            native.update_convection_step2(**update)
            expected = {key: native.get_field(*key).copy() for key in names}
            for key, value in before.items():
                native.put_field(*key, value)
            call = SimpleNamespace(slot='convection', dt=300., itimestep=step,
                                   options={'config_len_disp': 0.})
            call.pools = provider.begin(call, descriptor)
            kf.launch(call)
            provider.commit()
            native.update_convection_step2(**update)
            differences = {f'{pool}.{name}': word_diff(
                native.get_field(pool, name).copy(), expected[pool, name])
                for pool, name in names}
            differences.update({f'prepared.{name}': word_diff(
                native.get_variable(name)[0].copy(), prepared_before[name])
                for name in kf._PREPARED_FIELDS})
            row = dict(step=step, live_displacement=displacement, differences=differences)
            rows.append(row)
            print(json.dumps(row), flush=True)
        native.deallocate_convection(configs='configs')
    result = dict(schema='mpas-kf-real-device-provider-driver/v1',
                  process_count=1, execution_host=socket.gethostname(),
                  owned_columns=owned, halo_columns=nc-owned, steps=len(rows),
                  backend='hex-cuda', forecast_proven=False, cli_selection_proven=False,
                  warm_rain='branch irrelevant with native F_QI=true',
                  differences=sum(sum(row['differences'].values()) for row in rows), rows=rows,
                  artifacts={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (library, loader, Path(__file__), Path(kf.__file__),
                                          Path(inspect.getfile(NativeDevicePoolViews)))})
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--loader', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(bool(replay(args.library, args.loader, args.receipt)['differences']))
