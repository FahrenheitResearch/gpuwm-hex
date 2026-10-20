"""Replay the whole pristine ozone module at its actual nested initializer call.

The linker bridge is a proposed framework change, not a shipping forecast hook.
Rust decodes tables and real NetCDF latitudes; Python copies ABI storage only.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from gpuwm.netcdf_bridge import open_dataset
from hexcore.mpasphys import Backend, SlotCall, StepFlags
from hexcore.mpasphys import radshared_v842 as rad


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ptr(array):
    return array.ctypes.data_as(ct.POINTER(ct.c_float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'wrapped', 'tables', 'decoder', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--init', type=Path, action='append', default=[])
    parser.add_argument('--backend', choices=('native-cpu', 'hex-cuda'), required=True)
    parser.add_argument('--skip-initializer-control', action='store_true',
                        help='Negative control: leave poisoned callback outputs unwritten')
    args = parser.parse_args()
    assert not args.skip_initializer_control or args.backend == 'hex-cuda'
    original = ct.CDLL(str(args.oracle.resolve()))
    wrapped = ct.CDLL(str(args.wrapped.resolve()))
    for library in (original, wrapped):
        library.ozone_init.argtypes = [ct.c_int] * 2 + [ct.POINTER(ct.c_float)] * 3
        library.ozone_init.restype = None
        library.ozone_time.argtypes = [ct.c_int] * 2 + [ct.c_float] + [ct.POINTER(ct.c_float)] * 2
        library.ozone_time.restype = None
        library.ozone_update.argtypes = [ct.c_int] * 5 + [ct.POINTER(ct.c_float)] * 2
        library.ozone_update.restype = None
    wrapped.native_ozone_initializer_set.argtypes = [ct.c_void_p]
    wrapped.native_ozone_initializer_set.restype = None
    for name in ('native_ozone_initializer_status', 'native_ozone_initializer_calls'):
        getattr(wrapped, name).argtypes = []
        getattr(wrapped, name).restype = ct.c_int
    wrapped.ozone_callback_storage.argtypes = [ct.POINTER(ct.c_int)] * 2 + [ct.POINTER(ct.c_void_p)] * 3
    wrapped.ozone_callback_storage.restype = None
    native_initializer = getattr(wrapped, '__mpas_atmphys_o3climatology_MOD_init_o3climatology')
    native_initializer.argtypes = [ct.c_void_p] * 2
    native_initializer.restype = None
    resources = dict(rust_decoder=dict(path=str(args.decoder.resolve()), sha256=sha(args.decoder)),
        table_directory=str(args.tables.resolve()), nOznLevels=59,
        table_sha256={name: sha(args.tables / name) for name in
                     ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL')})
    decoded = rad.load_ozone_resources(resources, np)
    device = None
    if args.backend == 'hex-cuda':
        import cupy as cp
        device = rad.load_ozone_resources(resources, cp)
    latitudes = decoded['lat_ozone'].copy()
    # Exact final table latitude has its separate complete native-defect witness.
    latitudes[-1] += np.float32(.5)
    latitudes = np.r_[latitudes, np.float32(-95), np.float32(95)].astype('f4')
    latitudes *= np.float32(np.float32(3.1415926) / np.float32(180))
    cases = [('all table intervals and both out-of-range branches', latitudes)]
    for path in args.init:
        with open_dataset(path) as dataset:
            latitude = np.ascontiguousarray(dataset.variables['latCell'][:], dtype='f4')
        cases.append((str(path.resolve()), latitude))
    results = []
    # The native date module's DATA calendar state persists across calls.
    # Keep the same state across every case in this complete replay.
    cuda_calendar_state = rad.OzoneCalendarState()
    def compare(case, name, expected, actual):
        count = int(np.count_nonzero(expected.view('u4') != actual.view('u4')))
        results.append(dict(case=case, name=name, words=int(expected.size), differing_words=count))
        assert count == 0, (case, name, count)
    old = Path.cwd()
    os.chdir(args.tables.resolve())
    try:
        for case, latitude in cases:
            nc, nl = latitude.size, 59
            expected_pin = np.empty(nl, dtype='f4')
            expected = np.empty((12, nl, nc), dtype='f4', order='F')
            original.ozone_init(nc, nl, ptr(latitude), ptr(expected_pin), ptr(expected))
            actual_pin = np.empty_like(expected_pin)
            actual = np.empty_like(expected, order='F')
            wrapped.native_ozone_initializer_set(None)
            wrapped.ozone_init(nc, nl, ptr(latitude), ptr(actual_pin), ptr(actual))
            compare(case, 'unchanged native fallback monthly', expected, actual)
            compare(case, 'unchanged native fallback pressure', expected_pin, actual_pin)
            errors, received = [], []
            @ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)
            def callback(mesh, atm_input):
                try:
                    assert mesh and atm_input
                    received.append((mesh, atm_input))
                    if device is None:
                        native_initializer(mesh, atm_input)
                    else:
                        counts = (ct.c_int(), ct.c_int())
                        addresses = [ct.c_void_p() for _ in range(3)]
                        wrapped.ozone_callback_storage(*[ct.byref(x) for x in (*counts, *addresses)])
                        assert (counts[0].value, counts[1].value) == (nc, nl)
                        def storage(address, shape):
                            data = ct.cast(address, ct.POINTER(ct.c_float))
                            return np.ctypeslib.as_array(data, shape=(int(np.prod(shape)),)).reshape(shape, order='F')
                        host_latitude = storage(addresses[0], (nc,))
                        host_pressure = storage(addresses[1], (nl,))
                        host_monthly = storage(addresses[2], (12, nl, nc))
                        # Fresh Fortran allocations may reuse an earlier fallback's
                        # storage. Poison output-only pools before upload so an
                        # omitted CUDA write cannot inherit a native expected word.
                        host_pressure.fill(np.float32(-1.e30))
                        host_monthly.fill(np.float32(-2.e30))
                        pool = dict(nCells=nc, mesh=dict(latCell=cp.asarray(host_latitude)),
                            atm_input=dict(pin=cp.asarray(host_pressure),
                                ozmixm=cp.ascontiguousarray(cp.asarray(host_monthly))),
                            radshared_tables=device, radshared={})
                        call = SlotCall('cloudiness', 'cld_fraction', Backend.HEX_CUDA,
                            0, 0., 10., StepFlags(), pool, {})
                        if not args.skip_initializer_control:
                            rad.init_ozone_slot(call)
                        np.copyto(host_pressure, cp.asnumpy(pool['atm_input']['pin']))
                        np.copyto(host_monthly, cp.asnumpy(pool['atm_input']['ozmixm']))
                    return 0
                except BaseException as error:
                    errors.append(error)
                    return 1
            wrapped.native_ozone_initializer_set(ct.cast(callback, ct.c_void_p))
            try:
                wrapped.ozone_init(nc, nl, ptr(latitude), ptr(actual_pin), ptr(actual))
                assert not errors and wrapped.native_ozone_initializer_status() == 0, errors
                assert wrapped.native_ozone_initializer_calls() == len(received) == 1
            finally:
                wrapped.native_ozone_initializer_set(None)
            compare(case, args.backend + ' nested initializer monthly', expected, actual)
            compare(case, args.backend + ' nested initializer pressure', expected_pin, actual_pin)
            if device is not None:
                # Native ABI is [month,level,cell]; the scheme interface is
                # [cell,level,month]. These copies only change storage order.
                monthly_device = cp.ascontiguousarray(cp.asarray(actual).transpose(2, 1, 0))
            for julian in (0., 14.5, 44., 74., 104.875, 135., 165., 196., 227., 257., 288., 318., 349.75, 364.5, 365.25, 730.125):
                a, b = (np.empty((nl, nc), dtype='f4', order='F') for _ in range(2))
                original.ozone_time(nc, nl, julian, ptr(expected), ptr(a))
                wrapped.ozone_time(nc, nl, julian, ptr(actual), ptr(b))
                compare(case, 'calendar ' + str(julian), a, b)
                if device is not None:
                    value = rad.o3climatology_from_MPAS(julian, monthly_device)
                    compare(case, 'CUDA calendar ' + str(julian), a,
                            np.asfortranarray(cp.asnumpy(value).T))
            for year, month, day in ((2000, 2, 29), (2001, 3, 1), (2100, 2, 28), (2100, 3, 1), (2000, 12, 31)):
                a, b = (np.empty((nl, nc), dtype='f4', order='F') for _ in range(2))
                original.ozone_update(nc, nl, year, month, day, ptr(expected), ptr(a))
                wrapped.ozone_update(nc, nl, year, month, day, ptr(actual), ptr(b))
                compare(case, 'date ' + str((year, month, day)), a, b)
                if device is not None:
                    value = rad.update_o3climatology(
                        f'{year:04d}-{month:02d}-{day:02d}', monthly_device,
                        calendar_state=cuda_calendar_state)
                    compare(case, 'CUDA date ' + str((year, month, day)), a,
                            np.asfortranarray(cp.asnumpy(value).T))
            print(json.dumps(dict(complete_case=case, nCells=nc, differing_words=0)), flush=True)
    finally:
        os.chdir(old)
    receipt = dict(schema='p21-whole-ozone-nested-bridge-replay/v1', process_count=1,
        backend=args.backend, results=results, differing_words=sum(x['differing_words'] for x in results),
        compared_words=sum(x['words'] for x in results), native_expected_never_cuda_inputs=True,
        output_allocations_poisoned=args.backend == 'hex-cuda',
        skip_initializer_negative_control=args.skip_initializer_control,
        artifacts={str(p.resolve()): sha(p) for p in (args.oracle, args.wrapped, args.decoder, Path(rad.__file__), Path(__file__))},
        proposed_framework_bridge_only=True, production_nested_hook_executed=False, forecast_acceptance=False)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
