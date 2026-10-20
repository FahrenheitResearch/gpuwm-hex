"""Observe real forecast MM5 calls against pristine strict standalone Fortran.

This diagnostic invokes the real CLI and leaves every CUDA output untouched.
It cannot replace complete verify-mpas forecast acceptance. Device transfers
are column-oracle transport; weather files remain the Rust driver's data path.
MPAS physics_mmm/sf_sfclayrev.F90:27-1115;
physics_wrf/module_sf_sfclay.F:14-964.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--audit-transactions', action='store_true',
                        help='Observe every actual native-pool upload and its readback')
    parser.add_argument('--retain-prepared-input-storage', action='store_true',
                        help='Diagnostic: omit verified unchanged prepared-input reallocations')
    parser.add_argument('forecast_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import cupy as cp
    import numpy as np
    from hexcore.mpasphys import sfclay_mm5_v842 as mm5
    from hexcore.cli import main as real_cli
    from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
    library = args.oracle / 'liboracle.so'
    native = ctypes.CDLL(str(library))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    native.oracle_run.argtypes = [ptr, ptr] + [ctypes.c_int] * 7
    native.oracle_init()
    original = mm5.launch_columns
    receipt = dict(scope='Real forecast live-column diagnostic, no output replacement',
                   process_count=1, forecast_acceptance=False,
                   oracle_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                   cuda_source_sha256=hashlib.sha256(mm5.CUDA_SOURCE.encode()).hexdigest(),
                   module_sha256=hashlib.sha256(Path(mm5.__file__).read_bytes()).hexdigest(),
                   calls=[], transactions=[], excluded_words=[],
                   retain_prepared_input_storage=args.retain_prepared_input_storage)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def seal():
        temporary = args.receipt.with_suffix('.writing')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(args.receipt)

    def observed(packed, constants, **options):
        expected = cp.asnumpy(packed)
        c = cp.asnumpy(constants)
        row = dict(index=len(receipt['calls']), scheme=options['scheme'],
                   columns=packed.shape[1], input_nonfinite_words=int(
                       np.count_nonzero(~np.isfinite(expected))), fields={})
        native.oracle_run(expected, c, packed.shape[1],
                          int(options['scheme'] == mm5.SCHEMES[0]),
                          options.get('isfflx', 1), options.get('isftcflx', 0),
                          options.get('iz0tlnd', 0), options.get('scm_force_flux', 0),
                          options.get('shalwater_z0', 0))
        result = original(packed, constants, **options)
        actual = cp.asnumpy(packed)
        for index, field in enumerate(mm5.FIELDS):
            a, e = actual[index].view(np.uint32), expected[index].view(np.uint32)
            different = np.flatnonzero(a != e)
            row['fields'][field] = dict(words=int(a.size),
                differing_words=int(different.size),
                first_differing_index=int(different[0]) if different.size else None,
                actual_first_word=int(a[different[0]]) if different.size else None,
                native_first_word=int(e[different[0]]) if different.size else None,
                nonfinite_actual_words=int(np.count_nonzero(~np.isfinite(actual[index]))),
                nonfinite_native_words=int(np.count_nonzero(~np.isfinite(expected[index]))))
        row['differing_words'] = sum(v['differing_words'] for v in row['fields'].values())
        receipt['calls'].append(row)
        seal()
        return result

    mm5.launch_columns = observed
    original_commit = NativeDevicePoolViews.commit

    # MPAS mpas_atmphys_interface.F:304-305: audit ABI transport, no replacement.
    def observed_commit(provider):
        retained = {}
        if args.retain_prepared_input_storage:
            # MPAS driver_sfclayer.F:1016-1088 reads these atmospheric arrays.
            # Verify unchanged words before isolating the adapter's reallocation.
            for key, (host, packed, lower) in list(provider.buffers.items()):
                if key[0] != 'prepared' or key[1] not in (
                        'u_p', 'v_p', 't_p', 'qv_p', 'pres_hyd_p', 'dz_p', 'psfc_p'):
                    continue
                expected = (host[:packed.shape[1], :, 0].T
                            if host.ndim == 3 else host[:packed.shape[0], 0])
                actual = packed.get()
                if not np.array_equal(np.ascontiguousarray(expected).view(np.uint32),
                                      np.ascontiguousarray(actual).view(np.uint32)):
                    raise ValueError('Prepared input changed: isolation would drop scientific words')
                retained[str(key)] = dict(words=int(actual.size), differing_words=0)
                del provider.buffers[key]
        snapshots = [(key, packed.get().copy(), selection)
                     for key, (host, packed, selection) in provider.buffers.items()]
        result = original_commit(provider)
        fields = {}
        for key, expected, selection in snapshots:
            if key[0] == 'prepared':
                host, lower = provider.native.get_variable(key[1])
                if lower != selection:
                    raise ValueError('Native prepared bounds changed during commit')
                actual = (host[:expected.shape[1], :, 0].T
                          if host.ndim == 3 else host[:expected.shape[0], 0])
            else:
                actual = provider.native.get_field(key[0], key[1], level=key[2])[selection]
            a = np.ascontiguousarray(actual).view(np.uint32).ravel()
            e = np.ascontiguousarray(expected).view(np.uint32).ravel()
            different = np.flatnonzero(a != e)
            fields[str(key)] = dict(words=int(a.size), differing_words=int(different.size),
                first_differing_index=int(different[0]) if different.size else None,
                native_first_word=int(a[different[0]]) if different.size else None,
                device_first_word=int(e[different[0]]) if different.size else None,
                nonfinite_words=int(np.count_nonzero(~np.isfinite(actual))))
        receipt['transactions'].append(dict(index=len(receipt['transactions']), fields=fields,
            unchanged_prepared_inputs_retained=retained,
            differing_words=sum(v['differing_words'] for v in fields.values())))
        seal()
        return result

    if args.audit_transactions or args.retain_prepared_input_storage:
        NativeDevicePoolViews.commit = observed_commit
    command = args.forecast_arguments
    if command[:1] == ['--']:
        command = command[1:]
    receipt['real_cli_arguments'] = command
    seal()
    try:
        result = real_cli(command)
        receipt['real_cli_exit'] = result
        return result
    except BaseException as error:
        receipt['real_cli_exception'] = repr(error)
        raise
    finally:
        mm5.launch_columns = original
        NativeDevicePoolViews.commit = original_commit
        native.oracle_finalize()
        seal()


if __name__ == '__main__':
    raise SystemExit(main())
