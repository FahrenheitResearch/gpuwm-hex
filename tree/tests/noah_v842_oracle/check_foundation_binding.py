"""Whole Noah slot through actual f3 pools and f4's device provider.

Compare every pool word against the pristine standalone driver fixture, including
halos, all four precipitation/albedo option pairs and full atmospheric strides.
This is a column integration witness, not a forecast proof.
"""
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hexcore.mpasphys import noah_v842 as noah
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.mpasphys.driver_v842 import Backend, SlotCall
from hexcore.mpasphys.native_backend import NativePhysics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--table-directory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--backend', choices=('host', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.backend == 'cuda':
        import cupy as device
    else:
        device = np
    base = Path(__file__).resolve().parent
    decoder = base / 'table_decode'
    resources = noah.load_native_resources(dict(
        table_directory=str(args.table_directory), landuse='MODIFIED_IGBP_MODIS_NOAH',
        soil_type='STAS', rust_decoder=dict(path=str(decoder),
            sha256=hashlib.sha256(decoder.read_bytes()).hexdigest())), device)
    rows = []
    # The native fixture used cells 2:9. Move that contiguous solve interval to
    # the prefix, with the same inactive cells retained as halos.
    order = np.array([2, 3, 4, 5, 6, 7, 8, 0, 1, 9, 10])
    sizes = {'soil': 4, 'atmosphere': 2, 'pressure': 2}
    offsets = {}
    width = 0
    for name, shape in noah._MPAS_INPUT_FIELDS.items():
        if shape != 'integer_surface':
            offsets[name] = width
            width += sizes.get(shape, 1)
    with NativePhysics(args.library, test_only=True) as native:
        # MODIS has a table but no native isurban assignment. Preserve the
        # prior MODIFIED_IGBP_MODIS_NOAH word instead of rejecting it.
        for classification, mode in ((label, mode) for label in
                ('MODIFIED_IGBP_MODIS_NOAH', 'MODIS') for mode in range(4)):
            fixture = np.load(Path('n1out') / f'pipeline-{mode}.npz')
            fields, expected, metadata = {}, {}, []
            for name, shape in noah._MPAS_INPUT_FIELDS.items():
                if shape == 'integer_surface':
                    value = fixture['input_int'][order, 0 if name == 'ivgtyp' else 1].copy()
                    want = value.copy()
                else:
                    size = sizes.get(shape, 1)
                    k = offsets[name]
                    value = fixture['input'][order, k:k+size].copy()
                    want = fixture['native'][order, k:k+size].copy()
                    if size == 1:
                        value, want = value[:, 0], want[:, 0]
                    elif shape in ('atmosphere', 'pressure'):
                        depth = 55 if shape == 'atmosphere' else 56
                        padded = np.full((11, depth), np.float32(-999))
                        padded[:, :size] = value
                        value = padded
                        padded = np.full((11, depth), np.float32(-999))
                        padded[:, :size] = want
                        want = padded
                if name in noah._NATIVE_PREPARED_FIELDS:
                    variable = noah._NATIVE_PREPARED_FIELDS[name]
                    native.put_variable(variable, value[..., None])
                    continue
                pool, field = noah._NATIVE_POOL_FIELDS[name]
                if shape == 'soil':
                    value, want = value.T.copy(), want.T.copy()
                native.put_field(pool, field, value)
                fields[name] = pool, field
                expected[name] = want
                metadata.append(dict(pool=pool, name=field, pool_name=field,
                    dimensions=['nSoilLevels', 'nCells'] if shape == 'soil' else ['nCells']))
            native.put_text_field('sfc_input', 'mminlu', classification)
            ct.c_int.in_dll(native.lib, '__mpas_atmphys_landuse_MOD_isurban').value = 13
            native.put_field('sfc_input', 'isice', np.array(15, np.int32))
            native.put_variable('dt_pbl', np.array(60, np.float32))
            native.put_variable('xice_threshold', np.array(.55, np.float32))
            layout = SimpleNamespace(dimensions={'nCells': 11, 'nCellsSolve': 7},
                                     fields=metadata, pools={'sfc_input', 'diag_physics'},
                                     configs={})
            provider = NativeDevicePoolViews(SimpleNamespace(native=native, layout=layout), device)
            provider.noah_v842_resources = {**resources, 'landuse': classification}
            options = fixture['options']
            call = SlotCall('land_surface', 'sf_noah', Backend.HEX_CUDA, 2, 60, 999,
                None, provider, dict(config_sfc_albedo=int(options[3]),
                    config_microp_scheme='mp_thompson' if options[4] else 'off',
                    config_convection_scheme='cu_ntiedtke' if options[5] else 'off'))
            arguments = provider.begin(call, dict(module='noah_v842', bind='bind_native_pools'))
            if args.backend == 'cuda':
                noah.run_slot(replace(call, pools=arguments))
            else:
                run_host_slot(arguments, call, base)
            provider.commit()
            differences = {}
            for name, (pool, field) in fields.items():
                actual = native.get_field(pool, field)
                # Native to-MPAS also assigns unchanged fields; compare all of
                # them, not just the production output-name list.
                count = int(np.count_nonzero(actual.view(np.uint32) != expected[name].view(np.uint32)))
                if count:
                    differences[name] = count
            rows.append(dict(options=mode, classification=classification, native_isurban=arguments['isurban'], columns=7, process_count=1,
                native_execution='standalone, no MPI', atmospheric_levels=55,
                pressure_interfaces=56, halo_cells=4, all_pool_fields=len(fields),
                differing_words=sum(differences.values()), by_field=differences,
                native_dt_pbl=float(arguments['dt_pbl']), door_dt_ignored=call.dt))
    result = dict(scope='actual f3 ABI/f4 provider whole-slot column witness; no forecast proof',
        backend=args.backend,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        production_sha256=hashlib.sha256(Path(noah.__file__).read_bytes()).hexdigest(), results=rows)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if any(row['differing_words'] for row in rows):
        raise SystemExit(1)


def run_host_slot(arguments, call, base):
    """Execute the unchanged complete host-glibc port on actual bound fields.

    This debug witness proves the pool binding without claiming CUDA execution.
    """
    lib = ct.CDLL(str(base / 'libnoah_host.so'))
    fields = {}
    for name, shape in noah._MPAS_INPUT_FIELDS.items():
        array = arguments['fields'].get(name, arguments['fields']['skintemp'])
        fields[name] = np.ascontiguousarray(array.T if shape in ('soil', 'atmosphere', 'pressure') else array)
    count = arguments['count']
    x = np.empty((count, 79), np.float32)
    ix = np.empty((count, 7), np.int32)
    options = call.options
    fn = lib.noah_from_mpas
    fn.argtypes = [ct.c_void_p] * (2 + len(fields)) + [ct.c_int] * 4 + [ct.c_float] * 2 + [ct.c_int] * 6
    fn(x.ctypes.data, ix.ctypes.data, *(a.ctypes.data for a in fields.values()),
        fields['t'].shape[1], fields['pres2_hyd'].shape[1], 0, count,
        arguments['dt_pbl'], arguments['xice_threshold'],
        options['config_microp_scheme'] != 'off', options['config_convection_scheme'] != 'off',
        arguments['isurban'], arguments['isice'], call.itimestep, options['config_sfc_albedo'])
    fn = lib.replay_driver
    fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]
    fn(x.ctypes.data, ix.ctypes.data, arguments['tables'].ctypes.data, count)
    status = np.zeros(count, np.int32)
    fn = lib.noah_to_mpas
    fn.argtypes = [ct.c_void_p] * (2 + len(noah._MPAS_OUTPUT_FIELDS)) + [ct.c_int] * 2
    fn(x.ctypes.data, status.ctypes.data,
        *(fields[name].ctypes.data for name in noah._MPAS_OUTPUT_FIELDS), 0, count)
    for name in noah._MPAS_OUTPUT_FIELDS:
        if noah._MPAS_INPUT_FIELDS[name] == 'soil':
            np.copyto(arguments['fields'][name], fields[name].T)
        else:
            np.copyto(arguments['fields'][name], fields[name])


if __name__ == '__main__':
    main()
