"""Replay complete SW driver boundaries with the proposed p21 radius consumer.

MPAS mpas_atmphys_driver_radiation_sw.F:355-794,870-948. Native captured
carriers feed the real owner binder and full CUDA engine. Rust owns record
decode/encode/comparison; POST data never feed the CUDA calculation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class CaptureProvider(dict):
    """Device pool views and scalar/table ABI views from original PRE records."""

    def __init__(self, values, cp, engine, dimensions, calendar):
        super().__init__(values)
        self.values = self
        self.cp = cp
        self.rrtmg_sw_v842_resources = engine
        self.session = SimpleNamespace(layout=SimpleNamespace(dimensions=dimensions))
        self.calendar = calendar
        self.native = self

    def get_table(self, name):
        return np.asarray(self.calendar[name.split('.')[-1]]), {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('module', 'views', 'transport', 'table', 'table-decoder', 'work', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from hexcore.mpasphys import radshared_v842
    spec = importlib.util.spec_from_file_location(
        'hexcore.mpasphys.rrtmg_sw_radius_candidate', args.module.resolve())
    port = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    products = json.loads((args.views / 'input-index.json').read_text())
    assert len(products) == 2 and all(r['native_process_count'] == 1 for r in products)
    args.work.mkdir(exist_ok=False)
    engine = port.MPASShortwave.from_mpas_data(args.table, decoder=args.table_decoder,
        decoded_directory=args.work / 'decoded-tables')
    rows = []
    for number, row in enumerate(products):
        directory, prefix = Path(row['directory']), row['prefix']
        for name, original in row['files'].items():
            assert sha(directory / name) == original['sha256']
        output = args.work / str(number)
        output.mkdir()
        def path(name, phase='pre'):
            return directory / (prefix + '.' + phase + '.' + name + '.bin')
        def metadata(name):
            return json.loads(subprocess.check_output(
                [str(args.transport), 'metadata', str(path(name)), '-']))
        def real_scalar(name):
            # Rust owns native record decoding. This is a typed scalar ABI
            # view, without host interpolation or weather-field arithmetic.
            raw = output / (name + '.scalar.raw')
            meta = json.loads(subprocess.check_output(
                [str(args.transport), 'decode', str(path(name)), str(raw)]))
            assert meta['code'] == 1 and meta['width'] in (4, 8)
            data = np.frombuffer(raw.read_bytes(), dtype='<f' + str(meta['width']))
            assert data.size == 1
            value = data[0]
            raw.unlink()
            return value
        cache = {}
        def field(name):
            if name not in cache:
                raw = output / (name + '.raw')
                meta = json.loads(subprocess.check_output(
                    [str(args.transport), 'decode', str(path(name)), str(raw)]))
                assert meta['code'] == 1 and meta['width'] == 4
                cache[name] = cp.asarray(np.frombuffer(raw.read_bytes(), dtype='<f4').reshape(meta['shape'], order='F'))
                raw.unlink()
            return cache[name]
        options = {name.split('.')[-1]: metadata(name) for name in row['inputs']
                   if name.startswith('block.configs.')}
        prepared = {name.split('.')[-1]: field(name)[:, :, 0].T for name in row['inputs']
                    if name.startswith('state.mpas_atmphys_vars.') and name.endswith('_p')}
        values = {pool: {name.split('.')[1]: field(name) for name in row['inputs']
                        if name.startswith(pool + '.') and name not in (
                            'atm_input.nCellsSolve', 'atm_input.nVertLevels')}
                  for pool in ('diag_physics', 'tend_physics', 'atm_input', 'sfc_input', 'mesh')}
        scalar_state = {n: metadata('state.mpas_atmphys_vars.' + n)
                        for n in ('has_reqc', 'has_reqi', 'has_reqs')}
        scalar_state['dt_radtsw'] = real_scalar('state.mpas_atmphys_vars.dt_radtsw')
        calendar = {n: real_scalar('state.mpas_atmphys_manager.' + n)
                    for n in ('curr_julday', 'gmt')}
        calendar['julday'] = metadata('state.mpas_atmphys_manager.julday')
        # MPAS mpas_atmphys_constants.F:89-90, strict GNU folded parameter words.
        # Embed the FP32 words; no host weather arithmetic or libm calculation.
        constants = {n: np.asarray(b, dtype='<u4').view('<f4')[()]
                     for n, b in [('degrad', 0x3c8efa35), ('dpd', 0x3f7c7e3f)]}
        values.update(prepared=prepared, scalar_state=scalar_state, constants=constants)
        dimensions = {n: metadata('atm_input.' + n) for n in ('nCellsSolve', 'nVertLevels')}
        provider = CaptureProvider(values, cp, engine, dimensions, calendar)
        call = SimpleNamespace(pools=provider, options=options,
                               time_seconds=real_scalar('argument.xtime_s'))
        port.bind_native_pools(provider, call)
        port.shortwave_slot(call)
        arrays = {}
        for name in row['outputs']:
            pool, key = name.split('.')
            value = provider[pool][key]
            encoded = output / (name + '.bin')
            flat = value.transpose(tuple(range(value.ndim - 1, -1, -1))).ravel()
            subprocess.run([str(args.transport), 'encode', str(path(name)), '-', str(encoded)],
                input=flat.get().tobytes(), check=True)
            result = json.loads(subprocess.check_output(
                [str(args.transport), 'compare', str(path(name, 'post')), str(encoded)]))
            result.update(reference_sha256=row['files'][path(name, 'post').name]['sha256'],
                          replay_sha256=sha(encoded))
            arrays[name] = result
        rows.append(dict(source_dump_root_sha256=row['source_dump_root_sha256'],
            options=options, nCellsSolve=dimensions['nCellsSolve'], arrays=arrays))
        shutil.rmtree(output)
        print(json.dumps(dict(complete_native_boundary=row['source_dump_root_sha256'],
            arrays=len(arrays), differing_words=sum(a['different_words'] for a in arrays.values()))), flush=True)
    summary = dict(schema='p21-complete-coupled-sw-radius-replay/v1', process_count=1,
        scope='Both complete native SW driver pool captures; all15 written outputs including halos',
        rows=rows, compared_words=sum(a['words'] for r in rows for a in r['arrays'].values()),
        differing_words=sum(a['different_words'] for r in rows for a in r['arrays'].values()),
        native_expected_never_cuda_inputs=True, production_applied=False, forecast_proven=False,
        artifacts={str(p.resolve()): sha(p) for p in (args.module, args.transport, args.table,
            args.table_decoder, args.views / 'input-index.json', Path(radshared_v842.__file__), Path(__file__))})
    args.receipt.write_text(json.dumps(summary, indent=2) + '\n')
    raise SystemExit(0 if summary['differing_words'] == 0 else 1)


if __name__ == '__main__':
    main()
