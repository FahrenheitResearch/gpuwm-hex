"""Replay both complete native LW driver boundaries with p21 radii consumed.

MPAS mpas_atmphys_driver_radiation_lw.F:297-719,805-884. Rust alone decodes,
encodes and compares records. No POST output enters the CUDA input pools.
The complete proposed owner module is an explicit artifact, not a mock.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('module', 'views', 'transport', 'table', 'table-decoder', 'work', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from hexcore.mpasphys import radshared_v842
    spec = importlib.util.spec_from_file_location(
        'hexcore.mpasphys.rrtmg_lw_radius_candidate', args.module.resolve())
    port = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    products = json.loads((args.views / 'input-index.json').read_text())
    assert len(products) == 2 and all(r['native_process_count'] == 1 for r in products)
    args.work.mkdir(exist_ok=False)
    tables = port.load_tables_device(args.table, decoder=args.table_decoder)
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
        cache = {}
        def field(name):
            if name not in cache:
                raw = output / (name + '.raw')
                meta = json.loads(subprocess.check_output(
                    [str(args.transport), 'decode', str(path(name)), str(raw)]))
                assert meta['code'] == 1 and meta['width'] == 4
                view = np.frombuffer(raw.read_bytes(), dtype='<f4').reshape(meta['shape'], order='F')
                cache[name] = cp.asarray(view)
                raw.unlink()
            return cache[name]
        options = {name.split('.')[-1]: metadata(name) for name in row['inputs']
                   if name.startswith('block.configs.')}
        prepared = {}
        for name in ('pres_hyd', 'pres2_hyd', 't', 't2', 'dz', 'pi', 'xlat'):
            value = field('state.mpas_atmphys_vars.' + name + '_p')
            prepared[name] = value[:, :, 0].T if value.ndim == 3 else value[:, 0]
        radiation = {name: field('state.mpas_atmphys_vars.' + name + '_p')[:, :, 0]
                     for name in ('qvrad', 'qcrad', 'qirad', 'qsrad')}
        pools = {pool: {name.split('.')[1]: field(name) for name in row['inputs']
                       if name.startswith(pool + '.') and name != 'atm_input.nCellsSolve'}
                 for pool in ('diag_physics', 'tend_physics', 'atm_input', 'sfc_input')}
        pools.update(nCellsSolve=metadata('atm_input.nCellsSolve'), prepared=prepared,
            radshared=radiation, rrtmg_lw_tables=tables, icloud=1,
            julday=metadata('state.mpas_atmphys_manager.julday'),
            radius_flags=tuple(metadata('state.mpas_atmphys_vars.' + n)
                               for n in ('has_reqc', 'has_reqi', 'has_reqs')))
        port.radiation_lw_slot(SimpleNamespace(pools=pools, options=options))
        arrays = {}
        for name in row['outputs']:
            pool, key = name.split('.')
            value = pools[pool][key]
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
            options=options, nCellsSolve=pools['nCellsSolve'], arrays=arrays))
        shutil.rmtree(output)
        print(json.dumps(dict(complete_native_boundary=row['source_dump_root_sha256'],
            arrays=len(arrays), differing_words=sum(a['different_words'] for a in arrays.values()))), flush=True)
    summary = dict(schema='p21-complete-coupled-lw-radius-replay/v1', process_count=1,
        scope='Both complete native LW driver pool captures; all16 outputs including halos',
        rows=rows, compared_words=sum(a['words'] for r in rows for a in r['arrays'].values()),
        differing_words=sum(a['different_words'] for r in rows for a in r['arrays'].values()),
        native_expected_never_cuda_inputs=True, production_applied=False, forecast_proven=False,
        artifacts={str(p.resolve()): sha(p) for p in (args.module, args.transport, args.table,
            args.table_decoder, args.views / 'input-index.json', Path(radshared_v842.__file__), Path(__file__))})
    args.receipt.write_text(json.dumps(summary, indent=2) + '\n')
    raise SystemExit(0 if summary['differing_words'] == 0 else 1)


if __name__ == '__main__':
    main()
