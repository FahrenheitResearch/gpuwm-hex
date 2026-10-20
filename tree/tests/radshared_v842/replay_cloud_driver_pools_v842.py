"""Execute the production cloud slot on original native private-pool inputs.

MPAS mpas_atmphys_driver_cloudiness.F:92-234. Rust owns record/scalar decode,
encode and word comparison; CUDA owns profile layout and arithmetic. Expected
POST records enter only the independent comparison after slot execution.
All eight output arrays include untouched workspace/halo words. This proves
driver scientific outputs, not complete hidden state or a whole forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import cupy as cp
import numpy as np

from hexcore.mpasphys import radshared_v842 as rad


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(exist_ok=False)
    rows = []
    state = 'state.mpas_atmphys_vars.'
    for number, entry in enumerate(json.loads(args.index.read_text())):
        work = args.work / str(number)
        work.mkdir()
        root, prefix = Path(entry['directory']), entry['prefix']
        def path(name, phase='pre'):
            return root / (prefix + '.' + phase + '.' + name + '.bin')
        def authenticate(name, phase='pre'):
            p = path(name, phase)
            expected = entry['files'][p.name]
            assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256']
            return p
        def metadata(name):
            return json.loads(subprocess.check_output([str(args.transport),
                'metadata', str(authenticate(name)), '-']))
        decoded = {}
        def field(name):
            if name not in decoded:
                raw = work / (name + '.raw')
                header = json.loads(subprocess.check_output([str(args.transport),
                    'decode', str(authenticate(name)), str(raw)]))
                assert header['code'] == 1 and header['width'] == 4
                # These are already Rust-decoded words, not a Python file decoder.
                host = np.frombuffer(raw.read_bytes(), dtype='<f4').reshape(header['shape'], order='F')
                decoded[name] = cp.ascontiguousarray(cp.asarray(host))
                raw.unlink()
            return decoded[name]
        nc = metadata('block.dimensions.nCellsSolve')
        kts, kte = (metadata(state + n) for n in ('kts', 'kte'))
        its, ite = (metadata(state + n) for n in ('its', 'ite'))
        assert (its, ite, kts) == (1, nc, 1)
        assert metadata(state + 'jts') == metadata(state + 'jte') == 1
        def public(name):
            original = field(state + name + '_p')
            assert original.ndim == 3 and original.shape[2] == 1
            return cp.ascontiguousarray(original[:, :, 0].T)
        prepared = {n: public(n)[:kte] for n in
                    ('qv', 'qc', 'qi', 'qs', 't', 'pres', 'pres_hyd', 'rho')}
        radiation = {n: public(n) for n in ('cldfrac', 'qvrad', 'qcrad', 'qirad', 'qsrad')}
        radiation.update({n: field(state + n + '_p')[:, 0] for n in ('dx', 'xland')})
        pools = dict(prepared=prepared,
            radiation_prepared={n: a[:kte] if a.ndim == 2 else a for n, a in radiation.items()},
            mesh={'meshDensity': field('mesh.meshDensity')},
            sfc_input={'xland': field('sfc_input.xland')},
            diag_physics={'cldfrac': field('diag_physics.cldfrac')},
            nCellsSolve=nc, config_len_disp=field('block.configs.config_len_disp').item(),
            f_qc=metadata(state + 'f_qc'), f_qi=metadata(state + 'f_qi'))
        rad.cloudiness_slot(SimpleNamespace(pools=pools,
            scheme=metadata('block.configs.config_radt_cld_scheme')))
        candidates = {'diag_physics.cldfrac': pools['diag_physics']['cldfrac']}
        candidates.update({state + n + '_p': a.T[:, :, None] if a.ndim == 2 else a[:, None]
                           for n, a in radiation.items()})
        candidates.update({n: field(n) for n in entry['unchanged_inputs']})
        arrays = {}
        for name, candidate in candidates.items():
            encoded = work / (name + '.bin')
            # GPU arranges Fortran serialization; Rust encodes the original schema.
            payload = cp.asnumpy(candidate.ravel(order='F')).tobytes()
            subprocess.run([str(args.transport), 'encode', str(path(name)), '-', str(encoded)],
                           input=payload, check=True)
            result = json.loads(subprocess.check_output([str(args.transport), 'compare',
                str(authenticate(name, 'post')), str(encoded)]))
            result.update(reference_sha256=entry['files'][path(name, 'post').name]['sha256'],
                          replay_sha256=sha(encoded), output=name in entry['outputs'])
            arrays[name] = result
        rows.append(dict(original_dump_root_sha256=entry['original_dump_root_sha256'],
            prefix=prefix, native_process_count=1, replay_process_count=1,
            nCellsSolve=nc, nVertLevels=kte, arrays=arrays,
            all_outputs_bitwise_equal=all(a['bitwise_equal'] for a in arrays.values())))
        print(prefix, rows[-1]['all_outputs_bitwise_equal'], flush=True)
        shutil.rmtree(work)
    arrays = [a for row in rows for a in row['arrays'].values()]
    receipt = dict(schema='p21-native-cloud-driver-pool-replay/v1',
        scope='all eight native cloud-driver written arrays and ten preserved scientific inputs',
        process_count=1, captures=rows, hidden_state_complete=False, forecast_proven=False,
        index_sha256=sha(args.index), transport_sha256=sha(args.transport),
        source_sha256=sha(rad.__file__), replay_sha256=sha(__file__),
        all_outputs_bitwise_equal=all(a['bitwise_equal'] for a in arrays),
        words=sum(a['words'] for a in arrays), differing_words=sum(a['different_words'] for a in arrays))
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if receipt['all_outputs_bitwise_equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
