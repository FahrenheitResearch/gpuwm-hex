#!/usr/bin/env python3
"""Complete real output handoff; fork mpas_atm_core.F:919-1002 at 1d6e370e."""
import hashlib
import json

import numpy as np


def replay(root, prefix, target, library, manifest_path):
    import cupy as cp
    from rrfs_driver_dump_replay import authenticate, Records
    from hexcore.mpasphys.rrfs.driver_rrfs import (
        StrictPhysicsKernelCache, compute_native_output_diagnostics,
        reset_native_diagnostics,
    )
    manifest_sha = authenticate(root, prefix, manifest_path,
                                source='src/core_atmosphere/mpas_atm_core.F')
    manifest = json.loads(manifest_path.read_text())
    producer = json.loads((manifest_path.parent / 'producer.json').read_text())
    passive = json.loads((manifest_path.parent / 'passivity.json').read_text())
    bindings = [('producer.json', 'producer_attempt_sha256'),
                ('passivity.json', 'passivity_receipt_sha256'),
                ('input-request.json', 'input_request_sha256')]
    if (not manifest['observer_passivity_proven'] or not passive['bitwise_equal']
            or passive['different_words'] != 0
            or producer['host'] != passive['host']
            or producer['native_sha256'] != manifest['native_binary_sha256']
            or producer['process_count'] != 1
            or producer['observer_sha256'] != manifest['observer_sha256']
            or producer['source_sha256'] != manifest['observer_source_sha256']
            or producer['input_request_sha256'] != manifest['input_request_sha256']
            or any(row['exit'] != 0 for row in producer['runs'].values())
            or any(hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest()
                   != manifest[key] for name, key in bindings)):
        raise ValueError('Complete native output producer/passivity changed')
    pre = {p.name[len(prefix + '.pre.'):-4]: p for p in root.glob(prefix + '.pre.*.bin')}
    post = {p.name[len(prefix + '.post.'):-4]: p for p in root.glob(prefix + '.post.*.bin')}
    if not pre or set(pre) != set(post):
        raise ValueError('Complete output PRE/POST inventory differs')
    codec = Records(library)
    loaded = {key: codec.load(path)[0] for key, path in pre.items()}
    replacement = {}
    if prefix.startswith('boundary__atm_compute_output_diagnostics.'):
        nc, nz = int(loaded['mesh.nCells']), int(loaded['mesh.nVertLevels'])
        time_lev = int(loaded['argument.time_lev'])

        def key(pool, name):
            return pool + '.' + name + '.t' + str(time_lev if pool == 'state' else 1)

        def device(name):
            return cp.asarray(np.ascontiguousarray(loaded[name][:nz, :nc]))

        fields = {name: device(key(pool, name)) for pool, name in (
            ('state', 'theta_m'), ('state', 'rho_zz'), ('mesh', 'zz'),
            ('diag', 'pressure_base'), ('diag', 'pressure_p'))}
        index = int(loaded['state.index_qv']) - 1
        qv = cp.asarray(np.ascontiguousarray(loaded[key('state', 'scalars')][index, :nz, :nc]))
        results = compute_native_output_diagnostics(fields=fields, qv=qv,
                                                   kernel_cache=StrictPhysicsKernelCache())
        for name, value in results.items():
            record = key('diag', name)
            old = cp.asarray(np.ascontiguousarray(loaded[record]))
            old[:nz, :nc] = value
            replacement[record] = cp.asnumpy(old)
    elif prefix.startswith('boundary__atm_reset_diagnostics.'):
        # Bind the actual captured allocation metadata to the native pool ABI.
        # This transport never invokes a scheme or supplies expected POST data.
        class CapturedNativeOutputPools:
            def __init__(self):
                self._fields = {
                    ('diag_physics', name.split('.')[1]): (value.dtype, value.shape, 1)
                    for name, value in loaded.items() if name.startswith('diag_physics.')
                }

            def put_field(self, pool, name, value, *, nlevels):
                record = pool + '.' + name + '.t1'
                assert nlevels == self._fields[(pool, name)][2]
                replacement[record] = value

        reset_native_diagnostics(CapturedNativeOutputPools())
    else:
        raise ValueError('Unknown complete physics output boundary')
    target.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for name, path in pre.items():
        actual = target / post[name].name
        codec.post(path, actual, replacement.get(name))
        comparisons[name] = codec.compare(post[name], actual)
    return {
        'schema': 'r4.real-complete-physics-output/v1',
        'scope': 'all real output preparation/reset calls observed during stated first720s/MPI1; every direct pool getter and complete associated reset allocation',
        'bitwise_equal': all(row['bitwise_equal'] for row in comparisons.values()),
        'words': sum(row['words'] for row in comparisons.values()),
        'different_words': sum(row['different_words'] for row in comparisons.values()),
        'output_records': len(comparisons), 'manifest_sha256': manifest_sha,
        'native_capture_host': producer['host'],
        'replaced_records': sorted(replacement), 'comparisons': comparisons,
    }
