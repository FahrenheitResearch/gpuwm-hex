"""Check launcher bindings against the published f4 Registry and f3 ABI.

Metadata validation only; no mock callback or forecast acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-tree', type=Path, required=True)
    parser.add_argument('--mpas-source', type=Path, required=True)
    parser.add_argument('--abi', type=Path, required=True)
    parser.add_argument('--port', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--native-backend-source', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('wsm6_binding_authority', args.port)
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    sys.path.insert(0, str(args.foundation_tree / 'src'))
    from hexcore.mpasphys import resolve_namelist
    from hexcore.mpasphys.pool_v842 import RegistryLayout
    abi = json.loads(args.abi.read_text())
    fixtures = []
    for suite in ('none', 'mesoscale_reference'):
        text = f'&physics\nconfig_physics_suite="{suite}"\nconfig_microp_scheme="mp_wsm6"\n/'
        layout = RegistryLayout(args.mpas_source, resolve_namelist(text), text,
                                dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6,
                                     maxEdges2=12, vertexDegree=3, nVertLevels=55))
        rows = {(r['pool'], r['pool_name']): r for r in layout.fields}
        for name, pool in port.NATIVE_FIELD_POOLS.items():
            row = rows[pool, name]
            assert row['kind'] == ('integer' if name == 'i_rainnc' else 'real'), (pool, name)
            assert row['nlevels'] >= (2 if pool == 'state' else 1), (pool, name)
        state = rows['state', 'scalars']
        assert state['dimensions'] == ['num_scalars', 'nVertLevels', 'nCells']
        assert state['nlevels'] == 2
        species = {name: layout.indices['state', 'index_' + name] for name in port.VOLUME_NAMES[1:7]}
        assert len(set(species.values())) == 6 and min(species.values()) >= 1
        assert rows['tend_physics', 'mp_top_level']['dimensions'] == []
        assert rows['mesh', 'zgrid']['shape'][0] == 56
        assert set(port.FIELD_NAMES) | {'i_rainnc'} == set(port.NATIVE_FIELD_POOLS) | set(species)
        fixtures.append(dict(suite=suite, species_indices=species,
                             bound_fields=len(port.NATIVE_FIELD_POOLS), state_time_level=2))
    constants = {n.lower() for n in port.NATIVE_CONSTANT_NAMES if n is not None} | {'rvord', 'rcv', 'p0'}
    missing_constants = sorted(constants - abi['constants'].keys())
    scalars = {'dt_microp', 'dt_dyn', 'l_acrain', 'l_diags', 'has_reqc', 'has_reqi', 'has_reqs'}
    missing_scalars = sorted(scalars - abi['variables'].keys())
    assert not missing_constants, missing_constants
    assert not missing_scalars, missing_scalars
    assert all(abi['variables'][n]['rank'] == 0 for n in scalars)
    result = dict(scope=__doc__.strip(), fixtures=fixtures, missing_constants=missing_constants,
                  missing_scalars=missing_scalars,
                  hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.port, args.abi,
                                    args.mpas_source / 'src/core_atmosphere/Registry.xml',
                                    args.foundation_tree / 'src/hexcore/mpasphys/device_pools_v842.py')})
    if args.native_library is not None:
        import numpy as np
        assert args.native_backend_source is not None
        spec = importlib.util.spec_from_file_location('native_binding_backend', args.native_backend_source)
        backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = backend
        spec.loader.exec_module(backend)
        spec = importlib.util.spec_from_file_location('wsm6_binding_replay', Path(__file__).with_name('replay.py'))
        replay = importlib.util.module_from_spec(spec)
        sys.modules['hexcore.mpasphys.wsm6_v842'] = port
        spec.loader.exec_module(replay)
        with backend.NativePhysics(args.native_library, test_only=True) as native:
            values = np.asarray([native.get_constant(name) if name else 0
                                 for name in port.NATIVE_CONSTANT_NAMES], dtype=np.float32)
        actual, expected = values.view(np.uint32), replay.CONSTANTS.view(np.uint32)
        result['compiled_constants'] = dict(actual_words=actual.tolist(), expected_words=expected.tolist(),
                                           differing_words=int(np.count_nonzero(actual != expected)))
        result['hashes'][str(args.native_library)] = hashlib.sha256(args.native_library.read_bytes()).hexdigest()
        result['hashes'][str(args.native_backend_source)] = hashlib.sha256(args.native_backend_source.read_bytes()).hexdigest()
        assert result['compiled_constants']['differing_words'] == 0, result['compiled_constants']
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print('native Registry/ABI binding schema passed:', len(fixtures), 'decks')


if __name__ == '__main__':
    main()
