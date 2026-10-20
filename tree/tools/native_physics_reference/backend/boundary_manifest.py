"""Map explicitly selected f1 pool captures to native replay calls.

Rust decodes capture metadata and scientific payloads. This only selects paths
and maps native pool member kinds; it never fills missing native hidden state.
The caller supplies the exact driver arguments and ordered allocation calls.
MPAS pool authority: src/framework/mpas_pool_types.inc:7-11.
"""
import argparse
import hashlib
import json
from pathlib import Path

from oracle_replay import Capture, transport


def build_manifest(library, specification, *, sealed_files=None):
    """Select all present captured storage, retaining native time levels."""
    lib = transport(library)
    root = Path(specification['capture_root']).resolve()
    prefix = specification['boundary_prefix']
    if not prefix.endswith('.'):
        raise ValueError('boundary_prefix must end with a dot')
    metadata = []
    unassociated = []

    def verify(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if sealed_files is not None:
            expected = sealed_files[path.name]
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            if digest != expected:
                raise ValueError(f'pool capture differs from f1 seal: {path}')
        return digest

    def scalar(path):
        digest = verify(path)
        with Capture(lib, path) as capture:
            value = capture.scalar()
        metadata.append({'capture': str(path), 'sha256': digest})
        return value

    def members(phase):
        selected = []
        if sealed_files is None:
            paths = sorted(root.glob(prefix + phase + '.*.member_kind.bin'))
        else:
            paths = sorted(root / name for name in sealed_files
                           if name.startswith(prefix + phase + '.') and name.endswith('.member_kind.bin'))
        if not paths:
            raise ValueError(f'no captured {phase} pool metadata for {prefix}')
        for path in paths:
            label = path.name[len(prefix + phase + '.'):-len('.member_kind.bin')]
            pool, name = label.rsplit('.', 1)
            kind = scalar(path)
            base = root / (prefix + phase + '.' + label)
            if not scalar(Path(str(base) + '.descriptor.present.bin')):
                unassociated.append({'phase': phase, 'pool': pool, 'name': name,
                                     'reason': 'pool descriptor is unassociated'})
                continue
            if kind == 1004:
                levels = scalar(Path(str(base) + '.time_levels.bin'))
                if not scalar(Path(str(base) + '.field_descriptor.present.bin')):
                    unassociated.append({'phase': phase, 'pool': pool, 'name': name,
                                         'reason': 'field descriptor is unassociated'})
                    continue
                payloads = [(level, Path(str(base) + (f'.t{level}' if levels > 1 else '') + '.bin'))
                            for level in range(1, levels + 1)]
                for level, payload in payloads:
                    present = scalar(payload.with_suffix('.present.bin'))
                    if not present:
                        unassociated.append({'phase': phase, 'pool': pool, 'name': name,
                                             'level': level, 'reason': 'field storage is unassociated'})
                        continue
                    if not payload.is_file():
                        raise ValueError(f'missing associated {phase} field capture: {payload}')
                    verify(payload)
                    with Capture(lib, payload) as capture:
                        role = 'text_field' if capture.header[0] == 5 else 'field'
                    selected.append({'role': role, 'pool': pool, 'name': name,
                                     'capture': str(payload), 'nlevels': levels, 'level': level})
            elif phase == 'pre' and kind in (1005, 1006):
                verify(Path(str(base) + '.bin'))
                selected.append({'role': 'config' if kind == 1005 else 'dimension',
                                 'pool': pool, 'name': name, 'capture': str(base) + '.bin'})
            elif kind not in (1004, 1005, 1006, 1007, 1008):
                raise ValueError(f'unknown native pool member kind {kind}: {path}')
        return selected

    inputs, outputs = members('pre'), members('post')
    shared = specification.get('shared_state')
    if shared:
        abi = json.loads(Path(library).with_name('abi.json').read_text())
        shared_root = Path(shared['capture_root']).resolve()
        sealed = json.loads((shared_root / 'manifest.json').read_text())
        files = sealed['files']
        shared_count = 0
        # f1 sealed manifests map relative filenames to metadata including SHA256.
        for name in abi['variables']:
            path = shared_root / (shared['prefix'] + name + '.bin')
            if not path.exists():
                continue
            expected = files[path.name]
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'shared capture differs from f1 seal: {path}')
            inputs.append({'role': 'variable', 'name': name, 'capture': str(path)})
            shared_count += 1
        if not shared_count:
            raise ValueError(f'no shared physics state captures for {shared["prefix"]}')
    before = specification.get('before', [])
    calls = [dict(call) for call in before]
    driver = {'entry': specification['entry'], 'arguments': specification['arguments'],
              'outputs': outputs}
    if calls:
        calls[0]['inputs'] = inputs + calls[0].get('inputs', [])
    else:
        driver['inputs'] = inputs
    calls.append(driver)
    calls.extend(specification.get('after', []))
    return {'schema': 'mpas-native-replay-v1', 'process_count': specification['process_count'],
            'forecast_proof': False, 'input_completeness': specification['input_completeness'],
            'capture_selection': specification, 'pool_metadata': metadata,
            'unassociated_storage': unassociated, 'calls': calls}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--specification', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.library, json.loads(Path(args.specification).read_text()))
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
