"""Seal the consumed prepared files and a metadata-defined forecast command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(root, paths):
    result = []
    for value in sorted(set(paths)):
        path = root / value
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'input must be a contained regular file: {value}')
        if path.name in ('.env', '.env.local', 'auth.json') or not path.is_file():
            raise ValueError(f'input is missing or forbidden: {value}')
        result.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    if not result:
        raise ValueError('empty prepared input inventory')
    return result


def seal(root, metadata):
    if (root / 'pin.json').exists():
        raise ValueError('immutable pin already exists')
    entries = inventory(root, metadata['input_paths'])
    config = metadata['configuration'] | {'prepared_content_sha256': hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    (root / 'case.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
    launcher = Path(__file__).with_name('bench_case.py')
    (root / 'launch.py').write_bytes(launcher.read_bytes())
    pin = dict(schema='woof-bench-pin-v1', case=metadata['case'],
        argv=['{python}', '{input}/launch.py', '--input', '{input}', '--output', '{output}', '--cache', '{cache}'],
        timeline_argv=['{python}', '-m', 'gpuwm.cli', 'timeline', '{ledger}'],
        required_stages=[{'stage': 'forecast.integrate', 'phase': phase} for phase in ('begin', 'end')],
        artifacts_subdir='artifacts', prepared_content_sha256=config['prepared_content_sha256'],
        scope='prepared forecast, private COLD or WARM runtime caches',
        files=entries + inventory(root, ['case.json', 'launch.py']))
    (root / 'pin.json').write_text(json.dumps(pin, sort_keys=True, indent=2) + '\n')
    return dict(pin_sha256=sha(root / 'pin.json'), files=len(pin['files']),
        bytes=sum(entry['bytes'] for entry in pin['files']), prepared_content_sha256=config['prepared_content_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('metadata', type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.root, json.loads(args.metadata.read_text())), indent=2))
