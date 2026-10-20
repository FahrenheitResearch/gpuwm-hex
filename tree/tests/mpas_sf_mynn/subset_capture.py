#!/usr/bin/env python3
"""Copy and authenticate a sealed f1 MYNN record subset; no field decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    parent = manifest.pop('dump_root_sha256')
    if hashlib.sha256(canonical(manifest)).hexdigest() != parent:
        raise ValueError('f1 source manifest root differs from its seal')
    sites = [s for s in manifest['sites'] if s['callee']['module'] == 'module_sf_mynn']
    keys = {s['site'] for s in sites}
    captures = {k: v for k, v in manifest['captures'].items() if v['site'] in keys}
    if not captures:
        raise ValueError('sealed source has no MYNN calls')
    selected = {k: v for k, v in manifest['files'].items()
                if any(k.startswith(prefix + '.') for prefix in captures)}
    args.destination.mkdir(parents=True, exist_ok=False)
    for name, metadata in selected.items():
        path = source / name
        if path.parent != source or path.is_symlink() or not path.is_file():
            raise ValueError('record name does not identify a regular source child')
        if path.stat().st_size != metadata['bytes'] or digest(path) != metadata['sha256']:
            raise ValueError('sealed record changed: ' + name)
        output = args.destination / name
        shutil.copyfile(path, output)
        if digest(output) != metadata['sha256']:
            raise ValueError('copied record differs: ' + name)
    for site in sites:
        site['scheme'] = 'sf_mynn'
    manifest.update(sites=sites, captures=captures, files=selected,
                    parent_dump_root_sha256=parent,
                    parent_manifest_sha256=digest(manifest_path),
                    scheme_alias={'native_callee': 'sfclay_mynn', 'selector': 'sf_mynn'})
    manifest['dump_root_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (args.destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'captures': len(captures), 'records': len(selected),
                      'bytes': sum(v['bytes'] for v in selected.values()),
                      'parent_dump_root_sha256': parent,
                      'dump_root_sha256': manifest['dump_root_sha256']}))


if __name__ == '__main__':
    main()
