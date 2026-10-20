#!/usr/bin/env python
"""Stage only this site's authenticated sealed records for lane replay.

Manifest authentication and file staging only; Rust decodes all weather words.
MPAS mpas_atmphys_driver_convection.F:582-607.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / 'manifest.json').read_text())
    parent = manifest.pop('dump_root_sha256')
    if hashlib.sha256(canonical(manifest)).hexdigest() != parent:
        raise ValueError('source manifest seal differs')
    manifest['sites'] = [s for s in manifest['sites']
                         if s['scheme'] == 'cu_ntiedtke_driver']
    sites = {s['site'] for s in manifest['sites']}
    manifest['captures'] = {k: v for k, v in manifest['captures'].items()
                            if v['site'] in sites}
    if not manifest['captures']:
        raise ValueError('sealed source has no New Tiedtke captures')
    manifest['files'] = {k: v for k, v in manifest['files'].items()
                         if any(k.startswith(p + '.') for p in manifest['captures'])}
    args.destination.mkdir(parents=True, exist_ok=False)
    for name, info in manifest['files'].items():
        source = args.source / name
        if source.parent != args.source or source.is_symlink():
            raise ValueError('unsafe capture member')
        destination = args.destination / name
        shutil.copy2(source, destination)
        if destination.stat().st_size != info['bytes'] or hashlib.sha256(destination.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('capture member differs from source seal')
    manifest['parent_dump_root_sha256'] = parent
    manifest['dump_root_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (args.destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    args.receipt.write_text(json.dumps({
        'source': str(args.source.resolve()), 'parent_root': parent,
        'subset_root': manifest['dump_root_sha256'],
        'captures': list(manifest['captures']), 'files': len(manifest['files']),
        'bytes': sum(v['bytes'] for v in manifest['files'].values()),
        'all_selected_member_hashes_verified': True,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
