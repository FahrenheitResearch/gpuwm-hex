"""Authenticate an owned capture subset of a published ORCL0001 dump.

No numeric records are decoded or changed. Parent seal and every copied member
hash are retained so bounded remote replays can reference the canonical capture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def select(source, destination, site):
    from sys import path
    path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from column_oracle_replay import canonical, hash_file
    manifest = json.loads((source / 'manifest.json').read_text())
    parent = manifest.pop('dump_root_sha256')
    if hashlib.sha256(canonical(manifest)).hexdigest() != parent:
        raise ValueError('Published parent seal differs')
    captures = {n:v for n,v in manifest['captures'].items() if v['site'] == site}
    if not captures:
        raise ValueError('Published site is absent')
    files = {n:v for n,v in manifest['files'].items()
             if any(n.startswith(prefix+'.') for prefix in captures)}
    destination.mkdir(parents=True, exist_ok=False)
    for name, info in files.items():
        if Path(name).name != name:
            raise ValueError('Non-flat oracle member')
        src, dst = source/name, destination/name
        if src.is_symlink() or src.stat().st_size != info['bytes'] or hash_file(src) != info['sha256']:
            raise ValueError('Published member differs: '+name)
        shutil.copy2(src, dst)
        if hash_file(dst) != info['sha256']:
            raise ValueError('Owned copy differs: '+name)
    manifest.update(files=files, captures=captures,
                    selection=dict(parent_dump_root_sha256=parent,
                                   parent_manifest_sha256=hash_file(source/'manifest.json'),
                                   sites=[site], members_changed=False))
    manifest['dump_root_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'destination'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--site', required=True)
    args = parser.parse_args()
    select(args.source, args.destination, args.site)
