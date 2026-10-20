"""Copy and authenticate one scheme from a sealed f1 capture.

Only opaque files and capture metadata are handled here. Rust owns field
decoding. Preserve the parent seal so these subsets remain traceable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def select(source, target, receipt):
    manifest = json.loads((source/'manifest.json').read_text())
    parent_hash = manifest.pop('dump_root_sha256')
    if hashlib.sha256(canonical(manifest)).hexdigest() != parent_hash:
        raise ValueError('parent capture manifest seal mismatch')
    sites = [s for s in manifest['sites'] if s['scheme'] == 'cu_grell_freitas']
    selected = {s['site'] for s in sites}
    captures = {k: v for k, v in manifest['captures'].items() if v['site'] in selected}
    files = {k: v for k, v in manifest['files'].items()
             if any(k.startswith(prefix+'.') for prefix in captures)}
    if not captures or not files:
        raise ValueError('parent has no GF captures')
    target.mkdir(parents=True, exist_ok=False)
    for name, entry in files.items():
        path = source/name
        if path.resolve().parent != source.resolve() or path.is_symlink():
            raise ValueError('invalid parent capture member')
        content = path.read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('parent GF member differs from seal')
        shutil.copyfile(path, target/name)
    manifest.update(files=files, captures=captures, sites=sites)
    manifest['scheme_aliases'] = {'gf_mpas': 'cu_grell_freitas'}
    manifest['parent_dump_root_sha256'] = parent_hash
    manifest['dump_root_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    receipt.write_text(json.dumps(dict(source=str(source.resolve()), target=str(target.resolve()),
                                      parent_dump_root_sha256=parent_hash,
                                      dump_root_sha256=manifest['dump_root_sha256'],
                                      captures=list(captures), files=len(files),
                                      bytes=sum(v['bytes'] for v in files.values()),
                                      forecast_proven=False), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    select(a.source, a.target, a.receipt)
