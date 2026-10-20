"""Copy a sealed subset of f1 calls without changing the shared dump.

This is record transport/provenance only; no weather arrays are decoded.
The resulting manifest retains source/build pins and binds the parent seal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('--site', action='append', required=True)
    a = p.parse_args()
    original = json.loads((a.source / 'manifest.json').read_text())
    parent = original.pop('dump_root_sha256')
    if hashlib.sha256(canonical(original)).hexdigest() != parent:
        raise ValueError('parent seal mismatch')
    selected = {k: v for k, v in original['captures'].items() if v['site'] in a.site}
    if {v['site'] for v in selected.values()} != set(a.site):
        raise ValueError('requested callsite is absent')
    files = {k: v for k, v in original['files'].items()
             if k.split('.pre.')[0].split('.post.')[0] in selected}
    a.output.mkdir(parents=True, exist_ok=False)
    for name, info in files.items():
        source = a.source / name
        if source.resolve().parent != a.source.resolve() or source.is_symlink():
            raise ValueError('unsafe source member')
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if source.stat().st_size != info['bytes'] or digest != info['sha256']:
            raise ValueError('parent record changed: ' + name)
        shutil.copyfile(source, a.output / name)
    original.update(files=files, captures=selected,
                    parent_dump_root_sha256=parent,
                    selection_sites=sorted(set(a.site)))
    root = hashlib.sha256(canonical(original)).hexdigest()
    original['dump_root_sha256'] = root
    (a.output / 'manifest.json').write_text(json.dumps(original, indent=2) + '\n')
    a.receipt.write_text(json.dumps(dict(status='PASS', source=str(a.source),
        output=str(a.output), parent_dump_root_sha256=parent,
        selected_dump_root_sha256=root, sites=a.site, records=len(files),
        bytes=sum(v['bytes'] for v in files.values()), forecast_proven=False),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
