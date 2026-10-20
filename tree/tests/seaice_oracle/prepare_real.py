"""Copy authenticated sea-ice members from an f1 seal into lane-owned scratch.

No weather decoding or arithmetic. The parent manifest authenticates each copied
record; this deliberately makes a smaller seal with an explicit parent witness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def prepare(source, output, aliases=None):
    parent_bytes = (source/'manifest.json').read_bytes()
    manifest = json.loads(parent_bytes)
    parent_root = manifest.pop('dump_root_sha256')
    if hashlib.sha256(canonical(manifest)).hexdigest() != parent_root:
        raise ValueError('f1 parent seal hash mismatch')
    sites = {s['site']: s for s in manifest['sites']
             if Path(s['path']).stem == 'mpas_atmphys_driver_seaice'}
    captures = {p: c for p, c in manifest['captures'].items() if c['site'] in sites}
    if not captures:
        raise ValueError('f1 seal has no sea-ice captures')
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, info in manifest['files'].items():
        if not any(name.startswith(p+'.pre.') or name.startswith(p+'.post.') for p in captures):
            continue
        path = source/name
        if path.resolve().parent != source.resolve() or path.is_symlink():
            raise ValueError('unsafe capture member')
        raw = path.read_bytes()
        if len(raw) != info['bytes'] or hashlib.sha256(raw).hexdigest() != info['sha256']:
            raise ValueError('parent record hash mismatch: '+name)
        (output/name).write_bytes(raw)
        records[name] = info
    if aliases is not None:
        raw_aliases = aliases.read_bytes()
        alias_table = json.loads(raw_aliases)
        if not isinstance(alias_table, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in alias_table.items()):
            raise ValueError('scheme aliases must map names to names')
        manifest['scheme_aliases'] = {**manifest.get('scheme_aliases', {}), **alias_table}
        manifest['lane_alias_table_sha256'] = hashlib.sha256(raw_aliases).hexdigest()
    if 'parent_dump_root_sha256' in manifest:
        manifest['ancestor_dump_root_sha256'] = (manifest.get('ancestor_dump_root_sha256', [])
            + [manifest['parent_dump_root_sha256']])
    manifest.update(files=records, captures=captures, sites=list(sites.values()),
                    parent_dump_root_sha256=parent_root,
                    parent_manifest_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                    capture_scope='sea-ice subset; no other parent records validated')
    manifest['dump_root_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'parent_root': parent_root, 'captures': len(captures),
                      'records': len(records), 'bytes': sum(v['bytes'] for v in records.values())}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--aliases', type=Path)
    a = p.parse_args()
    prepare(a.source, a.output, a.aliases)
