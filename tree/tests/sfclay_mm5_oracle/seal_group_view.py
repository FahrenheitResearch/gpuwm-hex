"""Seal public scheme-group metadata over unchanged native capture records.

All member bytes and parent provenance are authenticated before hardlinking.
The original manifest and record files remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def seal(source, output, table, receipt):
    original_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(original_bytes)
    parent = manifest.pop('dump_root_sha256')
    if hashlib.sha256(canonical(manifest)).hexdigest() != parent:
        raise ValueError('parent seal mismatch')
    groups = json.loads(table.read_text())
    if not isinstance(groups, dict) or any(
            not isinstance(k, str) or not isinstance(v, list) or not v or
            any(not isinstance(member, str) for member in v)
            for k, v in groups.items()):
        raise ValueError('invalid public scheme group table')
    if {p.name for p in source.glob('*.bin')} != set(manifest['files']):
        raise ValueError('parent member inventory changed')
    output.mkdir(parents=True, exist_ok=False)
    for name, info in manifest['files'].items():
        member = source / name
        if member.resolve().parent != source.resolve() or member.is_symlink():
            raise ValueError('unsafe source member')
        with member.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if member.stat().st_size != info['bytes'] or digest != info['sha256']:
            raise ValueError('parent record changed: ' + name)
        os.link(member, output / name)
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest.update(scheme_groups=groups, parent_dump_root_sha256=parent,
                    scheme_group_table_sha256=table_sha)
    root = hashlib.sha256(canonical(manifest)).hexdigest()
    manifest['dump_root_sha256'] = root
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    receipt.write_text(json.dumps(dict(
        schema='mpas.mm5-group-view/v1', status='PASS', source=str(source),
        output=str(output), original_manifest_sha256=hashlib.sha256(original_bytes).hexdigest(),
        parent_dump_root_sha256=parent, dump_root_sha256=root,
        table=str(table), table_sha256=table_sha, scheme_groups=groups,
        records=len(manifest['files']),
        bytes=sum(row['bytes'] for row in manifest['files'].values()),
        record_bytes_changed=False, forecast_proven=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'table', 'receipt'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    seal(args.source, args.output, args.table, args.receipt)
