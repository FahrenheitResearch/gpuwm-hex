"""Authenticate and group immutable real records for f1 whole-capture replay.

Only filenames/manifest metadata change. Raw headers and payloads remain exact;
member origins and every parent root are recorded in the new sealed manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_real import canonical


def combine(inputs, output, receipt):
    rows = [r for p in inputs for r in json.loads(p.read_text())['runs']]
    groups = {}
    for row in rows:
        source = Path(row['dump'])
        raw = (source/'manifest.json').read_bytes()
        manifest = json.loads(raw)
        root = manifest.pop('dump_root_sha256')
        if hashlib.sha256(canonical(manifest)).hexdigest() != root:
            raise ValueError('parent seal hash mismatch')
        sites = {s['site']: s for s in manifest['sites']}
        for name, info in manifest['files'].items():
            member = source/name
            if member.resolve().parent != source.resolve() or member.is_symlink():
                raise ValueError('unsafe parent member')
            data = member.read_bytes()
            if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError('parent record hash mismatch')
        for prefix, capture in manifest['captures'].items():
            site = sites[capture['site']]
            key = canonical([site, manifest['pin_set_sha256'], manifest['source_sha256'],
                             manifest['recipe'], manifest.get('scheme_aliases', {})])
            groups.setdefault(key, []).append((row, source, manifest, root, raw, prefix, capture, site))
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for group_index, entries in enumerate(groups.values()):
        target = output/f'site-{group_index:03}'
        target.mkdir()
        manifest = dict(entries[0][2])
        records, captures, origins, parents = {}, {}, {}, []
        for index, (row, source, parent, root, raw, prefix, capture, site) in enumerate(entries):
            renamed = f'capture-{index:04}-'+prefix
            captures[renamed] = capture
            parents.append({'dump': str(source), 'dump_root_sha256': root,
                'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'capture': prefix,
                'ancestor_dump_root_sha256': parent.get('ancestor_dump_root_sha256', []),
                'parent_dump_root_sha256': parent.get('parent_dump_root_sha256')})
            for name, info in parent['files'].items():
                if not name.startswith((prefix+'.pre.', prefix+'.post.')):
                    continue
                new_name = renamed+name[len(prefix):]
                shutil.copyfile(source/name, target/new_name)
                records[new_name] = info
                origins[new_name] = {'parent_index': index, 'member': name}
        manifest.update(files=records, captures=captures, sites=[entries[0][7]],
            parent_seals=parents, member_origins=origins,
            capture_scope='authenticated union of same-site calls; raw headers/payloads unchanged',
            hidden_state_complete=all(e[2]['hidden_state_complete'] for e in entries),
            argument_complete=all(e[2]['argument_complete'] for e in entries), forecast_proven=False)
        for name in ('parent_dump_root_sha256', 'parent_manifest_sha256', 'ancestor_dump_root_sha256'):
            manifest.pop(name, None)
        manifest['dump_root_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
        (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        runs.append({'dump': str(target), 'native_executable': entries[0][0]['native_executable'],
                     'dump_root_sha256': manifest['dump_root_sha256'], 'calls': len(entries),
                     'site': entries[0][7]['site']})
    receipt.write_text(json.dumps({'kind': 'authenticated same-site raw-record union',
        'inputs': [str(p) for p in inputs], 'runs': runs}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    a = parser.parse_args()
    combine(a.inputs, a.out, a.receipt)
