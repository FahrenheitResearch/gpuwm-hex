#!/usr/bin/env python3
"""Index published native captures by source interface and selected step.

This reads sealed metadata only. It does not decode arrays or turn a recorded
interface replay into a hidden-state or forecast acceptance claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


RECORD = re.compile(r'^(.*)\.s(\d+)\.e(\d+)\.r(\d+)\.(pre|post|init)\.')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def index(campaign, owner):
    products = []
    for publication in sorted(campaign.glob('*/*/*/PUBLICATION.json')):
        pub = json.loads(publication.read_text())
        if pub.get('owner') != owner:
            continue
        root = publication.parent
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.pop('dump_root_sha256')
        actual = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                           separators=(',', ':')).encode()).hexdigest()
        manifest['dump_root_sha256'] = expected
        if actual != expected:
            raise ValueError(f'dump manifest root hash mismatch: {root}')
        for field in ('pin_set_sha256', 'dump_root_sha256'):
            if pub[field] != manifest[field]:
                raise ValueError(f'publication/manifest {field} mismatch: {root}')
        provenance_path = root / 'INPUT-PROVENANCE.json'
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        if provenance and provenance.get('dump_root_sha256') != expected:
            raise ValueError(f'input provenance root hash mismatch: {root}')
        process_count = manifest.get('process_count')
        side_count = provenance.get('process_count')
        if process_count is not None and side_count is not None and process_count != side_count:
            raise ValueError(f'input provenance process count mismatch: {root}')
        if process_count is None:
            process_count = side_count
        sites = {}
        for name, member in manifest['files'].items():
            match = RECORD.match(name)
            if match is None:
                continue
            site, step, event, rank, phase = match.groups()
            row = sites.setdefault(site, {'steps': set(), 'ranks': set(),
                                        'captures': set(), 'records': 0, 'bytes': 0,
                                        'phases': {}})
            row['steps'].add(int(step))
            row['ranks'].add(int(rank))
            row['captures'].add((int(step), int(event), int(rank)))
            row['records'] += 1
            row['bytes'] += member['bytes']
            row['phases'][phase] = row['phases'].get(phase, 0) + 1
        retirement = root / 'RETIREMENT.json'
        products.append({
            'path': str(root.resolve()), 'suite': pub['suite'],
            'pin_set_sha256': pub['pin_set_sha256'],
            'dump_root_sha256': pub['dump_root_sha256'],
            'manifest_sha256': digest(manifest_path),
            'publication_sha256': digest(publication),
            'retired': retirement.exists(),
            'retirement_sha256': digest(retirement) if retirement.exists() else None,
            'input_provenance_sha256': digest(provenance_path) if provenance else None,
            'scheme_aliases': manifest.get('scheme_aliases', {}),
            'scheme_groups': manifest.get('scheme_groups', {}),
            'capture_recipe': manifest.get('capture_recipe', manifest.get('recipe')),
            'capture_process_count': process_count,
            'process_count_source': 'manifest' if manifest.get('process_count') is not None
            else 'INPUT-PROVENANCE.json' if side_count is not None else None,
            'hidden_state_complete': manifest.get('hidden_state_complete', False),
            'sites': {site: dict(records=row['records'], bytes=row['bytes'],
                                 steps=sorted(row['steps']), ranks=sorted(row['ranks']),
                                 capture_count=len(row['captures']), phases=row['phases'])
                      for site, row in sorted(sites.items())},
        })
    return {'schema': 'mpas-oracle-column-dump-index-v1', 'owner': owner,
            'products': products, 'forecast_proven': False,
            'scope': 'Published sealed metadata; retired products retain provenance only. '
                     'Replay verifies every selected raw member before execution.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = index(args.campaign_root, args.owner)
    text = json.dumps(result, indent=2) + '\n'
    if len(text.encode()) >= 10_000_000:
        raise ValueError('index exceeds receipt size limit')
    args.out.write_text(text)
    print(json.dumps({'products': len(result['products']), 'index': str(args.out)}))


if __name__ == '__main__':
    main()
