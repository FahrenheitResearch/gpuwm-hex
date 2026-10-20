"""Seal selected immutable f1 captures without duplicating their bulk bytes.

Only manifest metadata and filesystem links are processed here. The f1 Rust
word helper decodes capture headers; every selected member is hashed by f1's
sealer and checked against the original authenticated manifest. Unselected
records are not read and no completeness claim is added to the inventory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def subset(source, output, step, site, door, words, pins, member_directory=None):
    spec = importlib.util.spec_from_file_location('thompson_f1_subset_sealer', door)
    f1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f1)
    raw = (source / 'manifest.json').read_bytes()
    original = json.loads(raw)
    expected = original.pop('dump_root_sha256')
    assert hashlib.sha256(f1.canonical(original)).hexdigest() == expected
    original['dump_root_sha256'] = expected
    prefixes = [p for p, c in original['captures'].items()
                if c['site'] == site and f'.s{step:010d}.' in p]
    if not prefixes:
        raise ValueError('selected native call site/step has no capture')
    selected = {n: info for n, info in original['files'].items()
                if any(n.startswith(p + '.pre.') or n.startswith(p + '.post.') for p in prefixes)}
    if output.exists():
        raise ValueError('subset output must be fresh')
    output.mkdir(parents=True)
    for name, info in selected.items():
        roots = [source, *(member_directory or [])]
        candidate = next((root / name for root in roots if (root / name).is_file()), None)
        if candidate is None:
            raise ValueError(f'sealed native member is unavailable: {name}')
        candidate = f1.safe_file(candidate.parent, name)
        with candidate.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != info['sha256'] or candidate.stat().st_size != info['bytes']:
            raise ValueError(f'native member differs from original sealed capture: {name}')
        os.link(candidate, output / name)
    inventory = {k: original[k] for k in ('sites', 'source_sha256', 'argument_complete',
                                          'hidden_state_complete')}
    inventory['sites'] = [r for r in inventory['sites'] if r['site'] == site]
    inventory.update(original.get('build_provenance', {}))
    inventory['scheme_aliases'] = {'thompson': 'mp_gt_driver'}
    inventory['input_provenance'] = dict(original_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        original_dump_root_sha256=expected, original_inventory_sha256=original['inventory_sha256'],
        source_input_provenance=original.get('input_provenance'),
        selected_step=step, process_count=1,
        scope='Selected single-process native capture records; no forecast or full-inventory claim')
    inventory_path = output / 'inventory.json'
    inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
    sealed = f1.seal(output, inventory_path, pins, words, original['recipe'])
    assert sealed['files'] == selected
    assert set(sealed['captures']) == set(prefixes)
    print(json.dumps(dict(dump=str(output), captures=len(prefixes), files=len(selected),
        bytes=sum(r['bytes'] for r in selected.values()), process_count=1,
        dump_root_sha256=sealed['dump_root_sha256'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'output', 'door', 'words', 'pins'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--site', default='mp_gt_driver__driver_microphysics__L391')
    parser.add_argument('--member-directory', type=Path, action='append', default=[])
    subset(**vars(parser.parse_args()))
