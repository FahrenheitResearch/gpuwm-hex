#!/usr/bin/env python3
"""Bind legacy dump reference metadata without changing any observed word.

The original manifest stays immutable. A new owned bank hard-links its payloads
and seals authenticated source identity using the current common transport.
This operation establishes provenance only, not producer passivity or replay.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

from pin_sets import provenance, reference_build


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def harness():
    path = Path(__file__).resolve().parents[2] / 'column_oracle_replay.py'
    spec = importlib.util.spec_from_file_location('rebind_column_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_inventory(manifest, inventory, inventory_sha, native, native_sha, selected, pins):
    """Reject relabelling unless the original source and strict referee agree."""
    if manifest.get('schema') != 'mpas-column-dump-v1' or manifest.get('recipe') != 'gnu-strict':
        raise ValueError('legacy rebinding requires a sealed strict native dump')
    if manifest.get('inventory_sha256') != inventory_sha:
        raise ValueError('original dump inventory differs')
    for key in ('sites', 'source_sha256', 'argument_complete', 'hidden_state_complete'):
        if manifest.get(key) != inventory.get(key):
            raise ValueError('original dump source/interface differs: ' + key)
    for key, expected in manifest.get('build_provenance', {}).items():
        if key not in inventory or inventory[key] != expected:
            raise ValueError('original dump build provenance differs: ' + key)
    if (inventory.get('reference') != selected['reference'] or
            inventory.get('model_commit') != selected['model_commit'] or
            inventory.get('submodule_commits') != selected['submodule_commits'] or
            inventory.get('pin_set_sha256') != selected['pin_set_sha256']):
        raise ValueError('original inventory pinned reference differs')
    if (inventory.get('strict_reference_manifest_sha256') != native_sha or
            inventory.get('strict_reference_binary_sha256') != native['binary_sha256']):
        raise ValueError('original dump strict referee differs')
    original_reference = manifest.get('reference')
    if original_reference is not None and original_reference != selected['reference']:
        raise ValueError('refusing a conflicting explicit reference')
    pins_sha = hashlib.sha256(pins.encode()).hexdigest()
    if manifest.get('pins') != pins or manifest.get('pin_set_sha256') not in (pins_sha, selected['pin_set_sha256']):
        raise ValueError('original dump pins differ')
    return dict(inventory)


def rebind(dump, inventory, pins, source, pin_set, native_reference, words, out, evidence=()):
    module = harness()
    original = module.verify(dump)
    selected = provenance(source, pin_set)
    native, _ = reference_build(native_reference, pin_set)
    recorded = json.loads(inventory.read_text())
    bound = bind_inventory(original, recorded, sha(inventory), native,
                           sha(native_reference / 'MANIFEST.json'), selected, pins.read_text())
    for relative, expected in recorded['source_sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(source / path) != expected:
            raise ValueError('pinned native source differs: ' + relative)
    if out.exists() or out.is_symlink():
        raise ValueError('rebinding requires a fresh owned output bank')
    if dump.stat().st_dev != out.parent.stat().st_dev:
        raise ValueError('rebinding requires hard links on the original filesystem')
    proof = dict(schema='mpas-native-dump-reference-binding-v1',
                 original_manifest_sha256=sha(dump / 'manifest.json'),
                 original_dump_root_sha256=original['dump_root_sha256'],
                 original_inventory_sha256=sha(inventory),
                 native_reference_manifest_sha256=sha(native_reference / 'MANIFEST.json'),
                 native_reference_binary_sha256=native['binary_sha256'],
                 source_identity=selected, pins_document_sha256=sha(pins),
                 evidence=[dict(path=str(p.resolve()), sha256=sha(p)) for p in evidence],
                 original_reference=original.get('reference'),
                 original_pin_set_sha256=original['pin_set_sha256'],
                 member_words_changed=False, producer_passivity_proven=False,
                 native_replay_proven=False, forecast_proven=False)
    bound['input_provenance'] = dict(original_input_provenance=original.get('input_provenance'),
                                     reference_binding=proof)
    out.mkdir()
    try:
        for name in original['files']:
            os.link(module.safe_file(dump, name), out / name)
        bound_path = out / 'source-inventory.json'
        bound_path.write_text(json.dumps(bound, indent=2) + '\n')
        result = module.seal(out, bound_path, pins, words, 'gnu-strict',
                             process_count=original.get('process_count'), reference=selected['reference'])
        if result['files'] != original['files'] or result['captures'] != original['captures']:
            raise ValueError('rebinding changed observed records or call inventory')
        proof.update(rebound_manifest_sha256=sha(out / 'manifest.json'),
                     rebound_dump_root_sha256=result['dump_root_sha256'], records=len(result['files']))
        (out / 'reference-binding.json').write_text(json.dumps(proof, indent=2) + '\n')
    except BaseException:
        shutil.rmtree(out)
        raise
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dump', 'inventory', 'pins', 'source', 'native-reference', 'words', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pin-set', required=True)
    p.add_argument('--evidence', type=Path, action='append', default=[])
    print(json.dumps(rebind(**vars(p.parse_args())), indent=2))
