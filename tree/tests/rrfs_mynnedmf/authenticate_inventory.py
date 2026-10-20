"""Authenticate the complete selected record set, including every native POST.

The committed pins are qualified against the original SHA-sealed r1 manifests.
This prevents a shortened or edited selection from becoming a smaller proof.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PIN_PATH = Path(__file__).with_name('real_capture_pins.json')


def inventory_sha256(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def selected_records(manifest, capture):
    site_name = capture.split('.s')[0]
    sites = manifest['sites']
    site = next(s for s in sites if s['site'] == site_name) if isinstance(sites, list) else sites[site_name]
    arguments = {row['name'] for row in site['arguments']}
    records = {}
    for filename, declared in manifest['files'].items():
        if filename.startswith(capture+'.post.'):
            records[filename] = declared
        elif filename.startswith(capture+'.pre.'):
            name = filename[len(capture+'.pre.'):-4]
            if (name.removesuffix('.present') in arguments or
                    name.startswith('allocation.module_bl_mynnedmf_driver.mynnedmf_driver.') or
                    name.startswith(('local.', 'state.'))):
                records[filename] = declared
    return records


def validate_selection(selection, pins=None):
    pins = json.loads(PIN_PATH.read_text()) if pins is None else pins
    matches = [row for row in pins['captures']
               if row['source_manifest_sha256'] == selection['source_manifest_sha256']]
    if len(matches) != 1:
        raise RuntimeError('Native capture has no unique committed manifest authority')
    pin = matches[0]
    for name in ('capture', 'publication_sha256', 'source_manifest_sha256',
                 'source_dump_root_sha256', 'complete_archive_sha256'):
        if selection[name] != pin[name]:
            raise RuntimeError('Native capture provenance differs from committed authority: '+name)
    if inventory_sha256(selection['files']) != pin['inventory_sha256']:
        raise RuntimeError('Native record inventory differs from complete committed selection')
    posts = [name for name in selection['files'] if name.startswith(selection['capture']+'.post.')]
    if not posts or len(posts) != pin['post_records']:
        raise RuntimeError('Native POST inventory is incomplete')
    return {'native_inventory_sha256': pin['inventory_sha256'],
            'expected_post_records': len(posts),
            'native_manifest_sha256': pin['source_manifest_sha256']}


def qualify_native_manifest(selection, raw, pins=None):
    result = validate_selection(selection, pins)
    if hashlib.sha256(raw).hexdigest() != selection['source_manifest_sha256']:
        raise RuntimeError('Original native manifest SHA-256 differs from pinned authority')
    original = selected_records(json.loads(raw), selection['capture'])
    if original != selection['files']:
        raise RuntimeError('Selection omits or alters an original native record')
    return {**result, 'original_manifest_bytes': len(raw),
            'selected_records': len(original), 'every_native_post_selected': True}


def validate_case_matrix(root):
    pins = json.loads(PIN_PATH.read_text())
    selections = sorted(Path(root).glob('*/mynnedmf*.json'))
    documents = [json.loads(path.read_text()) for path in selections]
    expected = {row['source_manifest_sha256'] for row in pins['captures']}
    actual = [row['source_manifest_sha256'] for row in documents]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('The complete pinned real case matrix is required')
    return [validate_selection(row, pins) for row in documents]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selections')
    parser.add_argument('--selection')
    parser.add_argument('--qualify-stdin', action='store_true')
    args = parser.parse_args()
    if args.selections:
        result = validate_case_matrix(args.selections)
    else:
        selection = json.loads(Path(args.selection).read_text())
        result = (qualify_native_manifest(selection, sys.stdin.buffer.read())
                  if args.qualify_stdin else validate_selection(selection))
    print(json.dumps(result, sort_keys=True))
