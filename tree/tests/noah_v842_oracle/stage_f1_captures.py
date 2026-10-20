"""Copy a scheme subset of a sealed f1 root, preserving source provenance."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--scheme', default='lsm')
p.add_argument('--public-name', help='seal a public selector alias for the selected native scheme')
a = p.parse_args()
canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':')).encode()
source = json.loads((a.source / 'manifest.json').read_text())
original_hash = source.pop('dump_root_sha256')
assert hashlib.sha256(canonical(source)).hexdigest() == original_hash
sites = {s['site']: s for s in source['sites']}
captures = {k: v for k, v in source['captures'].items()
            if v['site'] in sites and sites[v['site']]['scheme'] == a.scheme}
assert captures, 'sealed root has no selected captures'
files = {k: v for k, v in source['files'].items()
         if any(k.startswith(prefix + '.') for prefix in captures)}
a.output.mkdir(parents=True, exist_ok=False)
for name, info in files.items():
    src = a.source / name
    assert src.resolve().parent == a.source.resolve() and not src.is_symlink()
    payload = src.read_bytes()
    assert len(payload) == info['bytes']
    assert hashlib.sha256(payload).hexdigest() == info['sha256']
    (a.output / name).write_bytes(payload)
subset = dict(source, files=files, captures=captures,
              sites=[s for s in source['sites'] if s['scheme'] == a.scheme],
              subset_provenance={'source_root_sha256': original_hash,
                                 'source_directory': str(a.source.resolve()),
                                 'all_selected_member_hashes_verified': True})
if a.public_name:
    aliases = dict(subset.get('scheme_aliases', {}))
    assert a.public_name not in aliases or aliases[a.public_name] == a.scheme
    assert a.public_name != a.scheme, 'public alias must differ from the native scheme'
    # This changes selector metadata only; native sites and captured bytes remain
    # bound to the parent root above. Never relabel the native call-site scheme.
    aliases[a.public_name] = a.scheme
    subset['scheme_aliases'] = aliases
    subset['subset_provenance']['added_public_selector'] = {a.public_name: a.scheme}
subset['dump_root_sha256'] = hashlib.sha256(canonical(subset)).hexdigest()
(a.output / 'manifest.json').write_text(json.dumps(subset, indent=2) + '\n')
print(json.dumps({'source_root_sha256': original_hash,
                  'subset_root_sha256': subset['dump_root_sha256'],
                  'captures': len(captures), 'records': len(files)}))
