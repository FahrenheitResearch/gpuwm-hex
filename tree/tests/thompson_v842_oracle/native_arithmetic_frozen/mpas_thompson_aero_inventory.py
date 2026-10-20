"""Give the actual aerosol call f1's semantic thompson_aero selector.

Only the selected site's scheme metadata changes. Call keys, argument schemas,
native source and instrumentation hashes are retained without modification.
"""
import argparse
import hashlib
import json
from pathlib import Path


def alias(source, output):
    raw = source.read_bytes()
    inventory = json.loads(raw)
    key = 'mp_gt_driver__driver_microphysics__L415'
    selected = [row for row in inventory['sites'] if row['site'] == key]
    if len(selected) != 1 or selected[0]['scheme'] != 'mp_gt_driver':
        raise ValueError('expected the actual native aerosol driver call site')
    selected[0]['scheme'] = 'thompson_aero'
    inventory['scheme_alias_provenance'] = dict(
        original_inventory_sha256=hashlib.sha256(raw).hexdigest(),
        original_scheme='mp_gt_driver', replay_scheme='thompson_aero',
        sites=[key])
    with output.open('x') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    alias(args.source, args.output)
