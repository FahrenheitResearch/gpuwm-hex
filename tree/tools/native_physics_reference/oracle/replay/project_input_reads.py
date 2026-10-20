#!/usr/bin/env python3
"""Project unused allocation-entry records from one witnessed native replay.

Every argument and module/local state record is retained. Only allocation PRE
records that no supplied native recipe opened are removed. This is a debugging
input projection, not a declaration of hidden-state or branch completeness.
Numeric record extraction and comparison remain in the Rust oracle helper.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil

TOOL = Path(__file__).resolve().parents[3] / 'column_oracle_replay.py'
spec = importlib.util.spec_from_file_location('column_oracle_replay', TOOL)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
OPEN = re.compile(r'openat\([^,]+,\s*("(?:[^"\\]|\\.)*"),\s*([^)]*)\)\s*=\s*(-?\d+)')
RECIPES = {'gnu-strict', 'gnu-o3', 'ifx-o3', 'ifx-valuesafe', 'coverage'}


def read_set(trace, input_root, members):
    """Read successful native file-open metadata; never inspect field payloads."""
    result = set()
    input_root = input_root.resolve()
    for line in trace.read_text().splitlines():
        match = OPEN.search(line)
        if not match:
            if 'openat(' in line and str(input_root) in line:
                raise ValueError('unparsed native input open')
            continue
        name, flags, fd = match.groups()
        path = Path(json.loads(name))
        if path.parent.resolve() != input_root:
            continue
        if int(fd) < 0:
            raise ValueError('native input open failed')
        if not path.name.endswith('.bin'):
            continue
        if 'O_RDONLY' not in flags:
            raise ValueError('native input was opened for writing')
        if path.name not in members:
            raise ValueError('native opened an unsealed input record')
        result.add(path.name)
    if not result:
        raise ValueError('trace contains no successful native input reads')
    return result


def project(source, traces_path, inventory_path, pins, helper, target):
    source = source.resolve()
    manifest = harness.verify(source)
    traces = json.loads(traces_path.read_text())
    if traces['source_dump_root_sha256'] != manifest['dump_root_sha256']:
        raise ValueError('native input trace belongs to another dump')
    prefix = traces['capture']
    if set(manifest['captures']) != {prefix}:
        raise ValueError('input projection requires exactly one selected capture')
    if {r['recipe'] for r in traces['runs']} != RECIPES or len(traces['runs']) != len(RECIPES):
        raise ValueError('input projection requires all five native recipes exactly once')
    opened = set()
    runs = []
    for run in traces['runs']:
        command = run['command']
        if run['exit'] != 0 or len(command) != 4 or command[3] != 'column':
            raise ValueError('native trace did not complete source-bound column replay')
        if Path(command[1]).resolve() != source / prefix or Path(run['input_root']).resolve() != source:
            raise ValueError('native trace input root changed')
        exe, trace = Path(command[0]), Path(run['trace'])
        if harness.hash_file(exe) != run['executable_sha256'] or harness.hash_file(trace) != run['trace_sha256']:
            raise ValueError('native executable or input trace hash changed')
        receipt_path = Path(run['replay_receipt'])
        if harness.hash_file(receipt_path) != run['replay_receipt_sha256']:
            raise ValueError('native replay receipt hash changed')
        receipt = json.loads(receipt_path.read_text())
        if (receipt['dump_root_sha256'] != manifest['dump_root_sha256'] or
                receipt['executable_sha256'] != run['executable_sha256'] or
                receipt['replay_recipe'] != run['recipe'] or
                receipt['execution_mode'] != 'column' or
                [c['capture'] for c in receipt['captures']] != [prefix] or
                not receipt['comparison_complete']):
            raise ValueError('native replay receipt does not bind this complete column execution')
        if run['recipe'] in {'gnu-strict', 'coverage'} and not receipt['all_outputs_bitwise_equal']:
            raise ValueError('strict or coverage native column witness differs')
        members = read_set(trace, source, manifest['files'])
        opened.update(members)
        runs.append(dict(recipe=run['recipe'], execution_host=run['execution_host'],
                         executable_sha256=run['executable_sha256'],
                         trace_sha256=run['trace_sha256'],
                         replay_receipt_sha256=run['replay_receipt_sha256'],
                         opened_records=len(members)))
    allocation = prefix + '.pre.allocation.'
    dropped = {name for name in manifest['files'] if name.startswith(allocation) and name not in opened}
    if not dropped:
        raise ValueError('no unused allocation-entry records to project')
    kept = set(manifest['files']) - dropped
    inventory = json.loads(inventory_path.read_text())
    if harness.hash_file(inventory_path) != manifest['inventory_sha256']:
        raise ValueError('source inventory differs from the sealed capture')
    provenance = dict(schema='mpas-native-allocation-input-read-projection-v1',
        parent_dump_root_sha256=manifest['dump_root_sha256'],
        parent_manifest_sha256=harness.hash_file(source/'manifest.json'),
        native_traces_sha256=harness.hash_file(traces_path), capture=prefix,
        removed_records=len(dropped), removed_bytes=sum(manifest['files'][n]['bytes'] for n in dropped),
        retained_records=len(kept), retained_bytes=sum(manifest['files'][n]['bytes'] for n in kept),
        native_runs=runs, hidden_state_complete=False, coverage_complete=False,
        scope='Five completed source-bound column executions only; every argument, expected POST, '
              'module/local state and native-opened allocation input retained. '
              'All projected recipes must be replayed and independently compared before publication.')
    inventory = dict(inventory)
    inventory['input_provenance'] = dict(inventory.get('input_provenance', {}),
                                          allocation_input_projection=provenance)
    target.mkdir()
    for name in sorted(kept):
        shutil.copyfile(harness.safe_file(source, name), target/name)
    projected_inventory = target/'inventory.json'
    projected_inventory.write_text(json.dumps(inventory, indent=2)+'\n')
    sealed = harness.seal(target, projected_inventory, pins, helper,
                           manifest['recipe'], manifest.get('process_count'))
    provenance['dump_root_sha256'] = sealed['dump_root_sha256']
    (target/'PROJECTION.json').write_text(json.dumps(provenance, indent=2)+'\n')
    shutil.copyfile(traces_path, target/'INPUT-TRACES.json')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'traces', 'inventory', 'pins', 'words', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    result = project(a.source, a.traces, a.inventory, a.pins, a.words, a.out)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
