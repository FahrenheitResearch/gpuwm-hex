"""Classify the complete coverage ledger using original caller CFG contracts.

MPAS module_bl_mynn.F:485-593 constructs contiguous descriptors for
bl_mynn.F90:126-164. MPAS mpas_atmphys_driver_pbl.F:877-911 supplies the
WRF optional argument contract. Scientific predicates remain unconstrained.
No native word or synthetic profile is excluded, and every original gap stays
in the resulting ledger, including any branch the native caller can reach.
"""
import argparse
import copy
import json
import re
from pathlib import Path

from map_complete_profile_coverage_v104 import graphs, profile_blocks, sha


def main(a):
    result = copy.deepcopy(json.loads(a.classified.read_text()))
    mapping = json.loads(a.mapping.read_text())
    contract = json.loads(a.contract.read_text())
    assert sha(a.classified) == mapping['classified_ledger_sha256']
    assert mapping['identical_complete_native_graphs']
    assert sha(a.profile / 'admission.json') == mapping['profile_admission_sha256']
    for path, digest in contract['native_sources'].items():
        assert sha(Path(path)) == digest
    original = {n: next(a.profile.glob('run_adapter-' + n + '.*.profile_estimate')).read_text()
                for n in ('bl_mynn.F90', 'module_bl_mynn.F')}
    # Read the actual compiled ABI order, not the keyword order in the source.
    signature = re.search(r'^void bl_mynn_run \((.*)\)$', original['bl_mynn.F90'], re.M)[1]
    formals = [re.search(r'(\w+)$', part)[1] for part in signature.split(', ')]
    caller = original['module_bl_mynn.F'].split(';; Function mynn_bl_driver ', 1)[1]
    actual_calls = re.findall(r'^.*\] bl_mynn_run \((.*)\);$', caller, re.M)
    assert actual_calls
    calls = [re.sub(r'\[module_bl_mynn\.F:[^\]]*\] ', '', call).split(', ')
             for call in actual_calls]
    assert all(call == calls[0] for call in calls)
    assert len(formals) == len(calls[0])
    actuals = dict(zip(formals, calls[0]))
    descriptors = {}
    for field, actual in actuals.items():
        if re.fullmatch(r'&parm\.\d+', actual):
            parm = actual[1:]
            strides = re.findall(re.escape(parm) + r'\.dim\[0\]\.stride = ([^;]+);', caller)
            data = re.findall(re.escape(parm) + r'\.data = ([^;]+);', caller)
            assert strides and set(strides) == {'1'}, (field, strides)
            assert data and all(value != '0B' for value in data), (field, data)
            descriptors[field] = dict(actual=actual, stride=1, data_assignments=sorted(set(data)))
    assert len(descriptors) == 82
    for field, spec in contract['calls'][1]['optional_arguments'].items():
        assert (actuals[field] != '0B') == spec['present']

    known, witnesses, reachable = {}, {}, {}
    for name, routine in [('bl_mynn.F90', 'bl_mynn_run'),
                           ('module_bl_mynn.F', 'mynn_bl_driver')]:
        symbol = '__' + ('bl_mynn' if name == 'bl_mynn.F90' else 'module_bl_mynn') + '_MOD_' + routine
        conditions = profile_blocks(original[name])[symbol]
        graph = graphs((a.mapping.parent / ('run_adapter-' + Path(name).stem + '.gcno.dump')).read_text())[symbol]['arcs']
        fixed, proof = {}, {}
        for block, condition in conditions.items():
            # Later optimization can change an unrelated block. Use only
            # predicates whose two original gcno successors agree exactly.
            successors = graph.get(block, [])
            if len(successors) != 2 or sorted(x['destination'] for x in successors) != sorted([
                    condition['true_destination'], condition['false_destination']]):
                continue
            expression = condition['expression']
            pointer = re.fullmatch(r'(\w+)_\d+\(D\) != 0B', expression)
            field = pointer[1] if pointer else None
            if pointer:
                if name == 'bl_mynn.F90' and field in actuals:
                    fixed[block] = actuals[field] != '0B'
                    proof[block] = dict(kind='original caller descriptor presence', argument=field,
                                        actual=actuals[field])
                elif name == 'module_bl_mynn.F' and field in contract['calls'][0]['optional_arguments']:
                    fixed[block] = contract['calls'][0]['optional_arguments'][field]['present']
                    proof[block] = dict(kind='pinned MPAS driver optional presence', argument=field)
                continue
            value = re.fullmatch(r'(_\d+) != (0B|0)', expression)
            if name != 'bl_mynn.F90' or not value:
                continue
            load = re.search(re.escape(value[1]) + r' = .*?\b(\w+)_\d+\(D\)->(data|dim\[0\]\.stride);',
                             condition['full_block'])
            if load and load[1] in descriptors:
                fixed[block] = True
                proof[block] = dict(kind='allocated native contiguous descriptor', argument=load[1],
                                    member=load[2], caller_descriptor=descriptors[load[1]],
                                    limitation='Data presence assumes successful native allocation; allocation failure does not enter this scheme.')
        seen, pending = set(), [0]
        while pending:
            block = pending.pop()
            if block in seen:
                continue
            seen.add(block)
            destinations = [arc['destination'] for arc in graph.get(block, [])]
            if block in fixed:
                condition = conditions[block]
                destination = condition['true_destination' if fixed[block] else 'false_destination']
                assert destination in destinations
                destinations = [destination]
            pending.extend(destinations)
        known[name] = fixed
        witnesses[name] = proof
        reachable[name] = seen
    rows = {(r['gap']['file'], r['gap']['line'], r['gap']['branch']): r for r in mapping['rows']}
    changed = []
    for gap in result['uncovered_edges']:
        if not gap['classification'].startswith('OPEN'):
            continue
        key = (gap['file'], gap['line'], gap['branch'])
        row = rows[key]
        name = gap['file']
        if name not in known or not row['mapped'] or row['untaken_arm'] is None:
            continue
        block = row['gcov_arc']['block']
        if block not in reachable[name]:
            evidence = dict(reason='Original compiler block is unreachable after applying the pinned caller presence and descriptor contracts. All scientific predicates remain unconstrained.')
        elif block in known[name] and known[name][block] != row['untaken_arm']['predicate_value']:
            evidence = dict(reason='This exact untaken gcov arm contradicts the original caller descriptor or optional argument contract.',
                            caller_proof=witnesses[name][block])
        else:
            continue
        gap.update(evidence)
        gap['native_profile_mapping'] = row
        gap['classification'] = 'UNREACHABLE in pinned MPAS caller interfaces: authenticated compiler ABI'
        changed.append(key)
    result['counts'] = {n: sum(r['classification'].startswith(n) for r in result['uncovered_edges'])
                        for n in ('OPEN', 'UNREACHABLE')}
    result['profile_interface_audit'] = dict(scope=__doc__, process_count=1,
        mapping_sha256=sha(a.mapping), contract_sha256=sha(a.contract),
        source_profile_sha256={n: sha(next(a.profile.glob('run_adapter-' + n + '.*.profile_estimate')))
                               for n in original},
        compiled_actual_arguments=actuals, allocated_contiguous_descriptors=descriptors,
        fixed_predicates=witnesses,
        reachable_blocks={n: sorted(s) for n, s in reachable.items()},
        newly_classified=changed, excluded_words=[], full_branch_coverage=False)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('classified', 'mapping', 'contract', 'profile', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    main(p.parse_args())
