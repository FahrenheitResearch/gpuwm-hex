#!/usr/bin/env python3
"""Audit captured module declarations against pinned source, without a closure claim.

The old Thompson inventory silently omitted COMPLEX SAVE declarations in
physics_mmm/mp_radar.F90:43. Comparing only the recorded variable list could
never discover that defect. Reparse source before checking PRE/POST membership.
Framework state, local SAVE state and external pointer aliases remain separate
obligations; this audit never changes hidden_state_complete.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ORACLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORACLE))
from state import modules
from fortran_source import routines
from local_state import local_variables
from allocation_state import patch_entry_storage


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(variable):
    normalize = lambda value: re.sub(r'\s+', '', value.lower())
    return dict(type=normalize(variable['type']),
                dims=[normalize(d) for d in variable['dims']],
                attrs=normalize(variable['attrs']))


def compare_declarations(current, recorded):
    """Both roots and emitted leaves matter: unsupported roots must stay visible."""
    differences = []
    for field in ('roots', 'variables'):
        expected = {v['name']: signature(v) for v in current[field]}
        actual = {v['name']: signature(v) for v in recorded.get(field, [])}
        for name in sorted(set(expected) | set(actual)):
            if expected.get(name) != actual.get(name):
                differences.append(dict(field=field, name=name,
                                        source=expected.get(name), inventory=actual.get(name)))
        if field=='variables':
            for variable in current[field]:
                old=next((v for v in recorded.get(field,[]) if v['name']==variable['name']),{})
                for graph in ('graph_clock','graph_timer','graph_stream'):
                    if variable.get(graph) and variable[graph]!=old.get(graph):
                        differences.append(dict(field=graph[6:]+'_graph_schema',name=variable['name'],
                            reason='native nested graph declarations or source hashes changed'))
    return differences


def membership(variables, prefix, phase, filenames, present, read_record=None):
    """Check conditional native leaves using their captured logical controls.

    The Rust reader interprets presence and logical payloads. Source-bound
    ASSOCIATED/ALLOCATED controls and scalar logical components can authorize
    absent descendants; missing or unsupported controls never do. Typed clock
    graphs and external alias semantics retain their separate obligations.
    """
    expected, missing, conditional, inactive, graphs = set(), set(), [], [], []
    declared = {(module, var['name'].lower()): var for module, var in variables}
    controls, visiting = {}, set()

    def record(module, name):
        return f'{prefix}.{phase}.state.{module}.{name}.bin'

    def logical(member):
        expected.add(member)
        if member not in filenames:
            missing.add(member)
            return None
        value = present(member)
        if type(value) is not bool:
            raise ValueError('native conditional state control is not a decoded logical')
        return value

    def condition(module, expression):
        key = (module, expression)
        if key in controls:
            return controls[key]
        if key in visiting:
            return None
        visiting.add(key)
        pointer = re.fullmatch(r'(associated|allocated)\s*\(\s*([\w%]+)\s*\)',
                               expression, re.I)
        name = pointer[2].lower() if pointer else expression.lower().strip()
        var = declared.get((module, name))
        value = None
        if var and not var.get('graph_clock') and not var.get('graph_timer') and not var.get('graph_stream'):
            eligible = active(module, var)
            if eligible is True:
                if pointer:
                    storage = 'pointer' if pointer[1].lower() == 'associated' else 'allocatable'
                    if storage in var['attrs']:
                        value = logical(record(module, var['name'] + '.present'))
                elif (re.fullmatch(r'[\w%]+', expression.strip()) and
                      var['type'].lower().startswith('logical') and not var['dims'] and
                      not any(a in var['attrs'] for a in ('pointer', 'allocatable'))):
                    value = logical(record(module, var['name']))
        visiting.remove(key)
        controls[key] = value
        return value

    def active(module, var):
        # Source emits nested IFs in this order. A captured false outer control
        # skips inner controls whose native objects do not exist.
        for expression in var.get('conditions', []):
            value = condition(module, expression)
            if value is not True:
                return value
        return True

    for module, variable in variables:
        name = f"state.{module}.{variable['name']}"
        stem = f'{prefix}.{phase}.{name}'
        if variable.get('graph_clock') or variable.get('graph_timer') or variable.get('graph_stream'):
            timer=bool(variable.get('graph_timer'));stream=bool(variable.get('graph_stream'))
            if read_record is None:
                conditional.append(dict(name=name, conditions=variable.get('conditions', []),
                                        reason='typed native graph membership requires a topology audit'))
            else:
                if stream:
                    from stream_graph import audit_membership
                elif timer:
                    from timer_graph import audit_membership
                else:
                    from clock_graph import audit_membership
                graph = audit_membership(variable['graph_stream' if stream else 'graph_timer' if timer else 'graph_clock'], prefix, phase, name,
                                         filenames, read_record)
                graphs.append(dict(name=name, kind='stream' if stream else 'timer' if timer else 'clock', audit=graph))
                missing.update(graph['missing'])
            continue
        eligibility = active(module, variable)
        if eligibility is None:
            conditional.append(dict(name=name, conditions=variable.get('conditions', []),
                                    reason='native conditional control is missing, unsupported or cyclic'))
            continue
        if eligibility is False:
            inactive.append(dict(name=name, conditions=variable.get('conditions', [])))
            continue
        if variable.get('container_pointer'):
            logical(stem + '.present.bin')
            alias = stem + '.alias.bin'
            expected.add(alias)
            if alias not in filenames:
                missing.add(alias)
            continue
        guard = any(attr in variable['attrs'] for attr in ('pointer', 'allocatable'))
        if guard:
            flag = stem + '.present.bin'
            if logical(flag) is not True:
                continue
        value = stem + '.bin'
        expected.add(value)
        if value not in filenames:
            missing.add(value)
    return dict(expected_records=len(expected) + sum(g['audit']['expected_records'] for g in graphs), missing=sorted(missing),
                conditional_unverified=conditional,
                inactive_source_guarded_variables=inactive,
                alias_semantics_complete=False, clock_graphs=[g for g in graphs if g['kind']=='clock'],
                timer_graphs=[g for g in graphs if g['kind']=='timer'],
                stream_graphs=[g for g in graphs if g['kind']=='stream'],
                complete=not missing and not conditional and all(g['audit']['complete'] for g in graphs))


def local_inventory(source_root, module_names, module_infos, recorded):
    """Reparse SAVE/DATA/COMMON hooks, including silently missing routines."""
    expected = {}
    for relative in sorted({module_infos[n]['path'] for n in module_names}):
        for routine in routines(source_root / relative, source_root):
            if routine.module not in module_names:
                continue
            variables, issues = local_variables(routine)
            if not variables and not issues:
                continue
            expected[(routine.module, routine.name)] = dict(
                module=routine.module, routine=routine.name, path=relative,
                line=routine.start + 1, variables=variables, issues=issues)
    actual = {(r['module'], r['routine']): r for r in recorded
              if r['module'] in module_names}
    rows = []
    for key in sorted(set(expected) | set(actual)):
        src, old = expected.get(key), actual.get(key)
        differences = []
        if src is None or old is None:
            differences.append('source or recorded local hook is missing')
        else:
            differences.extend(compare_declarations(
                dict(roots=src['variables'], variables=src['variables']),
                dict(roots=old['variables'], variables=old['variables'])))
            if src['issues'] != old['issues']:
                differences.append(dict(source_issues=src['issues'],
                                        inventory_issues=old['issues']))
        rows.append(dict(module=key[0], routine=key[1], source=src,
                         inventory_present=old is not None, differences=differences))
    return rows


def reconcile_local_issues(module, issues, hooks):
    """Separate supported local storage hooks from unsupported declarations.

    state.modules() reports SAVE/DATA before it knows whether generated local
    hooks exist. Reparse and match the whole native routine inventory first;
    observed pairs and numerical execution are assessed separately below.
    """
    unresolved, represented = [], []
    known = {(h['module'], h['routine']): h for h in hooks}
    for issue in issues:
        hook = known.get((module, issue.get('routine')))
        source = hook['source'] if hook else None
        supported = bool(source and not hook['differences'] and not source['issues'])
        reason = issue.get('reason')
        covered = supported and (
            reason == 'COMMON/DATA local state requires capture' or
            reason == 'local SAVE state requires capture' and
            issue.get('name') in {v['name'] for v in source['variables']})
        (represented if covered else unresolved).append(issue)
    return unresolved, represented


def local_membership(hooks, prefix, filenames, accessors=()):
    """Missing pairs are failures; an entirely unobserved routine stays unknown."""
    rows = []
    known = set()
    primed = {(a['module'], a['routine']) for a in accessors}
    for hook in hooks:
        source = hook['source']
        if source is None:
            continue
        stems = [f"local.{hook['module']}.{hook['routine']}.{v['name']}"
                 for v in source['variables']]
        expected = {f'{prefix}.{phase}.{stem}.bin'
                    for stem in stems for phase in ('pre', 'post')}
        known.update(expected)
        observed = expected.intersection(filenames)
        rows.append(dict(module=hook['module'], routine=hook['routine'],
                         observed=bool(observed), expected_records=len(expected),
                         missing=sorted(expected - observed) if observed else [],
                         unsupported=source['issues'],
                         static_accessor_possible=(hook['module'], hook['routine']) in primed,
                         scope=('Observed original SAVE storage; accessor records do not prove numerical execution'
                                if (hook['module'], hook['routine']) in primed and observed else
                                'Observed first-entry/final-exit records only' if observed
                                else 'No local records: execution is unproven')))
    observed_all = {n for n in filenames
                    if n.startswith((prefix + '.pre.local.', prefix + '.post.local.'))}
    return dict(routines=rows, unexpected=sorted(observed_all - known),
                observed_pairs_complete=not (observed_all - known) and
                all(not r['missing'] for r in rows),
                execution_coverage_complete=all(r['observed'] and not r['unsupported'] and
                                                not r['static_accessor_possible']
                                                for r in rows))


def entry_inventory(source_root, module_names, recorded, requests):
    """Reparse declared automatic inputs; old inventories cannot hide omissions."""
    expected = []
    selected = [r for r in requests if r['module'] in module_names]
    for relative in sorted({r['path'] for r in selected}):
        original = source_root / relative
        _, rows = patch_entry_storage(original, original.read_text(),
                                      [r for r in selected if r['path'] == relative])
        expected.extend(rows)
    key = lambda r: (r['module'], r['routine'], r.get('name'))
    old = {key(r): r for r in recorded if r['module'] in module_names}
    native = {key(r): r for r in expected}
    fields = ('record', 'type', 'dims', 'line', 'end_line', 'capture', 'issues')
    rows = []
    for identity in sorted(set(native) | set(old), key=str):
        src, prior = native.get(identity), old.get(identity)
        differences = ([dict(field=field, source=src.get(field), inventory=prior.get(field))
                        for field in fields if src.get(field) != prior.get(field)]
                       if src is not None and prior is not None else
                       ['source or recorded automatic entry input is missing'])
        rows.append(dict(module=identity[0], routine=identity[1], name=identity[2],
                         source=src, inventory_present=prior is not None,
                         differences=differences))
    return rows


def entry_membership(hooks, prefix, filenames, callee):
    """An automatic input is PRE only; a POST name cannot satisfy it."""
    rows = []
    for hook in hooks:
        source = hook['source']
        if source is None:
            continue
        stem = f"{prefix}.pre.allocation.{source['record']}.n"
        matched = sorted(n for n in filenames
                         if re.fullmatch(re.escape(stem) + r'\d{10}\.bin', n))
        ordinals = [int(n[len(stem): -4]) for n in matched]
        direct = (callee['module'], callee['name']) == (hook['module'], hook['routine'])
        rows.append(dict(record=source['record'], observed=bool(matched),
                         records=matched, required_at_callee_entry=direct,
                         missing=direct and not matched,
                         ordinals_complete=ordinals == list(range(1, len(ordinals) + 1)),
                         scope='Captured automatic PRE inputs only; other allocator storage unproven'))
    return dict(inputs=rows, observed_inputs_complete=all(
        not r['missing'] and r['ordinals_complete'] for r in rows),
        execution_coverage_complete=all(r['observed'] for r in rows))


def audit(root, inventory_path, source_root, scheme, words):
    spec = importlib.util.spec_from_file_location('column_oracle', ORACLE.parent.parent / 'column_oracle_replay.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    manifest = harness.verify(root)
    if sha(inventory_path) != manifest['inventory_sha256']:
        raise ValueError('source inventory does not belong to the sealed dump')
    inventory = json.loads(inventory_path.read_text())
    sites = {s['site']: s for s in manifest['sites']}
    selected = harness.selected_schemes(manifest, scheme)
    captures = {p: c for p, c in manifest['captures'].items()
                if c['site'] in sites and
                (c['site'] in selected or sites[c['site']]['scheme'] in selected)}
    if not captures:
        raise ValueError('no captured call for the requested scheme')
    current = modules(source_root, ('src/core_atmosphere/dynamics/mpas_atm_iau.F',
                                    'src/core_atmosphere/mpas_atm_core.F'))
    recorded = {m['name']: m for m in inventory['module_state']}
    names = sorted({name for c in captures.values() for name in sites[c['site']]['state_modules']})
    audits = {}
    for name in names:
        if name not in current or name not in recorded:
            raise ValueError(f'missing module inventory: {name}')
        relative = current[name]['path']
        if manifest['source_sha256'].get(relative) != sha(source_root / relative):
            raise ValueError(f'changed or unbound native module source: {relative}')
        audits[name] = dict(path=relative, source_sha256=sha(source_root / relative),
                           source_declarations=len(current[name]['roots']),
                           source_leaves=len(current[name]['variables']),
                           differences=compare_declarations(current[name], recorded[name]),
                           source_parser_issues=current[name]['issues'])
    local_hooks = local_inventory(source_root, set(names), current,
                                  inventory.get('local_state', []))
    for name, module in audits.items():
        raw = module['source_parser_issues']
        unresolved, represented = reconcile_local_issues(name, raw, local_hooks)
        module['source_parser_observations'] = raw
        module['represented_local_storage_observations'] = represented
        module['source_parser_issues'] = unresolved
    request_path = ORACLE / 'entry-storage.v1.json'
    entry_hooks = entry_inventory(source_root, set(names), inventory.get('entry_storage', []),
                                  json.loads(request_path.read_text())['requests'])
    rows = []
    for prefix, capture in captures.items():
        variables = [(name, var) for name in sites[capture['site']]['state_modules']
                     for var in current[name]['variables']]
        phases = {phase: membership(variables, prefix, phase, manifest['files'],
                                   lambda member: harness.words(words, 'present', root / member, '-'),
                                   lambda operation, member, indices: harness.words(
                                       words, operation, root / member, indices))
                  for phase in ('pre', 'post')}
        rows.append(dict(capture=prefix, phases=phases,
                         local_state=local_membership(local_hooks, prefix, manifest['files'],
                                                      inventory.get('local_state_accessors', [])),
                         automatic_entry=entry_membership(entry_hooks, prefix, manifest['files'],
                                                          sites[capture['site']]['callee'])))
    result = dict(schema='mpas-oracle-source-module-state-audit-v1', scheme=scheme,
                  dump_root_sha256=manifest['dump_root_sha256'], inventory_sha256=sha(inventory_path),
                  capture_process_count=manifest.get('process_count'), modules=audits, captures=rows,
                  declaration_inventory_matches=all(not m['differences'] for m in audits.values()),
                  capture_membership_complete=all(p['complete'] for r in rows for p in r['phases'].values()),
                  source_parser_issues_present=any(m['source_parser_issues'] for m in audits.values()),
                  local_state_source_inventory=local_hooks,
                  local_state_inventory_matches=all(not h['differences'] for h in local_hooks),
                  observed_local_pairs_complete=all(r['local_state']['observed_pairs_complete']
                                                    for r in rows),
                  local_execution_coverage_complete=all(r['local_state']['execution_coverage_complete']
                                                       for r in rows),
                  automatic_entry_source_inventory=entry_hooks,
                  automatic_entry_inventory_matches=all(not h['differences'] for h in entry_hooks),
                  observed_automatic_inputs_complete=all(
                      r['automatic_entry']['observed_inputs_complete'] for r in rows),
                  automatic_entry_execution_coverage_complete=all(
                      r['automatic_entry']['execution_coverage_complete'] for r in rows),
                  audit_source_sha256={p.name: sha(p) for p in
                                      (Path(__file__), ORACLE / 'state.py', ORACLE / 'fortran_source.py',
                                       ORACLE / 'local_state.py', ORACLE / 'allocation_state.py', ORACLE / 'clock_graph.py', ORACLE / 'timer_graph.py', ORACLE / 'stream_graph.py',
                                       request_path)},
                  input_hidden_state_complete=manifest['hidden_state_complete'],
                  hidden_state_complete=False, forecast_proven=False,
                  scope='Known module and local SAVE/DATA/COMMON declarations plus observed PRE/POST pairs; '
                        'unobserved routines, framework globals, external aliases and conditional derived state '
                        'are not certified')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dump', 'inventory', 'source-root', 'words', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scheme', required=True)
    args = parser.parse_args()
    result = audit(args.dump, args.inventory, args.source_root, args.scheme, args.words)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if (result['declaration_inventory_matches'] and result['capture_membership_complete']
                 and result['local_state_inventory_matches'] and result['observed_local_pairs_complete']
                 and result['automatic_entry_inventory_matches']
                 and result['observed_automatic_inputs_complete']
                 and not result['source_parser_issues_present']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
