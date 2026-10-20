"""Audit the Thompson capture inventory against native source declarations.

The f1 global completeness flag remains authoritative. This metadata audit names
every declaration in the Thompson module dependency closure and checks every
available captured call; it does not prove pointer/allocator state or forecasts.
"""
import argparse
import hashlib
import json
import importlib.util
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output, mpas_source, source_parser):
    spec = importlib.util.spec_from_file_location('thompson_source_audit_parser', source_parser)
    parser = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    inventory_path = root / 'scratch/thompson-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    site = next(s for s in inventory['sites'] if s['scheme'] == 'thompson')
    modules = {m['name']: m for m in inventory['module_state']
               if m['name'] in site['state_modules']}
    assert not site['issues']
    assert all(not m['issues'] for m in modules.values())
    locals_ = [r for r in inventory['local_state'] if r['module'] in modules]
    assert not locals_, 'Local saved state needs an explicit declaration audit'
    declarations = []
    source_declarations = []
    native_source_hashes = {}
    for module in modules.values():
        path = mpas_source / module['path']
        native_source_hashes[module['path']] = digest(path)
        in_module = False
        native_variables = []
        parameters = set()
        for statement in parser.statements(parser.active_source(path.read_text())):
            text = statement.text.lower()
            if text.startswith('module '):
                in_module = text.split()[1] == module['name']
            elif in_module:
                if text == 'contains':
                    break
                if text.startswith('parameter'):
                    parameters.update(entity.split('=', 1)[0].strip() for entity in
                                      parser.split_top(text[text.index('(')+1:text.rindex(')')]))
                for variable in parser.declarations(statement.text):
                    if 'parameter' not in variable.attrs:
                        native_variables.append(dict(module=module['name'], name=variable.name,
                            type=variable.type, source=module['path'], line=statement.start+1))
        source_declarations.extend(variable for variable in native_variables
                                   if variable['name'] not in parameters)
        for variable in module['variables']:
            assert not variable.get('conditions')
            assert 'pointer' not in variable['attrs']
            declarations.append({'module': module['name'], 'name': variable['name'],
                                 'type': variable['type'], 'attrs': variable['attrs'],
                                 'dims': variable['dims'], 'source': module['path'],
                                 'line': variable['line']})
    inventoried = {(v['module'], v['name']) for v in declarations}
    missing_declarations = [v for v in source_declarations
                            if (v['module'], v['name']) not in inventoried]
    assert inventoried <= {(v['module'], v['name']) for v in source_declarations}
    repairs = json.loads((root / 'receipts/own-scratch-relocation-alias-repair.json').read_text())['rows']
    rows = []
    for specification in json.loads((root / 'scratch/current-real-native-coverage-spec.json').read_text()):
        source = specification['source']
        for repair in repairs:
            for old in (repair['path'], repair['old_resolved_path']):
                if source == old or source.startswith(old + '/'):
                    source = repair['new_target'] + source[len(old):]
        manifest_path = Path(source) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['source_sha256'] == inventory['source_sha256']
        captured_site = next(s for s in manifest['sites'] if s['site'] == site['site'])
        assert captured_site['state_modules'] == site['state_modules']
        assert captured_site['arguments'] == site['arguments']
        assert not captured_site['issues']
        calls = []
        for prefix, capture in manifest['captures'].items():
            if capture['site'] != site['site']:
                continue
            if specification.get('steps') and not any(
                    f'.s{step:010d}.' in prefix for step in specification['steps']):
                continue
            phase_counts = {}
            for phase in ('pre', 'post'):
                start = prefix + '.' + phase + '.state.'
                names = {name[len(start):-4] for name in manifest['files']
                         if name.startswith(start) and name.endswith('.bin')}
                known = set()
                for declaration in declarations:
                    name = declaration['module'] + '.' + declaration['name']
                    known.add(name)
                    if 'allocatable' in declaration['attrs']:
                        known.add(name + '.present')
                        assert name + '.present' in names, (source, prefix, phase, name)
                    else:
                        assert name in names, (source, prefix, phase, name)
                assert not names - known, (source, prefix, phase, names - known)
                assert all(v['module'] + '.' + v['name'] not in names
                           for v in missing_declarations)
                phase_counts[phase] = len(names)
            calls.append({'capture': prefix, 'state_records': phase_counts})
        assert calls, specification['label']
        rows.append({'label': specification['label'], 'manifest_sha256': digest(manifest_path),
                     'sealed_inventory_sha256': manifest['inventory_sha256'],
                     'uses_exact_original_inventory': manifest['inventory_sha256'] == digest(inventory_path),
                     'declared_hidden_state_complete': manifest['hidden_state_complete'],
                     'capture_process_count': manifest.get('capture_process_count'), 'calls': calls})
    assert len(rows) == 15
    result = {'acceptance': False, 'forecast_proven': False,
              'scope': 'Every original inventoried variable has capture metadata; native source declarations omitted by that inventory remain explicitly missing. Allocation-presence payloads are not decoded.',
              'source_parser_sha256': digest(source_parser),
              'native_source_sha256': native_source_hashes,
              'native_declaration_count': len(source_declarations),
              'native_declarations_missing_from_inventory': missing_declarations,
              'source_declarations_complete': not missing_declarations,
              'global_hidden_state_complete_overridden': False,
              'inventory_sha256': digest(inventory_path), 'source_sha256': inventory['source_sha256'],
              'site': site['site'], 'modules': site['state_modules'],
              'declaration_count': len(declarations), 'declarations': declarations,
              'local_saved_declaration_count': len(locals_),
              'omitted_optional_arguments': [v['name'] for v in site['omitted_optional_arguments']],
              'collection_count': len(rows), 'capture_count': sum(len(r['calls']) for r in rows),
              'rows': rows}
    output.write_text(json.dumps(result, indent=2) + '\n')
    assert output.stat().st_size < 10_000_000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpas-source', type=Path, required=True)
    parser.add_argument('--source-parser', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve(), args.mpas_source.resolve(),
          args.source_parser.resolve())
