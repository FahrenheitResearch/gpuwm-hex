#!/usr/bin/env python3
"""Isolate compiled hooks by restoring complete native module routine bodies.

This emits an owned diagnostic build, never refreshes the frozen model, and
does not claim capture completeness. Module state serializers remain available
to existing driver hooks. Every native routine body is copied byte for byte.
Compile the emitted Makefile with make -j4 (through HEX_BUILD_QUEUE on a shared host).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fortran_source import routines, statements, split_top
from state import module_anchors


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_bodies(original, instrumented, module_names):
    native = {(r.module, r.name): r for r in routines(original, original.parent)
              if r.module in module_names}
    generated = {(r.module, r.name): r for r in routines(instrumented, instrumented.parent)}
    lines = instrumented.read_text().splitlines(keepends=True)
    raw = original.read_text().splitlines(keepends=True)
    spans = []
    for key, routine in native.items():
        if key not in generated:
            raise ValueError('native routine absent from instrumented source: ' + repr(key))
        target = generated[key]
        spans.append((target.start, target.end, ''.join(raw[routine.start:routine.end]), key))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise ValueError('overlapping native routine bodies need an enclosing-module adapter')
    for start, end, text, _ in reversed(spans):
        lines[start:end] = [text]
    return ''.join(lines), [dict(module=key[0], routine=key[1],
                               source_sha256=hashlib.sha256(text.encode()).hexdigest())
                           for _, _, text, key in spans]


def restore_private_procedures(original, generated_text, module_names):
    """Retain serializer exports without promoting native private procedures."""
    native_text = original.read_text()
    procedure_names = {(r.module, r.name) for r in routines(original, original.parent)}
    restored = {}
    for name in sorted(module_names):
        native_contains, _ = module_anchors(native_text, name)
        native_start = next(st.start for st in statements(native_text)
                            if re.fullmatch(r'module\s+' + re.escape(name), st.text, re.I))
        offset = sum(len(line) for line in native_text.splitlines(keepends=True)[:native_start])
        native_spec = list(statements(native_text[offset:native_contains]))
        if not any(re.fullmatch('private', st.text, re.I) for st in native_spec):
            continue
        exports = set()
        for st in native_spec:
            match = re.fullmatch(r'public\s*(?:::)?\s+(.+)', st.text, re.I)
            if match:
                exports.update(s.lower() for s in split_top(match[1]))
        private = {routine for module, routine in procedure_names
                   if module == name and routine not in exports}
        contains, _ = module_anchors(generated_text, name)
        lines = generated_text.splitlines(keepends=True)
        header_lines = len(generated_text[:contains].splitlines())
        start = next(st.start for st in statements(generated_text)
                     if re.fullmatch(r'module\s+' + re.escape(name), st.text, re.I))
        changes = []
        removed = set()
        for st in statements(generated_text):
            if not start <= st.start < header_lines:
                continue
            match = re.fullmatch(r'public\s*(?:::)?\s+(.+)', st.text, re.I)
            if not match:
                continue
            names = split_top(match[1])
            removed.update(s.lower() for s in names if s.lower() in private)
            kept = [s for s in names if s.lower() not in private]
            if kept != names:
                changes.append((st.start, st.end, ' public :: ' + ','.join(kept) + '\n' if kept else ''))
        for begin, end, replacement in reversed(changes):
            lines[begin:end] = [replacement]
        generated_text = ''.join(lines)
        restored[name] = sorted(removed)
    return generated_text, restored


def seal_variant(out):
    """Bind the compiled variant to unchanged prepared sources and libraries."""
    out = out.resolve()
    manifest_path = out / 'variant.json'
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['rows']:
        if sha(out / Path(row['path']).name) != row['variant_sha256']:
            raise ValueError('prepared native body changed: ' + row['path'])
    for name, digest in manifest['dependencies'].items():
        if sha(Path(name)) != digest:
            raise ValueError('prepared build dependency changed: ' + name)
    if sha(out / 'Makefile') != manifest['makefile_sha256']:
        raise ValueError('prepared compile/link recipe changed')
    products = [out / 'atmosphere_model']
    products += [out / name for name in manifest.get('libraries', ['libdycore.a'])]
    products += [out / (Path(row['path']).stem + '.o') for row in manifest['rows']]
    result = dict(schema='mpas-compiled-hook-passivity-artifacts-v1',
                  variant_manifest_sha256=sha(manifest_path),
                  artifacts={p.name: sha(p) for p in products},
                  forecast_proven=False)
    (out / 'artifacts.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def parent_dependencies(parent, manifest):
    """Resolve frozen source snapshots without accepting changed build inputs.

    An older variant may have read a live generator path. Its prepared source
    is an exact archived copy, bound by the compiled variant's manifest. Only
    that matching snapshot can replace a subsequently edited source path;
    changed flags, libraries or arbitrary dependencies still fail.
    """
    dependencies=[];archived=[]
    for name,digest in manifest['dependencies'].items():
        path=Path(name)
        if path.is_file() and sha(path)==digest:
            dependencies.append(path)
            continue
        matches=[row for row in manifest['rows']
                 if row.get('source_path')==name and row.get('source_sha256')==digest
                 and row['variant_sha256']==digest]
        if len(matches)!=1:
            raise ValueError('sealed parent build dependency changed: '+name)
        snapshot=parent/Path(matches[0]['path']).name
        if not snapshot.is_file() or sha(snapshot)!=digest:
            raise ValueError('sealed parent source snapshot changed: '+name)
        dependencies.append(snapshot)
        archived.append(dict(original_path=name,snapshot_path=str(snapshot),sha256=digest))
    return dependencies,archived


def prepare_pristine_objects(parent, publication, audit_path, object_names, out):
    """Relink audited pristine objects without rebuilding the reference.

    This diagnostic isolates differences in untouched, Registry-generated
    objects as well as instrumentation. It never changes the frozen build.
    """
    from run_passivity_variant import authenticate_variant
    parent,out=parent.resolve(),out.resolve()
    if out.exists():
        raise ValueError('diagnostic variant output must be fresh')
    parent_seal=parent/'artifacts.json'
    parent_binding=authenticate_variant(parent/'atmosphere_model',parent_seal)
    parent_manifest=json.loads((parent/'variant.json').read_text())
    inherited_dependencies,archived_dependencies=parent_dependencies(parent,parent_manifest)
    published=json.loads(publication.read_text())
    audit=json.loads(audit_path.read_text())
    if sha(audit_path)!=published['audit_sha256'] or not audit['all_ok'] or audit['recipe']!='gnu-strict':
        raise ValueError('pristine objects require their complete strict native audit')
    native={r['name']:r for r in published['rows']}
    rows=[]
    for name in object_names:
        if Path(name).name!=name or not name.endswith('.o') or name not in native:
            raise ValueError('pristine object is absent from the publication')
        source=publication.resolve().parent/name
        matches=[r for r in audit['objects'] if Path(r['object']).name==name]
        if len(matches)!=1 or sha(source)!=native[name]['sha256'] or sha(source)!=matches[0]['sha256']:
            raise ValueError('pristine object differs from its strict native audit')
        member=subprocess.check_output(['ar','t',str(parent/'libdycore.a')],text=True).splitlines()
        if name not in member:
            raise ValueError('pristine object is absent from the parent model library')
        rows.append(dict(path=matches[0]['object'],source_path=str(source),
                         source_sha256=sha(source),variant_sha256=sha(source),
                         native_library='libdycore.a',pristine_compiled_object=True))
    if len(set(object_names))!=len(object_names) or not rows:
        raise ValueError('pristine object list must be nonempty and unique')
    link=shlex.split((parent/'Makefile').read_text().splitlines()[-1].strip())
    if '-o' not in link or link[link.index('-o')+1]!=str(parent/'atmosphere_model'):
        raise ValueError('sealed parent model link recipe is missing')
    link[link.index('-o')+1]=str(out/'atmosphere_model')
    link.insert(1,'-L'+str(out))
    out.mkdir(parents=True)
    make=['.PHONY: all','all: atmosphere_model']
    for row in rows:
        name=Path(row['path']).name
        make += [name+':','\tcp '+shlex.quote(row['source_path'])+' '+shlex.quote(str(out/name))]
    make += ['libdycore.a: '+' '.join(object_names),
             '\tcp '+shlex.quote(str(parent/'libdycore.a'))+' '+shlex.quote(str(out/'libdycore.a')),
             '\tar r '+shlex.quote(str(out/'libdycore.a'))+' '+ ' '.join(shlex.quote(str(out/name)) for name in object_names),
             '\tranlib '+shlex.quote(str(out/'libdycore.a')),
             'atmosphere_model: libdycore.a','\t'+shlex.join(link)]
    (out/'Makefile').write_text('\n'.join(make)+'\n')
    dependencies=[publication.resolve(),audit_path.resolve(),parent/'variant.json',parent_seal,parent/'Makefile']
    dependencies += [parent/name for name in json.loads(parent_seal.read_text())['artifacts']]
    dependencies += [Path(row['source_path']) for row in rows]
    dependencies += inherited_dependencies
    manifest=dict(schema='mpas-compiled-hook-passivity-variant-v1',
                  prepare_tool_sha256=sha(Path(__file__)),rows=rows,libraries=['libdycore.a'],
                  inventory_sha256=parent_manifest['inventory_sha256'],audit_sha256=parent_manifest['audit_sha256'],
                  parent_variant_seal_sha256=parent_binding,
                  archived_parent_dependencies=archived_dependencies,
                  pristine_object_publication_sha256=sha(publication),pristine_object_audit_sha256=sha(audit_path),
                  pristine_binary_sha256=published['native_binary_sha256'],
                  dependencies={str(p):sha(p) for p in dependencies},
                  makefile_sha256=sha(out/'Makefile'),forecast_proven=False,captured_hidden_state_complete=False)
    (out/'variant.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def prepare(build, source, audit_path, link_log, modules, out,
            transport_source=None, transport_previous_sha256=None, native_visibility=False,
            source_paths=(), parent_variant=None, whole_native_source=False, pool_source=None,
            pool_previous_sha256=None):
    build, source, out = build.resolve(), source.resolve(), out.resolve()
    if out.exists():
        raise ValueError('diagnostic variant output must be fresh')
    inventory_path = build / 'oracle-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit['all_ok'] or audit['recipe'] != 'gnu-strict':
        raise ValueError('variant requires a complete strict native object audit')
    for row in audit['objects']:
        if sha(build / row['object']) != row['object_sha256']:
            raise ValueError('frozen audited object changed: ' + row['object'])
    selected = [m for m in inventory['module_state'] if m['name'] in modules]
    if {m['name'] for m in selected} != set(modules):
        raise ValueError('requested module is absent from the frozen inventory')
    if any(not m['path'].startswith('src/core_atmosphere/') for m in selected):
        raise ValueError('diagnostic replacement is restricted to the dycore library')
    extra_names = {}
    for relative in source_paths:
        member = Path(relative)
        if member.is_absolute() or '..' in member.parts or relative not in inventory['generated_sha256']:
            raise ValueError('source must be a frozen generated member: ' + relative)
        if not relative.startswith(('src/core_atmosphere/', 'src/framework/', 'src/external/esmf_time_f90/')):
            raise ValueError('source is outside supported native model libraries')
        extra_names[relative] = {r.module for r in routines(source / relative, source)}
    library_sources = {p.name: p.resolve() for p in (build / 'src').glob('*.a')}
    library_sources.update({p.name: p.resolve() for p in (build / 'src/external/esmf_time_f90').glob('*.a')})
    original_members = {name: set(subprocess.check_output(['ar', 't', str(path)],text=True).splitlines())
                        for name,path in library_sources.items()}
    parent_binding = None
    if parent_variant:
        from run_passivity_variant import authenticate_variant
        parent_variant = parent_variant.resolve()
        parent_seal = parent_variant / 'artifacts.json'
        parent_binding = authenticate_variant(parent_variant / 'atmosphere_model', parent_seal)
        parent_manifest = json.loads((parent_variant / 'variant.json').read_text())
        if parent_manifest['inventory_sha256'] != sha(inventory_path) or parent_manifest['audit_sha256'] != sha(audit_path):
            raise ValueError('parent variant belongs to a different frozen native build')
        parent_artifacts = json.loads(parent_seal.read_text())['artifacts']
        for name in library_sources:
            if name in parent_artifacts:
                library_sources[name] = parent_variant / name
    rows, commands = [], []
    paths = {m['path'] for m in selected} | set(extra_names)
    transport_path = 'src/core_atmosphere/physics/mpas_oracle_dump.F'
    owned_sources = {}
    if transport_source:
        if not transport_previous_sha256:
            raise ValueError('transport diagnostic requires the frozen transport source hash')
        paths.add(transport_path)
        owned_sources[transport_path] = transport_source.resolve()
    if pool_source:
        if not pool_previous_sha256:
            raise ValueError('pool diagnostic requires the frozen pool source hash')
        pool_path='src/core_atmosphere/physics/mpas_oracle_pool.F'
        paths.add(pool_path)
        owned_sources[pool_path]=pool_source.resolve()
    for relative in paths - owned_sources.keys():
        if sha(source / relative) != inventory['source_sha256'].get(relative):
            raise ValueError('original native source differs from the frozen pin set: ' + relative)
    out.mkdir(parents=True)
    for relative in sorted(paths):
        names = {m['name'] for m in selected if m['path'] == relative} | extra_names.get(relative, set())
        is_transport = relative in owned_sources
        original = owned_sources[relative] if is_transport else source / relative
        objects = [r for r in audit['objects'] if Path(r['object']).stem == original.stem
                   and not r.get('identical_copy_of')]
        if len(objects) != 1:
            raise ValueError('native source has ambiguous compiled object: ' + relative)
        row = objects[0]
        native_object = Path(row['object']).name
        native_library = ('libframework.a' if relative.startswith('src/framework/') else
                          'libesmf_time.a' if relative.startswith('src/external/esmf_time_f90/') else 'libdycore.a')
        if native_object not in original_members[native_library]:
            raise ValueError('audited object is absent from its native library: ' + relative)
        args = shlex.split(row['command'])
        origin_arg = args[args.index('-c') + 1]
        compiled = (build / row['compiled_object']).parent / origin_arg
        expected = (transport_previous_sha256 if relative == transport_path and transport_source else
                    pool_previous_sha256 if relative in owned_sources and pool_source else
                    inventory['generated_sha256'].get(relative))
        if sha(compiled) != expected:
            raise ValueError('compiled source differs from the frozen inventory: ' + relative)
        # ESMF also contains aggregate modules consisting only of USEs and
        # declarations. There is no native executable body to restore, but
        # active callers still need the generated state serializer exports.
        # Only explicit whole-source restoration removes those exports.
        text, bodies = (original.read_text(), []) if is_transport or whole_native_source else restore_bodies(original, compiled, names)
        restored_visibility = {}
        if native_visibility and not is_transport and not whole_native_source:
            text, restored_visibility = restore_private_procedures(original, text, names - {''})
        target = out / original.name
        target.write_text(text)
        args[args.index('-c') + 1] = str(target)
        if '-o' in args:
            args[args.index('-o') + 1] = str(out / Path(row['object']).name)
        else:
            args += ['-o', str(out / Path(row['object']).name)]
        args += ['-J', str(out), '-I', str(out)]
        # All relative include paths retain their original object-directory
        # meaning. Build runs elsewhere and never overwrites native .mod files.
        cwd = (build / row['compiled_object']).parent
        args = [str((cwd / a).resolve()) if i and args[i-1] == '-I' and not Path(a).is_absolute()
                else '-I' + str((cwd / a[2:]).resolve()) if a.startswith('-I') and a[2:] and not Path(a[2:]).is_absolute()
                else a for i, a in enumerate(args)]
        commands.append((out / Path(row['object']).name, args))
        rows.append(dict(path=relative, native_library=native_library, source_sha256=sha(original),
                         source_path=str(original), transport=is_transport,
                         whole_native_source=whole_native_source and not is_transport,
                         restored_private_procedures=restored_visibility,
                         previous_generated_sha256=sha(compiled), variant_sha256=sha(target),
                         modules=sorted(names), native_bodies=bodies,
                         original_command=row['command'], variant_command=args))
    links = [shlex.split(line) for line in link_log.read_text().splitlines()
             if line.startswith('mpif90 ') and '-o atmosphere_model ' in line]
    if not links:
        raise ValueError('actual native model link command missing')
    link = links[-1]
    link[link.index('-o')+1] = str(out / 'atmosphere_model')
    expanded = []
    for arg in link:
        if arg == 'driver/*.o':
            expanded.extend(str(p) for p in sorted((build / 'src/driver').glob('*.o')))
        elif arg == '-L.':
            expanded += ['-L' + str(out), '-L' + str(build / 'src')]
        elif arg.startswith(('-L', '-I')) and arg[2:] and not Path(arg[2:]).is_absolute():
            expanded.append(arg[:2] + str((build / 'src' / arg[2:]).resolve()))
        else:
            expanded.append(arg)
    if not any('/driver/' in a and a.endswith('.o') for a in expanded):
        raise ValueError('native model driver object list is empty')
    make = ['.PHONY: all', 'all: atmosphere_model']
    # The whole selected module set is compiled in its supplied dependency
    # order; modules in a common source are a single compilation unit.
    ordered = sorted(commands, key=lambda c: min(
        [-1] if c[0].stem == 'mpas_oracle_dump' else
        [modules.index(m) if m in modules else len(modules)
         for r in rows if Path(r['path']).stem == c[0].stem for m in r['modules']],
        default=len(modules)))
    previous = None
    for obj, args in ordered:
        make += [obj.name + ':' + (' ' + previous.name if previous else ''),
                 '\t' + shlex.join(args)]
        previous = obj
    libraries = {row['native_library'] for row in rows}
    if parent_variant:
        libraries.update(name for name in library_sources if name in parent_artifacts)
    for name in sorted(libraries):
        lib=out/name
        objects=[p for p,_ in commands if any(Path(row['path']).stem==p.stem and row['native_library']==name for row in rows)]
        make += [name + ': ' + ' '.join(p.name for p in objects),
                 '\tcp ' + shlex.quote(str(library_sources[name])) + ' ' + shlex.quote(str(lib))]
        if objects:
            make += ['\tar r ' + shlex.quote(str(lib)) + ' ' + ' '.join(shlex.quote(str(p)) for p in objects),
                     '\tranlib ' + shlex.quote(str(lib))]
    # ESMF has an explicit -L path in the recorded link, so put the owned
    # variants first for every replaced native library, not only libdycore.
    expanded.insert(1,'-L'+str(out))
    make += ['atmosphere_model: ' + ' '.join(sorted(libraries)), '\t' + shlex.join(expanded)]
    (out / 'Makefile').write_text('\n'.join(make) + '\n')
    manifest = dict(schema='mpas-compiled-hook-passivity-variant-v1',
                    prepare_tool_sha256=sha(Path(__file__)),
                    inventory_sha256=sha(inventory_path), audit_sha256=sha(audit_path),
                    original_model_sha256=sha(build / 'atmosphere_model'),
                    source_paths=list(source_paths), parent_variant_seal_sha256=parent_binding,
                    whole_native_source=whole_native_source,
                    libraries=sorted(libraries),
                    modules=modules, rows=rows, captured_hidden_state_complete=False,
                    forecast_proven=False, numerical_routine_bodies='pristine-native-bytes')
    dependencies = [inventory_path, audit_path.resolve(), link_log.resolve()]
    dependencies += list((build / 'src').glob('*.a'))
    dependencies += list((build / 'src/external/esmf_time_f90').glob('*.a'))
    dependencies += list((build / 'src/external/SMIOL').glob('*.a'))
    for argument in expanded:
        if argument.startswith('-L') and len(argument) > 2:
            directory = Path(argument[2:])
            if directory != out:
                dependencies += list(directory.glob('libpnetcdf.a'))
    dependencies += list((build / 'src/driver').glob('*.o'))
    dependencies += [Path(row['source_path']) for row in rows]
    if parent_variant:
        dependencies += [parent_variant / name for name in ('variant.json', 'artifacts.json', 'libdycore.a', 'atmosphere_model')]
    dependencies += [library_sources[name] for name in libraries]
    manifest['dependencies'] = {str(p.resolve()): sha(p) for p in dependencies}
    manifest['makefile_sha256'] = sha(out / 'Makefile')
    (out / 'variant.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path)
    for name in ('build', 'source', 'audit', 'link-log', 'out'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--module', action='append')
    parser.add_argument('--transport-source', type=Path)
    parser.add_argument('--pool-source', type=Path)
    parser.add_argument('--pool-previous-sha256')
    parser.add_argument('--transport-previous-sha256')
    parser.add_argument('--native-visibility', action='store_true')
    parser.add_argument('--source-file', action='append', default=[])
    parser.add_argument('--parent-variant', type=Path)
    parser.add_argument('--whole-native-source', action='store_true',
                        help='replace complete source including module headers and serializers; diagnostic only')
    parser.add_argument('--pristine-object-publication',type=Path)
    parser.add_argument('--pristine-object-audit',type=Path)
    parser.add_argument('--pristine-object',action='append')
    args = parser.parse_args()
    if args.seal:
        seal_variant(args.seal)
        raise SystemExit(0)
    if args.pristine_object:
        if any(p is None for p in (args.parent_variant,args.pristine_object_publication,args.pristine_object_audit,args.out)):
            parser.error('pristine object replacement requires parent-variant, publication, audit and out')
        prepare_pristine_objects(args.parent_variant,args.pristine_object_publication,args.pristine_object_audit,
                                 args.pristine_object,args.out)
        raise SystemExit(0)
    if any(getattr(args, name) is None for name in ('build', 'source', 'audit', 'link_log', 'out')) or not (args.module or args.source_file or args.transport_source or args.pool_source):
        parser.error('preparation requires build, source, audit, link-log, out and module or transport-source')
    prepare(args.build, args.source, args.audit, args.link_log, args.module or [], args.out,
            args.transport_source, args.transport_previous_sha256, args.native_visibility,
            args.source_file, args.parent_variant, args.whole_native_source,args.pool_source,
            args.pool_previous_sha256)
