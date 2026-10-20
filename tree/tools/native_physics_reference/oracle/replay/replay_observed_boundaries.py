#!/usr/bin/env python3
"""Replay complete native routines against authenticated external pool captures.

Native source is compiled whole with a declaration-only private export. The
existing audited framework implements every pool lookup; captured PRE records
initialize actual module storage and queried fields. Rust owns packet metadata
and exact comparison. This does not assert a complete pool/alias/callback graph.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fortran_source import routines
from pin_sets import provenance
try:
    from .reproduce_archived_replay import authority
except ImportError:
    from reproduce_archived_replay import authority

TABLE = Path(__file__).with_name('observed_boundaries.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def flat(name):
    if Path(name).name != name or name in ('.', '..') or '\\' in name:
        raise ValueError('observed boundary member escapes bank')
    return name


def authenticate(bank, source, spec):
    for name, expected in spec['metadata'].items():
        if sha(bank / flat(name)) != expected:
            raise ValueError('observed native metadata differs: ' + name)
    native = json.loads((bank / 'manifest.json').read_text())
    producer = json.loads((bank / 'producer.json').read_text())
    passive = json.loads((bank / 'passivity.json').read_text())
    if (native['schema'] != spec['native_bank_schema'] or
            native['model_commit'] != spec['model_commit'] or
            native['input_origin'] != 'native-tap' or
            native['process_count'] != 1 or producer['process_count'] != 1 or
            not native['observer_passivity_proven'] or
            not passive['bitwise_equal'] or passive['different_words'] != 0 or
            producer['host'] != passive['host'] or
            producer['native_sha256'] != native['native_binary_sha256'] or
            producer['observer_sha256'] != native['observer_sha256'] or
            producer['input_request_sha256'] != native['input_request_sha256'] or
            any(row['exit'] != 0 for row in producer['runs'].values())):
        raise ValueError('observed native producer/passivity identity differs')
    for name, key in (('producer.json', 'producer_attempt_sha256'),
                      ('passivity.json', 'passivity_receipt_sha256'),
                      ('input-request.json', 'input_request_sha256')):
        if spec['metadata'][name] != native[key]:
            raise ValueError('observed native provenance binding differs')
    for relative, expected in native['source_sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(source / path) != expected:
            raise ValueError('observed native source differs: ' + relative)
    if sha(source / spec['source']) != spec['source_sha256']:
        raise ValueError('observed routine source differs')
    found = set()
    records = {}
    for capture, row in native['captures'].items():
        flat(capture)
        key = capture.split('.s', 1)[0]
        if key not in spec['routines']:
            raise ValueError('observed boundary is outside replay table')
        found.add(key)
        members = row['files']
        if {p.name for p in (bank / 'dump').glob(capture + '.*.bin')} != set(members):
            raise ValueError('observed complete packet inventory differs')
        pre = {}; post = {}
        for name, expected in members.items():
            if sha(bank / 'dump' / flat(name)) != expected:
                raise ValueError('observed native packet changed: ' + name)
            if name.startswith(capture + '.pre.'):
                pre[name[len(capture + '.pre.'):-4]] = name
            elif name.startswith(capture + '.post.'):
                post[name[len(capture + '.post.'):-4]] = name
            else:
                raise ValueError('observed native phase differs')
        if not pre or set(pre) != set(post):
            raise ValueError('observed PRE/POST record inventory differs')
        records[capture] = dict(key=key, pre=pre, post=post)
    if found != set(spec['routines']):
        raise ValueError('complete observed routine pair required')
    return native, producer, passive, records


def export_source(source, spec):
    """Keep every native routine body, rather than reconstructing its loops."""
    path = source / spec['source']
    raw = path.read_text()
    parsed = {r.name: r for r in routines(path, source)}
    wanted = [row['routine'] for row in spec['routines'].values()]
    for row in spec['routines'].values():
        routine = parsed[row['routine']]
        if routine.module != spec['module'] or routine.args != row['arguments']:
            raise ValueError('native boundary interface differs')
    matches = list(re.finditer(r'^\s*contains\s*$', raw, re.M | re.I))
    if len(matches) != 1:
        raise ValueError('native module export anchor is ambiguous')
    at = matches[0].start()
    export = '\n public :: ' + ','.join(wanted) + '\n'
    changed = raw[:at] + export + raw[at:]
    if changed[:at] + changed[at + len(export):] != raw:
        raise ValueError('native body changed during private export')
    return changed, [dict(name=name, line=parsed[name].start + 1,
                         end_line=parsed[name].end) for name in wanted]


def program(spec, row, headers):
    """Allocate from PRE headers, invoke original native bodies, emit all POST."""
    declarations = []
    loads = []
    emits = []
    pool_names = set()
    arguments = {}
    imports = []
    types = {1: 'real(kind=RKIND)', 2: 'real(kind=8)', 3: 'integer',
             4: 'logical', 6: 'integer(kind=8)'}
    for ordinal, (name, header) in enumerate(sorted(headers.items())):
        code = header['code']; rank = len(header['shape'])
        symbol = 'bank_value_' + str(ordinal)
        category = name.split('.')[0]
        local = name.split('.')[-1]
        if code == 5:
            if rank:
                raise ValueError('observed character array unsupported')
            typ = 'character(len=' + str(header['character_length']) + ')'
        elif code in types:
            typ = types[code]
        else:
            raise ValueError('observed primitive type unsupported')
        if category == 'vars':
            imports.append(symbol + ' => ' + local)
            value = symbol
            if rank:
                loads.append(' if(allocated(' + value + ')) deallocate(' + value + ')')
        elif category == 'argument':
            declarations.append(typ + ' :: ' + symbol)
            arguments[local] = symbol
            value = symbol
        elif '.configs.' in name:
            declarations.append(typ + ' :: ' + symbol)
            loads.append(' call mpas_pool_add_config(configs,\'' + local + '\',' + symbol + ')')
            value = symbol
        elif len(name.split('.')) == 3 and name.endswith('.t1'):
            pool, field, _ = name.split('.')
            pool_names.add(pool)
            field_kind = {1: 'Real', 3: 'Integer'}.get(code)
            if not field_kind or rank not in range(1, 6):
                raise ValueError('observed field type/rank unsupported')
            declarations.append(' type(field' + str(rank) + 'D' + field_kind + '),pointer :: ' + symbol)
            loads.extend([' allocate(' + symbol + ')',
                          ' ' + symbol + '%fieldName=\'' + field + '\'',
                          ' ' + symbol + '%isActive=.true.',
                          ' ' + symbol + '%dimNames=\'\'',
                          ' ' + symbol + '%dimSizes=[' + ','.join(map(str, header['shape'])) + ']'])
            value = symbol + '%array'
        else:
            raise ValueError('observed pool record shape unsupported: ' + name)
        path = "trim(prefix)//'.pre." + name + ".bin'"
        if rank:
            bounds = [str(lo) + ':' + str(lo + extent - 1)
                      for lo, extent in zip(header['lower'], header['shape'])]
            loads.append(' allocate(' + value + '(' + ','.join(bounds) + '))')
        # Config values must be read before the pool copies them.
        read = ' call oracle_read(' + path + ',' + value + ')'
        if '.configs.' in name:
            loads.insert(len(loads) - 1, read)
        else:
            loads.append(read)
        if category not in ('vars', 'argument') and '.configs.' not in name:
            loads.append(' call mpas_pool_add_field(' + pool + ',\'' + field + '\',' + symbol + ')')
        lower = ',lower=lbound(' + value + ')' if rank else ''
        # Restore only captured metadata/IDs, never any expected POST word.
        emits.append(' call oracle_restore_context(' + path + ',event)')
        emits.append(" call oracle_write('" + row['key'] + "','post','" + name + "'," + value +
                     ',event' + lower + ',cell_axis=' + str(header['cell_axis']) +
                     ',level_axis=' + str(header['level_axis']) + ')')
    routine = spec['routines'][row['key']]
    pools = {'configs'} | (set(routine['arguments']) - set(arguments)) | pool_names
    declarations.append(' type(mpas_pool_type),pointer :: ' + ','.join(sorted(pools)))
    actuals = [arguments.get(name, name) for name in routine['arguments']]
    return '\n'.join(['! Complete native fork boundary replay; no numerical body rewrite.',
                     'program observed_boundary_replay', ' use mpas_kind_types',
                     ' use mpas_pool_routines', ' use mpas_oracle_dump',
                     ' use ' + spec['module'] + ',only:' + routine['routine'],
                     ' use ' + spec['state_module'] + ',only:' + ','.join(imports),
                     ' implicit none', ' character(len=4096) :: prefix,output',
                     ' integer :: event', *declarations,
                     ' call get_command_argument(1,prefix)', ' call get_command_argument(2,output)',
                     *[' call mpas_pool_create_pool(' + name + ',97)' for name in sorted(pools)],
                     *loads, ' call ' + routine['routine'] + '(' + ','.join(actuals) + ')',
                     ' call oracle_set_output(trim(output))', *emits,
                     'end program observed_boundary_replay', ''])


def prepare(bank, source, selection, index, words, out):
    spec = json.loads(TABLE.read_text())[selection]
    native, producer, passive, records = authenticate(bank, source, spec)
    data = authority(index)
    inventory = json.loads((Path(data['build']) / 'oracle-inventory.json').read_text())
    if (provenance(source, inventory['pin_set'])['model_commit'] != spec['model_commit'] or
            inventory['strict_reference_binary_sha256'] != native['native_binary_sha256']):
        raise ValueError('native framework referee differs from observed bank')
    changed, ranges = export_source(source, spec)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'native-module.F').write_text(changed)
    audit = json.loads(Path(data['audit']).read_text())
    obj = spec['source'].rsplit('.', 1)[0] + '.o'
    matched = [r for r in audit['objects'] if r['object'] == obj]
    if len(matched) != 1 or not matched[0]['flags_ok']:
        raise ValueError('original full native module audit required')
    # Absolute include paths in the actual audited native command bind modules.
    includes = [arg for arg in shlex.split(matched[0]['command']) if arg.startswith('-I/')]
    flags = ['-cpp', '-D_MPI', '-DMPAS_USE_MPI_F08', '-DSINGLE_PRECISION', '-Dmpas',
             '-ffree-line-length-none', '-ffree-form', '-fconvert=big-endian']
    try:
        from .prepare_standalone import RECIPES
    except ImportError:
        from prepare_standalone import RECIPES
    flags += RECIPES[data.get('recipe', 'gnu-strict')][1] + includes
    recorded_link = json.loads(Path(data['link_manifest']).read_text())
    libraries = recorded_link.get('native_libraries', [str(Path(data['build']) / name) for name in
        ('src/libdycore.a', 'src/libops.a', 'src/libframework.a', 'src/external/esmf_time_f90/libesmf_time.a')])
    link = ['-Wl,--start-group', *libraries, '-Wl,--end-group', *recorded_link['external_link_args']]
    manifests = []
    for number, (capture, row) in enumerate(sorted(records.items())):
        headers = {name: json.loads(subprocess.check_output(
            [str(words), 'header', str(bank / 'dump' / member)], text=True))
            for name, member in row['pre'].items()}
        target = out / ('replay_boundary_' + str(number) + '.F90')
        target.write_text(program(spec, row, headers))
        manifests.append(dict(capture=capture, executable='replay_boundary_' + str(number),
                              pre=row['pre'], post=row['post'], headers=headers,
                              generated_source_sha256=sha(target),
                              post_count=len(row['post'])))
    compiler = shlex.quote(data['compiler'])
    all_flags = ' '.join(map(shlex.quote, flags))
    lines = ['all: ' + ' '.join(r['executable'] for r in manifests),
             'native-module.o: native-module.F',
             '\t' + compiler + ' ' + all_flags + ' -c $< -o $@']
    for row in manifests:
        name = row['executable']
        lines += [name + ': ' + name + '.F90 native-module.o',
                  '\t' + compiler + ' ' + all_flags + ' $< native-module.o ' +
                  ' '.join(map(shlex.quote, link)) + ' -o $@']
    (out / 'Makefile').write_text('\n'.join(lines) + '\n')
    manifest = dict(schema='complete-native-observed-boundary-build-v1', selection=selection,
                    recipe=data.get('recipe', 'gnu-strict'), bank=str(bank.resolve()),
                    source=str(source.resolve()), input_index=str(index.resolve()),
                    bank_manifest_sha256=sha(bank / 'manifest.json'), table_sha256=sha(TABLE),
                    source_sha256=spec['source_sha256'], native_routines=ranges,
                    native_module_generated_sha256=sha(out / 'native-module.F'),
                    declaration_only_private_export=True, numerical_body_changed=False,
                    input_index_sha256=sha(index), source_tools_sha256=sha(Path(__file__)), makefile_sha256=sha(out/'Makefile'),
                    scope=spec['scope'], original_bank_completeness={k: native.get(k) for k in
                    ('full_pool_contents_captured', 'hidden_state_complete', 'callback_state_complete')},
                    producer=producer, passivity_words=passive['words'], calls=manifests)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def run(prepared,products,words,out):
 data=json.loads((prepared/'manifest.json').read_text());bank=Path(data['bank'])
 spec=json.loads(TABLE.read_text())[data['selection']]
 native,producer,passive,records=authenticate(bank,Path(data['source']),spec)
 if sha(TABLE)!=data['table_sha256'] or sha(Path(data['input_index']))!=data['input_index_sha256']:
  raise ValueError('observed replay table or build authority changed')
 authority(Path(data['input_index']))
 if data['source_tools_sha256']!=sha(Path(__file__)) or data['makefile_sha256']!=sha(prepared/'Makefile'):
  raise ValueError('observed replay generator or compile command changed')
 if set(records)!={r['capture']for r in data['calls']}:raise ValueError('observed capture list incomplete')
 for row in data['calls']:
  if records[row['capture']]['pre']!=row['pre'] or records[row['capture']]['post']!=row['post']:
   raise ValueError('observed output inventory differs')
 if sha(bank/'manifest.json')!=data['bank_manifest_sha256'] or sha(prepared/'native-module.F')!=data['native_module_generated_sha256']:
  raise ValueError('observed bank or generated native module changed')
 original=json.loads((bank/'manifest.json').read_text())
 out.mkdir(parents=True,exist_ok=False)
 rows=[];normalizations=[];different=0;total=0;arrays=0
 for row in data['calls']:
  folder=out/row['capture'];folder.mkdir();inputs=folder/'inputs';inputs.mkdir();expected=folder/'expected';expected.mkdir();actual=folder/'actual';actual.mkdir()
  members=original['captures'][row['capture']]['files']
  for phase in ('pre','post'):
   for name,member in row[phase].items():
    source=bank/'dump'/member
    if sha(source)!=members[member]:raise ValueError('observed packet changed: '+member)
    target=(inputs if phase=='pre' else expected)/member
    header=json.loads(subprocess.check_output([str(words),'header',str(source)],text=True))
    if header['code']==5:
     note=json.loads(subprocess.check_output([str(words),'normalize-character',str(source),str(target)],text=True))
     normalizations.append(dict(member=member,original_sha256=sha(source),normalized_sha256=sha(target),proof=note))
    else:
     os.link(source,target)
  executable=products/flat(row['executable'])
  if sha(prepared/(row['executable']+'.F90'))!=row['generated_source_sha256']:
   raise ValueError('observed native replay program changed')
  runtime=dict(os.environ)
  for name in list(runtime):
   if name.startswith('MPAS_ORACLE_'):del runtime[name]
  if data['recipe']=='coverage':runtime.update(GCOV_PREFIX=str((out/'profiles').resolve()),GCOV_PREFIX_STRIP='0')
  with (folder/'native.log').open('w') as log:
   process=subprocess.run([str(executable.resolve()),str((inputs/row['capture']).resolve()),str((actual/row['capture']).resolve())],cwd=folder,env=runtime,stdout=log,stderr=subprocess.STDOUT)
  comparisons={}
  expected_names=set(row['post'].values());actual_names={p.name for p in actual.glob('*.bin')}
  if process.returncode!=0 or expected_names!=actual_names:
   result=dict(capture=row['capture'],exit=process.returncode,expected_files=sorted(expected_names),actual_files=sorted(actual_names),comparison_complete=False)
   (folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
   raise RuntimeError('native whole handoff failed: '+str(folder/'native.log'))
  for name,member in row['post'].items():
   compare=json.loads(subprocess.check_output([str(words),'compare',str(expected/member),str(actual/member)],text=True))
   compare.update(original_post_sha256=members[member],expected_sha256=sha(expected/member),actual_sha256=sha(actual/member))
   if compare.get('schema_mismatch') or compare.get('different_words') is None:
    raise ValueError('observed output schema differs: '+name)
   compare['comparison_complete']=True
   total+=compare['words'];different+=compare['different_words'];arrays+=1;comparisons[name]=compare
  result=dict(capture=row['capture'],exit=0,comparison_complete=True,comparisons=comparisons,executable_sha256=sha(executable))
  (folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
 summary=dict(physics_mode='exact',mode_authority='Unchanged pinned native routine bodies; no hex door or defect-fix mode invoked.',schema='complete-native-observed-boundary-replay-v1',recipe=data['recipe'],calls=len(rows),arrays=arrays,compared_words=total,differing_words=different,comparison_complete=True,bitwise_equal=different==0,process_count=1,host=os.uname().nodename,bank_manifest_sha256=data['bank_manifest_sha256'],build_manifest_sha256=sha(prepared/'manifest.json'),words_sha256=sha(words),normalizations=normalizations,scope=data['scope'],native_routines=data['native_routines'],calls_receipts={r['capture']:sha(out/r['capture']/'comparison.json')for r in rows})
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in ('normalizations','calls_receipts')},indent=2))
 return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    build = commands.add_parser('prepare')
    for name in ('bank','source','index','words','out'):
        build.add_argument('--'+name,type=Path,required=True)
    build.add_argument('--selection',required=True)
    execute = commands.add_parser('run')
    for name in ('prepared','products','words','out'):
        execute.add_argument('--'+name,type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command')
    result=prepare(**args) if command=='prepare' else run(**args)
    if command=='prepare':
        print(json.dumps(dict(selection=result['selection'],calls=len(result['calls']),source_ranges=result['native_routines']),indent=2))
    else:
        raise SystemExit(0 if result['bitwise_equal'] else 1)
