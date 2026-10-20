"""Replay framework globals and represented calendars from native physics-init.

Native sources: mpas_io_units.F:29, mpas_pool_routines.F:108,
mpas_timekeeping.F:22-23 and ESMF_CalendarMod.F90:56-63,
122-124, mpas_timekeeping_types.inc:19-38 and ESMF_ClockMod.F90:59-82.
External domain-root aliases and unobserved local SAVE remain unproven.
"""
from hexcore import site_runner
import hashlib
import json
from pathlib import Path
import subprocess
from replay.prepare_native import prepare


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(build, log, dump, out, environment, capture_process_count=None):
    if capture_process_count is not None and (type(capture_process_count) is not int or capture_process_count<1):
        raise ValueError('native capture process count must be positive')
    inventory=json.loads((build/'oracle-inventory.json').read_text())
    state=next(m for m in inventory['module_state'] if m['name']=='mpas_timekeeping')
    names=[v['name'] for v in state['variables']]
    if set(names)!={'thecalendar','yearwidth'} or state['issues']:
        raise ValueError('framework source state changed; whole-module replay needs regeneration')
    selected=[m for m in inventory['module_state']
              if (m['path'].startswith(('src/framework/','src/external/')) and m['variables'])
              or any(v.get('graph_clock') or v.get('container_pointer') and 'esmf_calendar' in v['type']
                     for v in m['variables'])]
    selected.sort(key=lambda m:(any(v.get('graph_clock') for v in m['variables']),m['name']))
    out.mkdir()
    sources=out/'sources';sources.mkdir()
    program=sources/'replay_framework_calendar.F90'
    imports=''.join(f" use {m['name']}, only: oracle_restore_{m['name']},oracle_state_{m['name']}\n"
                    for m in selected if m['name']!='mpas_timekeeping')
    restores=''.join(f"  call oracle_restore_{m['name']}(trim(prefix))\n" for m in selected)
    snapshots=''.join(f"  call oracle_state_{m['name']}('framework_restore','post',event)\n" for m in selected)
    program.write_text('''program replay_framework_calendar
 use mpas_timekeeping, only: TheCalendar,yearWidth,oracle_restore_mpas_timekeeping,oracle_state_mpas_timekeeping
 use mpas_oracle_dump, only: oracle_begin,oracle_set_context
'''+imports+''' implicit none
 character(len=4096) :: prefix
 integer :: event,iteration
 call get_command_argument(1,prefix)
 call oracle_set_context([1],0,0)
 do iteration=1,2
  TheCalendar=-huge(1);yearWidth=-huge(1)
'''+restores+'''
  call oracle_begin('framework_restore',event)
'''+snapshots+'''
 enddo
end program
''')
    native=out/'native'
    metadata=prepare(build,log,native,sources)
    words_source=Path(__file__).resolve().parents[1]/'oracle_words.rs'
    with (native/'Makefile').open('a') as stream:
        stream.write(f'all: oracle_words\noracle_words: {words_source}\n\trustc --edition=2021 -O $< -o $@\n')
    words=native/'oracle_words'
    with (out/'build.log').open('w') as stream:
        subprocess.run(site_runner.build_command(['make','-j4','-C',str(native)]),env=environment,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    suffix='.init.state.mpas_timekeeping.'
    prefixes=sorted({p.name.split(suffix)[0] for p in dump.glob('state__physics_init*'+suffix+'*.bin')})
    if not prefixes:raise ValueError('physics_init emitted no framework globals')
    rows=[]
    stems=tuple('state.'+m['name']+'.' for m in selected)
    for number,prefix in enumerate(prefixes):
        work=out/str(number);work.mkdir()
        inputs=work/'inputs';inputs.mkdir()
        results=work/'results';results.mkdir()
        originals={p.name.split('.init.',1)[1][:-4]:p for p in dump.glob(prefix+'.init.*.bin')
                   if p.name.split('.init.',1)[1].startswith(stems)}
        for name,source in originals.items():
            (inputs/('capture.pre.'+name+'.bin')).symlink_to(source)
        run_environment=dict(environment,MPAS_ORACLE_DIR=str(results),MPAS_ORACLE_STEPS='0',
                      MPAS_ORACLE_KEYS='framework_restore')
        with (work/'run.log').open('w') as stream:
            subprocess.run([str(native/'replay_framework_calendar'),str(inputs/'capture')],
                           env=run_environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        comparisons={}
        emitted={p.name.split('.post.',1)[1][:-4] for p in results.glob('*.post.*.bin')}
        if emitted!=set(originals):raise ValueError('framework restored state record membership changed')
        for name,original in originals.items():
            records=sorted(results.glob('*.post.'+name+'.bin'))
            if len(records)!=2:raise ValueError('both mutation/restore passes must emit every global')
            comparisons[name]=[]
            for record in records:
                result=subprocess.run([str(words),'compare',str(original),str(record)],
                                      capture_output=True,text=True,check=True,env=environment)
                value=json.loads(result.stdout)
                if not value['bitwise_equal']:raise ValueError('framework global restore differs')
                comparisons[name].append(dict(record=record.name,sha256=sha(record),comparison=value))
        rows.append(dict(native_init_prefix=prefix,
                         native_init_sha256={n:sha(p) for n,p in originals.items()},
                         arrays=comparisons))
    receipt=dict(schema='f1-native-framework-calendar-whole-state-restore-v2',
                 replay_process_count=1,capture_process_count=capture_process_count,
                 native_source='src/framework/mpas_timekeeping.F:22-23',
                 native_source_sha256=inventory['source_sha256'][state['path']],
                 native_inventory_sha256=sha(build/'oracle-inventory.json'),
                 native_binary_sha256=sha(build/'atmosphere_model'),
                 replay_binary_sha256=sha(native/'replay_framework_calendar'),
                 stream_comparator_source_sha256=sha(words_source),
                 stream_comparator_binary_sha256=sha(words),
                 replay_build=metadata,mutable_globals=names,captures=rows,
                 modules=[dict(name=m['name'],source=m['path'],roots=m['roots'],
                               source_sha256=inventory['source_sha256'][m['path']]) for m in selected],
                 all_bitwise_equal=True,global_hidden_state_complete=False,
                 scope='Supported mutable MPAS/ESMF framework globals, internal native clock graph and represented calendar aliases, actual native physics_init; unsupported framework roots, external domain-root aliases and unobserved local SAVE remain unproven')
    target=out.parent/'framework-state-restore.json'
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    return target
