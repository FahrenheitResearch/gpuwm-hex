#!/usr/bin/env python3
"""Observe complete nested modules while replaying authenticated real inputs.

The parent and child are whole native source closures. Every ordinary/observed
parent output is compared to the original pristine native capture. These are
standalone calls, never new forecasts or new pristine forecast capture authority.
Python handles provenance and processes; Rust/Fortran handle scientific packets.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from replay_external_calls import legacy_captures,compare_outputs,sha,write


def execute(a):
    parent=json.loads((a.parent_build/'replay-spec.json').read_text())
    child=json.loads((a.child_build/'replay-spec.json').read_text())
    observer_spec=json.loads(a.observer_spec.read_text())
    if observer_spec['callee']!=child['callee']:
        raise ValueError('nested observer and replay target different native routines')
    for path,digest in observer_spec['source_sha256'].items():
        if child['pristine_source_sha256'].get(path)!=digest:
            raise ValueError('nested observer native source differs from complete replay')
    observer_sources={}
    for path,digest in observer_spec['generated_sha256'].items():
        source=a.observer_spec.parent/path
        if sha(source)!=digest:raise ValueError('nested observer generated ABI source changed')
        observer_sources[str(source.resolve())]=digest
    for path,digest in child['pristine_source_sha256'].items():
        if parent['pristine_source_sha256'].get(path)!=digest:
            raise ValueError('parent and child do not share the pristine native source closure')
    for build,spec in [(a.parent_build,parent),(a.child_build,child)]:
        for path,digest in spec['generated_source_sha256'].items():
            if sha(build/'gnu-strict'/path)!=digest:raise ValueError('standalone numerical closure changed')
        if sha(build/'gnu-strict/replay.F90')!=spec['replay_source_sha256']:
            raise ValueError('standalone replay program changed')
    a.out.mkdir(parents=True,exist_ok=False)
    dependencies={str(p.resolve()):sha(p) for p in [a.index,a.parent_build/'replay-spec.json',
        a.child_build/'replay-spec.json',a.observer_spec,a.observer,a.words,a.parent_build/'gnu-strict/replay']}
    dependencies.update(observer_sources)
    # Keep all original authority files as immutable dependencies, not merely
    # an unauthenticated assertion of a prior successful parent replay.
    dependencies.update(json.loads(a.index.read_text())['dependencies'])
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPAS_ORACLE_SCHEMES='*')
    rows=[]
    for number,(deck,mode,dump,call,evidence) in enumerate(legacy_captures(a.index,parent,a.words,a.out/'inputs')):
        root=a.out/f'parent{number:03}';root.mkdir()
        files=sorted(dump.glob('*.bin'));before={p.name:sha(p) for p in files}
        comparisons={};complete=True;changed=[]
        captures=root/'captures'
        for observed in (False,True):
            label='observed' if observed else 'ordinary';output=root/label;output.mkdir()
            env['MPAS_ORACLE_DIR']=str(output.resolve())
            command=[str((a.parent_build/'gnu-strict/replay').resolve()),
                     str((dump/call['prefix']).resolve()),str((output/'replay').resolve())]
            if observed:command=[str(a.observer.resolve()),command[0],str(captures.resolve()),'--',*command[1:]]
            run=subprocess.run(command,env=env,capture_output=True,text=True)
            (root/(label+'.log')).write_text(run.stdout+run.stderr)
            arrays,output_complete=compare_outputs(parent['site']['arguments'],dump,call['prefix'],output,a.words)
            same=run.returncode==0 and output_complete
            different=sum(c.get('different_words') or 0 for c in arrays.values())
            comparisons[label]=dict(exit=run.returncode,comparison_complete=same,differing_words=different,
                                    compared_words=sum(c.get('words',0) for c in arrays.values()),arrays=arrays)
            complete=complete and same and different==0
            changed.extend(p.name for p in files if sha(p)!=before[p.name])
            for p in output.glob('*.bin'):p.unlink()
        write(root/'parent-comparison.json',comparisons)
        if not complete or changed:raise ValueError('nested observer changed a complete real parent replay: '+str(root))
        calls=[json.loads(l) for l in (captures/'rank0-calls.jsonl').read_text().splitlines()]
        entries=[c for c in calls if c['phase']=='pre'];returns=[c for c in calls if c['phase']=='post']
        if not entries or [(c['call'],c['step']) for c in entries]!=[(c['call'],c['step']) for c in returns]:
            raise ValueError('nested native entries/returns incomplete')
        # Source-selected internal inputs restored in the parent must be the
        # exact bytes that the child observer records, with no invented flags.
        hidden={};probes=[p for p in parent['hidden_local_restore_points'] if p['name']==child['callee']['name']]
        if len(probes)>1 or (probes and len(probes[0]['labels'])!=len(entries)):
            raise ValueError('nested real hidden-entry witness is not complete')
        for probe in probes:
            argument=observer_spec['arguments'][probe['argument']]['name']
            for entry,(_,name) in zip(entries,probe['labels']):
                reference=dump/(call['prefix']+'.pre.'+name+'.bin')
                actual=captures/(entry['prefix']+'.pre.'+argument+'.bin')
                result=json.loads(subprocess.check_output([str(a.words),'compare',str(reference),str(actual)],text=True))
                if not result['bitwise_equal']:raise ValueError('nested restored native entry words changed')
                hidden[name]=result
        row=dict(parent_row=number,deck=deck,mode=mode,parent_call=call['prefix'],
            parent_capture_authority=evidence,parent_input_sha256=before,
            parent_comparison_sha256=sha(root/'parent-comparison.json'),child_calls=len(entries),
            dump=str(captures.resolve()),capture_sha256={str(p.relative_to(captures)):sha(p) for p in captures.rglob('*') if p.is_file()},
            hidden_entry_comparisons=hidden,selected_hidden_restore_points=len(probes),
            comparison_complete=complete,changed_inputs=changed,
            differing_words=sum(c['differing_words'] for c in comparisons.values()),
            compared_words=sum(c['compared_words'] for c in comparisons.values()),
            observer_standalone_passive=True,observer_forecast_passive=False,forecast_proven=False)
        write(root/'capture-manifest.json',row)
        rows.append(dict(manifest=str((root/'capture-manifest.json').resolve()),sha256=sha(root/'capture-manifest.json'),
                         child_calls=row['child_calls'],comparison_complete=complete,
                         differing_words=row['differing_words'],compared_words=row['compared_words']))
        dependencies[str((root/'parent-comparison.json').resolve())]=sha(root/'parent-comparison.json')
    result=dict(schema='native-nested-standalone-capture-v1',rows=rows,dependencies=dependencies,
        child_pristine_source_sha256=child['pristine_source_sha256'],process_count=1,
        child_calls=sum(r['child_calls'] for r in rows),
        comparison_complete=bool(rows) and all(r['comparison_complete'] for r in rows),
        differing_words=sum(r['differing_words'] for r in rows),compared_words=sum(r['compared_words'] for r in rows),
        forecast_proven=False,observer_forecast_passive=False,hidden_state_complete=False,
        runner_sha256=sha(Path(__file__)),
        scope='Complete nested native modules from authenticated real forecast inputs; parent standalone passivity only.')
    write(a.out/'summary.json',result)
    return int(not result['comparison_complete'] or result['differing_words']!=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-build','child-build','observer','observer-spec','words','index','out'):
        p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(execute(p.parse_args()))
