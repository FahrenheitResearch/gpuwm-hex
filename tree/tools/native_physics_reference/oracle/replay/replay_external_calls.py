"""Replay every full native call captured by the source-derived external tap.

The Rust comparator consumes packets; Python only authenticates metadata/files
and launches native executables. A nonpassive observer's packets are not admitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def legacy_captures(index_path,spec,words,work):
    """Admit existing tagged external packets with their producer's evidence.

    Python authenticates opaque bytes and schedules Rust normalization only.
    Aliases here are source bindings validated against both captured phases;
    they are explicitly not a claim that the legacy observer logged pointers.
    """
    index=json.loads(index_path.read_text())
    observer=json.loads(Path(index['observer_spec']).read_text())
    if observer['source_sha256']!=spec['pristine_source_sha256'][spec['callee']['path']]:
        raise ValueError('legacy observer source differs from complete native replay')
    if observer['arguments']!=[a['name'] for a in spec['site']['arguments']]:
        raise ValueError('legacy observed native ABI differs from replay')
    for path,digest in index['dependencies'].items():
        if sha(Path(path))!=digest:raise ValueError('legacy capture authority changed: '+path)
    for deck,definition in enumerate(index['decks']):
        root=Path(definition['root']);summary=json.loads((root/'summary.json').read_text())
        parent=json.loads((root.parent/'summary.json').read_text())
        if parent['native_binary_sha256']!=index['native_binary_sha256'] or parent['watch_sha256']!=index['observer_sha256']:
            raise ValueError('legacy native executable or observer identity changed')
        if summary['process_count']!=1 or len(summary['runs'])!=3 or len(summary['comparisons'])!=3:
            raise ValueError('legacy capture requires its measured MPI1 ordinary/two-observed scope')
        if any(r['exit'] or r['changed_inputs'] for r in summary['runs']):
            raise ValueError('legacy native forecast incomplete or changed its ingress')
        if any(not c['complete'] or c['different_words'] or c.get('runtime_input_different_words',0)
               for c in summary['comparisons']):
            raise ValueError('legacy observer was not passive over every forecast word')
        for run in summary['runs'][1:]:
            original=root/run['mode'];recorded=run['wrapper_capture_files']
            if set(recorded)!={p.name for p in original.glob('*.bin')} or any(sha(original/n)!=h for n,h in recorded.items()):
                raise ValueError('legacy packet inventory or words changed')
            calls=[json.loads(l) for l in (original/'rank0-calls.jsonl').read_text().splitlines()]
            expected=summary.get('expected_wrapper_calls',len(calls)//2)
            if len(calls)!=2*expected or [(c['call'],c['step'],c['phase']) for c in calls]!=[
                (i,i+1,phase) for i in range(expected) for phase in ('pre','post')]:
                raise ValueError('legacy wrapper entry/return sequence incomplete')
            for call in calls[::2]:
                if call['hidden_state_complete'] or call['global_cell_ids_known']:
                    raise ValueError('legacy external capture cannot assert complete hidden state or global IDs')
                prefix=call['prefix'];dump=work/f'deck{deck}-{run["mode"]}-call{call["call"]}'
                dump.mkdir(parents=True,exist_ok=False)
                files=sorted(original.glob(prefix+'.*.bin'))
                for p in files:
                    subprocess.run([str(words),'normalize-external',str(p),str(dump/p.name)],check=True,capture_output=True,text=True)
                for probe in spec['hidden_local_restore_points']:
                    for _,name in probe['labels']:
                        if not (dump/(prefix+'.pre.'+name+'.bin')).is_file():
                            raise ValueError('captured native internal entry missing: '+name)
                observed_aliases={name:target for name,target in spec['argument_aliases'].items()
                                  if (dump/(prefix+'.pre.'+name+'.bin')).exists()}
                call=dict(call,argument_aliases=observed_aliases,
                          alias_authority='native driver source binding and exact PRE/POST alias packet words; legacy observer did not log addresses')
                evidence=dict(source_index_sha256=sha(index_path),producer_summary_sha256=sha(root/'summary.json'),
                              original_packet_sha256={p.name:recorded[p.name] for p in files},
                              native_binary_sha256=index['native_binary_sha256'],observer_sha256=index['observer_sha256'],
                              process_count=1,observer_forecast_passive=True)
                yield deck,run['mode'],dump,call,evidence
                shutil.rmtree(dump)


def compare_outputs(arguments,dump,prefix,out,words):
    """Compare all declared outputs, including mandatory optional-presence records."""
    arrays={};complete=True
    for a in arguments:
        if a['intent']=='in':continue
        labels=[a['name']+'.present',a['name']] if a['optional'] else [a['name']]
        for label in labels:
            reference=dump/(prefix+'.post.'+label+'.bin')
            actual=out/('replay.post.'+label+'.bin')
            if not reference.exists() and not actual.exists() and a['optional'] and label==a['name']:
                flag=dump/(prefix+'.post.'+a['name']+'.present.bin')
                presence=subprocess.run([str(words),'present',str(flag)],capture_output=True,text=True)
                if presence.returncode==0 and presence.stdout.strip()=='false':continue
            if not reference.exists() or not actual.exists():
                arrays[label]=dict(status='missing',reference_present=reference.exists(),replay_present=actual.exists());complete=False;continue
            run=subprocess.run([str(words),'compare',str(reference),str(actual)],capture_output=True,text=True)
            if run.returncode:
                arrays[label]=dict(status='incompatible',error=run.stderr);complete=False
            else:
                comparison=json.loads(run.stdout)
                if comparison.get('comparison_complete',True) is False or comparison.get('different_words') is None:
                    complete=False
                arrays[label]=dict(status='compared' if comparison.get('different_words') is not None else 'incompatible',**comparison)
    if not arrays:complete=False
    return arrays,complete


def standalone_captures(authority,spec):
    """Admit nested standalone observations without claiming forecast passivity.

    Their producer authenticates the original forecast inputs and compares both
    ordinary and observed parent calls against every original native output.
    """
    index=json.loads(authority.read_text())
    if index.get('schema')!='native-nested-standalone-capture-v1' or not index.get('comparison_complete'):
        raise ValueError('incomplete standalone parent capture authority')
    if (index.get('differing_words')!=0 or index.get('forecast_proven') is not False
            or index.get('observer_forecast_passive') is not False):
        raise ValueError('standalone parent must be exact and explicitly outside forecast proof')
    if index['child_pristine_source_sha256']!=spec['pristine_source_sha256']:
        raise ValueError('standalone child native source differs from replay')
    for path,digest in index['dependencies'].items():
        if sha(Path(path))!=digest:raise ValueError('standalone capture dependency changed: '+path)
    for definition in index['rows']:
        if 'manifest' in definition:
            manifest=Path(definition['manifest'])
            if sha(manifest)!=definition['sha256']:raise ValueError('nested standalone call manifest changed')
            row=json.loads(manifest.read_text())
        else:row=definition
        if not row['comparison_complete'] or row['differing_words'] or row['changed_inputs']:
            raise ValueError('standalone parent observation failed complete word passivity')
        if row.get('forecast_proven') is not False or row.get('observer_forecast_passive') is not False:
            raise ValueError('standalone capture cannot assert pristine forecast passivity')
        dump=Path(row['dump'])
        recorded={str(p.relative_to(dump)):sha(p) for p in dump.rglob('*') if p.is_file()}
        if recorded!=row['capture_sha256']:raise ValueError('nested standalone packet inventory changed')
        calls=[json.loads(l) for l in (dump/'rank0-calls.jsonl').read_text().splitlines()]
        pre=[c for c in calls if c['phase']=='pre'];post=[c for c in calls if c['phase']=='post']
        if len(pre)!=row['child_calls'] or [(c['call'],c['step']) for c in pre]!=[(c['call'],c['step']) for c in post]:
            raise ValueError('nested standalone entry/return sequence incomplete')
        for call in pre:
            if call['hidden_state_complete'] or call['global_cell_ids_known']:
                raise ValueError('nested external capture cannot assert hidden-state or global-ID completeness')
            yield row['parent_row'], 'nested', dump, call, dict(
                standalone_authority_sha256=sha(authority),
                parent_native_capture_authority=row['parent_capture_authority'],
                parent_comparison_sha256=row['parent_comparison_sha256'],
                observer_standalone_passive=True,observer_forecast_passive=False,process_count=1)


def execute(args):
    spec=json.loads((args.build/'replay-spec.json').read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    for recipe,definition in spec['recipes'].items():
        root=args.build/recipe
        for path,digest in spec['generated_source_sha256'].items():
            if sha(root/path)!=digest:raise ValueError('native replay source changed: '+path)
        if sha(root/'replay.F90')!=spec['replay_source_sha256']:raise ValueError('replay program changed')
    if getattr(args,'standalone_index',None):
        authority=args.standalone_index
        captures=standalone_captures(authority,spec)
    elif args.legacy_index:
        authority=args.legacy_index
        captures=legacy_captures(authority,spec,args.words,args.out/'normalized-inputs')
    else:
        authority=args.observations/'forecasts/summary.json'
        observation=json.loads(authority.read_text())
        if observation['process_count']!=1:
            raise ValueError('standalone native context replay requires its recorded MPI1 scope')
        for row in observation['rows']:
            if not row.get('observer_passive') or row.get('entry_replay_intervention'):
                raise ValueError('external observer has not passed complete forecast passivity')
            for label in ('pristine-observer','variant-observer'):
                c=row['observer_comparisons'][label]
                if not c['comparison_complete'] or not c['same_output_inventory'] or c['differing_words']:
                    raise ValueError('incomplete or nonpassive pristine observer forecast')
        source_manifest=json.loads((args.observations/'source-provenance.json').read_text())
        for path,digest in source_manifest['files'].items():
            if sha(args.observations/path)!=digest:raise ValueError('observed ABI/observer source changed: '+path)
        captures=[]
        for deck,row in enumerate(observation['rows']):
            for mode in ('pristine-observed','variant-observed'):
                dump=args.observations/'forecasts'/str(deck)/(mode+'-entries')
                recorded=next(r for r in row['runs'] if r['label']==mode).get('observer_outputs')
                if not recorded or {str(p.relative_to(dump)):sha(p) for p in dump.rglob('*') if p.is_file()}!=recorded:
                    raise ValueError('native observed packets differ from their forecast run manifest')
                calls=[json.loads(l) for p in sorted(dump.glob('rank*-calls.jsonl')) for l in p.read_text().splitlines() if l]
                pre=[c for c in calls if c['phase']=='pre'];post=[c for c in calls if c['phase']=='post']
                if not pre or [(c['call'],c['step']) for c in pre]!=[(c['call'],c['step']) for c in post]:
                    raise ValueError('incomplete native wrapper call sequence')
                captures.extend((deck,mode,dump,c,dict(observer_manifest_sha256=sha(args.observations/'source-provenance.json'))) for c in pre)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPAS_ORACLE_SCHEMES='*')
    results=[]
    for deck,mode,dump,call,evidence in captures:
        expected_aliases={name:target for name,target in spec['argument_aliases'].items()
                          if (dump/(call['prefix']+'.pre.'+name+'.bin')).exists()}
        if call['argument_aliases']!=expected_aliases:
            raise ValueError('observed native pointer alias layout differs from source-bound replay')
        prefix=call['prefix']
        files=sorted(dump.glob(prefix+'.*.bin'))
        input_hashes={p.name:sha(p) for p in files}
        aliases={}
        for alias,target in call['argument_aliases'].items():
            for phase in ('pre','post'):
                run=subprocess.run([str(args.words),'compare',str(dump/(prefix+'.'+phase+'.'+alias+'.bin')),str(dump/(prefix+'.'+phase+'.'+target+'.bin'))],capture_output=True,text=True,check=True)
                comparison=json.loads(run.stdout)
                if not comparison['bitwise_equal']:raise ValueError('captured aliased buffer schema or words differ')
                aliases[phase+'.'+alias]=comparison
        for recipe in (args.recipe or spec['recipes']):
            if recipe not in spec['recipes']:raise ValueError('native recipe was not built: '+recipe)
            out=args.out/f'deck{deck}-{mode}-call{call["call"]}-{recipe}';out.mkdir()
            executable=args.build/recipe/'replay'
            env['MPAS_ORACLE_DIR']=str(out.resolve())
            command=[str(executable.resolve()),str((dump/prefix).resolve()),str((out/'replay').resolve())]
            process=subprocess.run(command,env=env,capture_output=True,text=True)
            (out/'replay.log').write_text(process.stdout+process.stderr)
            arrays,output_complete=compare_outputs(spec['site']['arguments'],dump,prefix,out,args.words)
            complete=process.returncode==0 and output_complete
            changed=[p.name for p in files if sha(p)!=input_hashes[p.name]]
            if changed:complete=False
            item=dict(capture=prefix,deck=deck,mode=mode,replay_recipe=recipe,replay_process_count=1,
                      executable_sha256=sha(executable),dump_root_sha256=hashlib.sha256(json.dumps(input_hashes,sort_keys=True).encode()).hexdigest(),
                      capture_sha256=input_hashes,source_manifest_sha256=sha(args.build/'replay-spec.json'),
                      capture_authority=evidence,
                      comparison_complete=complete,exit=process.returncode,arrays=arrays,changed_inputs=changed,
                      argument_aliases=call['argument_aliases'],alias_comparisons=aliases,
                      compared_words=sum(c.get('words',0) for c in arrays.values()),
                      differing_words=sum(c.get('different_words') or 0 for c in arrays.values()),
                      hidden_state_complete=False,forecast_proven=False,
                      scope=('Nested standalone module calls from authenticated real forecast inputs; ordinary and observed parent outputs exact; no observer forecast-passivity claim.'
                             if getattr(args,'standalone_index',None) else
                             'Pristine observer capture admitted only in its measured complete-forecast passivity scope; full native module replay, not Hex acceptance.'))
            write(out/'comparison.json',item)
            results.append(dict(comparison_path=str((out/'comparison.json').relative_to(args.out)),
                                comparison_sha256=sha(out/'comparison.json'),
                                capture=prefix,deck=deck,mode=mode,replay_recipe=recipe,
                                comparison_complete=complete,exit=process.returncode,
                                compared_words=item['compared_words'],differing_words=item['differing_words'],
                                arrays=len(arrays),forecast_proven=False,hidden_state_complete=False))
            # Replayed packet data is disposable after the word receipt;
            # original observer packets remain canonical on the capture host.
            for p in out.glob('*.bin'):p.unlink()
    result=dict(schema='native-external-full-call-replay-v1',rows=results,
        observation_receipt_sha256=sha(authority),
                words_tool_sha256=sha(args.words),runner_sha256=sha(Path(__file__)),
                comparison_complete=bool(results) and all(r['comparison_complete'] for r in results),
                compared_words=sum(r['compared_words'] for r in results),
                differing_words=sum(r['differing_words'] for r in results),
                arrays=sum(r['arrays'] for r in results),forecast_proven=False,coverage_complete=False)
    write(args.out/'summary.json',result)
    return int(not result['comparison_complete'] or result['differing_words']!=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--observations',type=Path)
    sources.add_argument('--legacy-index',type=Path,help='Hash-bound original tagged external captures and completed passivity receipts')
    sources.add_argument('--standalone-index',type=Path,help='Hash-bound nested standalone calls with complete parent-output comparisons; not forecast passivity')
    p.add_argument('--recipe',action='append',choices=['gnu-strict','gnu-o3','coverage'],
                   help='recipes to replay; strict is the native identity authority')
    for name in ('build','words','out'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(execute(p.parse_args()))
