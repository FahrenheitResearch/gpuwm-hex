#!/usr/bin/env python3
"""Regenerate, compile and word-compare real external native captures.

The given checkout supplies every tool/source template. Native inputs remain
opaque files authenticated against the producer's receipts. This runs standalone
module replay at its fixed captured process count, never a new forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

from hexcore import site_runner


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def compile_directory(a,directory,env,log):
    command=([str(a.build_launcher.resolve()),str(directory.resolve())] if a.build_launcher else
             site_runner.build_command(['make','-j4','-C',str(directory.resolve())]))
    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def retire_packets(authority):
    """Authenticate disposable packets against the successful capture manifest."""
    files=[]
    for definition in json.loads(authority.read_text())['rows']:
        manifest=Path(definition['manifest'])
        if sha(manifest)!=definition['sha256']:raise ValueError('capture manifest changed before retirement')
        row=json.loads(manifest.read_text());root=Path(row['dump'])
        actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        if actual!=row['capture_sha256']:raise ValueError('native packets changed before retirement')
        files.extend(p for p in root.rglob('*.bin') if p.is_file())
    retired=dict(files=len(files),bytes=sum(p.stat().st_size for p in files),retired=True,
                 capture_authority_sha256=sha(authority))
    for p in files:p.unlink()
    return retired


def execute(a):
    a.out.mkdir(parents=True,exist_ok=False)
    oracle=a.checkout.resolve()/'tree/tools/native_physics_reference/oracle'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPAS_UNIT='r1-rrfs-native-tap')
    generation=a.out/'generated'
    command=[sys.executable,str(oracle/'generate.py'),'--source',str(a.source.resolve()),
             '--out',str(generation.resolve()),'--pin-set',a.pin_set,
             '--external-observer',a.callee,'--external-native-replay','--external-replay-recipe',a.recipe]
    for hidden in a.hidden:command+=['--external-hidden',hidden]
    if a.preserve_native_calls:command+=['--external-preserve-calls']
    with (a.out/'generate.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    build=generation/'external-replay'/a.callee
    helpers=a.out/'tools';helpers.mkdir()
    (helpers/'oracle_words.rs').write_bytes((oracle/'oracle_words.rs').read_bytes())
    (helpers/'Makefile').write_text('all: oracle_words\noracle_words: oracle_words.rs\n\trustc --edition=2021 -O $< -o $@\n')
    for name,directory in [('words',helpers),('native',build)]:
        with (a.out/(name+'-build.log')).open('w') as log:
            compile_directory(a,directory,env,log)
    with (a.out/'replay.log').open('w') as log:
        process=subprocess.run([sys.executable,str(oracle/'replay/replay_external_calls.py'),
             '--legacy-index',str(a.index.resolve()),'--build',str(build.resolve()),
             '--words',str((helpers/'oracle_words').resolve()),'--out',str((a.out/'replays').resolve()),
             '--recipe',a.recipe],env=env,stdout=log,stderr=subprocess.STDOUT)
    summary=a.out/'replays/summary.json'
    data=json.loads(summary.read_text()) if summary.is_file() else {}
    result=dict(schema='real-native-external-replay-reproduction-v1',recipe=a.recipe,process_count=1,
                exit=process.returncode,comparison_complete=data.get('comparison_complete',False),
                differing_words=data.get('differing_words'),compared_words=data.get('compared_words'),
                capture_authority_sha256=sha(a.index),checkout=str(a.checkout.resolve()),
                generated_spec_sha256=sha(build/'replay-spec.json'),runner_sha256=sha(Path(__file__)),
                tools_sha256={str(p.relative_to(oracle)):sha(p) for p in oracle.rglob('*')
                              if p.is_file() and p.suffix in {'.py','.rs','.json','.F','.F90'}},
                summary_sha256=sha(summary) if summary.is_file() else None,
                forecast_proven=False,hidden_state_complete=False)
    result['build_launcher_sha256']=sha(a.build_launcher) if a.build_launcher else None
    result['all_selected_modules_complete']=False
    (a.out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    failed=process.returncode!=0 or not result['comparison_complete'] or result['differing_words']!=0
    if failed:return 1
    result['total_compared_words']=result['compared_words']
    result['total_differing_words']=result['differing_words']
    result['nested_replays']=[]
    for ordinal,callee in enumerate(a.nested_callee):
        prefix='nested' if ordinal==0 else 'nested-'+callee
        # Regenerate the nested observer and its complete native module using the
        # same checkout. The original parent forecasts are never re-run.
        nested=a.out/(prefix+'-generated')
        with (a.out/(prefix+'-generate.log')).open('w') as log:
            subprocess.run([sys.executable,str(oracle/'generate.py'),'--source',str(a.source.resolve()),
                '--out',str(nested.resolve()),'--pin-set',a.pin_set,'--external-observer',callee,
                '--external-native-replay','--external-replay-recipe',a.recipe],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        child=nested/'external-replay'/callee
        observer=nested/'external'/callee
        for name,directory in [('child',child),('observer',observer)]:
            with (a.out/(prefix+'-'+name+'-build.log')).open('w') as log:
                compile_directory(a,directory,env,log)
        with (a.out/(prefix+'-capture.log')).open('w') as log:
            subprocess.run([sys.executable,str(oracle/'replay/capture_nested_external_calls.py'),
                '--parent-build',str(build.resolve()),'--child-build',str(child.resolve()),
                '--observer',str((observer/'observer').resolve()),'--observer-spec',str((observer/'spec.json').resolve()),
                '--words',str((helpers/'oracle_words').resolve()),'--index',str(a.index.resolve()),
                '--out',str((a.out/(prefix+'-captures')).resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (a.out/(prefix+'-replay.log')).open('w') as log:
            run=subprocess.run([sys.executable,str(oracle/'replay/replay_external_calls.py'),
                '--standalone-index',str((a.out/(prefix+'-captures/summary.json')).resolve()),
                '--build',str(child.resolve()),'--words',str((helpers/'oracle_words').resolve()),
                '--out',str((a.out/(prefix+'-replays')).resolve()),'--recipe',a.recipe],env=env,stdout=log,stderr=subprocess.STDOUT)
        summary=a.out/(prefix+'-replays/summary.json')
        data=json.loads(summary.read_text()) if summary.is_file() else {}
        replayed=dict(callee=callee,exit=run.returncode,
            comparison_complete=data.get('comparison_complete',False),differing_words=data.get('differing_words'),
            compared_words=data.get('compared_words'),arrays=data.get('arrays'),summary_sha256=sha(summary) if summary.is_file() else None,
            capture_authority_sha256=sha(a.out/(prefix+'-captures/summary.json')),observer_forecast_passive=False)
        result['nested_replays'].append(replayed)
        result['total_compared_words']+=replayed['compared_words'] or 0
        result['total_differing_words']+=replayed['differing_words'] or 0
        if ordinal==0:result['nested_replay']=replayed
        (a.out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
        if run.returncode!=0 or not data.get('comparison_complete') or data.get('differing_words')!=0:return 1
        if a.retire_packets:
            replayed['packet_retirement']=retire_packets(a.out/(prefix+'-captures/summary.json'))
            (a.out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    result['all_selected_modules_complete']=True
    (a.out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkout','source','index','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin-set',choices=['rrfs','mpas-v842'],required=True)
    p.add_argument('--callee',required=True)
    p.add_argument('--hidden',action='append',default=[])
    p.add_argument('--preserve-native-calls',action='store_true',help='Preserve standalone parent call boundaries; every parent output must still match the original native capture')
    p.add_argument('--retire-packets',action='store_true',help='Authenticate and retire newly generated nested packets after complete successful replay; retain manifests and comparisons')
    p.add_argument('--nested-callee',action='append',default=[],help='Also observe and replay the complete nested native module from the authenticated real parent inputs')
    p.add_argument('--build-launcher',type=Path,help='Optional host build adapter: invoked with the freshly generated build directory, must return its built executables there')
    p.add_argument('--recipe',choices=['gnu-strict','gnu-o3','coverage'],default='gnu-strict')
    raise SystemExit(execute(p.parse_args()))
