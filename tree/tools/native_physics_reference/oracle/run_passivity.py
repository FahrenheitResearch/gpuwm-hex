#!/usr/bin/env python3
"""Run an owned instrumented binary on an authenticated native forecast deck.

No model source is rebuilt here. File staging and hashing are orchestration;
all field decoding and comparison is delegated to the supplied Rust comparator.
A numeric match and a complete-file byte match are reported separately.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def native_baseline(path,output):
    """Use either an oracle baseline or r0's published forecast protocol."""
    baseline=read(path)
    if baseline.get('schema')!='mpas.rrfs-native-forecast/v1':return baseline
    if baseline['recipe']!='gnu-strict' or not baseline['completed']:
        raise ValueError('r0 baseline must be a completed strict native forecast')
    hashes={**baseline['inputs'],**baseline['resolved_configuration']['files']}
    inputs={}
    for name,digest in hashes.items():
        member=Path(name)
        if not member.parts or member.is_absolute() or '..' in member.parts:
            raise ValueError('unsafe r0 runtime input name')
        source=(output/name).resolve(strict=True)
        if sha(source)!=digest:
            raise ValueError('r0 baseline runtime input bytes changed')
        inputs[name]={'source':str(source),'sha256':digest}
    return {'exit':baseline['exit_status'],'process_count':baseline['ranks'],
            'native_binary_sha256':baseline['binary_sha256'],'argv':baseline['argv'],
            'inputs':inputs,'outputs':baseline['outputs']}


def run(a):
    baseline=native_baseline(a.baseline_receipt,a.baseline_output)
    inventory=read(a.inventory);audit=read(a.audit)
    if a.process_count<1:raise ValueError('process count must be positive')
    if baseline['exit']!=0 or baseline['process_count']!=a.process_count:
        raise ValueError('baseline must be a complete forecast at the same process count')
    if inventory['strict_reference_binary_sha256']!=baseline['native_binary_sha256']:
        raise ValueError('instrumented source tree and baseline native binaries differ')
    if not audit['all_ok'] or audit['recipe']!='gnu-strict':
        raise ValueError('instrumented binary needs the complete strict object audit')
    binaries=read(a.build_artifacts)
    digest=sha(a.binary)
    if not any(Path(r['path']).name==a.binary.name and r['sha256']==digest for r in binaries):
        raise ValueError('instrumented executable is absent from build provenance')
    if not any(Path(r['path']).name==a.inventory.name and r['sha256']==sha(a.inventory) for r in binaries):
        raise ValueError('source inventory is absent from executable build provenance')
    if a.out.exists() or a.receipt.exists():raise ValueError('run and receipt must be fresh')
    expected={name:hash_ for name,hash_ in baseline['outputs'].items() if name.endswith('.nc')}
    if not expected:raise ValueError('baseline contains no field outputs')
    for name,digest in expected.items():
        if Path(name).name!=name or sha(a.baseline_output/name)!=digest:
            raise ValueError('baseline output bytes changed')
    a.out.mkdir(parents=True)
    for name,row in baseline['inputs'].items():
        source=Path(row['source'])
        member=Path(name)
        if not member.parts or member.is_absolute() or '..' in member.parts or sha(source)!=row['sha256']:
            raise ValueError('native runtime input bytes changed')
        (a.out/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,a.out/name)
    argv=list(baseline['argv']);argv[-1]=str(a.binary.resolve())
    env={k:v for k,v in os.environ.items() if not k.startswith('MPAS_ORACLE_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    mpi_lib=Path(argv[0]).parent.parent/'lib'
    env['LD_LIBRARY_PATH']=str(mpi_lib)+':'+env.get('LD_LIBRARY_PATH','')
    if a.keys is not None:
        if a.dump is None:raise ValueError('enabled capture requires an explicit dump directory')
        if a.dump.exists():raise ValueError('dump must be fresh')
        known={s['site'] for s in inventory['sites']}|{b.get('key',b.get('name','')) for b in inventory['boundaries']}
        if a.keys!='*' and not set(a.keys.split(',')).issubset(known):
            raise ValueError('capture key is absent from the authenticated inventory')
        a.dump.mkdir(parents=True)
        env.update(MPAS_ORACLE_DIR=str(a.dump.resolve()),MPAS_ORACLE_KEYS=a.keys)
        if a.steps is not None:env['MPAS_ORACLE_STEPS']=a.steps
    with (a.out/'stdout.log').open('w') as log:
        result=subprocess.run(argv,cwd=a.out,env=env,stdout=log,stderr=subprocess.STDOUT)
    changed_inputs=[n for n,r in baseline['inputs'].items() if sha(a.out/n)!=r['sha256']]
    # Filename times and deck timestep provide the comparison schedule. Python
    # never opens a field container; the Rust reader checks stored xtime.
    deck=(a.out/'namelist.atmosphere').read_text()
    start=re.search(r"config_start_time\s*=\s*['\"]([^'\"]+)['\"]",deck,re.I)
    dt=re.search(r'config_dt\s*=\s*([\d.eEdD+-]+)',deck,re.I)
    if not start or not dt:raise ValueError('native start time/timestep are required')
    initial=datetime.datetime.strptime(start[1],'%Y-%m-%d_%H:%M:%S')
    timestep=float(dt[1].replace('d','e').replace('D','e'))
    samples=[]
    for name in sorted(expected):
        match=re.fullmatch(r'(.+)\.(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})\.nc',name)
        if not match:raise ValueError('supply a native stream filename with an explicit output time')
        time=datetime.datetime.strptime(match[2],'%Y-%m-%d_%H.%M.%S')
        step=(time-initial).total_seconds()/timestep
        if step<0 or not step.is_integer():raise ValueError('output time does not lie on the native timestep')
        samples.append({'stream':match[1],'steps':[int(step)],'times':[time.strftime('%Y-%m-%d_%H:%M:%S')],
            'strict':str((a.baseline_output/name).resolve()),'hex':str((a.out/name).resolve())})
    request=a.out/'compare-request.json';compare=a.out/'compare.json'
    write(request,{'schema':'mpas-stream-compare-v1','samples':samples})
    c=subprocess.run([str(a.comparator.resolve()),'--request',str(request.resolve()),'--out',str(compare.resolve())],capture_output=True,text=True)
    comparison=read(compare) if c.returncode==0 else {'fields':[],'error':c.stdout+c.stderr}
    fields=comparison.get('fields',[])
    complete=bool(fields) and all(r['status']=='compared' for r in fields)
    numeric_equal=complete and all(r['differing_words']==0 for r in fields)
    actual={p.name:sha(p) for p in a.out.glob('*.nc') if p.name not in baseline['inputs']}
    file_equal=actual==expected
    passed=result.returncode==0 and not changed_inputs and file_equal and numeric_equal
    receipt={'schema':'mpas-oracle-passivity-v1','host':socket.gethostname(),'process_count':a.process_count,
        'baseline_receipt_sha256':sha(a.baseline_receipt),'baseline_output_root':str(a.baseline_output.resolve()),
        'native_binary_sha256':baseline['native_binary_sha256'],'instrumented_binary_sha256':sha(a.binary),
        'inventory_sha256':sha(a.inventory),'object_audit_sha256':sha(a.audit),'build_artifacts_sha256':sha(a.build_artifacts),
        'runner_sha256':sha(Path(__file__)),'comparator_sha256':sha(a.comparator),'inputs':baseline['inputs'],
        'argv':argv,'capture':{'keys':a.keys,'steps':a.steps,'dump':None if a.dump is None else str(a.dump.resolve())},
        'exit':result.returncode,'changed_inputs':changed_inputs,'expected_output_sha256':expected,'output_sha256':actual,
        'all_output_files_byte_equal':file_equal,'comparison_complete':complete,'all_field_words_equal':numeric_equal,
        'compared_words':sum(r['compared_words'] for r in fields if r['status']=='compared'),
        'differing_words':sum(r['differing_words'] for r in fields if r['status']=='compared'),
        'missing_or_incompatible_samples':sum(r['status']!='compared' for r in fields),
        'comparison':comparison,'passivity_byte_equal':passed}
    write(a.receipt,receipt)
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline_receipt','baseline_output','binary','inventory','audit','build_artifacts','out','receipt','comparator']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--process-count',type=int,required=True)
    p.add_argument('--dump',type=Path)
    p.add_argument('--keys')
    p.add_argument('--steps')
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
