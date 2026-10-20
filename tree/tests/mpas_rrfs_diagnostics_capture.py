"""Capture complete pristine diagnostics calls, then prove observer passivity.

Reuses r1's authenticated real GSL case and Rust whole-stream comparator. Only
orchestrates opaque files, hashes, native children and metadata; Rust reads all
weather values. MPI1 and unchanged one-hour/five-step native deck, not another
24h Hex forecast. Numerical source/executable/stream lists remain unchanged.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2) + '\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--namelist',type=Path,required=True)
    p.add_argument('--observer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--passivity-runner',type=Path,required=True)
    args=p.parse_args()
    pack=json.loads((args.pack/'pack.json').read_text())
    for member,digest in pack['files'].items():
        if member.startswith(('inputs/','bin/','lib/')) and sha(args.pack/member)!=digest:
            raise ValueError('native pack changed: '+member)
    binary=args.pack/'bin/atmosphere_model'
    if sha(binary)!=pack['native_binary_sha256']:
        raise ValueError('pristine binary hash mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    args.work.mkdir(parents=True,exist_ok=False)
    # Source field labels are metadata only; no array is initialized here.
    from hexcore.mpasdiag.convective_rrfs import FIELDS
    labels=args.out/'fields.txt';labels.write_text('\n'.join(FIELDS)+'\n')
    sources={str(Path(__file__).resolve()):sha(__file__),
             str(args.observer.resolve()):sha(args.observer),
             str(args.observer.with_suffix('.rs').resolve()):sha(args.observer.with_suffix('.rs')),
             str(args.passivity_runner.resolve()):sha(args.passivity_runner)}
    write(args.out/'request.json',dict(schema='mpas.r12-native-diagnostics-capture/v1',
        host=socket.gethostname(),process_count=1,native_binary_sha256=sha(binary),
        pack_manifest_sha256=sha(args.pack/'pack.json'),namelist_sha256=sha(args.namelist),
        fields=list(FIELDS),tools=sources,
        scope='original GSL supporting real case; five native steps; no Hex forecast'))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             LD_LIBRARY_PATH=str((args.pack/'lib').resolve())+':'+os.environ.get('LD_LIBRARY_PATH',''))
    runs=[];roots=[]
    for label in ('ordinary','observed'):
        root=args.work/label;root.mkdir()
        for name,digest in pack['inputs'].items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(args.pack/'inputs'/name,target)
            target.chmod(target.stat().st_mode|0o200)
            if sha(target)!=digest:raise ValueError('staged input mismatch '+name)
        shutil.copyfile(args.namelist,root/'namelist.atmosphere')
        inputs={n:sha(root/n) for n in pack['inputs']}
        argv=[str((args.pack/'bin/mpiexec').resolve()),'-np','1']
        if label=='observed':argv += [str(args.observer.resolve()),str(binary.resolve()),str((args.out/'captures').resolve()),str(labels.resolve())]
        else:argv += [str(binary.resolve())]
        with (args.out/(label+'.log')).open('w') as log:
            proc=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        outputs={str(f.relative_to(root)):sha(f) for f in root.rglob('*.nc') if str(f.relative_to(root)) not in inputs}
        changed=[n for n,h in inputs.items() if sha(root/n)!=h]
        runs.append(dict(label=label,argv=argv,exit=proc.returncode,inputs=inputs,outputs=outputs,changed_inputs=changed))
        roots.append(root)
        for f in root.glob('log.atmosphere.*'):shutil.copy2(f,args.out/(label+'-'+f.name))
        write(args.out/'runs.json',runs)
    spec=importlib.util.spec_from_file_location('r1_passivity',args.passivity_runner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import re
    text=args.namelist.read_text()
    initial=datetime.datetime.strptime(re.search(r"config_start_time\s*=\s*'([^']+)'",text)[1],'%Y-%m-%d_%H:%M:%S')
    dt=float(re.search(r'config_dt\s*=\s*([\d.]+)',text)[1])
    args.comparator=args.pack/'bin/rw_mpas_verify'
    comparison=module.compare_runs(args,args.out,roots,runs,initial,dt,'passivity-')
    captures={str(f.relative_to(args.out/'captures')):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in (args.out/'captures').rglob('*') if f.is_file()}
    write(args.out/'capture-manifest.json',dict(schema='mpas.r12-real-diagnostics-captures/v1',
        native_binary_sha256=sha(binary),native_source_pin='1d6e370e7b43a332074809f170532a216c2212fb',
        host=socket.gethostname(),process_count=1,passivity=comparison,
        observer_numeric_writes=False,fields=list(FIELDS),files=captures))
    passed=bool(captures) and comparison['comparison_complete'] and comparison['differing_words']==0
    write(args.out/'summary.json',dict(runs=runs,**comparison,captured_files=len(captures),observer_passive=passed))
    if passed:
        for root,run in zip(roots,runs):
            for name,h in {**run['inputs'],**run['outputs']}.items():
                if sha(root/name)!=h:raise ValueError('runtime file changed before retirement')
            shutil.rmtree(root)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
