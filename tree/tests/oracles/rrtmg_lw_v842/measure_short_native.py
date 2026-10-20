"""Execute the entire short-layer boundary matrix with pristine MPAS physics.
Opaque input/output hashes only; numerical records are not decoded in Python.
"""
import argparse,hashlib,json,os,socket,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bounds',action='store_true');p.add_argument('--two-process-only',action='store_true');a=p.parse_args()
root=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
def stamp(path):
 return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
rows=[]
variants=[('bounds',1)] if a.bounds else [('two-process',2)] if a.two_process_only else [('repeat1',1),('repeat2',1),('two-process',2)]
start=time.time()
for variant,np in variants:
 for nl in range(1,5):
  for nc in (1,3):
   for scenario in range(36):
    dest=out/variant/f'nl{nl}-nc{nc}-case{scenario:02}';dest.mkdir(parents=True)
    cmd=[str(root/'short_mcica'),str(dest),str(nl),str(nc),str(scenario)]
    if np==2:cmd=['mpirun','--bind-to','none','-n','2',*cmd]
    with (dest/'run.log').open('wb') as log:
     try:rc=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=30).returncode
     except subprocess.TimeoutExpired:rc=124
    row={'variant':variant,'layers':nl,'columns_per_process':nc,'scenario':scenario,'overlap':scenario%6,'rng_in':scenario//12,'decorrelation':(scenario//6)%2,'process_count':np,'physics_decomposition':False,'command':cmd,'returncode':rc,'inputs':[stamp(x) for x in sorted(dest.glob('*-input.bin'))],'outputs':[stamp(x) for x in sorted(dest.glob('*-output.bin'))],'log':stamp(dest/'run.log')}
    rows.append(row)
    (out/'progress.json').write_text(json.dumps({'host':socket.gethostname(),'completed':len(rows),'last':row},indent=2)+'\n')
manifest={'schema':'p22-short-native-defect-measurement-v47','host':socket.gethostname(),'started_epoch':start,'finished_epoch':time.time(),'process_counts':sorted({r['process_count'] for r in rows}),'standalone_synthetic_only':True,'published_full_model_proof':False,'module':stamp(root/'src/module_ra_rrtmg_lw.F'),'executable':stamp(root/'short_mcica'),'harness':stamp(root/'src/short_mcica.F90'),'makefile':stamp(root/'Makefile'),'runs':rows}
(out/'matrix.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'matrix':str(out/'matrix.json'),'actual_runs':len(rows),'with_outputs':sum(len(r['outputs'])==r['process_count'] for r in rows),'nonzero_exits':sum(r['returncode']!=0 for r in rows)}),flush=True)
