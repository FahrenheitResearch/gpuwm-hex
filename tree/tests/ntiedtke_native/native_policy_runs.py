"""Measure the published strict native option on repeated and MPI runs.

NATIVE-DEFECT-POLICY.md item 2. This is a native-only witness, not hex forecast
acceptance. Field comparison is the published f0 Rust comparator. The supplied
published deck controls duration and every stream's output schedule.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('host');p.add_argument('--label',default='none');p.add_argument('--output',type=Path,required=True);p.add_argument('--deck',type=Path);p.add_argument('--build',type=Path);p.add_argument('--timeout',type=int,default=180);p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path('scratch/native-policy').resolve();a.output.mkdir(exist_ok=True,parents=True)
base=[sys.executable,str(root/'run_forecast.py'),'--build',str(a.build.resolve() if a.build else root/'build'),'--deck',str(a.deck.resolve() if a.deck else root/'deck'),'--case',str(root/'case'),'--tables',str(root/'tables'),'--mpi',os.environ.get('MPAS_POLICY_MPI','mpiexec'),'--pin-set-hash','fc599c7e472fb5996957d95b4021a42f116f7b044dbd0a532943b271293ebc40']
manifest=[]
for name,ranks in (('rank1-a',1),('rank1-b',1),('rank4',4)):
 out=root/(a.host+'-'+a.label+'-'+name)
 timed_out=False
 with (a.output/(name+'.log')).open('a' if a.resume else 'w') as log:
  if a.resume and (out/'MANIFEST.json').is_file():
   process=None
  else:
   process=subprocess.Popen(base+['--ranks',str(ranks),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   if process is not None: process.wait(timeout=a.timeout)
  except subprocess.TimeoutExpired:
   timed_out=True
   os.killpg(process.pid,signal.SIGTERM)
   try: process.wait(timeout=5)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGKILL);process.wait()
 if process is not None and process.returncode:
  for log in ('native.stdout.log','log.atmosphere.0000.out'):
   if (out/log).exists(): (a.output/(name+'-'+log)).write_bytes((out/log).read_bytes())
  manifest.append({'name':name,'ranks':ranks,'exit':process.returncode,'timed_out':timed_out,'forecast_completed':False})
  continue
 manifest.append({'name':name,'ranks':ranks,'manifest':json.loads((out/'MANIFEST.json').read_text())})
 for reference in [root/(a.host+'-'+a.label+'-rank1-a')] if name!='rank1-a' and manifest[0].get('forecast_completed',True) else []:
  destination=a.output/(name+'-compare.json')
  result=subprocess.run([sys.executable,str(root/'compare_runs.py'),str(reference),str(out),'--comparator',str(root/'compare'),'--out',str(destination)])
  if result.returncode not in (0,1): raise RuntimeError('native comparison failed to produce a witness')
  manifest[-1]['comparison_receipt']=str(destination)
(a.output/'native-policy-runs.json').write_text(json.dumps({'schema':'native-ntiedtke-policy-witness/v1','host':a.host,'label':a.label,'acceptance':False,'runs':manifest},indent=2)+'\n')
