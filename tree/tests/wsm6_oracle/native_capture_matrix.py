"""Orchestrate six source-bound native fixtures; all weather work is native/Rust."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--payload',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--pin-set-hash',required=True)
p.add_argument('--capture-key',default='wsm6__driver_microphysics__L439')
a=p.parse_args();payload=a.payload.resolve();out=a.output.resolve()
out.mkdir(parents=True,exist_ok=True)
spec=json.loads((payload/'payload.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
for name,entry in spec['files'].items():
 dest=payload/name
 if not dest.exists() and entry.get('parts'):
  with dest.open('wb') as f:
   subprocess.run(['cat',*[str(payload/x) for x in entry['parts']]],stdout=f,check=True)
 if sha(dest)!=entry['sha256']:raise ValueError('copied authority hash changed: '+name)
for variant in ('plain','oracle'):(payload/variant/'atmosphere_model').chmod(0o755)
(payload/'mpas-native-word-compare').chmod(0o755)
rows=[]
mpi=str(payload/'mpich/bin/mpiexec')
os.environ['LD_LIBRARY_PATH']=str(payload/'mpich/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
(out/'runtime.json').write_text(json.dumps({'mpiexec':mpi,'sha256':sha(Path(mpi).resolve()),
 'version':subprocess.run([mpi,'--version'],capture_output=True,text=True).stdout,
 'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH','')},indent=2)+'\n')
def save(complete):
 (out/'summary.json').write_text(json.dumps({'scope':spec['scope'],'host':os.uname().nodename,
  'process_count':1,'complete':complete,'forecast_acceptance':False,
  'payload_sha256':sha(payload/'payload.json'),'cases':rows},indent=2)+'\n')
for case in spec['cases']:
 key=case['key'];caseout=out/key;caseout.mkdir(exist_ok=True)
 for variant in ('plain','oracle'):
  run=caseout/variant
  if (run/'MANIFEST.json').is_file():continue
  if run.exists():
   if not (run/'log.atmosphere.0000.out').exists() and ('error while loading shared libraries' in (run/'native.stdout.log').read_text() or (run/'native.stdout.log').stat().st_size==0):
    reason='mpi-library-failure' if (run/'native.stdout.log').stat().st_size else 'mpi-path-failure'
    run.rename(caseout/(variant+'-'+reason))
    shutil.copyfile(caseout/(variant+'-runner.log'),caseout/(variant+'-'+reason+'.log'))
   elif (run/'log.atmosphere.0000.out').exists() and 'badly nested XML tag "?xml"' in (run/'log.atmosphere.0000.out').read_text():
    run.rename(caseout/(variant+'-xml-declaration-failure'))
    shutil.copyfile(caseout/(variant+'-runner.log'),caseout/(variant+'-xml-declaration-failure.log'))
   else:raise RuntimeError('incomplete native run retained for inspection: '+str(run))
  env=os.environ.copy()
  for name in ('MPAS_ORACLE_DIR','MPAS_ORACLE_KEYS','MPAS_ORACLE_STEPS'):env.pop(name,None)
  if variant=='oracle':
   dumps=caseout/'dumps';dumps.mkdir(exist_ok=True)
   env.update(MPAS_ORACLE_DIR=str(dumps),MPAS_ORACLE_KEYS=a.capture_key,MPAS_ORACLE_STEPS='1')
  cmd=[sys.executable,str(payload/'run_forecast.py'),'--build',str(payload/variant),
   '--deck',str(payload/case['deck']),'--case',str(payload/case['case']),
   '--out',str(run),'--tables',str(payload/variant),'--mpi',
   mpi,
   '--ranks','1','--pin-set-hash',a.pin_set_hash]
  print('RUN',key,variant,flush=True)
  with (caseout/(variant+'-runner.log')).open('w') as log:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError(f'native fixture failed: {key}/{variant}, status {rc}')
 comparison=caseout/'native-instrumentation-identity.json'
 cmd=[sys.executable,str(payload/'compare_runs.py'),str(caseout/'plain'),str(caseout/'oracle'),
      '--comparator',str(payload/'mpas-native-word-compare'),'--out',str(comparison)]
 with (caseout/'comparison.log').open('w') as log:
  rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError('native instrumentation changed output words: '+key)
 result=json.loads(comparison.read_text())
 if not result['all_identical']:raise ValueError(key+' nonidentical native instrumentation')
 dumps=caseout/'dumps'
 # The f1 writer uses per-call .pre/.post records; inspect the original member
 # names without interpreting their weather payloads.
 members={str(x.relative_to(caseout)):sha(x) for x in sorted(dumps.iterdir()) if x.is_file()}
 if not members:raise RuntimeError('native WSM6 capture absent: '+key)
 rows.append({'case':case,'comparison':key+'/'+comparison.name,
  'comparison_sha256':sha(comparison),'all_native_output_words_identical':True,
  'native_output_files':len(result['files']),'capture_members':members,
  'plain_manifest_sha256':sha(caseout/'plain/MANIFEST.json'),
  'oracle_manifest_sha256':sha(caseout/'oracle/MANIFEST.json')})
 save(False)
 print('COMPLETE',key,len(result['files']),'native output files',len(members),'capture members',flush=True)
save(True)
