"""Short LTS-off history byte check. This records no benchmark timings."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
R=W/'prepared'
BASE='49b4a1b012333b512934f68ed98a5cf77d9a6e1e'

def parity_command(python,out,engine,manifest):
    member=manifest['members'][0]
    return [python,'-m','hexcore.cli','forecast','--mesh',manifest['mesh'],
        '--grid',member.get('grid',manifest['grid']),
        '--static',member.get('static',manifest['static']),
        '--init',member['init'],'--init-source',member['source'],
        '--start-time',manifest['start_time'],'--hours',str(1/12),'--dt','18.75',
        '--history-every-minutes','5','--out',str(out/'history'),
        '--scratch',str(out/'scratch'),'--gpuwm-checkout',engine]

def main():
    source=W/'parity-base/tree'
    if not source.is_dir(): raise ValueError('baseline must be staged before the GPU lease')
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1:
        raise ValueError('byte check requires one granted card')
    records={}
    manifest=json.loads((R/'ensemble.json').read_text())
    attempt=W/'parity'/f'attempt-{len(list((W/"parity").glob("attempt-*")))+1:03d}'
    for label,tree in [('base',source),('lane',W/'hex/tree')]:
        out=attempt/label
        out.mkdir(parents=True)
        command=parity_command(sys.executable,out,os.environ['E'],manifest)
        env=dict(os.environ,PYTHONPATH=str(tree/'src')+':'+os.environ['E'],
                 GPUWM_HEX_MESH_ROWS=str(R/'mesh-rows.json'),HEXPUB=str(tree),HEXEXP=str(tree),
                 HEX_CAPABILITY_CHECKOUT=str(tree))
        with (out/'forecast.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{label} byte-check failed; {out}/forecast.log')
        files={}
        for path in sorted((out/'history').glob('cuda-history.*.nc')):
            with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
            files[path.name]=dict(sha256=sha,bytes=path.stat().st_size)
        if len(files)!=2: raise ValueError('expected frame zero and five minutes')
        records[label]=dict(command=command,history=files,returncode=result.returncode)
    equal=records['base']['history']==records['lane']['history']
    (W/'evidence/lts-off-byte-parity.json').write_text(json.dumps(dict(base=BASE,
        passed=equal,arms=records,scope='LTS off, same engine and inputs, two native history files; no timings'),indent=2)+'\n')
    if not equal: raise ValueError('LTS-off history bytes changed')

if __name__=='__main__': main()
