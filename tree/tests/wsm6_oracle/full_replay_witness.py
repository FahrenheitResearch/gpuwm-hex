"""Replay all stored WSM6 columns and all outputs with the host port witness.

Each expected POST record is read only by the independent Rust comparator,
after the executable has finished producing results from PRE inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport',type=Path,required=True)
    p.add_argument('--host-library',type=Path,required=True)
    p.add_argument('--words',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--capture',nargs=2,action='append',required=True)
    a=p.parse_args()
    a.scratch.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,WSM6_REPLAY_TRANSPORT=str(a.transport.resolve()),
             WSM6_REPLAY_HOST_LIBRARY=str(a.host_library.resolve()),WSM6_REPLAY_BACKEND='host-witness')
    adapter=Path(__file__).with_name('full_replay_adapter.py')
    results=[]
    result={'scope':__doc__.strip(),'execution_backend':'host-port-glibc-witness',
            'eligible_for_cuda_or_forecast_acceptance':False,'debug_processes':1,
            'adapter_sha256':sha(adapter),'transport_sha256':sha(a.transport),
            'host_library_sha256':sha(a.host_library),'word_helper_sha256':sha(a.words),
            'captures':results,'complete':False}
    for index,(root,prefix) in enumerate(a.capture):
        root=Path(root).resolve()
        manifest=json.loads((root/'manifest.json').read_text())
        out=a.scratch/str(index)
        subprocess.run([sys.executable,str(adapter),str(root/prefix),str(out/prefix)],env=env,check=True)
        expected={n for n in manifest['files'] if n.startswith(prefix+'.post.')}
        actual={path.name for path in out.glob('*.bin')}
        if expected!=actual:
            raise ValueError(f'incomplete result: missing={expected-actual}, extra={actual-expected}')
        rows={}
        for name in sorted(expected):
            source=root/name;candidate=out/name
            if sha(source)!=manifest['files'][name]['sha256']:
                raise ValueError('expected record changed after sealed capture')
            values=json.loads(subprocess.check_output([str(a.words.resolve()),'compare',str(source),str(candidate)],text=True))
            rows[name[len(prefix+'.post.'):-4]]=values
        case={'capture':prefix,'dump_root_sha256':manifest['dump_root_sha256'],
              'outputs':rows,'different_words':sum(v['different_words'] for v in rows.values()),
              'compared_words':sum(v['words'] for v in rows.values())}
        results.append(case)
        a.receipt.write_text(json.dumps(result,indent=2)+'\n')
        if case['different_words']:
            raise ValueError('host port differs; retain own outputs to debug')
        shutil.rmtree(out)  # Only this invocation's results, after sealed receipt.
        print(prefix,len(rows),'outputs:',case['compared_words'],'words equal',flush=True)
    result['complete']=True
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
