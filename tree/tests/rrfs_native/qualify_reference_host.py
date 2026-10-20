"""Qualify complete real fork work boundaries on the reference host.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:
2225-2366,2646-2984,3613-4023,5148-6409,7398-7446 at 1d6e370e.
Only process/filesystem orchestration and receipt metadata live here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lane',type=Path,required=True)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    matrix=json.loads(args.matrix.read_text())
    lane=args.lane.resolve();out=args.out.resolve()
    if socket.gethostname()!=matrix['reference_host']:
        raise ValueError('reference-host capture invoked on the wrong host')
    tools=Path(__file__).resolve().parent
    for field in ('instrumented','producer_contract','comparator','inventory'):
        if sha(lane/matrix[field])!=matrix[field+'_sha256']:
            raise ValueError('reference-host producer dependency changed: '+field)
    producer=json.loads((lane/matrix['producer_contract']).read_text())
    if producer['fork_pin']!=matrix['source_commit'] or producer['native_arithmetic_edits']:
        raise ValueError('reference-host producer changed the pinned native arithmetic')
    out.mkdir(parents=True,exist_ok=False)
    result=[]
    for case in matrix['cases']:
        inputs=lane/case['inputs_manifest']
        if sha(inputs)!=case['inputs_manifest_sha256']:
            raise ValueError('real input manifest changed')
        root=out/case['id'];root.mkdir()
        command=[sys.executable,str(tools/'capture_dycore_first_step.py'),
            '--inputs',str(inputs),'--case',str(lane/case['case']),
            '--work-root',str(root/'runs'),'--instrumented',str(lane/matrix['instrumented']),
            '--mpi',str(lane/matrix['mpi']),'--runtime',str(lane/matrix['runtime']),
            '--steps',str(case['steps']),'--keys',*case['keys']]
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,str(tools/'compare_capture_passivity.py'),
            '--runs',str(root/'runs'),'--comparator',str(lane/matrix['comparator']),
            '--out',str(root/'passivity.json'),'--scope',case['scope']+'; every stored output word, reference-host observer controls'],check=True)
        bank=root/'runs/active/captures';records=[];calls={};post=[]
        for path in sorted(bank.glob('*.bin')):
            records.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path)))
            if '.pre.' in path.name:
                prefix=path.name.split('.pre.')[0];key=prefix.split('.s')[0]
                calls.setdefault(key,set()).add(prefix)
            elif '.post.' in path.name:post.append(path)
        manifest=dict(schema='rrfs-dycore-real-capture/v1',source_commit=matrix['source_commit'],
            records=records,record_count=len(records),bytes=sum(row['bytes'] for row in records))
        manifest_path=root/'manifest.json'
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        inventory=json.loads(subprocess.check_output([str(lane/matrix['inventory']),
            '--inventory',*[str(path) for path in post]],text=True))
        assert inventory['records']==len(post)
        (root/'post-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        count={key:len(prefixes) for key,prefixes in sorted(calls.items())}
        if count!=case['calls'] or inventory['words']!=case['compared_words']:
            raise ValueError('reference-host full call/output inventory differs')
        contract=dict(schema='rrfs-complete-work-replay/v1',source_commit=matrix['source_commit'],
            manifest_sha256=sha(manifest_path),calls=count,compared_words=inventory['words'],
            output_records=len(post),scope=case['scope'],
            canonical_bank=str(Path(matrix['lane_root'])/bank.relative_to(lane)),
            capture_host=socket.gethostname(),process_count=1,
            passivity_receipt_sha256=sha(root/'passivity.json'))
        (root/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        result.append(dict(case=case['id'],manifest_sha256=contract['manifest_sha256'],
            calls=sum(count.values()),post_records=len(post),compared_words=inventory['words'],
            observer_passive=True))
        (out/'qualification.json').write_text(json.dumps(dict(source_commit=matrix['source_commit'],
            capture_host=socket.gethostname(),process_count=1,cases=result,
            scope='reference-host native/passivity qualification; CUDA module replay separately required'),indent=2)+'\n')
        print(json.dumps(result[-1]),flush=True)


if __name__=='__main__':main()
