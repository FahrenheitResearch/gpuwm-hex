"""Retain strict native bytes for every required CAM forecast proof deck.

Earlier native field digests remain valid evidence; retained bytes are needed
by the actual whole-stream comparator. No reference executable is rebuilt.
Python manages metadata/files and subprocesses, never numerical model fields.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import os
import xml.etree.ElementTree as ET


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    from hexcore.config import parse_fortran_namelist
    plan = json.loads(args.plan.read_text())
    selected = list(enumerate(plan['required_cases']))
    if args.only_case is not None:
        if not 0 <= args.only_case < len(selected):
            raise ValueError('Native case number is outside the required matrix')
        selected = [selected[args.only_case]]
    native = args.build/'atmosphere_model'
    if digest(native) != args.binary_sha256:
        raise ValueError('Pinned strict native executable differs')
    prepared = json.loads(args.prepared.read_text())['records']
    rows = {(row['source'],row['variant'],row['options']['config_camrad_abs_update']):row
            for row in prepared}
    runner = Path(__file__).resolve().parents[2]/'tools/native_physics_reference/run_forecast.py'
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(
        scope='retained strict native 24h CAM proof references; no CUDA identity claim',
        complete=False, process_count=1, host=socket.gethostname(),
        binary_sha256=args.binary_sha256, pin_set_hash=args.pin_set_hash,
        required_plan_sha256=digest(args.plan), orchestration_sha256=digest(Path(__file__)),
        runtime_library_path=os.environ.get('LD_LIBRARY_PATH',''),
        selected_cases=[n for n,_ in selected], cases=[])
    identity = (summary['required_plan_sha256'], summary['binary_sha256'], summary['host'])
    if identity != (digest(args.plan), args.binary_sha256, socket.gethostname()):
        raise ValueError('Native matrix checkpoint plan, binary or host changed')
    if summary['selected_cases'] != [n for n,_ in selected]:
        raise ValueError('Native matrix checkpoint selected cases changed')
    for ordinal, (number, required) in enumerate(selected):
        if ordinal < len(summary['cases']):
            row=summary['cases'][ordinal]
            if row['id'] != required['id'] or digest(Path(row['publication'])) != row['publication_sha256']:
                raise ValueError('Native completed checkpoint changed')
            continue
        key=(required['source'],required['variant'],required['options']['config_camrad_abs_update'])
        source=Path(rows[key]['namelist']).resolve()
        for path,name in ((source,'namelist'),(source.with_name('streams.atmosphere'),'streams')):
            if digest(path) != required[name+'_sha256']:
                raise ValueError('Prepared native scientific input differs')
        root=args.output/str(number)
        deck=root/'deck'
        deck.mkdir(parents=True,exist_ok=False)
        for name in ('namelist.atmosphere','streams.atmosphere'):
            shutil.copyfile(source.with_name(name),deck/name)
        groups=parse_fortran_namelist(source.read_text())
        options={name:value for group in groups.values() for name,value in group.items()}
        parts=required['source'].split('/')
        case=args.cases/parts[1]/parts[2]/'init-modis20-v2'
        streams=ET.parse(deck/'streams.atmosphere').getroot()
        inputs=[stream for stream in streams if stream.get('name')=='input']
        if len(inputs)!=1:raise ValueError('Native deck lacks one input stream')
        runtime=dict(mesh=parts[1],start=options['config_start_time'],
                     duration=options['config_run_duration'],dt=options['config_dt'],
                     init=inputs[0].get('filename_template'))
        (root/'INDEX.json').write_text(json.dumps(dict(runtime=runtime),indent=2)+'\n')
        output=root/'native'
        command=[sys.executable,str(runner),'--build',str(args.build.resolve()),
                 '--tables',str(args.tables.resolve()),'--case',str(case.resolve()),
                 '--deck',str(deck.resolve()),'--out',str(output.resolve()),
                 '--mpi',str(args.mpi.resolve()),'--ranks','1','--pin-set-hash',args.pin_set_hash]
        with (root/'run.log').open('w') as log:
            executed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if executed.returncode:
            (root/'failure.json').write_text(json.dumps(dict(id=required['id'],
                process_count=1,host=socket.gethostname(),returncode=executed.returncode,
                command=command,run_log_sha256=digest(root/'run.log'),
                runtime_library_path=os.environ.get('LD_LIBRARY_PATH','')),indent=2)+'\n')
            raise ValueError('Native whole deck failed '+required['id']+'; log '+str(root/'run.log'))
        manifest=json.loads((output/'MANIFEST.json').read_text())
        if (manifest['ranks']!=1 or manifest['binary_sha256']!=args.binary_sha256
                or manifest['runtime']!=runtime or not manifest['outputs']):
            raise ValueError('Native complete forecast manifest differs')
        files={name:digest(output/name) for name in
               ['MANIFEST.json','namelist.atmosphere','streams.atmosphere']
               + sorted(manifest['outputs'])
               + [p.name for p in sorted(output.glob('log.atmosphere.*.out'))]}
        publication=output/'PUBLICATION.json'
        publication.write_text(json.dumps(dict(schema='mpas.native-cache-publication/v1',
            description='owned pinned strict CAM 24h native proof reference',
            files=files),indent=2)+'\n')
        summary['cases'].append(dict(id=required['id'],number=number,
            process_count=1,host=socket.gethostname(),runtime=runtime,
            publication=str(publication.resolve()),publication_sha256=digest(publication),
            files=len(manifest['outputs']),command=command,returncode=executed.returncode,
            native_log_sha256=digest(output/'log.atmosphere.0000.out'),
            cuda_forecast_identity=False))
        summary['complete']=len(summary['cases'])==len(selected)
        summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        print(required['id'],len(manifest['outputs']),'strict native files retained',flush=True)
    return 0 if summary['complete'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','build','tables','cases','mpi','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--binary-sha256',required=True)
    parser.add_argument('--pin-set-hash',required=True)
    parser.add_argument('--only-case',type=int,help='one whole deck, copied back before another native run')
    args=parser.parse_args()
    raise SystemExit(run(args))
