"""Actual native 12h stop/restart controls; mpas_atm_core.F:109-150,894-902.

Run the published strict executable at exactly one process. Native endpoints
are authenticated original f0 files. Rust owns file decoding and word grading.
The native-pool adapter's restart is consumed by native itself. This control
supports, and does not replace, the computed hex restart forecast proof.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from hexcore.config import parse_fortran_namelist
from hexcore.mpasphys import resolve_namelist
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import RegistryLayout
from hexcore.restart_v842 import NativePoolStreams, read_stream, write_restart


def digest(path):
    """Receipt provenance only; no weather-field conversion."""
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def compare(args, label, directory, endpoint, step, *, strict_directory=None):
    """Native stream comparison; mpas_io_streams.F:2430-3251."""
    publication=json.loads((args.reference/'PUBLICATION.json').read_text())
    files=sorted(args.reference.glob('*.'+endpoint.strftime('%Y-%m-%d_%H.%M.%S')+'.nc'))
    assert files, 'authenticated native endpoint absent'
    templates={s.attrib['name']:s.attrib['filename_template']
               for s in ET.parse(args.reference/'streams.atmosphere').getroot()
               if 'filename_template' in s.attrib}
    samples=[]
    for path in files:
        assert digest(path)==publication['files'][path.name],path.name
        stream=path.name.split('.')[0]
        strict=path if strict_directory is None else strict_directory/path.name
        sample=dict(stream=stream,strict=str(strict),hex=str(directory/path.name),
                    steps=[step],times=[endpoint.strftime('%Y-%m-%d_%H:%M:%S')])
        if stream=='surface':
            sample['clock_witness']=dict(filename_template=templates[stream])
        samples.append(sample)
    request=args.receipts/(label+'-request.json')
    request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1',samples=samples),indent=2)+'\n')
    output=args.receipts/(label+'-words.json')
    subprocess.run([str(args.comparator),'--request',str(request),'--out',str(output)],check=True)
    raw=json.loads(output.read_text());rows=raw['fields']
    bad=[r for r in rows if r['status']!='compared' or r['differing_words']]
    result=dict(label=label,process_count=1,endpoint=endpoint.isoformat(),fields=len(rows),
        compared_words=sum(r['compared_words'] for r in rows if r['status']=='compared'),
        differing_words=sum(r['differing_words'] for r in rows if r['status']=='compared'),
        incompatible_samples=sum(r['status']!='compared' for r in rows),
        passed=not bad,first_difference=bad[0] if bad else None,receipt_sha256=digest(output))
    print(json.dumps(result),flush=True)
    return result


def reencode(args, source, target, start, text):
    """Whole actual native pool restore; mpas_atm_core.F:129-150,894-902."""
    with read_stream(source,scratch=args.scratch/'metadata',select=('xtime',)) as metadata:
        dimensions=dict(metadata.dimensions);attrs=dict(metadata.global_attrs)
    layout=RegistryLayout(args.source,resolve_namelist(text),text,dimensions)
    layout.configs.update(config_start_time=start.strftime('%Y-%m-%d_%H:%M:%S'),config_do_restart=True)
    layout.configs['config_do_DAcycling']=args.da_cycling
    with NativePhysics(args.library,test_only=True) as native:
        for row in layout.fields:
            for level in range(1,row['nlevels']+1):
                if row['kind']=='character':
                    native.put_text_field(row['pool'],row['pool_name'],'',level=level,nlevels=row['nlevels'])
                else:
                    native.put_field(row['pool'],row['pool_name'],
                        np.zeros(row['shape'],dtype=np.float32 if row['kind']=='real' else np.int32),
                        level=level,nlevels=row['nlevels'])
        adapter=NativePoolStreams(native,layout,args.reference/'streams.atmosphere',args.scratch/'pool-output',
                                  list_directory=args.native,global_attrs=attrs)
        restored=adapter.restore(source,scratch=args.scratch/'decode')
        stream=adapter.streams['restart']
        write_restart(target,adapter.mesh,adapter.fields(stream),start,initial_time=adapter.initial,
            stream=stream,streams=adapter.streams,active_packages=layout.packages,
            global_attrs=attrs,field_attrs=adapter.field_attrs,scratch=args.scratch/'encode')
        return dict(fields=len(restored),source_sha256=digest(source),output_sha256=digest(target),
                    original_initial_time=adapter.initial.isoformat(),serial_process_count=1)


def compare_continuations(args, strict, candidate, start, dt):
    """Every native continuation output; mpas_io_streams.F:2430-3251.

    These are actual generated native files from the preceding control arm,
    authenticated by the retained run and per-file hashes. Rust owns decoding,
    clock validation and every word comparison. No straight-run claim follows.
    """
    templates={s.attrib['name']:s.attrib['filename_template']
               for s in ET.parse(args.reference/'streams.atmosphere').getroot()
               if 'filename_template' in s.attrib}
    samples=[];inventory={}
    for path in sorted(strict.glob('*.nc')):
        stream=path.name.split('.')[0]
        assert stream in templates,'unexpected actual native output stream'
        stamp=datetime.strptime(path.name[len(stream)+1:-3],'%Y-%m-%d_%H.%M.%S')
        elapsed=(stamp-start).total_seconds()
        assert elapsed>=0 and elapsed/dt==int(elapsed/dt),'native output clock not on step'
        sample=dict(stream=stream,strict=str(path),hex=str(candidate/path.name),
                    steps=[int(elapsed/dt)],times=[stamp.strftime('%Y-%m-%d_%H:%M:%S')])
        if stream=='surface':
            sample['clock_witness']=dict(filename_template=templates[stream])
        samples.append(sample);inventory[path.name]=digest(path)
    assert samples,'no actual native continuation outputs retained'
    request=args.receipts/'continuation-pair-request.json'
    request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1',samples=samples),indent=2)+'\n')
    output=args.receipts/'continuation-pair-words.json'
    subprocess.run([str(args.comparator),'--request',str(request),'--out',str(output)],check=True)
    rows=json.loads(output.read_text())['fields']
    result=dict(process_count=1,samples=len(samples),field_samples=len(rows),
                compared_words=sum(r['compared_words'] for r in rows if r['status']=='compared'),
                differing_words=sum(r['differing_words'] for r in rows if r['status']=='compared'),
                incompatible_samples=sum(r['status']!='compared' for r in rows),
                passed=all(r['status']=='compared' and not r['differing_words'] for r in rows),
                actual_native_outputs_sha256=inventory,receipt_sha256=digest(output))
    print(json.dumps(result),flush=True)
    return result


def main():
    """Complete native driver experiment; mpas_atm_core.F:109-150,589-701."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','native','tables','source','library','comparator','mpi','scratch','receipts'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--da-cycling',action='store_true',
                        help='exercise native coupled-diagnostic initialization on restart')
    args=parser.parse_args()
    assert os.access(args.comparator,os.X_OK),'word comparator must be executable'
    args.scratch=args.scratch.resolve();args.receipts=args.receipts.resolve()
    assert not args.scratch.exists(),'run scratch must be fresh'
    args.scratch.mkdir(parents=True);args.receipts.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.reference/'MANIFEST.json').read_text())
    publication=json.loads((args.reference/'PUBLICATION.json').read_text())
    assert manifest['ranks']==1
    assert digest(args.native/'atmosphere_model')==manifest['binary_sha256']
    text=(args.reference/'namelist.atmosphere').read_text()
    assert digest(args.reference/'namelist.atmosphere')==publication['files']['namelist.atmosphere']
    assert digest(args.reference/'streams.atmosphere')==publication['files']['streams.atmosphere']
    configs={k:v for g in parse_fortran_namelist(text).values() for k,v in g.items()}
    start=datetime.strptime(configs['config_start_time'],'%Y-%m-%d_%H:%M:%S')
    duration=timedelta(days=1);half=start+duration/2;end=start+duration
    assert configs['config_run_duration']=='1_00:00:00','control needs actual 24h deck'
    half_text,count=re.subn(r"(config_run_duration\s*=\s*)'[^']*'",r"\1'0_12:00:00'",text)
    assert count==1
    resumed,count=re.subn(r"(config_start_time\s*=\s*)'[^']*'",
                         lambda m:m.group(1)+repr(half.strftime('%Y-%m-%d_%H:%M:%S')),half_text)
    assert count==1
    assert 'config_do_restart' not in configs,'original control already restarts'
    resumed+='\n&restart\nconfig_do_restart=.true.\nconfig_do_DAcycling='+(
        '.true.' if args.da_cycling else '.false.')+'\n/\n'
    summary=dict(schema='o1-native-physics-restart-control-v1',process_count=1,host=socket.gethostname(),
        suite=configs['config_physics_suite'],config_do_DAcycling=args.da_cycling,
        native_binary_sha256=digest(args.native/'atmosphere_model'),
        publication_sha256=digest(args.reference/'PUBLICATION.json'),
        orchestration_sha256=digest(__file__),
        stream_adapter_sha256=digest(Path(__file__).parent.parent/'src/hexcore/restart_v842.py'),scope='Actual native forecast controls; no hex-computed forecast claim',
        cases=[],passed=False)
    restart_name='restart.'+half.strftime('%Y-%m-%d_%H.%M.%S')+'.nc'
    saved=args.scratch/'native-restart.nc';encoded=args.scratch/'pool-restart.nc'
    resumed_endpoint=args.scratch/'native-resumed-endpoint';resumed_endpoint.mkdir()
    try:
        for label in ('half','resumed-native','resumed-pool'):
            run=args.scratch/label;run.mkdir()
            (run/'namelist.atmosphere').write_text(half_text if label=='half' else resumed)
            shutil.copyfile(args.reference/'streams.atmosphere',run/'streams.atmosphere')
            for root in (args.native,args.tables):
                for asset in root.iterdir():
                    if asset.is_file() and (asset.suffix in ('.TBL','.DBL') or
                       asset.name.startswith(('RRTMG_','CCN_','stream_list.'))):
                        target=run/asset.name
                        if not target.exists():target.symlink_to(asset.resolve())
            # The original publication pins every runtime input; no synthetic init.
            for name in manifest['inputs']:
                if name.endswith(('.nc','.graph.info')):
                    candidates=list((args.source.parent/'cases').rglob(name))
                    candidates=[p for p in candidates if digest(p)==manifest['inputs'][name]]
                    assert candidates,'published runtime input unavailable: '+name
                    (run/name).symlink_to(candidates[0].resolve())
            if label!='half':
                (run/restart_name).symlink_to(saved if label=='resumed-native' else encoded)
                (run/'restart_timestamp').write_text(half.strftime('%Y-%m-%d_%H:%M:%S')+'\n')
            log=args.receipts/(label+'-run.log')
            with log.open('w') as output:
                result=subprocess.run([str(args.mpi),'-n','1',str(args.native/'atmosphere_model')],
                    cwd=run,stdout=output,stderr=subprocess.STDOUT)
            for model_log in run.glob('log.atmosphere.*'):
                shutil.copyfile(model_log,args.receipts/(label+'-'+model_log.name))
            if result.returncode:
                summary['cases'].append(dict(label=label,passed=False,exit_code=result.returncode,log_sha256=digest(log)))
                raise RuntimeError('actual native forecast failed: '+label)
            endpoint=half if label=='half' else end
            grade=compare(args,label,run,endpoint,int((endpoint-start).total_seconds()/configs['config_dt']))
            grade.update(log_sha256=digest(log),namelist_sha256=digest(run/'namelist.atmosphere'))
            summary['cases'].append(grade)
            if label=='resumed-native':
                files=[p for p in run.glob('*.nc') if not p.is_symlink()]
                summary['resumed_native_endpoint_sha256']={p.name:digest(p) for p in files}
                for path in files:shutil.copyfile(path,resumed_endpoint/path.name)
            elif label=='resumed-pool':
                summary['native_consumption_equivalence']=compare(args,'pool-versus-native',run,end,
                    int((end-start).total_seconds()/configs['config_dt']),strict_directory=resumed_endpoint)
                summary['native_consumption_full_forecast_equivalence']=compare_continuations(
                    args,resumed_endpoint,run,start,configs['config_dt'])
            if label=='half':
                shutil.copyfile(run/restart_name,saved)
                summary['pool_reencode']=reencode(args,saved,encoded,half,text)
            shutil.rmtree(run)
        summary['passed']=(all(x['passed'] for x in summary['cases']) and len(summary['cases'])==3
                           and summary['native_consumption_full_forecast_equivalence']['passed'])
    except Exception as error:
        summary['failure']=str(error)
    finally:
        (args.receipts/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        shutil.rmtree(args.scratch)
    print(json.dumps(summary),flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
