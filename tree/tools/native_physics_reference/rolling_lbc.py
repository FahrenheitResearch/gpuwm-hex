#!/usr/bin/env python3
"""Generate native case9 clocks with shared reusable native full-field intermediates.

Native mpas_init_atm_cases.F:301-364 dispatches interpolation independently at
its current clock; :5434-6218 computes LBC fields from that time's full inputs.
Each invocation has its own start/stop clock. Native Time is therefore zero
in each file; xtime is the requested valid time. No field is rewritten.
"""
import argparse,json,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import timedelta
from pathlib import Path
from cache_cases import prepare,sha,stamp
from shared_ungrib import decode
from publish_cache import publish
from matrix_support import discard_copied_outputs,published
from verify_acceptance import validate as validate_acceptance
from frozen_download import fetch


def execute(row,work,shared,raw_manifest,init,ungrib,vtable,geog,mpi,ready):
    out=work/row['mesh']/row['start'].replace(':','.')
    existing=published(Path(row['publish']),out)
    if existing:
        manifest=json.loads((Path(row['publish'])/'MANIFEST.json').read_text())
        if manifest['case']!=row or manifest['init_sha256']!=sha(Path(init)):
            raise ValueError('completed native LBC contract differs')
        return existing
    out.mkdir(parents=True,exist_ok=True)
    pins={p['name']:p for p in raw_manifest['files']};segments=[];outputs={};inputs={};intermediates={}
    for i,original in enumerate(row['gribs']):
        time=(stamp(row['start'])+timedelta(seconds=i*row['interval_seconds'])).strftime('%Y-%m-%d_%H:%M:%S')
        segment=row.copy();segment.update(start=time,stop=time,surface_stop=time,shared_ungrib=str(shared),gribs=[original])
        native=work/'segments'/row['start'].replace(':','.')/row['mesh']/time.replace(':','.')
        if (native/'MANIFEST.json').is_file():
            m=json.loads((native/'MANIFEST.json').read_text())
        else:
            pin=pins[Path(original).name];grib=shared/(pin['name']+'.download')
            shared.mkdir(parents=True,exist_ok=True)
            if i==0 and not (shared/'MANIFEST.json').exists():
                analysis=Path(row['analysis'])
                source=next(analysis.glob('GFS:*'));(shared/source.name).symlink_to(source)
                initial={'schema':'mpas.shared-ungrib/v1','ungrib_sha256':sha(Path(ungrib)),
                         'vtable_sha256':sha(Path(vtable)),'files':{source.name:sha(source)},
                         'sources':{source.name:{**pin,'publication_sha256':sha(analysis/'PUBLICATION.json')}}}
                (shared/'MANIFEST.json').write_text(json.dumps(initial,indent=2)+'\n')
            if i:
                transport=fetch(pin,grib,work/'transport')
                (shared/(pin['name']+'.transport.json')).write_text(json.dumps(transport,indent=2)+'\n')
                if grib.stat().st_size!=pin['bytes'] or sha(grib)!=pin['sha256']:raise ValueError('downloaded LBC input differs from frozen raw pin')
                decode(grib,time,shared,ungrib,vtable)
            native=prepare(segment,work/'segments'/row['start'].replace(':','.'),init,ungrib,vtable,geog,mpi)
            m=json.loads((native/'MANIFEST.json').read_text())
            if grib.exists():grib.unlink()
        segments.append({'time':time,'native_manifest':m})
        for name,digest in m['files'].items():
            destination=out/name
            if destination.exists():
                if sha(destination)!=digest:raise ValueError('completed LBC output changed')
            else:(native/name).rename(destination)
            outputs[name]=digest
        for name,digest in m['intermediates'].items():
            intermediates[name]=digest
            # Other meshes and Rust replay consume the exact shared native bytes.
            # Retain these, while encoded GRIB bytes retire after decoding.
        inputs.update({str(Path(pin['name'])):pin['sha256'] for pin in pins.values() if pin['name']==Path(original).name})
        # A durable per-clock checkpoint survives restart after raw retirement.
        (out/'SEGMENTS.json').write_text(json.dumps(segments,indent=2)+'\n')
        print(json.dumps({'completed_lbc_time':time,'files':len(outputs)}),flush=True)
    manifest={'schema':'mpas.native-input-cache/v1','case':row,'init_sha256':sha(Path(init)),
              'geography_indices':{},'inputs':inputs,'files':outputs,'intermediates':intermediates,
              'decks':{},'native_clock_mode':'independent single-time case9; native Time=0; actual xtime retained',
              'native_segments_sha256':sha(out/'SEGMENTS.json')}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    publication=publish(out,Path(row['publish']),['lbc.*.nc','MANIFEST.json','SEGMENTS.json'],ready,row['description']+'; independent native case9 clocks, rolling full-field input')
    discard_copied_outputs(out,Path(row['publish']),outputs)
    return publication


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path)
    for key in ['work','shared','raw_manifest','init','ungrib','vtable','geog','ready']:p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--mpi',required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--acceptance-index',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2],default=1,
                   help='at most two distinct dates; one full intermediate per date')
    a=p.parse_args();results=[]
    checks=json.loads(a.acceptance_index.read_text())['checks']
    if not checks:raise ValueError('no acceptance receipts')
    for check in checks:
        errors=validate_acceptance(json.loads(Path(check['receipt']).read_text()),check['hours'],check['cells'],check['levels'],check['init_sha256'])
        if errors:raise ValueError(errors)
    pins=json.loads(a.raw_manifest.read_text())
    rows=json.loads(a.index.read_text())['cases']
    if not rows or len({row['start'] for row in rows})!=len(rows):
        raise ValueError('one rolling native LBC case per distinct date required')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(execute,row,a.work.resolve(),a.shared.resolve()/row['start'].replace(':','.'),pins,a.init,a.ungrib,a.vtable,a.geog,a.mpi,a.ready):row for row in rows}
        for future in as_completed(futures):
            row=futures[future];result=future.result()
            results.append({'case':row,'publication':result})
            a.receipt.write_text(json.dumps({'complete':len(results)==len(rows),'cases':results},indent=2)+'\n')
if __name__=='__main__':main()
