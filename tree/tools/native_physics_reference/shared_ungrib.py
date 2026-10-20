#!/usr/bin/env python3
"""Decode mesh-independent WPS inputs once, with an explicit source manifest.

Native surface authority: mpas_init_atm_surface.F:176-195,211-275 reads
LANDSEA and SST/SKINTEMP/SEAICE. Full-field initial/LBC inputs keep Vtable.GFS.
No weather fields are decoded or transformed by Python.
"""
import argparse
from datetime import datetime,timedelta
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time as clock
from cache_cases import sha,stamp,literal


def surface_vtable(source):
    lines=Path(source).read_text().splitlines()
    selected=[line for line in lines if len(line.split('|'))>5 and line.split('|')[4].strip() in {'SST','SKINTEMP','SEAICE','LANDSEA'}]
    names={line.split('|')[4].strip() for line in selected}
    if not {'SKINTEMP','SEAICE','LANDSEA'}.issubset(names):raise ValueError('surface Vtable misses required native fields')
    return '\n'.join(lines[:3]+selected+[lines[2]])+'\n'


def decode(grib,time,directory,ungrib,vtable,restore_missing=False):
    """One complete time, protected against simultaneous lane consumers."""
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.decode.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=directory/'MANIFEST.json'
        result=json.loads(manifest.read_text()) if manifest.exists() else {'schema':'mpas.shared-ungrib/v1','ungrib_sha256':sha(Path(ungrib)),'vtable_sha256':sha(Path(vtable)),'files':{},'sources':{}}
        if (result['ungrib_sha256'],result['vtable_sha256'])!=(sha(Path(ungrib)),sha(Path(vtable))):raise ValueError('shared decoder pins differ')
        name='GFS:'+stamp(time).strftime('%Y-%m-%d_%H')
        target=directory/name
        expected=result['files'].get(name)
        if expected is not None:
            if target.is_file():
                if sha(target)!=expected:raise ValueError('shared intermediate missing or changed')
                return target
            if not restore_missing:raise ValueError('shared intermediate missing or changed')
            pin=result['sources'][name]
            if sha(Path(grib))!=pin['sha256'] or Path(grib).stat().st_size!=pin['bytes']:
                raise ValueError('restoration raw source differs from original pin')
        source=Path(grib).resolve(strict=True)
        stage=directory/('.pending-'+name.replace(':','_'))
        if stage.exists():
            # The directory lock excludes another decoder. A stage left after
            # interruption is evidence, never an accepted intermediate.
            previous={}
            allowed={'GRIBFILE.AAA','Vtable','namelist.wps','ungrib.log','logfile.log',name}
            for file in stage.iterdir():
                if file.name not in allowed or (not file.is_file() and not file.is_symlink()):
                    raise ValueError('unrecognized interrupted native decoder file')
                previous[file.name]=({'symlink':str(file.readlink())} if file.is_symlink()
                                     else {'bytes':file.stat().st_size,'sha256':sha(file)})
            logs=directory/'logs';logs.mkdir(exist_ok=True)
            (logs/('interrupted-'+name.replace(':','_')+'-'+str(clock.time_ns())+'.json')).write_text(
                json.dumps({'accepted':False,'requested_time':time,'files':previous},indent=2)+'\n')
            shutil.rmtree(stage)
        stage.mkdir(exist_ok=False)
        try:
            (stage/'GRIBFILE.AAA').symlink_to(source)
            (stage/'Vtable').symlink_to(Path(vtable).resolve(strict=True))
            (stage/'namelist.wps').write_text(f"&share\n wrf_core = 'ARW'\n max_dom = 1\n start_date = {literal(time)}\n end_date = {literal(time)}\n interval_seconds = 10800\n/\n&ungrib\n out_format = 'WPS'\n prefix = 'GFS'\n/\n")
            with (stage/'ungrib.log').open('w') as log:subprocess.run([str(Path(ungrib).resolve())],cwd=stage,stdout=log,stderr=subprocess.STDOUT,check=True)
            products=list(stage.glob('GFS:*'))
            if len(products)!=1 or products[0].name!=name:raise ValueError('native decoder did not produce exactly the requested time')
            digest=sha(products[0])
            if expected is not None and digest!=expected:
                raise ValueError('restored native intermediate differs from original pin')
            products[0].rename(target);target.chmod(0o444)
            logs=directory/'logs';logs.mkdir(exist_ok=True)
            shutil.copy2(stage/'ungrib.log',logs/(name.replace(':','_')+'.log'))
            result['files'][name]=digest
            result['sources'][name]={'path':str(source),'bytes':source.stat().st_size,'sha256':sha(source),'time':time,'log_sha256':sha(stage/'ungrib.log')}
            temporary=manifest.with_suffix('.partial');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(manifest)
        finally:shutil.rmtree(stage)
        return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path)
    for key in ('ungrib','vtable','directory'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--surface',action='store_true');a=p.parse_args()
    vtable=a.vtable
    if a.surface:
        a.directory.mkdir(parents=True,exist_ok=True);vtable=a.directory/'Vtable.surface'
        text=surface_vtable(a.vtable)
        if vtable.exists() and vtable.read_text()!=text:raise ValueError('surface table changed')
        vtable.write_text(text)
    for row in json.loads(a.index.read_text())['inputs']:
        output=decode(row['grib'],row['time'],a.directory,a.ungrib,vtable)
        print(json.dumps({'time':row['time'],'intermediate':str(output),'bytes':output.stat().st_size}),flush=True)
    (a.directory/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha(a.directory/'MANIFEST.json'),'input_index_sha256':sha(a.index)})+'\n')
if __name__=='__main__':main()
