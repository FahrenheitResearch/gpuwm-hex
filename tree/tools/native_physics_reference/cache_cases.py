#!/usr/bin/env python3
"""Offline ungrib + native case 7/8/9 orchestration from an explicit input index.

Input JSON: {"cases": [{"mesh": str, "grid": path, "graph": path,
"start": "YYYY-MM-DD_HH:MM:SS", "stop": ..., "gribs": [paths],
"interval_seconds": 21600, "kind": "global"|"limited_area",
"levels": 55, "fg_levels": int, "surface_stop": ...}]}
No downloading, Python field transforms, or synthetic substitution.
MPAS authority: core_init_atmosphere/Registry.xml:56-84,145-162,289-319.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from generate_decks import literal
from verify_acceptance import validate as validate_acceptance
from native_completion import completed


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def run(argv,cwd,log):
    with (cwd/log).open('w') as f:
        subprocess.run(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)

def link(src,dst):
    src=Path(src).resolve(strict=True)
    if dst.exists() or dst.is_symlink():raise ValueError(f'refusing existing {dst}')
    dst.symlink_to(src)

def stamp(value):
    return datetime.strptime(value,'%Y-%m-%d_%H:%M:%S')

def contract(row):
    result={key:value for key,value in row.items() if key not in ('publish','description')}
    result['ranks']=row.get('ranks',1)
    return result

def validate_geography(geog):
    # Native static.F:397-403 selects this 20-class dataset. With-lakes data
    # introduces vegetation21, rejected by Noah module_sf_noahlsm.F:2441.
    index=Path(geog)/'modis_landuse_20class_30s/index'
    values={}
    for line in index.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key,value=line.split('=',1);values[key.strip()]=value.strip().strip('"')
    if values.get('category_min')!='1' or values.get('category_max')!='20' or values.get('iswater')!='17':
        raise ValueError('native MODIS input requires the actual 20-class dataset; with-lakes category21 aborts Noah')
    if values.get('mminlu')!='MODIFIED_IGBP_MODIS_NOAH':
        raise ValueError('native MODIS classification differs')
    return {str(p.resolve()):sha(p) for p in sorted(Path(geog).glob('*/index'))}

def validate(row,init,ungrib,vtable,geog):
    if row['kind'] not in ('global','limited_area'):raise ValueError('bad case kind')
    interval=row['interval_seconds']
    if interval not in (10800,21600):raise ValueError('expected requested 3 h/6 h inputs')
    start=stamp(row['start'])
    for key in ('stop','surface_stop'):
        span=(stamp(row[key])-start).total_seconds()
        if span<0 or span%interval:raise ValueError(f'{key} must align with input times')
    if row['levels']<1 or row['fg_levels']<1:raise ValueError('positive level counts required')
    required=[row['grid'],row['graph'],init,ungrib,vtable]
    if row.get('shared_ungrib'):
        directory=Path(row['shared_ungrib'])
        required.extend([directory/'MANIFEST.json',*shared_intermediates(row)])
    else:
        required.extend(row['gribs'])
    ranks=row.get('ranks',1)
    if not isinstance(ranks,int) or ranks<1:raise ValueError('positive native init rank count required')
    if ranks>1:required.append(row['graph']+f'.part.{ranks}')
    stages=row.get('stages',[7,8] if row['kind']=='global' else [9])
    if stages not in ([7],[8],[7,8],[9]) or (row['kind']=='global')==(stages==[9]):
        raise ValueError('invalid native initialization stages')
    if stages==[8]:required.append(row['initial'])
    for name in required:
        if not Path(name).is_file():raise FileNotFoundError(name)
    if not Path(geog).is_dir():raise FileNotFoundError(geog)
    if 7 in stages:validate_geography(geog)
    if not row['gribs']:raise ValueError('no GFS GRIB inputs')
    return required

def shared_intermediates(row):
    """Bind native inputs to completed shared decoding, never a pending file."""
    directory=Path(row['shared_ungrib'])
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    if manifest.get('schema')!='mpas.shared-ungrib/v1':raise ValueError('unsupported shared ungrib manifest')
    last=max(stamp(row['stop']),stamp(row['surface_stop']))
    start=stamp(row['start']);interval=row['interval_seconds']
    names=['GFS:'+(start+timedelta(seconds=i*interval)).strftime('%Y-%m-%d_%H')
           for i in range(int((last-start).total_seconds()/interval)+1)]
    paths=[]
    for name in names:
        path=directory/name
        if name not in manifest['files'] or not path.is_file():raise FileNotFoundError(path)
        if sha(path)!=manifest['files'][name]:raise ValueError('shared native intermediate changed: '+name)
        if name not in manifest.get('sources',{}):raise ValueError('shared native source provenance absent')
        paths.append(path)
    return paths

def prepare(row,work,init,ungrib,vtable,geog,mpi):
    mesh=row['mesh'];start=row['start'];stop=row['stop']
    interval=row['interval_seconds'];kind=row['kind']
    # Every supplied input is checked before ungrib or native jobs start.
    required=validate(row,init,ungrib,vtable,geog)
    out=work/mesh/start.replace(':','.')
    out.mkdir(parents=True,exist_ok=False)
    if row.get('shared_ungrib'):
        intermediate=[]
        for source in shared_intermediates(row):
            link(source,out/source.name);intermediate.append(out/source.name)
        (out/'ungrib.log').write_text('Reused completed mesh-independent native decoding: '+str(Path(row['shared_ungrib']).resolve())+'\n')
    else:
        intermediate=decode_local(row,out,ungrib,vtable)
    link(row['grid'],out/(mesh+'.grid.nc'));link(row['graph'],out/(mesh+'.graph.info'))
    if row.get('ranks',1)>1:
        link(row['graph']+f'.part.{row["ranks"]}',out/(mesh+f'.graph.info.part.{row["ranks"]}'))
    link(init,out/'init_atmosphere_model')
    stages=row.get('stages',[7,8] if kind=='global' else [9])
    if stages==[8]:link(row['initial'],out/(mesh+'.init.nc'))
    return execute_native(row,out,required,intermediate,init,geog,mpi)

def decode_local(row,out,ungrib,vtable):
    """Compatibility door; campaign case matrices use shared_ungrib."""
    start=row['start'];stop=row['stop'];interval=row['interval_seconds']
    link(ungrib,out/'ungrib.exe');link(vtable,out/'Vtable')
    for i,g in enumerate(row['gribs']):
        if i>=26**3:raise ValueError('too many ungrib inputs')
        suffix=''.join(chr(65+(i//(26**p))%26) for p in (2,1,0))
        link(g,out/('GRIBFILE.'+suffix))
    surface_stop=row['surface_stop']
    last=max(stamp(stop),stamp(surface_stop)).strftime('%Y-%m-%d_%H:%M:%S')
    (out/'namelist.wps').write_text('&share\n wrf_core = '+literal('ARW')+'\n max_dom = 1\n'
        f' start_date = {literal(start)}\n end_date = {literal(last)}\n interval_seconds = {interval}\n/\n'
        "&ungrib\n out_format = 'WPS'\n prefix = 'GFS'\n/\n")
    run([str((out/'ungrib.exe').resolve())],out,'ungrib.log')
    expected={('GFS:'+(stamp(start)+timedelta(seconds=i*interval)).strftime('%Y-%m-%d_%H'))
              for i in range(int((stamp(last)-stamp(start)).total_seconds()/interval)+1)}
    intermediate=list(out.glob('GFS:*'))
    actual={p.name for p in intermediate}
    if actual!=expected:raise ValueError(f'ungrib time mismatch: missing {sorted(expected-actual)}, extra {sorted(actual-expected)}')
    return intermediate

def streams_for_case(mesh,case,input_name,interval):
    # Native xml_stream_parser.c:482-492 validates names before package activation;
    # even an inactive output must not alias the case8 initialization input.
    streams=ET.Element('streams')
    for name,filename,package,output_interval in (
        ('input',input_name,None,None),('output',mesh+('.init.nc' if case==7 else '.initial_output.nc'),'initial_conds','initial_only' if case==7 else 'none'),
        ('ugwp_oro_data',mesh+'.ugwp_oro_data.nc','gwd_gsl_stage_out','initial_only' if case==7 else 'none'),
        ('surface',mesh+'.sfc_update.nc','sfc_update',str(interval) if case==8 else 'none'),
        ('lbc','lbc.$Y-$M-$D_$h.$m.$s.nc','lbcs',str(interval) if case==9 else 'none')):
        attrs={'name':name,'type':'input' if name=='input' else 'output','filename_template':filename}
        if name=='input':attrs['input_interval']='initial_only'
        else:attrs.update(output_interval=output_interval,packages=package)
        if name=='surface':attrs['filename_interval']='none'
        if name=='lbc':attrs['filename_interval']='output_interval'
        ET.SubElement(streams,'immutable_stream',attrs)
    return streams

def execute_native(row,out,required,intermediate,init,geog,mpi):
    """Execute native stages, also usable after a witnessed completed ungrib."""
    mesh=row['mesh'];start=row['start'];stop=row['stop'];surface_stop=row['surface_stop']
    interval=row['interval_seconds'];kind=row['kind']
    stages=row.get('stages',[7,8] if kind=='global' else [9])
    for case in stages:
        if case==8:input_name=mesh+'.init.nc';case_stop=surface_stop
        else:input_name=mesh+'.grid.nc';case_stop=start if case==7 else stop
        settings=(f'&nhyd_model\n config_init_case = {case}\n config_start_time = {literal(start)}\n config_stop_time = {literal(case_stop)}\n/\n'
                  f'&dimensions\n config_nvertlevels = {row["levels"]}\n config_nsoillevels = 4\n config_nfglevels = {row["fg_levels"]}\n config_nfgsoillevels = 4\n/\n'
                  f'&data_sources\n config_geog_data_path = {literal(str(Path(geog).resolve()))}\n config_met_prefix = \'GFS\'\n config_sfc_prefix = \'GFS\'\n config_fg_interval = {interval}\n config_use_spechumd = false\n/\n'
                  '&vertical_grid\n config_ztop = 30000.0\n config_nsmterrain = 1\n config_smooth_surfaces = true\n config_dzmin = 0.3\n config_nsm = 30\n config_tc_vertical_grid = true\n/\n'
                  '&preproc_stages\n'+f' config_static_interp = {str(case==7).lower()}\n config_native_gwd_static = {str(case==7).lower()}\n config_native_gwd_gsl_static = {str(case==7).lower()}\n config_vertical_grid = {str(case==7).lower()}\n config_met_interp = {str(case!=8).lower()}\n config_input_sst = true\n config_frac_seaice = true\n/\n'
                  '&io\n config_pio_num_iotasks = 0\n config_pio_stride = 1\n/\n'
                  f'&decomposition\n config_block_decomp_file_prefix = {literal(mesh+".graph.info.part.")}\n/\n')
        (out/'namelist.init_atmosphere').write_text(settings)
        ET.ElementTree(streams_for_case(mesh,case,input_name,interval)).write(out/'streams.init_atmosphere',encoding='unicode')
        shutil.copy2(out/'namelist.init_atmosphere',out/f'case-{case}.namelist')
        shutil.copy2(out/'streams.init_atmosphere',out/f'case-{case}.streams')
        run([mpi,'-np',str(row.get('ranks',1)),str((out/'init_atmosphere_model').resolve())],out,f'case-{case}.log')
        log=out/'log.init_atmosphere.0000.out'
        if not log.exists() or not completed(log.read_text(),'init_atmosphere'):
            raise ValueError(f'case {case} did not complete')
        shutil.copy2(log,out/f'case-{case}.model.log')
    products=[p for p in out.glob('*.nc') if not p.is_symlink()]
    if not products:raise ValueError('no native products')
    needed=set()
    if 7 in stages:needed.update({mesh+'.init.nc',mesh+'.ugwp_oro_data.nc'})
    if 8 in stages:needed.add(mesh+'.sfc_update.nc')
    if 9 in stages and not list(out.glob('lbc.*.nc')):raise ValueError('no native LBC products')
    if not needed.issubset({p.name for p in products}):raise ValueError('native input products are incomplete')
    manifest={'schema':'mpas.native-input-cache/v1','case':row,'init_sha256':sha(Path(init)),
              'geography_indices':validate_geography(geog) if 7 in stages else {},
              'inputs':{str(Path(p).resolve()):sha(Path(p)) for p in required},
              'files':{p.name:sha(p) for p in sorted(products)},
              'intermediates':{p.name:sha(p) for p in sorted(intermediate)},
              'decks':{p.name:sha(p) for p in sorted(out.glob('case-*'))},
              'directory_files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    if row.get('shared_ungrib'):
        manifest['shared_ungrib_manifest']=json.loads((Path(row['shared_ungrib'])/'MANIFEST.json').read_text())
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index',type=Path);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--acceptance-index',type=Path,required=True,
                   help='JSON checks list: receipt, hours, cells, levels, init_sha256')
    for name in ('init','ungrib','vtable','geog','mpi'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    checks=json.loads(a.acceptance_index.read_text())['checks']
    if not checks:raise ValueError('no dycore acceptance checks')
    for check in checks:
        errors=validate_acceptance(json.loads(Path(check['receipt']).read_text()),check['hours'],
                                   check['cells'],check['levels'],check['init_sha256'])
        if errors:raise ValueError(f'dycore acceptance failed: {errors}')
    rows=json.loads(a.index.read_text())['cases']
    if not rows:raise ValueError('empty case index')
    for row in rows:validate(row,a.init,a.ungrib,a.vtable,a.geog)
    for row in rows:
        print(prepare(row,a.work.resolve(),a.init,a.ungrib,a.vtable,a.geog,a.mpi))
if __name__=='__main__':main()
