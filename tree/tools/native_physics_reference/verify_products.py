#!/usr/bin/env python3
"""Witness complete native output file sets and stored clocks through Rust.

Hashes and clocks are receipts, not a forecast equivalence claim. Python plans
expected times; the Rust comparator inventory door decodes native NetCDF clocks.
Inactive compiled packages must be named explicitly by the caller.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from generate_decks import interval_seconds
from publish_cache import sha


def sequence(start, stop, interval):
    first=datetime.strptime(start,'%Y-%m-%d_%H:%M:%S')
    last=datetime.strptime(stop,'%Y-%m-%d_%H:%M:%S')
    span=int((last-first).total_seconds())
    if interval<=0 or span<0 or span%interval:raise ValueError('unaligned native cache clocks')
    return [(first+timedelta(seconds=i)).strftime('%Y-%m-%d_%H:%M:%S')
            for i in range(0,span+1,interval)]

def filename(template, time):
    stamp=datetime.strptime(time,'%Y-%m-%d_%H:%M:%S')
    for macro,fmt in [('Y','%Y'),('M','%m'),('D','%d'),('h','%H'),('m','%M'),('s','%S')]:
        template=template.replace('$'+macro,stamp.strftime(fmt))
    if '$' in template:raise ValueError('unsupported native filename macro')
    return template

def expected_forecast(runtime, streams, inactive):
    start=datetime.strptime(runtime['start'],'%Y-%m-%d_%H:%M:%S')
    stop=(start+timedelta(seconds=int(interval_seconds(runtime['duration'])))).strftime('%Y-%m-%d_%H:%M:%S')
    result={}
    for stream in streams:
        if 'output' not in stream.get('type',''):continue
        packages=stream.get('packages','').split(';') if stream.get('packages') else []
        if packages and all(package in inactive for package in packages):continue
        times=sequence(runtime['start'],stop,int(interval_seconds(stream.get('output_interval'))))
        # Native restart output begins after startup: the initial alarm is consumed
        # by its input read. Surface input_interval=none retains startup output.
        if 'input' in stream.get('type','') and stream.get('input_interval')=='initial_only':times=times[1:]
        for time in times:
            result.setdefault(filename(stream.get('filename_template'),time),[]).append(time)
    if not result:raise ValueError('no expected native output streams')
    return result

def verify(directory, comparator, inactive):
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    static_files=set();level_counts={};ranges={}
    if manifest['schema']=='mpas.native-forecast-cache/v1':
        actual=manifest['outputs']
        expected=expected_forecast(manifest['runtime'],ET.parse(directory/'streams.atmosphere').getroot(),inactive)
    elif manifest['schema']=='mpas.native-input-cache/v1':
        actual=manifest['files'];row=manifest['case'];stages=row.get('stages',[7,8])
        times=sequence(row['start'],row['surface_stop'] if stages==[8] else row['stop'],row['interval_seconds'])
        if stages==[8]:expected={row['mesh']+'.sfc_update.nc':times}
        elif stages==[9]:expected={'lbc.'+time.replace(':','.')+'.nc':[time] for time in times}
        elif stages==[7]:
            initial=row['mesh']+'.init.nc';static=row['mesh']+'.ugwp_oro_data.nc'
            expected={initial:[row['start']],static:[]};static_files.add(static)
            level_counts[initial]={'nVertLevels':row['levels'],'nVertLevelsP1':row['levels']+1,
                                   'nSoilLevels':4}
        else:raise ValueError('clock witness requires independently completed native stages')
    else:raise ValueError('unsupported native manifest')
    retired=None
    if manifest.get('outputs_retired'):
        clock=directory/'CLOCKS.json'
        if sha(clock)!=manifest['retired_clock_receipt_sha256']:raise ValueError('retired native clock receipt changed')
        retired=json.loads(clock.read_text())
        if (retired['schema']!='mpas.native-retired-output-clocks/v1' or not retired['complete']
                or retired['comparator_sha256']!=manifest['output_digest_tool_sha256']
                or {name:entry['sha256'] for name,entry in retired['files'].items()}!=actual):
            raise ValueError('retired native output coverage differs')
    errors=[];files=[]
    if set(actual)!=set(expected):errors.append('native output file set differs from requested stream clocks')
    for name,digest in sorted(actual.items()):
        path=directory/name
        if retired:
            inventory=retired['files'][name]['inventory']
            if not all(field.get('word_sha256') for field in inventory['variables']):raise ValueError('retired raw field digest absent')
        else:
            if sha(path)!=digest:errors.append('native output hash differs: '+name)
            process=subprocess.run([str(comparator),'--inventory',str(path)],capture_output=True,text=True,check=True)
            inventory=json.loads(process.stdout)
        fields=inventory['variables'];times=inventory['xtime']
        dimensions={dim:size for field in fields for dim,size in zip(field['dimensions'],field['shape'])}
        if not fields:errors.append('empty native stream: '+name)
        if any(field['field']=='xtime' for field in fields) and times!=expected.get(name):
            errors.append('stored native clock differs: '+name)
        if manifest['schema']=='mpas.native-input-cache/v1' and name not in static_files and not times:
            errors.append('native initialized or boundary clock is absent: '+name)
        time_shapes=[field['shape'][field['dimensions'].index('Time')] for field in fields if 'Time' in field['dimensions']]
        if not time_shapes and name not in static_files:errors.append('native output has no Time records: '+name)
        if time_shapes and any(count!=len(expected.get(name,[])) for count in time_shapes):
            errors.append('native stream record count differs: '+name)
        for dim,size in level_counts.get(name,{}).items():
            if dimensions.get(dim)!=size:errors.append('native initialization dimension differs: '+name+' '+dim)
        if name in level_counts:
            process=subprocess.run([str(comparator),'--i32-range','ivgtyp',str(path)],
                                   capture_output=True,text=True,check=True)
            ranges[name]=json.loads(process.stdout)
            # Observed with-lakes input aborts native Noah at
            # physics_wrf/module_sf_noahlsm.F:2441; its MODIS table has20 rows.
            if ranges[name]['maximum']>20:errors.append('native Noah vegetation category exceeds20: '+name)
        files.append({'file':name,'sha256':digest,'fields':len(fields),'xtime':times,
                      'dimensions':dimensions,'clock_storage':'static ancillary; no forecast clock' if name in static_files
                      else ('xtime' if times else 'filename; stream has no xtime variable')})
    if static_files and len({file['dimensions'].get('nCells') for file in files})!=1:
        errors.append('native initialization and static ancillary cell counts differ')
    return {'schema':'mpas.native-product-clocks/v1','directory':str(directory),'manifest_sha256':sha(directory/'MANIFEST.json'),
            'inactive_packages':inactive,'complete':not errors,'errors':errors,'files':files,'integer_ranges':ranges}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--comparator',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--inactive-package',action='append',default=[])
    a=p.parse_args();result=verify(a.directory,a.comparator.resolve(),a.inactive_package)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':result['complete'],'files':len(result['files']),'errors':result['errors']}))
    raise SystemExit(0 if result['complete'] else 1)

if __name__=='__main__':main()
