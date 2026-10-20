"""Seal completed case10 output inventories for the future f2 forecast door.

Python only binds hashes/XML/clock metadata; the existing Rust comparator reads
netCDF storage and xtime. This prepares native references, never Hex proof.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--comparator',type=Path,required=True)
    parser.add_argument('--pin-set',required=True)
    parser.add_argument('--hours',type=int,choices=(6,24),default=6)
    parser.add_argument('--native-executable',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    root=args.root.resolve();reader=args.comparator.resolve()
    for closure in ['3d_smagorinsky','prognostic_1.5_order']:
        directory=root/closure
        log=directory/'log.atmosphere.0000.out'
        if 'Finished running the atmosphere core' not in log.read_text():
            raise SystemExit('native forecast has no completion witness')
        namelist=directory/'namelist.atmosphere';streams=directory/'streams.atmosphere'
        init=directory/'x1.2562.init.nc'
        text=namelist.read_text()
        if not re.search(r'config_dt\s*=\s*60(?:\.0*)?\b',text):
            raise SystemExit('native timestep differs from this case10 fixture')
        duration=f'{args.hours//24}_{args.hours%24:02}:00:00'
        if not re.search(r"config_run_duration\s*=\s*['\"]"+duration+r"['\"]",text):
            raise SystemExit('native duration differs from requested inventory')
        declaration=ET.parse(streams).getroot()
        inputs={name:digest(path) for name,path in [('namelist',namelist),('streams',streams),('init',init)]}
        auxiliary={p.name:digest(p) for p in directory.iterdir() if p.is_file() and
                   (p.name.startswith('stream_list.') or p.suffix in ['.TBL','.DBL'] or p.name.endswith('.graph.info'))}
        outputs=[];inventory=[]
        for path in sorted(directory.glob('*.nc')):
            if path==init:continue
            metadata=json.loads(subprocess.check_output([str(reader),'--describe',str(path)],text=True))
            stream_name=path.name.split('.')[0]
            matching=[entry for entry in declaration if entry.get('name')==stream_name and 'output' in entry.get('type','')]
            if len(matching)!=1:raise SystemExit('output has no unique native stream')
            times=metadata['times']
            clock=None
            if times is None:
                # This fixture writes exactly one record/file with a complete
                # timestamp in the native filename; never invent a record.
                if metadata['records']!=1:raise SystemExit('missing xtime with multiple records')
                match=re.fullmatch(r'[^.]+\.(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})\.nc',path.name)
                if match is None:raise SystemExit('missing native stream clock')
                day,hour,minute,second=match.groups();times=[f'{day}_{hour}:{minute}:{second}']
                clock={'filename_template':matching[0].get('filename_template')}
            start=datetime(2023,3,31,12)
            seconds=[int((datetime.strptime(t,'%Y-%m-%d_%H:%M:%S')-start).total_seconds()) for t in times]
            if any(s<0 or s%60 for s in seconds):raise SystemExit('output clock is off native timestep')
            row={'stream':stream_name,'path':path.name,'sha256':digest(path),'steps':[s//60 for s in seconds],
                 'times':times,'records':metadata['records'],'time_source':metadata['time_source']}
            if clock:row.update(clock_witness=clock,time_source='STREAM-FILENAME+RECORD-COUNT')
            outputs.append(row);inventory.append({'file':path.name,**metadata})
        schedule={}
        for row in outputs:schedule.setdefault(row['stream'],[]).extend(row['steps'])
        steps=list(range(0,args.hours*60+1,360))
        expected={'diagnostics':steps,'output':steps,'restart':steps[1:],'surface':steps}
        if schedule!=expected:
            raise SystemExit('completed native output schedule is incomplete')
        manifest={'schema':'mpas-native-run-v1','pin_set':args.pin_set,'process_count':1,
                  'inputs':inputs,'auxiliary_inputs':auxiliary,'complete':True,'output_root':str(directory),
                  'outputs':outputs,'native_executable_sha256':digest(args.native_executable or root/'bin/atmosphere_model'),
                  'completion_log_sha256':digest(log),'host':socket.gethostname(),
                  'whole_forecast_proven':False}
        prefix=args.out/f'd2-les-{closure}'
        Path(str(prefix)+'-native.json').write_text(json.dumps(manifest,indent=2)+'\n')
        Path(str(prefix)+'-inventory.json').write_text(json.dumps({'process_count':1,'whole_forecast_proven':False,
            'reader_sha256':digest(reader),'fields':inventory,'expected_samples':schedule},indent=2)+'\n')
        print(closure+f': native {len(outputs)}-file/all-field inventory, one rank; no Hex comparison')


if __name__=='__main__':main()
