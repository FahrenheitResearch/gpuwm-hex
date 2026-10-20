"""Native from-MPAS -> whole Noah driver -> to-MPAS versus production launcher."""
from pathlib import Path
import argparse
import ctypes as C
import json
import os
import numpy as np
from hexcore.mpasphys.noah_v842 import _MPAS_INPUT_FIELDS,run_mpas_surface
p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');args=p.parse_args()
base=Path(__file__).resolve().parent
sizes={'soil':4,'atmosphere':2,'pressure':2}
offsets={};width=0
for key,shape in _MPAS_INPUT_FIELDS.items():
    if shape!='integer_surface':offsets[key]=width;width+=sizes.get(shape,1)
cells,start,count=11,2,7
results=[]
if not args.gpu:
    native=C.CDLL(str(base/'libnoah_oracle.so'));old=Path.cwd();os.chdir(base);native.read_tables();os.chdir(old)
else:import cupy as cp
for mode in range(4):
    path=Path('n1out')/f'pipeline-{mode}.npz'
    if not args.gpu:
        x=np.zeros((cells,width),np.float32);ix=np.tile(np.array([7,3],np.int32),(cells,1))
        values={'skintemp':292.,'tmn':285.,'xland':1.,'vegfra':40.,'shdmin':10.,'shdmax':80.,'snoalb':.8,'sfc_albedo':.2,'sfc_albbck':.2,'sfc_emiss':.96,'sfc_emibck':.96,'z0':.1,'znt':.1,'canwat':.1,'chs':.01,'cqs2':.01,'lai':2.,'br':-.1,'glw':300.,'gsw':400.,'qgh':.012,'rainncv':.01,'raincv':.02,'sr':.3,'t':[290.,289.],'qv':[.006,.005],'dz8w':[50.,70.],'pres2_hyd':[95000.,94000.],'dzs':[.1,.3,.6,1.],'tslb':[285.,285.,285.,285.],'smois':[.3,.3,.3,.3],'sh2o':[.3,.3,.3,.3]}
        for key,value in values.items():
            k=offsets[key];size=sizes.get(_MPAS_INPUT_FIELDS[key],1);x[:,k:k+size]=value
        ix[4,0]=15;x[5,offsets['xice']]=.8;x[6,offsets['xland']]=2.
        x[7,offsets['snow']]=40.;x[7,offsets['snowh']]=.2;x[7,offsets['snowc']]=.7
        options=np.array([13,15,2,mode%2,mode%2,mode//2],np.int32)
        packed=np.empty((count,79),np.float32);integers=np.empty((count,7),np.int32)
        tail=(cells,start,count,60.,.55,options.ctypes.data)
        fn=native.io_from;fn.argtypes=[C.c_void_p]*4+[C.c_int]*3+[C.c_float]*2+[C.c_void_p]
        fn(x.ctypes.data,ix.ctypes.data,packed.ctypes.data,integers.ctypes.data,*tail)
        fn=native.replay_driver;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int]
        fn(packed.ctypes.data,integers.ctypes.data,count)
        expected=x.copy();fn=native.io_to;fn.argtypes=[C.c_void_p]*4+[C.c_int]*3+[C.c_float]*2+[C.c_void_p]
        fn(expected.ctypes.data,ix.ctypes.data,packed.ctypes.data,integers.ctypes.data,*tail)
        np.savez(path,input=x,input_int=ix,options=options,native=expected)
        results.append({'options':mode,'columns':count,'native_pipeline_executed':True})
        continue
    z=np.load(path);x=z['input'];ix=z['input_int'];options=z['options'];tb=cp.asarray(np.load('n1out/driver-synthetic.npz')['table_bytes'])
    for atmosphere_levels,pressure_interfaces,level_first in (
        (a,p,l) for a,p in ((2,2),(55,56)) for l in (False,True)
    ):
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            fields={};expected_fields={}
            for key,shape in _MPAS_INPUT_FIELDS.items():
                if shape=='integer_surface':a=ix[:,0 if key=='ivgtyp' else 1].copy()
                else:
                    size=sizes.get(shape,1);k=offsets[key];a=x[:,k:k+size].copy()
                    expected=z['native'][:,k:k+size].copy()
                    depth=atmosphere_levels if shape=='atmosphere' else pressure_interfaces if shape=='pressure' else size
                    if depth>size:
                        # Native driver:788-791 reads only the lowest air layer
                        # and two pressure interfaces. Extra pool levels must
                        # survive, while device indexing uses the full stride.
                        padded=np.full((cells,depth),np.float32(-999.),np.float32)
                        padded[:,:size]=a;a=padded
                        padded=np.full((cells,depth),np.float32(-999.),np.float32)
                        padded[:,:size]=expected;expected=padded
                    expected_fields[key]=expected
                    if size==1:a=a[:,0]
                    elif level_first:a=a.T.copy()
                fields[key]=cp.asarray(a)
            status=run_mpas_surface(fields,tb,dt_pbl=60.,itimestep=2,isurban=13,isice=15,config_sfc_albedo=int(options[3]),xice_threshold=.55,microphysics_enabled=bool(options[4]),convection_enabled=bool(options[5]),start=start,count=count,stream=stream,level_first=level_first)
        stream.synchronize()
        differences={}
        for key,k in offsets.items():
            size=sizes.get(_MPAS_INPUT_FIELDS[key],1);got=cp.asnumpy(fields[key])
            if level_first and got.ndim==2:got=got.T.copy()
            want=expected_fields[key].reshape(got.shape)
            d=int((got.view(np.uint32)!=want.view(np.uint32)).sum())
            if d:differences[key]=d
        results.append({'options':mode,'columns':count,'level_first':level_first,'atmospheric_levels':atmosphere_levels,'pressure_interfaces':pressure_interfaces,'explicit_nonblocking_stream':True,'differing_words':sum(differences.values()),'by_field':differences,'error_columns':int(cp.count_nonzero(status).item())})
receipt=Path('receipts/pipeline-cuda.json' if args.gpu else 'n1out/pipeline-native.json');receipt.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if any(r.get('differing_words',0) or r.get('error_columns',0) for r in results):raise SystemExit(1)
