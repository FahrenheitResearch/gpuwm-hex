"""Complete native driver/update oracle against the full translated host module.

Requires the actual f3 strict library/loader artifact; never forecast proof.
MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:405-621,991-1099.
"""
from pathlib import Path
import sys,importlib.util,ctypes as C,json,hashlib,socket
from types import SimpleNamespace
import numpy as np
from support import *
from hexcore.mpasphys import kain_fritsch_v842 as kf
import argparse
p=argparse.ArgumentParser()
p.add_argument('--loader',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
p.add_argument('--backend',choices=['host-glibc','hex-cuda'],default='host-glibc')
p.add_argument('--live-displacement',action='store_true',
               help='exercise live native config updates with a stale zero namelist option')
args=p.parse_args()
src=args.loader
spec=importlib.util.spec_from_file_location('kf_native_bridge',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
library=args.library
native,host,tables=load_oracles(args.build_dir)
nz,nc,owned=55,3,2
columns=[fixture(nz,8,.95,300,2),fixture(nz,3,.3,285,0),fixture(nz)]
f=np.stack([x[0] for x in columns],axis=2)
s=np.stack([x[1] for x in columns],axis=2)
v=np.stack([x[2] for x in columns],axis=1)
s[:,:,2]=99;v[2:4,2]=99;v[4,2]=300;v[5:7,2]=99
rain=np.zeros(nc,np.float32);counter=np.zeros(nc,np.int32);rain[2]=99;counter[2]=99
rows=[]
if args.backend=='hex-cuda':
 import cupy as cp
 prepared={name:cp.asarray(f[i,:nz+(name=='w_p'),:owned].copy()) for i,name in enumerate(kf._PREPARED_FIELDS)}
 diag={'w0avg':cp.asarray(s[0,:nz,:owned].copy())}
 diag.update({name:cp.asarray(v[i,:owned].copy()) for i,name in enumerate(kf._SURFACE_POOL) if i>=2})
 tend={name:cp.asarray(s[i,:nz,:owned].copy()) for i,name in enumerate(kf.STATE) if i>=1}
 config={'params':cp.asarray(PARAMS),'flags':cp.asarray(np.array([5,1,0,0,1,1,1],np.int32))}
 config['tables']=kf.build_tables(config['params'])
 pools={'prepared':prepared,'diag_physics':diag,'tend_physics':tend,'mesh':{'areaCell':cp.asarray(v[1,:owned].copy()),'meshDensity':cp.asarray(np.ones(owned,np.float32))},'dt_dyn':60,'n_cu':5,'len_disp':np.float32(100000),'kain_fritsch':config}
 device_rain=cp.asarray(rain[:owned].copy());device_counter=cp.asarray(counter[:owned].copy())
with mod.NativePhysics(library,test_only=True) as b:
 for prefix,size,total in [('i',owned,nc),('j',1,1),('k',nz,nz+1)]:
  for scope in ['d','m','t']:
   b.put_variable(prefix+scope+'s',np.asarray(1,np.int32))
   b.put_variable(prefix+scope+'e',np.asarray(total if scope in ('d','m') else size,np.int32))
 for name,val in [('dt_dyn',60.),('dt_cu',300.)]:b.put_variable(name,np.asarray(val,np.float32))
 b.put_variable('n_cu',np.asarray(5,np.int32))
 for name in ('f_qr','f_qi','f_qs'):b.put_variable(name,np.asarray(True))
 b.put_variable('l_acrain',np.asarray(True))
 for name,val in [('config_convection_scheme','cu_kain_fritsch'),('config_len_disp',100000.),('config_do_restart',False),('config_gfconv_closure_deep',3),('config_gfconv_closure_shallow',3),('config_bucket_rainc',1.)]:b.config(name,val)
 b.pool('sfc_input')
 b.allocate_convection(configs='configs')
 for i,name in enumerate(kf._PREPARED_FIELDS):b.put_variable(name,f[i].T[:,:,None].copy())
 b.put_field('mesh','meshDensity',np.ones(nc,np.float32));b.put_field('mesh','areaCell',v[1])
 b.put_field('diag_physics','w0avg',s[0,:nz].copy())
 for i,name in enumerate(kf._SURFACE_POOL):
  if i>=2:b.put_field('diag_physics',name,v[i].copy())
 for i,name in enumerate(kf.STATE):
  if i>=1:b.put_field('tend_physics',name,s[i,:nz].copy())
 b.put_field('diag_physics','rainc',rain.copy());b.put_field('diag_physics','i_rainc',counter.copy())
 lut=b.lib.__module_cu_kfeta_MOD_kf_lutab;lut.argtypes=[C.POINTER(C.c_float)]*4
 const=[C.c_float(x) for x in PARAMS[12:]];lut(*[C.byref(x) for x in const])
 for step in range(1,6):
  if args.live_displacement:
   b.config('config_len_disp', (125000.,100000.,75000.,230000.,75000.)[step-1])
   displacement=b.get_real_config('config_len_disp')
  else:
   displacement=np.float32(100000)
  native.kf_native_dx(owned,displacement,np.ones(owned,np.float32),v[0,:owned])
  b.update_convection_step1(configs='configs',diag_physics='diag_physics',tend_physics='tend_physics',its=1,ite=owned)
  if args.backend=='hex-cuda':
   pools['len_disp']=displacement
   _,ds,dv,pp=kf.convection_from_MPAS(prepared,pools['mesh'],diag,tend,displacement)
   kf.update_convection_step1(ds,dv,60);kf.convection_to_MPAS(ds,dv,pp)
  else:
   for c in range(owned):
    sc,vc=s[:,:,c].copy(),v[:,c].copy();native.kf_native_step1(nz,60,sc,vc);s[:,:,c]=sc;v[:,c]=vc
  b.driver_convection(itimestep=step,configs='configs',mesh='mesh',sfc_input='sfc_input',diag_physics='diag_physics',tend_physics='tend_physics',its=1,ite=owned)
  flags=np.array([5,step if step==1 else step+1,0,0,1,1,1],np.int32)
  if args.backend=='hex-cuda':
   kf.launch(SimpleNamespace(pools=pools,dt=300,itimestep=step,options={'config_len_disp':0}))
  else:
   for c in range(owned):
    sc,vc=s[:,:,c].copy(),v[:,c].copy();a=np.zeros(1,np.uint8)
    host.kf_host_column(nz,1,f[:,:,c].copy(),sc,vc,a,PARAMS,flags,tables,np.empty(112*nz+1,np.float32));s[:,:,c]=sc;v[:,c]=vc
  b.update_convection_step2(configs='configs',diag_physics='diag_physics',its=1,ite=owned)
  if args.backend=='hex-cuda':
   kf.update_convection_step2(diag['cuprec'],device_rain,device_counter,60,1,True)
   s[0,:nz,:owned]=cp.asnumpy(diag['w0avg'])
   for i,name in enumerate(kf.STATE):
    if i>=1:s[i,:nz,:owned]=cp.asnumpy(tend[name])
   for i,name in enumerate(kf._SURFACE_POOL):
    if i>=2:v[i,:owned]=cp.asnumpy(diag[name])
   rain[:owned]=cp.asnumpy(device_rain);counter[:owned]=cp.asnumpy(device_counter)
  else:
   native.kf_native_step2(owned,60,1,1,v[3,:owned].copy(),rain[:owned],counter[:owned])
  counts={}
  counts['w0avg']=word_diff(b.get_field('diag_physics','w0avg').copy(),s[0,:nz].copy())
  for i,name in enumerate(kf.STATE):
   if i>=1:counts[name]=word_diff(b.get_field('tend_physics',name).copy(),s[i,:nz].copy())
  for i,name in enumerate(kf._SURFACE_POOL):
   if i>=2:counts[name]=word_diff(b.get_field('diag_physics',name).copy(),v[i].copy())
  counts['rainc']=word_diff(b.get_field('diag_physics','rainc').copy(),rain)
  counts['i_rainc']=word_diff(b.get_field('diag_physics','i_rainc').copy(),counter)
  rows.append({'step':step,'live_displacement_word':int(np.asarray(displacement).view(np.uint32)),
               'differences':counts});print(rows[-1],flush=True)
 b.deallocate_convection(configs='configs')
receipt={'schema':'mpas-kf-full-native-driver-v1','backend':args.backend,'native_library':str(library),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'owned_columns':owned,'halo_sentinels':1,'steps':5,'loader_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'f_qi':True,'warm_rain':'native uninitialized; branch irrelevant with f_qi=true','comparison':'complete pristine native driver and updates vs complete translated module and driver slice; not a forecast','rows':rows}
receipt['live_displacement_updates']=args.live_displacement
receipt['immutable_namelist_displacement']=0.
receipt['execution_host']=socket.gethostname()
receipt['process_count']=1
args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(any(sum(r['differences'].values()) for r in rows))
