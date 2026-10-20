"""Separate native-slot transfer diagnostic; never use as a throughput benchmark.

Run under the normal GPU ownership wrapper:
    python profile_native_slot_transfers.py --profile-receipt FILE forecast ...

The current commit implementation is pinned below because instrumentation repeats
its operations. Refuse a different implementation until the diagnostic is reviewed.
CPU byte comparisons are additional overhead; native writes are never omitted.
"""
import datetime,hashlib,inspect,json,sys,time,os,functools
from pathlib import Path
import numpy as np
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
if len(sys.argv)<3 or sys.argv[1]!='--profile-receipt':
 raise SystemExit('usage: profile_native_slot_transfers.py --profile-receipt FILE forecast ...')
destination=Path(sys.argv[2]).resolve(); del sys.argv[1:3]
original=NativeDevicePoolViews.commit;rows=[]
source=inspect.getsource(original)
if hashlib.sha256(source.encode()).hexdigest()!='c6f2a9a824f066f55f1d9c3729b9dcf9f1588e730df4345eafdf9bc4f3cbd2bf':
 raise SystemExit('commit source changed: review diagnostic operations before profiling')
context={'itimestep':0,'time_seconds':0.,'slot':'initialization','scheme':''}
inputs=[]
from hexcore.mpasphys.session_v842 import NativePhysicsSession
original_step=NativePhysicsSession.begin_step
def step(self,itimestep,time_seconds):
 context.update(itimestep=itimestep,time_seconds=time_seconds,slot='between-slots',scheme='')
 return original_step(self,itimestep,time_seconds)
NativePhysicsSession.begin_step=step
original_begin=NativeDevicePoolViews.begin
def begin(self,call,descriptor):
 context.update(itimestep=call.itimestep,time_seconds=call.time_seconds,slot=call.slot,scheme=call.scheme)
 return original_begin(self,call,descriptor)
NativeDevicePoolViews.begin=begin
for method in ('field','prepared'):
 def install(method):
  original_input=getattr(NativeDevicePoolViews,method)
  @functools.wraps(original_input)
  def observed(self,*args,**kwargs):
   before=len(self.buffers); started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
   result=original_input(self,*args,**kwargs)
   inputs.append(dict(operation=method,names=list(args),new_buffer=len(self.buffers)>before,bytes=int(getattr(result,'nbytes',0)),wall_seconds=time.monotonic()-tick,started_utc=started,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parallelism='CPU1 native getter plus one GPU transfer/packing; host wall includes existing waits only',**context))
   return result
  setattr(NativeDevicePoolViews,method,observed)
 install(method)
def commit(self):
 for key,(host,packed,selection) in self.buffers.items():
  started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
  if (key[0]!='prepared' and isinstance(packed,getattr(self.cp,'ndarray',())) and packed.dtype in (self.cp.float32,self.cp.int32)):
   value=packed.view(self.cp.uint32).get(order='F').view(packed.dtype)
  else:value=packed.get() if hasattr(packed,'get') else packed
  got=time.monotonic()
  if key[0]=='prepared':
   target=host[:value.shape[2],:,0,:] if host.ndim==4 else host[:value.shape[1],:,0] if host.ndim==3 else host[:value.shape[0],0]
   incoming=value.transpose(2,1,0) if host.ndim==4 else value.T if host.ndim==3 else value
  else:target=host[selection];incoming=value
  # CPU-only diagnostic at the already-existing get boundary. No device sync.
  unchanged=target.shape==incoming.shape and target.dtype==incoming.dtype and target.tobytes()==incoming.tobytes()
  compared=time.monotonic()
  target[...] = incoming
  packed_at=time.monotonic()
  if key[0]=='prepared':self.native.put_variable(key[1],host,lower=selection)
  else:self.native.put_field(key[0],key[1],host,level=key[2],nlevels=self.native._fields[key[0],key[1]][2])
  end=time.monotonic()
  rows.append(dict(**context,key=list(key),shape=list(value.shape),dtype=str(value.dtype),bytes=int(value.nbytes),unchanged=unchanged,started_utc=started,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),get_seconds=got-tick,diagnostic_compare_seconds=compared-got,host_assignment_seconds=packed_at-compared,native_write_seconds=end-packed_at,parallelism='One granted GPU existing get boundary; remaining stages CPU1; diagnostic comparison CPU1'))
 self.buffers.clear()
NativeDevicePoolViews.commit=commit
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
from hexcore.cli import main
try:raise SystemExit(main(sys.argv[1:]))
finally:
 r=dict(started_utc=started,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-tick,parallelism=dict(device='one granted GPU plus CPU1 native callbacks',requested_rust_threads=os.environ.get('RAYON_NUM_THREADS'),requested_omp_threads=os.environ.get('OMP_NUM_THREADS')),profile_only=True,original_commit_sha256=hashlib.sha256(source.encode()).hexdigest(),native_backend_sha256=hashlib.sha256(Path(inspect.getsourcefile(NativePhysicsSession.__init__)).with_name('native_backend.py').read_bytes()).hexdigest(),rows=rows,inputs=inputs,added_cuda_syncs=0,cuda_graphs=False,scope='Separate diagnostic copies unchanged data too; CPU bitwise comparison is instrumentation only. No extra CUDA synchronization. Unchanged observed data does not establish read-only intent.')
 destination.write_text(json.dumps(r,indent=2)+'\n')
