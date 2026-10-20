"""Measure standalone MPAS alternatives without assigning them forecast identity.

MPAS sf_sfclayrev.F90:713-721; module_sf_sfclay.F:781-789;
module_sf_sfclayrev.F:128,225; mpas_atmphys_driver_sfclayer.F:30-34.
The actual driver fixes IZ0TLND=0, ISFFLX=1, SCM_FORCE_FLUX=0.
This is not the published-executable/multi-rank exclusion witness policy needs.
"""
import argparse,ctypes,hashlib,importlib.util,json,os,socket,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('receipt',type=Path);p.add_argument('--kind', choices=('all','LH_HV','CZIL'), default='all');p.add_argument('--baseline-only', action='store_true');a=p.parse_args()
support=Path(__file__).resolve().parent.parent;spec=importlib.util.spec_from_file_location('mm5tests',support/'test_sfclay_mm5_v842.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
columns,c=t.columns.__wrapped__()
if a.baseline_only:columns=np.repeat(columns[:,:1],4,axis=1)
n=columns.shape[1]
native=ctypes.CDLL(str(a.oracle.resolve()/'liboracle.so'));native.oracle_init();native.oracle_run.argtypes=[t.PTR]*2+[ctypes.c_int]*7;native.oracle_run_wrapped.argtypes=[t.PTR]*2+[ctypes.c_int]*8
records=[]
for revised in (() if a.kind=='LH_HV' else (0,1)):
 for iz in (3,9):
  for prime in (None,0,1,2):
   x=columns.copy()
   if prime is not None:native.oracle_run(x,c,n,revised,1,0,prime,0,0)
   for iteration in range(3):
    x=columns.copy();native.oracle_run(x,c,n,revised,1,0,iz,0,0)
    controls=[]
    for alternative in (0,1,2):
     y=columns.copy();native.oracle_run(y,c,n,revised,1,0,alternative,0,0);controls.append(dict(iz0tlnd=alternative,differing_words=int(np.count_nonzero(x.view('u4')!=y.view('u4')))))
    records.append(dict(kind='CZIL',revised=revised,iz0tlnd=iz,prime=prime,iteration=iteration,sha256=hashlib.sha256(x.tobytes()).hexdigest(),controls=controls,words=x.view('u4').tolist()))
for opts in (() if a.kind=='CZIL' else ((0,0,0,0,0),(1,0,0,1,0))):
 for prime in (False,True):
  for iteration in range(4):
   x=columns.copy()
   if prime:
    y=columns.copy();native.oracle_run_wrapped(y,c,n,1,1,0,0,0,0,255)
   native.oracle_run_wrapped(x,c,n,1,*opts,255)
   records.append(dict(kind='LH_HV',revised=1,options=opts,prime=prime,iteration=iteration,lh_words=x[t.mm5.FIELDS.index('lh')].view('u4').tolist(),sha256=hashlib.sha256(x.tobytes()).hexdigest()))
r=dict(schema='mpas.mm5-native-undefined-probe/v1',scope='Standalone unchanged MPAS module, not published full native executable; source alternatives are private fixed parameters in actual MPAS driver, not selectable Registry options',host=socket.gethostname(),process_count=1,mpi_ranks=None,native_library_sha256=hashlib.sha256((a.oracle/'liboracle.so').read_bytes()).hexdigest(),columns=n,selection=dict(kind=a.kind,baseline_only=a.baseline_only),fields=t.mm5.FIELDS,records=records)
a.receipt.write_text(json.dumps(r,indent=2)+'\n');print('records',len(records),'host',r['host'])
native.oracle_finalize()
