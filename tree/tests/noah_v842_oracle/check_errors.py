"""Check native fatal branches and CUDA error propagation using isolated columns."""
from pathlib import Path
import argparse
import ctypes as C
import json
import os
import subprocess
import sys
import numpy as np
from hexcore.mpasphys.noah_v842 import COLUMN_LAYOUTS, table_dtype, replay_columns

CASES = {
    'soil': ('noah',2438,'too many input soil types'),
    'vegetation': ('noah',2441,'too many input landuse types'),
    'slope': ('noah',2444,'too many input slope types'),
    'roots': ('noah',2511,'too many root layers'),
    'snow': ('noah',515,'Physical snow depth'),
    'glacier_snow': ('glacial',231,'Physical snow depth'),
    'initial_soil': ('init',140,'out of range soil type'),
}
parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--case');args=parser.parse_args()
base=Path(__file__).resolve().parent
if args.case:
    cases=[args.case]
else:
    cases=list(CASES)
results=[]
for name in cases:
    scheme,line,message=CASES[name]
    if not args.gpu and not args.case:
        run=subprocess.run([sys.executable,__file__,'--case',name],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output=(run.stdout+run.stderr).decode(errors='replace')
        ok=run.returncode!=0 and message.lower() in output.lower()
        results.append({'case':name,'native_rejected':ok,'returncode':run.returncode,'expected_status':line})
        continue
    z=np.load(Path('n1out')/(scheme+'-synthetic.npz'))
    x=z['input'][:1].copy();ix=z['input_int'][:1].copy();tables=z['table_bytes'].copy()
    floats={n:i for i,n in enumerate(COLUMN_LAYOUTS[scheme]['float'])}
    ints={n:i for i,n in enumerate(COLUMN_LAYOUTS[scheme]['int'])}
    if name=='soil':ix[0,ints['SOILTYP']]=20
    if name=='initial_soil':ix[0,ints['ISLTYP']]=0;ix[0,ints['RESTART']]=0
    if name=='vegetation':ix[0,ints['VEGTYP']]=51
    if name=='slope':ix[0,ints['SLOPETYP']]=10
    if 'snow' in name:x[0,floats['SNEQV']]=.2;x[0,floats['SNOWH']]=.1
    if name=='roots':
        at=table_dtype().fields['NROTBL'][1]+6*4
        tables[at:at+4]=np.frombuffer(np.int32(5).tobytes(),np.uint8)
    if args.gpu:
        import cupy as cp
        status=cp.asnumpy(replay_columns(scheme,cp.asarray(x),cp.asarray(ix),cp.asarray(tables)))
        results.append({'case':name,'cuda_status':int(status[0]),'expected_status':line,'matched':int(status[0])==line})
    else:
        lib=C.CDLL(str(base/'libnoah_oracle.so'))
        before=Path.cwd();os.chdir(base);lib.read_tables();os.chdir(before)
        if name=='roots':
            roots=(C.c_int*50).in_dll(lib,'__module_sf_noahlsm_MOD_nrotbl');roots[6]=5
        fn=getattr(lib,'replay_'+scheme);fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int]
        fn(x.ctypes.data,ix.ctypes.data,1)
        raise SystemExit('Native unexpectedly accepted the invalid column')
if not args.case:
    path=Path('receipts/errors-cuda.json') if args.gpu else Path('n1out/errors-native.json')
    path.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
    if not all(r['matched'] if args.gpu else r['native_rejected'] for r in results):raise SystemExit(1)
