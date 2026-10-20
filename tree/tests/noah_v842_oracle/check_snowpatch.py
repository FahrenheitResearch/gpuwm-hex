"""Replay a preserved native snowpatch-melt compiler-fold regression column."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from hexcore.mpasphys import noah_v842 as noah

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu',action='store_true')
parser.add_argument('--receipt',type=Path,required=True)
args=parser.parse_args()
base=Path(__file__).resolve().parent
fixture=base/'fixtures/snowpatch-melt'
metadata=json.loads((fixture/'fixture.json').read_text())
for name,digest in metadata['files'].items():
    assert hashlib.sha256((fixture/name).read_bytes()).hexdigest()==digest
x=np.fromfile(fixture/'input.f32',dtype='<f4').reshape(1,79)
ix=np.fromfile(fixture/'input.i32',dtype='<i4').reshape(1,7)
table=np.fromfile(fixture/'tables.bin',dtype=np.uint8)
if args.gpu:
    import cupy as cp
    dx,dix,dt=map(cp.asarray,(x,ix,table))
    status=noah.replay_columns('driver',dx,dix,dt)
    x,ix,error=map(cp.asnumpy,(dx,dix,status))
    assert not error.any()
else:
    library=ct.CDLL(str(base/'libnoah_host.so'))
    function=library.replay_driver
    function.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]
    function(x.ctypes.data,ix.ctypes.data,table.ctypes.data,1)
output=args.receipt.parent/'snowpatch-words'
output.mkdir(parents=True,exist_ok=True)
(output/'fields.txt').write_text('float\ninteger\n')
for name,actual,reference in [('float',x,'native.f32'),('integer',ix,'native.i32')]:
    (output/(name+'.native.bin')).write_bytes((fixture/reference).read_bytes())
    (output/(name+'.cuda.bin')).write_bytes(actual.tobytes())
result=json.loads(subprocess.check_output([str(base/'compare_noah_call'),str(output)],text=True))
result.update(scope='preserved strict-native whole-driver column regression; no forecast proof',
              process_count=1,backend='hex-cuda' if args.gpu else 'host-glibc',
              production_sha256=hashlib.sha256(Path(noah.__file__).read_bytes()).hexdigest(),
              fixture_sha256=hashlib.sha256((fixture/'fixture.json').read_bytes()).hexdigest())
args.receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(int(result['differing_words']!=0))
