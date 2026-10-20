#!/usr/bin/env python
"""Executable for f1 column_oracle_replay.py --scheme lsm --backend hex-cuda.

Only Rust reads/encodes typed weather records. Python transfers its packed ABI
to/from CUDA. Invoke through f1's slot wrapper, which owns the GPU reservation.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import cupy as cp
import numpy as np
from hexcore.mpasphys.noah_v842 import replay_columns

base=Path(__file__).resolve().parent
if len(sys.argv)!=3:
    raise SystemExit('usage: replay_f1_cuda.py INPUT_PREFIX OUTPUT_PREFIX')
source,output=map(Path,sys.argv[1:])
manifest=json.loads((source.parent/'manifest.json').read_text())
capture=manifest['captures'][source.name]
site=next(s for s in manifest['sites'] if s['site']==capture['site'])
if site['callee']['module']!='module_sf_noahdrv' or site['scheme']!='lsm':
    raise ValueError('requires MPAS Noah lsm call site')

# Enumerate names only. Expected output payloads never enter the adapter.
names=sorted(p.name[len(source.name+'.post.'):-4]
             for p in source.parent.glob(source.name+'.post.*.bin'))
with tempfile.TemporaryDirectory(prefix='noah-f1-',dir=output.parent) as tmp:
    packed=Path(tmp)/'packed'
    subprocess.run([str(base/'dump_transport'),'pack',str(source),str(packed),
                    str(base/'table_decode'),str(base)],check=True)
    def transfer(suffix,kind,width):
        raw=Path(str(packed)+suffix).read_bytes()
        return cp.asarray(np.frombuffer(raw,dtype=np.uint8)).view(kind).reshape(-1,width)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        x=transfer('.f32',cp.float32,79)
        ix=transfer('.i32',cp.int32,7)
        tables=transfer('.table',cp.uint8,5696).reshape(-1)
        locations=transfer('.locations',cp.int32,2)
        module_locations=transfer('.module_locations',cp.int32,4).reshape(-1)
        status=replay_columns('driver',x,ix,tables,locations=locations,
                              module_locations=module_locations,stream=stream)
        # Launch-status integers are host orchestration, not weather fields.
        # Download every actual status word; avoid compiling a diagnostic reduction.
        line=int(status.get().max().item()) if status.size else 0
        if line:
            raise RuntimeError(f'MPAS Noah rejected replay at native line {line}')
        Path(str(packed)+'.f32').write_bytes(cp.asnumpy(x).tobytes())
        Path(str(packed)+'.i32').write_bytes(cp.asnumpy(ix).tobytes())
        Path(str(packed)+'.module_locations').write_bytes(cp.asnumpy(module_locations).tobytes())
    listing=Path(tmp)/'outputs.txt';listing.write_text('\n'.join(names)+'\n')
    subprocess.run([str(base/'dump_transport'),'unpack',str(source),str(output),
                    str(packed),str(listing)],check=True)
