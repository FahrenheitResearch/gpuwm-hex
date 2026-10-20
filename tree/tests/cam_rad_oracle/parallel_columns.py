"""Emit a whole CAM replay candidate with independent columns in CUDA warps.

The MPAS routine bodies, operations and descriptor ownership are unchanged.
Each active thread retains its own complete arena. This is a qualification
candidate, never an untested change to the registered backend.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def prepare(source, output, receipt, batch_columns):
    if batch_columns < 32 or batch_columns % 32:
        raise ValueError('CAM column batch must contain complete CUDA warps')
    text = source.read_text()
    old = 'if(threadIdx.x) return;const unsigned long long slice=(bytes/gridDim.x)&~7ULL;\nCamArena arena(workspace+blockIdx.x*slice,slice);CamColumn cam(arena);'
    new = '''const int column=int(blockIdx.x*blockDim.x+threadIdx.x);
const int active=min(int(args[1135])-int(args[1134])+1-first_column,int(gridDim.x*blockDim.x));
if(column>=active) return;
const unsigned long long slice=(bytes/active)&~7ULL;
CamArena arena(workspace+column*slice,slice);CamColumn cam(arena);'''
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = 'cam.native_chunk_its=its;its+=first_column+int(blockIdx.x);ite=its;'
    assert text.count(old) == 1
    text = text.replace(old, 'cam.native_chunk_its=its;its+=first_column+column;ite=its;')
    old = 'if(first_column==0 && blockIdx.x==0) {'
    assert text.count(old) == 1
    text = text.replace(old, 'if(first_column==0 && column==0) {')
    old = "nc = min(64, int(arguments['ite']) - int(arguments['its']) + 1)"
    assert text.count(old) == 1
    text = text.replace(old, old.replace('min(64,', f'min({batch_columns},'))
    old = 'for start in range(0,columns,64):\n            kernel((min(64,columns-start),),(1,),'
    new = f'for start in range(0,columns,{batch_columns}):\n            kernel(((min({batch_columns},columns-start)+31)//32,),(32,),'
    assert text.count(old) == 1
    text = text.replace(old, new)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    spec = importlib.util.spec_from_file_location('cam_parallel_candidate', output)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    receipt.write_text(json.dumps(dict(
        scope='whole CAM independent-column launch candidate; execution unqualified',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(text.encode()).hexdigest(),
        cuda_source_sha256=hashlib.sha256(module.CUDA_SOURCE.encode()).hexdigest(),
        threads_per_block=32, batch_columns=batch_columns,
        numerical_routines_changed=False, promoted=False), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-columns', type=int, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.receipt, args.batch_columns)
