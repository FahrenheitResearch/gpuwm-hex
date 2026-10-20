#!/usr/bin/env python3
"""CUDA replay executable for f1's real column_oracle_replay.py.
MPAS: module_ra_rrtmg_lw.F:11574-12305; mpas_atmphys_driver_radiation_lw.F:865-885.
Weather record decoding/encoding belongs to the Rust oracle_transport helper.
Only PRE files are read. Hidden context is forwarded unchanged: this capture's
359 records are audited immutable across the native call, and the initialized
CUDA coefficient/lookup data have separate word-exact native validation.
This replay does not establish forecast identity or the remaining capture matrix.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import cupy as cp
from hexcore.mpasphys import rrtmg_lw_v842 as port

prefix,outprefix=map(Path,sys.argv[1:])
transport=Path(os.environ['MPAS_LW_RECORD_TRANSPORT'])
rawdir=outprefix.parent/(outprefix.name+'.inputs')
rawdir.mkdir()
prepath=lambda name:prefix.parent/(prefix.name+'.pre.'+name+'.bin')
outpath=lambda name:outprefix.parent/(outprefix.name+'.post.'+name+'.bin')
cache={}
def scalar(name):
    return int(subprocess.check_output([str(transport),'scalar',str(prepath(name)),'-'],text=True))
def field(name):
    if name not in cache:
        raw=rawdir/(name+'.raw')
        meta=json.loads(subprocess.check_output([str(transport),'decode',str(prepath(name)),str(raw)],text=True))
        if meta['code']!=1 or meta['width']!=4:raise ValueError('MPAS radiation fields require binary32 transport')
        # View the Rust-decoded canonical words; all rearrangement is on device.
        view=np.frombuffer(raw.read_bytes(),dtype='<f4').reshape(meta['shape'],order='F')
        cache[name]=cp.asarray(view)
    return cache[name]
if scalar('kts')!=1 or scalar('jts')!=1 or scalar('jte')!=1:
    raise ValueError('MPAS driver expects vertical start 1 and its single J row')
nm=scalar('kte');first=scalar('its')-scalar('ims');last=scalar('ite')-scalar('ims')+1
manifest=json.loads((prefix.parent/'manifest.json').read_text())
selected_column=manifest.get('recipe','').endswith('-native-selected-column')
model=cp.stack([field(name)[:,:nm,0] for name in port.MODEL_FIELDS],axis=1)
interfaces=cp.stack([field(name)[:,:nm+1,0] for name in ('p8w','t8w')],axis=1)
surface=cp.stack([field(name)[:,0] for name in port.SURFACE_FIELDS],axis=1)
# f1 replay/generate_replay.py:127-140 restores memory and tile bounds
# from the one-cell WRF array header for a source-bound selected column.
# The original whole-mesh scalar PRE records remain byte-exact in the dump.
if selected_column:
    if model.shape[0]!=1:raise ValueError('selected-column native replay requires one extracted cell')
    first,last=0,1
if not 0<=first<=last<=model.shape[0]:raise ValueError('native tile exceeds captured storage')
flags={name:scalar(name) for name in ('icloud','cldovrlp','idcor','julday',
            'has_reqc','has_reqi','has_reqs','o3input')}
names=port.FLUX_FIELDS+('rthratenlw','rre_cloud','rre_ice','rre_snow')
restored={name:field(name).copy() for name in names}
tables=port.load_tables_device(os.environ['MPAS_LW_TABLE_ASSET'],decoder=os.environ['MPAS_LW_TABLE_DECODER'])
for start in range(first,last,256):
    cols=slice(start,min(start+256,last))
    actual=port.rrtmg_lwrad_device(cp.ascontiguousarray(model[cols]),
        cp.ascontiguousarray(interfaces[cols]),cp.ascontiguousarray(surface[cols]),tables,
        **flags,pin=field('pin'),ozone=cp.ascontiguousarray(field('o3clim')[cols,:,0]))
    for name in names:
        if name in port.FLUX_FIELDS:restored[name][cols,0]=actual[name]
        elif name=='rthratenlw':restored[name][cols,:nm,0]=actual[name]
        elif flags['icloud']>0:
            mask=field('cldfra3d')[cols,:nm,0]>0
            restored[name][cols,:nm,0]=cp.where(mask,actual[name],restored[name][cols,:nm,0])
for name in names:
    array=restored[name]
    flat=array.transpose(tuple(range(array.ndim-1,-1,-1))).ravel()
    subprocess.run([str(transport),'encode',str(prepath(name)),'-',str(outpath(name))],
        input=flat.get().tobytes(),check=True)
# Preserve optional presence and immutable module/local input context.
for pre in prefix.parent.glob(prefix.name+'.pre.*.bin'):
    name=pre.name[len(prefix.name+'.pre.'):-4]
    if name.startswith(('state.','local.')) or (name.endswith('.present') and name[:-8] in names):
        subprocess.run([str(transport),'forward',str(pre),str(outpath(name))],check=True)
for raw in rawdir.iterdir():raw.unlink()
rawdir.rmdir()
print(json.dumps({'replayed_columns':last-first,'model_layers':nm,'driver_outputs':len(names),
                  'source_bound_selected_column':selected_column,
                  'cuda_source_sha256':__import__('hashlib').sha256((port._solver_source()+'\n'+
                   port._RANDOM_SOURCE+'\n'+port._DRIVER_SOURCE+'\n'+port._INIT_SOURCE).encode()).hexdigest()}))
