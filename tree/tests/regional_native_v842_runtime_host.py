"""Replay the complete regional native host binding on authenticated decks.

This checks actual Rust decoding, runtime geometry and native ABI allocation.
It uses the native provider, no mock pools, and makes no forecast identity claim.
"""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timedelta
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
from hexcore.cuda_regional_forecast_v842 import (
    load_native_regional_inputs, RegionalNativePhysicsSession)
from hexcore.cuda_regional_forecast_v841 import PaddedRegionalHostMesh
from hexcore.regional_v841 import derive_regional_masks
from hexcore.mpasphys.driver_v842 import resolve_namelist, interval_seconds
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import decode_registry_pools


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(manifest, source, reader, library, output):
    spec=json.loads(manifest.read_text())
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='d4.complete-regional-host-binding/v1',forecast_proof=False,
                host=socket.gethostname(),process_count=1,
                manifest_sha256=digest(manifest),native_library_sha256=digest(library),
                rust_reader_sha256=digest(reader),complete=False,decks=[])
    for entry in spec['decks']:
        for value in entry['inputs'].values():
            assert digest(value['path']) == value['sha256']
        init=Path(entry['inputs']['init']['path'])
        first=Path(entry['inputs']['lbc']['path'])
        text=Path(entry['inputs']['namelist']['path']).read_text()
        config=resolve_namelist(text)
        payload=output/entry['id']/'pools'
        layout=decode_registry_pools(reader,source,config,text,[init,first],payload)
        assert layout.configs['config_apply_lbcs']
        args=SimpleNamespace(init=init,grid=init)
        mesh,grid,profiles,state,reference,saved,names=load_native_regional_inputs(args,layout,payload)
        dims=layout.dimensions
        assert state.rho.shape == (dims['nVertLevels'],dims['nCells'])
        assert state.scalars.shape == (dims['num_scalars'],dims['nVertLevels'],dims['nCells'])
        masks=derive_regional_masks(mesh,np.dtype(np.float32))
        padded=PaddedRegionalHostMesh(mesh,masks)
        assert padded.n_cells == dims['nCells']+1
        assert padded.n_edges == dims['nEdges']+1
        with NativePhysics(library,test_only=True) as native:
            session=RegionalNativePhysicsSession(native,layout,payload,config,
                dt=layout.configs['config_dt'],start=layout.configs['config_start_time'],
                stop=(datetime.strptime(layout.configs['config_start_time'],'%Y-%m-%d_%H:%M:%S')+
                      timedelta(seconds=float(interval_seconds(layout.configs['config_run_duration'])))).strftime('%Y-%m-%d_%H:%M:%S'),
                stream_intervals={},
                post_transport=lambda call:None)
            sources=session.after_call('physics_get_tend',None)
            assert sources.rho.shape == (dims['nVertLevels'],dims['nCells']+1)
            assert sources.rho_u.shape == (dims['nVertLevels'],dims['nEdges']+1)
            assert sources.scalars.shape == (len(names),dims['nVertLevels'],dims['nCells']+1)
            assert session.device_field('state','scalars').shape == sources.scalars.shape
        result['decks'].append(dict(id=entry['id'],suite=config.suite,
            active_scalar_names=list(names),n_cells=dims['nCells'],n_edges=dims['nEdges'],
            padded_cells=padded.n_cells,padded_edges=padded.n_edges,passed=True,
            geometry_sha256={name:digest_buffer(mesh.arrays[name]) for name in
              ('edgeNormalVectors','localVerticalUnitVectors','cellTangentPlane','coeffs_reconstruct')}))
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(entry['id'],'passed',flush=True)
    result['complete']=True
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def digest_buffer(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','source','reader','library','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    replay(args.manifest,args.source,args.reader,args.library,args.output)
