#!/usr/bin/env python3
"""Replay captured fork MPAS-driver GFL arguments through the complete kernel.

Only PRE records are read. Rust owns ORCL decoding/encoding. This adapter is for
the actual fork driver, whose gdc/qc_cu and gdc2/qi_cu arguments alias. Synthetic
captures must explicitly supply their independent argument association. It writes
every inout argument, including held words outside the solved row/tile.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import cupy as cp
from abi import DIMENSIONS
from hexcore.mpasphys.rrfs import cu_gfl_bf6a4b0 as gfl


# Native DATA or initialized local SAVE objects with no executed writes.
# RRFS GFL deep.F90:2639-2794,5176-5192; sh.F90:80;
# MPAS/module_cu_gfl_mpas.F90:942-950.
IMMUTABLE_LOCALS={
    'module_cu_gfl_deep.get_zu_zd_pdf_fim.alpha',
    'module_cu_gfl_deep.get_zu_zd_pdf_fim.g_alpha',
    'module_cu_gfl_deep.fct1d3.vrbos',
    'module_cu_gfl_sh.cu_gfl_sh_run.make_calc_for_xk',
    'module_cu_gfl.calc_cldfraction_monan.zl0',
    'module_cu_gfl.calc_cldfraction_monan.zcsigma',
    'module_cu_gfl.calc_cldfraction_monan.zcsig_conv',
    'module_cu_gfl.calc_cldfraction_monan.only_convective_cloud_fraction',
}


def main():
    prefix,outprefix=map(Path,sys.argv[1:])
    transport=Path(os.environ['MPAS_GFL_RECORD_TRANSPORT'])
    rawdir=outprefix.parent/(outprefix.name+'.inputs');rawdir.mkdir()
    pre=lambda name:prefix.parent/(prefix.name+'.pre.'+name+'.bin')
    post=lambda name:outprefix.parent/(outprefix.name+'.post.'+name+'.bin')
    records={};meta={};presence={}
    def scalar(name):
        return int(subprocess.check_output([str(transport),'scalar',str(pre(name)),'-'],text=True))
    def load(name):
        raw=rawdir/(name+'.raw')
        info=json.loads(subprocess.check_output([str(transport),'decode',str(pre(name)),str(raw)],text=True))
        if info['width']!=4 or info['code'] not in (1,3,4):raise ValueError(f'GFL word kind mismatch: {name}')
        dtype='<u4' if info['code']==4 else '<i4' if info['code']==3 else '<f4'
        meta[name]=info
        records[name]=cp.asarray(np.frombuffer(raw.read_bytes(),dtype=dtype).reshape(info['shape'],order='F'))
        return records[name]
    dims={name:scalar(name) for name in DIMENSIONS}
    if dims['jms']!=1 or dims['jme']!=1 or dims['jts']!=1 or dims['jte']!=1:
        raise ValueError('capture does not have the fork MPAS driver bounds')
    nz=dims['kte'];first=dims['its']-dims['ims'];last=dims['ite']-dims['ims']+1
    raw=rawdir/'pbl_scheme.raw'
    subprocess.run([str(transport),'decode',str(pre('pbl_scheme')),str(raw)],check=True,capture_output=True)
    pbl=raw.read_bytes().decode('ascii').rstrip(' ')
    for name in gfl.FIELDS:
        flag=pre(name+'.present')
        present=not flag.exists() or subprocess.check_output([str(transport),'present',str(flag),'-'],text=True).strip()=='true'
        presence[name]=present
        if present:load(name)
    columns={}
    for name in gfl.FIELDS:
        if not presence[name]:columns[name]=None;continue
        value=records[name]
        if name in gfl.SCALAR_FIELDS:columns[name]=cp.ascontiguousarray(value.reshape(1))
        elif name in gfl.SURFACE_FIELDS:columns[name]=cp.ascontiguousarray(value[first:last,0])
        elif name=='chem3d_in':columns[name]=cp.ascontiguousarray(value[first:last,:,0,:].transpose(0,2,1))
        else:columns[name]=cp.ascontiguousarray(value[first:last,:,0])
    # The pinned driver's actual argument association, convection.F:687-706.
    # r1's generated synthetic program instead declares these as independent
    # arrays. Its runner supplies {} explicitly; never infer aliases from equal
    # array values, and never read POST words to initialize a replay.
    aliases=json.loads(os.environ.get('MPAS_GFL_CAPTURE_ARGUMENT_ALIASES',
                                     '{"gdc":"qc_cu","gdc2":"qi_cu"}'))
    for alias,target in aliases.items():
        if (alias,target) not in (('gdc','qc_cu'),('gdc2','qi_cu')):
            raise ValueError(f'unknown native condensate association: {alias} -> {target}')
        if presence[alias]:columns[alias]=columns[target]
    error_name='local.module_cu_gfl_deep.fct1d3.error'
    saved_error=cp.ascontiguousarray(load(error_name).reshape(1)) if pre(error_name).exists() else None
    result=gfl.launch_columns(columns,ncol=last-first,nz=nz,pbl_scheme=pbl,row_dx_first=columns['dxcell'][:1],native_dimensions=dims,saved_fct_error=saved_error)
    for name in gfl.OUTPUT_FIELDS:
        if presence[name]:
            value=columns[name];restored=records[name]
            if name in gfl.SURFACE_FIELDS:restored[first:last,0]=value
            elif name=='chem3d_in':restored[first:last,:,0,:]=value.transpose(0,2,1)
            else:restored[first:last,:,0]=value
            flat=restored.transpose(tuple(range(restored.ndim-1,-1,-1))).ravel()
            subprocess.run([str(transport),'encode',str(pre(name)),'-',str(post(name))],input=flat.get().tobytes(),check=True)
        if pre(name+'.present').exists():
            subprocess.run([str(transport),'forward',str(pre(name+'.present')),str(post(name+'.present'))],check=True)
    for path in prefix.parent.glob(prefix.name+'.pre.local.*.bin'):
        name=path.name[len(prefix.name+'.pre.'):-4];key=name.removeprefix('local.')
        if key=='module_cu_gfl_deep.cu_gfl_deep_run.iversion':
            # Native assignment is unconditional, deep.F90:1420.
            subprocess.run([str(transport),'encode',str(path),'-',str(post(name))],input=np.array(0,dtype='<i4').tobytes(),check=True)
        elif key=='module_cu_gfl_deep.fct1d3.error':
            subprocess.run([str(transport),'encode',str(path),'-',str(post(name))],input=result['saved_fct_error'].get().tobytes(),check=True)
        elif key in IMMUTABLE_LOCALS:
            subprocess.run([str(transport),'forward',str(path),str(post(name))],check=True)
        else:raise ValueError(f'uncaptured GFL saved-state contract: {name}')
    unknown=list(prefix.parent.glob(prefix.name+'.pre.state.*.bin'))
    if unknown:raise ValueError(f'GFL module state requires an explicit contract: {unknown[0].name}')
    for path in rawdir.iterdir():path.unlink()
    rawdir.rmdir()
    print(json.dumps({'scheme':gfl.SCHEME,'source_pin':gfl.SOURCE_PIN,'columns':last-first,'levels':nz,
                      'argument_aliases':aliases,'native_vertical_bounds':{n:dims[n] for n in ('kms','kme','kds','kde','kts','kte')},
                      'scope':'captured fork wrapper, not whole-forecast acceptance'}))


if __name__=='__main__':main()
