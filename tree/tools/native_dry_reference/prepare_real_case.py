#!/usr/bin/env python3
"""Remap a real MPAS init onto a native real-case vertical-grid template.

This is a controlled dry numerics experiment, not an operational initialization.
Nearest source column, linear interpolation in geometric height, endpoint
extrapolation held constant. Reconstruct source edge winds to cell vectors,
then project at target edges. Remove vertical wind in all arms; remove moisture
unless --moisture qv (then qv is remapped like theta and floored at zero).
No forecast state is assimilated or nudged after time zero.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def xyz(lat,lon):
    return np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))


def interpolate(values, source_height, target_height):
    result=np.empty(target_height.shape,np.float32)
    for i in range(len(result)):
        if not np.all(np.diff(source_height[i])>0): raise ValueError('source vertical height is not increasing')
        result[i]=np.interp(target_height[i],source_height[i],values[i])
    return result


def add_synthetic_condensate(dst):
    """Write five SYNTHETIC condensate fields derived from the file's own qv.

    Deterministic and sharp-edged on purpose: physics is off, so these are
    transport and mass-loading tracers, chosen to exercise the monotonic
    limiter at horizontal and vertical edges.
    """
    z=np.asarray(dst['zgrid'][:],np.float64); h=(z[:,1:]+z[:,:-1])/2
    qv=np.asarray(dst['qv'][0],np.float64)
    latc=np.asarray(dst['latCell'][:],np.float64)[:,None]
    ice=(h>6000)&(h<12000)&(qv>1e-4)
    fields={'qc':np.where(h<8000,0.02*qv,0.0),
            'qr':np.where(qv>0.01,0.01*qv,0.0),
            'qi':np.where(ice,2e-5,0.0),
            'qs':np.where(ice&(latc>np.deg2rad(30)),1e-5,0.0),
            'qg':np.where((qv>0.01)&(h<4000),5e-6,0.0)}
    for name,value in fields.items():
        if name not in dst.variables: dst.createVariable(name,'f4',('Time','nCells','nVertLevels'))
        dst[name][0]=value.astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','template','out','receipt'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-time',required=True)
    p.add_argument('--moisture',choices=('dry','qv','condensate'),default='dry',
                   help='dry (default, the dycore-match case) or qv: remap source water vapour too, '
                        'so native and hex both run moist dynamics with qv transported, physics off; '
                        'condensate: qv plus five SYNTHETIC condensate fields (the source has none) for '
                        'a native build that allocates qc..qg with physics off (registry_condensate.py)')
    a=p.parse_args()
    if a.out.exists(): raise ValueError('refusing to overwrite an existing prepared input')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.template,a.out)
    with Dataset(a.source) as src,Dataset(a.out,'r+') as dst:
        src.set_auto_mask(False); dst.set_auto_mask(False)
        if 'Time' not in dst.dimensions: dst.createDimension('Time',None)
        if 'nSoilLevels' not in dst.dimensions: dst.createDimension('nSoilLevels',4)
        for name,dims in {
          'rho':('Time','nCells','nVertLevels'),'theta':('Time','nCells','nVertLevels'),
          'rho_base':('Time','nCells','nVertLevels'),'theta_base':('Time','nCells','nVertLevels'),
          'qv':('Time','nCells','nVertLevels'),'u':('Time','nEdges','nVertLevels'),
          'w':('Time','nCells','nVertLevelsP1')}.items():
            if name not in dst.variables: dst.createVariable(name,'f4',dims)
        for name in ('u_init','v_init'):
            if name not in dst.variables: dst.createVariable(name,'f4',('nVertLevels',))
            if name in src.variables and src[name].shape==dst[name].shape:
                dst[name][:]=src[name][:]
            else:
                dst[name][:]=0
        lat,lon=np.asarray(src['latCell'][:]),np.asarray(src['lonCell'][:])
        tree=cKDTree(xyz(lat,lon))
        cell=tree.query(xyz(dst['latCell'][:],dst['lonCell'][:]))[1]
        edge=tree.query(xyz(dst['latEdge'][:],dst['lonEdge'][:]))[1]
        heights=np.asarray(src['zgrid'][:],np.float64)
        source_height=(heights[:,1:]+heights[:,:-1])/2
        z=np.asarray(dst['zgrid'][:],np.float64)
        target_height=(z[:,1:]+z[:,:-1])/2
        for name in ('rho','theta'):
            values=np.asarray(src[name][0],np.float64)[cell]
            dst[name][0]=interpolate(values,source_height[cell],target_height)
        for name in ('rho_base','theta_base'):
            values=np.asarray(src[name][0],np.float64)
            if name=='rho_base': values=values*np.asarray(src['zz'][:],np.float64)
            remapped=interpolate(values[cell],source_height[cell],target_height)
            if name=='rho_base': remapped=remapped/np.asarray(dst['zz'][:],np.float64)
            dst[name][0]=remapped
        # The dry case zeroes every water species. --moisture qv is the
        # stated moist experiment: qv is remapped like theta (and floored at
        # zero, as native's own per-step clip would do after one step), which
        # activates native atm_compute_moist_coefficients and the qv coupling
        # in theta_m. Condensate stays zero: the source analysis carries none.
        for name in ('qv','qc','qr','qi','qs','qg','w'):
            if name in dst.variables: dst[name][:]=0
        if a.moisture in ('qv','condensate'):
            values=np.asarray(src['qv'][0],np.float64)[cell]
            dst['qv'][0]=np.maximum(interpolate(values,source_height[cell],target_height),np.float32(0))
        if a.moisture=='condensate':
            add_synthetic_condensate(dst)
        src_u=np.asarray(src['u'][0],np.float64)
        slots=np.asarray(src['edgesOnCell'][:],np.int64)[edge]-1
        counts=np.asarray(src['nEdgesOnCell'][:],np.int64)[edge]
        coeff=np.asarray(src['coeffs_reconstruct'][:],np.float64)[edge]
        vector=np.zeros((len(edge),src_u.shape[1],3),np.float64)
        for j in range(slots.shape[1]):
            active=j<counts
            vector[active]+=src_u[slots[active,j],:,None]*coeff[active,j,None,:]
        slat,slon=lat[edge],lon[edge]
        east=-vector[:,:,0]*np.sin(slon[:,None])+vector[:,:,1]*np.cos(slon[:,None])
        north=-(vector[:,:,0]*np.cos(slon[:,None])+vector[:,:,1]*np.sin(slon[:,None]))*np.sin(slat[:,None])+vector[:,:,2]*np.cos(slat[:,None])
        cells=np.asarray(dst['cellsOnEdge'][:],np.int64)-1
        if np.any(cells<0): raise ValueError('forecast experiment requires a global mesh')
        edge_height=(target_height[cells[:,0]]+target_height[cells[:,1]])/2
        east=interpolate(east,source_height[edge],edge_height)
        north=interpolate(north,source_height[edge],edge_height)
        angle=np.asarray(dst['angleEdge'][:],np.float64)[:,None]
        dst['u'][0]=east*np.cos(angle)+north*np.sin(angle)
        rho=np.asarray(dst['rho'][0],np.float64);theta=np.asarray(dst['theta'][0],np.float64)
        if not np.isfinite(rho).all() or not np.isfinite(theta).all() or np.min(rho)<=0:
            raise ValueError('invalid regridded state')
        dst.setncattr('dry_real_state_source_time',a.source_time)
        dst.setncattr('dry_real_state_preparation',__doc__)
        dimensions={k:len(v) for k,v in dst.dimensions.items()}
    record={'schema':'gpuwm-hex.dry-real-init/v1','source_time':a.source_time,
            'moisture':a.moisture,'source_sha256':digest(a.source),'template_sha256':digest(a.template),
            'init_sha256':digest(a.out),'dimensions':dimensions,'method':__doc__,
            'source':str(a.source),'out':str(a.out)}
    a.receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
