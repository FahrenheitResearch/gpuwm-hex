"""Rust wire for complete native fork deformation initialization.

RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e.
Coordinates and all geometry arithmetic remain in the Rust bridge.
"""
import os,shutil,struct,subprocess
import numpy as np

# MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:373-614 at 1d6e370e.
def select_deformation_weights_rrfs(mesh, *, on_a_sphere=True, sphere_radius, bridge=None):
    """Keep initialized pool weights; atmosphere never rebuilds these fields.

    The initial-condition producer computes the weights before a regional cut.
    Repeating its routine on the cut would zero cells whose neighbor is beyond
    the local pool (advection.F:813-824), losing valid inherited coefficients.
    Geometry-only callers can still explicitly use the complete initializer.
    """
    arrays=getattr(mesh,'arrays',{})
    stored={name:getattr(mesh,name,arrays.get(name)) for name in ('defc_a','defc_b')}
    if all(value is None for value in stored.values()):
        return initialize_deformation_weights_rrfs(mesh,on_a_sphere=on_a_sphere,
            sphere_radius=sphere_radius,bridge=bridge)
    counts=getattr(mesh,'nEdgesOnCell',arrays.get('nEdgesOnCell'))
    edges=getattr(mesh,'edgesOnCell',arrays.get('edgesOnCell'))
    shape=(len(counts),edges.shape[1])
    for name,value in stored.items():
        if value is None or value.shape!=shape or value.dtype!=np.dtype('float32'):
            raise ValueError(name+' must carry the initialized RKIND pool at the mesh extent')
    return stored

# MPAS-Model-rrfs/src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e.
def initialize_deformation_weights_rrfs(mesh,*,on_a_sphere=True,sphere_radius,bridge=None):
    """Fork MPAS-Model-rrfs/src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e."""
    # MPAS-Model-rrfs/src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e.
    def get(name):
        arrays=getattr(mesh,'arrays',{})
        return getattr(mesh,name,arrays.get(name))
    nc=len(get('nEdgesOnCell'));ne=len(get('cellsOnEdge'));nv=len(get('xVertex'));me=get('edgesOnCell').shape[1]
    from .engines import bundled_program
    executable=(bridge or os.environ.get('RW_MPAS_RRFS_DEFORMATION') or bundled_program('rw_mpas_rrfs_deformation')
                or shutil.which('rw_mpas_rrfs_deformation'))
    if executable is None:raise FileNotFoundError('rw_mpas_rrfs_deformation Rust bridge is not installed')
    payload=struct.pack('<8sQQQQif',b'RRFSDEF2',nc,ne,nv,me,int(on_a_sphere),sphere_radius)
    for name in ('nEdgesOnCell','edgesOnCell','cellsOnCell','verticesOnCell','cellsOnEdge'):
        # Wire connectivity is the existing zero-based host mesh representation.
        payload+=np.ascontiguousarray(get(name),dtype='<i4').tobytes()
    for name in ('xCell','yCell','zCell','xVertex','yVertex','zVertex'):
        value=np.asarray(get(name))
        if value.dtype not in (np.dtype('float32'),np.dtype('float64')):raise TypeError(name+' must be a native real field')
        # netCDF doubles narrow to RKIND inside Rust, as the native pool reader does.
        payload+=struct.pack('<i',value.dtype.itemsize)
        payload+=np.ascontiguousarray(value,dtype='<f'+str(value.dtype.itemsize)).tobytes()
    output=subprocess.run([str(executable)],input=payload,capture_output=True,check=True).stdout
    if len(output)!=4*nc*me*4:raise ValueError('deformation bridge returned an invalid word count')
    words=np.frombuffer(output,dtype='<f4').copy().reshape(4,nc,me)
    return dict(zip(('defc_a','defc_b','cell_gradient_coef_x','cell_gradient_coef_y'),words))


# MPAS-Model-rrfs/src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e.
def initialize_edge_trigonometry_rrfs(angle,bridge=None):
    """Fork time integration 5802–5803 at 1d6e370e; GNU RKIND fixed geometry."""
    value=np.asarray(angle)
    if value.dtype not in (np.dtype('float32'),np.dtype('float64')):raise TypeError('angleEdge must be a real field')
    from .engines import bundled_program
    executable=(bridge or os.environ.get('RW_MPAS_RRFS_DEFORMATION') or bundled_program('rw_mpas_rrfs_deformation')
                or shutil.which('rw_mpas_rrfs_deformation'))
    if executable is None:raise FileNotFoundError('rw_mpas_rrfs_deformation Rust bridge is not installed')
    payload=struct.pack('<8sQi',b'RRFSTRG1',value.size,value.dtype.itemsize)+np.ascontiguousarray(value,dtype='<f'+str(value.dtype.itemsize)).tobytes()
    output=subprocess.run([str(executable),'--edge-trig'],input=payload,capture_output=True,check=True).stdout
    if len(output)!=value.size*8:raise ValueError('edge trig bridge returned an invalid word count')
    return np.frombuffer(output,dtype='<f4').copy().reshape(2,value.size)
