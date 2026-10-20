"""Rust preparation of the fork's vertical-velocity damping coefficients.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1266-1309 at 1d6e370e.
Python carries the wire arrays; the complete source-order loop and host GNU
sinf/powf calls run in rw_mpas_rrfs_damping.
"""
import os
import shutil
import struct
import subprocess
import numpy as np

# MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1266-1309 at 1d6e370e.
def build_rrfs_vertical_velocity_damping(zgrid, mesh_density, *, xnutr,
    damping_start_z, bridge=None):
    """Fork mpas_atm_core.F:1266-1309 at 1d6e370e; RKIND=binary32."""
    heights=np.asarray(zgrid); density=np.asarray(mesh_density)
    if heights.dtype!=np.float32 or density.dtype!=np.float32:
        raise TypeError('RRFS damping requires native single precision')
    if heights.ndim!=2 or heights.shape[0]<2 or density.shape!=(heights.shape[1],):
        raise ValueError('RRFS damping requires interface heights and one density per cell')
    nl=heights.shape[0]-1;nc=heights.shape[1]
    from .engines import bundled_program
    executable=(bridge or os.environ.get('RW_MPAS_RRFS_DAMPING') or bundled_program('rw_mpas_rrfs_damping')
                or shutil.which('rw_mpas_rrfs_damping'))
    if executable is None: raise FileNotFoundError('rw_mpas_rrfs_damping Rust bridge is not installed')
    payload=(struct.pack('<8sQQff',b'RRFSDSS1',nl,nc,float(xnutr),float(damping_start_z))+
        np.ascontiguousarray(heights,dtype='<f4').tobytes()+np.ascontiguousarray(density,dtype='<f4').tobytes())
    result=subprocess.run([str(executable)],input=payload,capture_output=True,check=True)
    if len(result.stdout)!=nl*nc*4: raise ValueError('RRFS damping bridge returned an invalid word count')
    return np.frombuffer(result.stdout,dtype='<f4').copy().reshape(nl,nc)
