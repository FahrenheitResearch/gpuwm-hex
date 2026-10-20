"""Shared input fixture for the complete pinned compute_dyn_tend_work replay.

Test input generation only; no model data path processing.
MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5150-6409 at 1d6e370e.
"""
import ctypes as ct
from pathlib import Path
import re
import numpy as np


DYNAMICS_CASES = [(2,0.,0.,0.,True),(2,2.,.05,0.,True),(2,.23,.015,0.,True),
                  (1,.4,0.,0.,True),(0,0.,0.,.07,True),(1,.5,0.,.11,False),
                  (1,.35,.011,.07,True),(2,.2,.015,.12,True)]


def dynamics_fixture(path):
    n=5;nc=ne=nv=n+1;nl=7;me=2;ns=1;vd=3
    lib=ct.CDLL(str(Path(path).resolve()))
    lib.set_dimensions(ct.c_int(nl),ct.c_int(ns),ct.c_int(me),ct.c_int(n),ct.c_int(n),ct.c_int(n))
    source=Path(path).with_name('oracle.f90').read_text()
    routine=source[source.index('   subroutine atm_compute_dyn_tend_work'):]
    signature=re.search(r'\((.*?)\)\s*bind',routine,re.S)[1]
    names=re.sub(r'[&\s]','',signature).lower().split(',')
    dims=dict(ncells=n,nedges=n,nvertices=n,nvertlevels=nl,maxedges=me,maxedges2=2*me,num_scalars=ns,vertexdegree=vd)
    a={};rng=np.random.default_rng(853)
    for line in routine.split('! Local variables')[0].splitlines():
        match=re.match(r'\s*(integer|real|logical).*?dimension\((.*?)\)\s*::\s*(.*)',line,re.I)
        if match:
            shape=tuple(eval(x,{'__builtins__':{}},dims) for x in match[2].lower().split(','))
            for name in match[3].lower().split(','):
                name=name.strip()
                if name in names:a[name]=np.zeros(shape,'i4' if match[1].lower()=='integer' else 'f4',order='F')
    for name in names:
        if name not in a:a[name]=0.0
    a.update(dims);a.update(nvertlevels_dummy=nl,maxedges_dummy=me,maxedges2_dummy=me*2,num_scalars_dummy=ns,moist_start=1,moist_end=1,ncellssolve=n,nedgessolve=n,
        cellstart=1,cellend=n,vertexstart=1,vertexend=n,edgestart=1,edgeend=n,cellsolvestart=1,cellsolveend=n,vertexsolvestart=1,vertexsolveend=n,edgesolvestart=1,edgesolveend=n,
        config_mix_full=1,config_rayleigh_damp_u=0,config_number_cam_damping_levels=3,config_number_rayleigh_damp_u_levels=3,rk_step=1,mixing_option=2,
        dt=.375,c_s=.125,config_len_disp=11.3,config_del4u_div_factor=10.,config_rayleigh_damp_u_timescale_days=5.,r_earth=6371220.,coef_3rd_order=.25,cf1=1.,cf2=0.,cf3=0.)
    for name in ('dvEdge','dcEdge','rho_edge','rho_zz','zz','cqw','cqu','rb','exner','pressure_b'):
        a[name.lower()][:]=rng.uniform(.3,1.7,a[name.lower()].shape).astype('f4')
    for name in ('u','v','ru','w','ke','pv_edge','divergence','vorticity','theta_m','defc_a','defc_b','weightsOnEdge','u_init','v_init','t_init','angleEdge'):
        a[name.lower()][:]=rng.uniform(-1,1,a[name.lower()].shape).astype('f4')
    for key,geometry in [('invdcedge','dcedge'),('invdvedge','dvedge')]:a[key][:]=np.float32(1)/a[geometry]
    a['invareacell'][:]=rng.uniform(.1,.4,nc).astype('f4');a['invareatriangle'][:]=rng.uniform(.07,.3,nv).astype('f4')
    a['meshscalingdel2'][:]=1;a['meshscalingdel4'][:]=1
    a['zgrid'][:]=np.arange(nl+1,dtype='f4')[:,None]*1000
    a['fzm'][:]=.6;a['fzp'][:]=.4;a['rdzw'][:]=.001;a['rdzu'][:]=.001
    a['nedgesoncell'][:n]=me;a['nedgesonedge'][:n]=me*2
    for e in range(n):
        a['cellsonedge'][:,e]=e+1,(e+1)%n+1;a['verticesonedge'][:,e]=e+1,(e+1)%n+1
        a['edgesonedge'][:,e]=[(e+j+1)%n+1 for j in range(me*2)]
    a['cellsonedge'][:,n]=nc;a['verticesonedge'][:,n]=nv
    a['edgesonedge'][:,n]=ne
    for c in range(n):
        a['edgesoncell'][:,c]=c+1,(c-1)%n+1;a['edgesoncell_sign'][:,c]=1,-1
        a['cellsoncell'][:,c]=(c+1)%n+1,(c-1)%n+1
        # Each listed edge must actually meet the vertex; duplicate an
        # incident edge for this small synthetic valence-three fixture.
        a['edgesonvertex'][:,c]=c+1,(c-1)%n+1,c+1
        # MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1214-1225 at 1d6e370e.
        for j,e in enumerate(a['edgesonvertex'][:,c]):a['edgesonvertex_sign'][j,c]=1 if a['verticesonedge'][1,e-1]==c+1 else -1
    a['edgesoncell'][:,n]=ne;a['cellsoncell'][:,n]=nc;a['edgesonvertex'][:,n]=ne
    return a, (n,nc,ne,nv,nl,me,ns,vd)
