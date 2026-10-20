"""Complete fork dynamics work boundary over native resident pools.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:
5148-6409 at 1d6e370e. Shared unchanged advection stencils are composed in
native order with the separate fork mixing and vector-momentum operators.
"""
from dataclasses import fields
from types import SimpleNamespace

from .cuda_driver_rrfs import CudaRRFSDycoreDriver
from .cuda_horizontal_rrfs import CudaHorizontalRRFS
from .cuda_dynamics_rrfs import accumulate_euler_w_cuda_rrfs
from .mixing_rrfs import RRFSMixingConfig


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
def compute_dyn_tend_work_cuda_rrfs(arguments, module_state, *, mesh, context, kernel_cache,
                                   edge_cos=None, edge_sin=None):
    """Run the complete work routine on an owned single-partition carrier.

    Open regional edges retain the native dummy pool row in resident storage.
    The caller preserves unused output storage from entry. All numerical
    operands, including dummy pools and saved work, come from native PRE.
    """
    import cupy as cp
    import numpy as np
    a=arguments
    nl,nc,ne=int(a['nvertlevels_dummy']),int(mesh.n_cells),int(mesh.n_edges)
    physical_nc,physical_ne,physical_nv=(int(a[name]) for name in ('ncells','nedges','nvertices'))
    if (nc,ne,mesh.n_vertices) not in ((physical_nc,physical_ne,physical_nv),
                                     (physical_nc+1,physical_ne+1,physical_nv+1)):
        raise ValueError('resident work storage must match original native pools')
    # This boundary currently admits complete owned MPI1 work intervals.
    # The separate production regional driver retains other halo partitions.
    if any(int(a[name])!=want for name,want in (
        ('cellstart',1),('cellend',physical_nc),('cellsolvestart',1),('cellsolveend',physical_nc),
        ('edgestart',1),('edgeend',physical_ne),('edgesolvestart',1),('edgesolveend',physical_ne),
        ('vertexstart',1),('vertexend',physical_nv),
        ('vertexsolvestart',1),('vertexsolveend',physical_nv))):
        raise ValueError('owned work carrier requires complete native MPI1 intervals')
    driver=object.__new__(CudaRRFSDycoreDriver)
    driver.cp=cp;driver.cache=kernel_cache
    driver.nlev=nl;driver.ncells=nc;driver.nedges=ne
    driver._kernels={};driver._physics_driver_kernels={}
    driver._rrfs_native_advection_coefficients=True
    driver.v841_context=context
    driver.config=SimpleNamespace(config_coef_3rd_order=a['coef_3rd_order'])
    driver.atmosphere=SimpleNamespace(mesh=mesh,
        vertical=SimpleNamespace(rdzw=a['rdzw'],rdzu=a['rdzu'],fzm=a['fzm'],fzp=a['fzp']),
        reference=SimpleNamespace(rho_base=a['rb']))
    driver.coefficients=SimpleNamespace(adv_coefs=cp.ascontiguousarray(a['adv_coefs'].T),
        adv_coefs_3rd=cp.ascontiguousarray(a['adv_coefs_3rd'].T),
        n_adv_cells_for_edge=a['nadvcellsforedge'],
        adv_cells_for_edge=cp.ascontiguousarray(a['advcellsforedge'].T-1))
    horizontal=CudaHorizontalRRFS(mesh,nl,context,kernel_cache=kernel_cache)
    current=SimpleNamespace(rho=a['rho_zz'],rho_u=a['ru'],rho_w=a['rw'])
    saved=SimpleNamespace(rho_u=a['ru_save'],rho_w=a['rw_save'])
    changed={};private={}
    hd=horizontal.mass_flux_divergence(a['ru'])
    changed['h_divergence']=hd
    euler_u=a['tend_u_euler'];euler_w=a['tend_w_euler'];euler_theta=a['tend_theta_euler']
    tend_rho=a['tend_rho']
    if int(a['rk_step'])==1:
        flag=cp.zeros(1,'i4')
        dpdz=driver._moist_dpdz_v841(a['rr_save'],module_state['qtot'],flag)
        private['dpdz']=dpdz
        tend_rho=horizontal.density_tendency(a['ru'],a['rw'],a['rdzw'],physics_tendency=module_state['tend_rho_physics'])
        changed['tend_rho']=tend_rho
        seed=horizontal.pressure_gradient_euler_tendency(a['pp'],dpdz,a['cqu'],a['zz'],a['zxu'])
        config={f.name:a[f.name] for f in fields(RRFSMixingConfig) if f.name in a}
        config['config_smagorinsky_coef']=a['c_s']
        config['config_h_ScaleWithMesh']=False  # native precomputed pools below
        mixed=horizontal.compute_dry_mixing_tendencies_v841(a['u'],a['v'],a['w'],a['theta_m'],
            a['rho_edge'],a['divergence'],a['vorticity'],dt=a['dt'],config=RRFSMixingConfig(**config),
            cam_damping_levels=int(a['config_number_cam_damping_levels']),n_cells_solve=physical_nc,
            initial_u_tendency=seed,mesh_scaling_del2=a['meshscalingdel2'],mesh_scaling_del4=a['meshscalingdel4'])
        euler_u=mixed.tend_u_euler
        euler_theta=mixed.tend_theta_euler
        euler_w=accumulate_euler_w_cuda_rrfs(mixed.tend_w_euler,a['pp'],dpdz,a['cqw'],
            a['rdzu'],a['fzm'],a['fzp'],cell_solve_count=physical_nc,kernel_cache=kernel_cache)
        if float(a['config_v_mom_eddy_visc2'])>0 or float(a['config_v_theta_eddy_visc2'])>0:
            from .cuda_vertical_mixing_rrfs import apply_vertical_mixing_cuda_rrfs
            apply_vertical_mixing_cuda_rrfs(mesh,context,RRFSMixingConfig(**config),
                zgrid=a['zgrid'],normal=a['u'],vertical=a['w'],theta=a['theta_m'],rho=a['rho_zz'],
                rho_edge=a['rho_edge'],rdzw=a['rdzw'],rdzu=a['rdzu'],theta_init=a['t_init'],
                edge_cos=edge_cos,edge_sin=edge_sin,
                tend_u=euler_u,tend_w=euler_w,tend_theta=euler_theta,kernel_cache=kernel_cache)
        changed.update(kdiff=mixed.kdiff,tend_u_euler=euler_u,tend_w_euler=euler_w,tend_theta_euler=euler_theta)
        private.update(delsq_u=mixed.delsq_u,delsq_w=mixed.delsq_w,delsq_theta=mixed.delsq_theta)
        if mixed.h_mom_eddy_visc4>0:
            private.update(delsq_divergence=mixed.delsq_divergence,delsq_vorticity=mixed.delsq_vorticity)
    tu=driver._vertical_u(a['u'],a['rw'])
    tu=driver._vector_momentum(a['u'],a['rho_edge'],a['pv_edge'],a['ke'],hd,initial_tendency=tu)
    if bool(a['config_rayleigh_damp_u']):
        from .cuda_upper_damping_v841 import apply_rayleigh_u_cuda
        apply_rayleigh_u_cuda(tu,a['rho_edge'],a['u'],levels=int(a['config_number_rayleigh_damp_u_levels']),
            days=a['config_rayleigh_damp_u_timescale_days'],cache=kernel_cache)
    driver._add(tu,euler_u);driver._add(tu,module_state['tend_ru_physics'])
    tw=driver._w_tendency(current,a['w']);driver._add(tw,euler_w)
    tt=driver._theta_tendency(current,a['theta_m'],saved,a['theta_m_save'],int(a['rk_step']))
    rate=cp.empty_like(tt);flag=cp.zeros(1,'i4')
    driver._launch_physics_driver('modified_theta_dynamics_rate_v841_f32',nc,
        (np.int32(nl),np.int32(nc),tt,tend_rho,a['rho_zz'],a['theta_m'],rate,flag))
    # Native6292 adds lagged diabatic heating before the saved mixing pool.
    driver._launch_physics_driver('add_lagged_diabatic_theta_v842_f32',nc,
        (np.int32(nl),np.int32(nc),a['rho_zz'],a['rt_diabatic_tend'],tt))
    driver._add(tt,euler_theta);driver._add(tt,module_state['tend_rtheta_physics'])
    changed.update(tend_u=tu,tend_w=tw,tend_theta=tt,rthdynten=rate)
    return changed,private
