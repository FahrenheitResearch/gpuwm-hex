"""Replay full fork compute_dyn_tend_work against RRFS horizontal/momentum kernels.

RRFSx/MPAS-Model src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409
at 1d6e370e. Native call executes the entire routine; zero vertical mass flux
isolates the changed horizontal subsystem without altering native arithmetic.
"""
import hashlib
import json,os
from pathlib import Path
import numpy as np
import pytest
from test_initialization_rrfs_native import call

pytestmark = pytest.mark.cuda


def test_full_native_dyn_tend_horizontal():
    cp=pytest.importorskip('cupy')
    path=os.environ.get('RRFS_DYNAMICS_ORACLE')
    if not path:pytest.skip('full native fork dynamics oracle is required')
    from hexcore.cuda_backend.containers import DeviceMesh,TransferStats
    from hexcore.cuda_compile_rrfs import RRFSKernelCache
    from hexcore.cuda_horizontal_rrfs import CudaHorizontalRRFS
    from hexcore.cuda_dynamics_rrfs import vector_momentum_tendency_cuda_rrfs
    from hexcore.cuda_v841 import CudaV841Context
    from hexcore.mixing_rrfs import RRFSMixingConfig
    from hexcore.cuda_vertical_mixing_rrfs import apply_vertical_mixing_cuda_rrfs
    from hexcore.deformation_rrfs import initialize_edge_trigonometry_rrfs
    from rrfs_native.dynamics_fixture import dynamics_fixture, DYNAMICS_CASES
    a, (n,nc,ne,nv,nl,me,ns,vd) = dynamics_fixture(path)
    def gpu(x):return cp.ascontiguousarray(cp.asarray(x))
    map_index={'cells_on_edge':'cellsonedge','edges_on_cell':'edgesoncell','n_edges_on_cell':'nedgesoncell','cells_on_cell':'cellsoncell','vertices_on_edge':'verticesonedge','edges_on_edge':'edgesonedge','n_edges_on_edge':'nedgesonedge','vertices_on_cell':'edgesoncell','edges_on_vertex':'edgesonvertex'}
    map_float={'edge_sign_on_cell':'edgesoncell_sign','weights_on_edge':'weightsonedge','dc_edge':'dcedge','dv_edge':'dvedge','lat_cell':'latcell','lat_edge':'latedge','angle_edge':'angleedge','defc_a':'defc_a','defc_b':'defc_b','f_edge':'fedge'}
    kw={k:gpu(a[v].T-1 if a[v].ndim==2 else a[v]-int(k!='n_edges_on_cell' and k!='n_edges_on_edge')) for k,v in map_index.items()}
    kw.update({k:gpu(a[v].T if a[v].ndim==2 else a[v]) for k,v in map_float.items()})
    kw.update(cells_on_vertex=gpu(np.full((nv,vd),nc-1,'i4')),area_cell=gpu(np.float32(1)/a['invareacell']),area_triangle=gpu(np.float32(1)/a['invareatriangle']),kite_areas_on_vertex=cp.ones((nv,vd),'f4'),
        lon_cell=cp.zeros(nc,'f4'),lon_edge=cp.zeros(ne,'f4'),mesh_density=cp.ones(nc,'f4'),f_vertex=cp.zeros(nv,'f4'),spec_zone_mask_edge=cp.zeros(ne,'f4'),
        n_cells=nc,n_edges=ne,n_vertices=nv,max_edges=me,max_edges2=me*2,vertex_degree=vd,nominal_min_dc=11.3,dtype=np.dtype('f4'),index_dtype=np.dtype('i4'),h2d=TransferStats(0,0))
    mesh=DeviceMesh(**kw)
    context=CudaV841Context(**{name:gpu(a[alias]) for name,alias in {'inv_area_cell':'invareacell','inv_area_triangle':'invareatriangle','inv_dc_edge':'invdcedge','inv_dv_edge':'invdvedge','u_init':'u_init','v_init':'v_init'}.items()},
        etp=cp.ones(nl,'f4'),etm=cp.ones(nl,'f4'),ewp=cp.ones(nl+1,'f4'),ewm=cp.ones(nl+1,'f4'),h2d=TransferStats(0,0))
    cache=RRFSKernelCache();horizontal=CudaHorizontalRRFS(mesh,nl,context,kernel_cache=cache)
    comparisons=[]
    def check(case,field,actual,want):
        got=cp.asnumpy(actual)[:,:n];want=want[:,:n]
        diff=int(np.count_nonzero(got.view('u4')!=want.view('u4')))
        if diff:
            key=tuple(np.argwhere(got.view('u4')!=want.view('u4'))[0]);print(case,field,key,got[key],want[key])
        comparisons.append(dict(case=case,field=field,words=got.size,differing_words=diff,
            differing_indices_zero_based=np.argwhere(got.view('u4') != want.view('u4')).tolist(),
            native_nonfinite_indices_zero_based=np.argwhere(~np.isfinite(want)).tolist(),
            native_word_sha256=hashlib.sha256(np.ascontiguousarray(want).tobytes()).hexdigest(),
            hex_word_sha256=hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest()))
    cases=DYNAMICS_CASES
    cs,sn=(gpu(x) for x in initialize_edge_trigonometry_rrfs(a['angleedge']))
    for option,cam,visc,vertical,full in cases:
        a.update(mixing_option=option,config_mix_full=int(full),config_v_mom_eddy_visc2=vertical,config_v_theta_eddy_visc2=vertical*.7,config_h_mom_eddy_visc2=.17,config_h_theta_eddy_visc2=.13,config_h_mom_eddy_visc4=.011,config_h_theta_eddy_visc4=.013)
        a['kdiff'][...] = 0
        a['config_mpas_cam_coef']=cam;a['config_visc4_2dsmag']=visc
        call(path,'atm_compute_dyn_tend_work',a)
        cfg=RRFSMixingConfig(config_len_disp=a['config_len_disp'],config_mpas_cam_coef=cam,config_visc4_2dsmag=visc,config_horiz_mixing=('off','2d_fixed','2d_smagorinsky')[option],config_mix_full=full,**{k:a[k] for k in ('config_h_mom_eddy_visc2','config_h_theta_eddy_visc2','config_h_mom_eddy_visc4','config_h_theta_eddy_visc4','config_v_mom_eddy_visc2','config_v_theta_eddy_visc2')})
        # The production driver seeds horizontal mixing with the pressure
        # gradient. Even zero pressure/buoyancy carries native signed zeros.
        # This fixture has qtot=0 in its native dimension setter.
        dpdz=-np.float32(9.80616)*(a['rb']*np.float32(0)+a['rr_save']*np.float32(1))
        pressure_seed=horizontal.pressure_gradient_euler_tendency(
            gpu(a['pp']),gpu(dpdz),gpu(a['cqu']),gpu(a['zz']),gpu(a['zxu']))
        mixed=horizontal.compute_dry_mixing_tendencies_v841(*[gpu(a[k]) for k in ('u','v','w','theta_m','rho_edge','divergence','vorticity')],dt=a['dt'],config=cfg,cam_damping_levels=3,n_cells_solve=n,initial_u_tendency=pressure_seed)
        apply_vertical_mixing_cuda_rrfs(mesh,context,cfg,zgrid=gpu(a['zgrid']),normal=gpu(a['u']),vertical=gpu(a['w']),theta=gpu(a['theta_m']),rho=gpu(a['rho_zz']),rho_edge=gpu(a['rho_edge']),rdzw=gpu(a['rdzw']),rdzu=gpu(a['rdzu']),theta_init=gpu(a['t_init']),edge_cos=cs,edge_sin=sn,tend_u=mixed.tend_u_euler,tend_w=mixed.tend_w_euler,tend_theta=mixed.tend_theta_euler,cell_solve_count=n,edge_solve_count=n,kernel_cache=cache)
        for name in ('kdiff','tend_u_euler','tend_w_euler','tend_theta_euler'):check((option,cam,visc,vertical,full),name,getattr(mixed,name),a[name])
        div=horizontal.mass_flux_divergence(gpu(a['ru']));check((option,cam,visc,vertical,full),'h_divergence',div,a['h_divergence'])
        base=vector_momentum_tendency_cuda_rrfs(mesh,context,normal_velocity=gpu(a['u']),rho_edge=gpu(a['rho_edge']),pv_edge=gpu(a['pv_edge']),kinetic_energy=gpu(a['ke']),horizontal_divergence=div,initial_tendency=cp.zeros_like(gpu(a['u'])),kernel_cache=cache)
        # Native adds the saved Euler source after vector momentum.
        check((option,cam,visc,vertical,full),'tend_u',base+mixed.tend_u_euler,a['tend_u'])
    out=os.environ.get('RRFS_RECEIPTS')
    if out:Path(out,'dynamics-horizontal.json').write_text(json.dumps(dict(
        tier='BITWISE SUBSYSTEM ONLY' if not any(x['differing_words'] for x in comparisons) else 'FAILED SUBSYSTEM COMPARISON',
        native_ranks=1,comparisons=comparisons,compile=cache.compile_manifest()),indent=2)+'\n')
    assert not any(x['differing_words'] for x in comparisons)
