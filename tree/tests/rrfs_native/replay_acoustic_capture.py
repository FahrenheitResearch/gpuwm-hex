"""Replay complete acoustic/scalar/dynamics boundaries from real PRE words.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:
2225-2366,2646-2984,3613-4023 at 1d6e370e. Rust owns raw record I/O/comparison.
Only PRE operands feed CUDA. Every captured POST field is compared in full.
"""
import argparse
import ctypes as ct
from dataclasses import fields
import json
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import time

import numpy as np


def validate_capture_bank(bank, manifest_path, contract):
    """Authenticate the complete stated case set before launching any kernels."""
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != contract['manifest_sha256']:
        raise ValueError('capture manifest authentication failed')
    manifest = json.loads(raw)
    if manifest['source_commit'] != contract['source_commit']:
        raise ValueError('capture source pin mismatch')
    if manifest.get('source_authority') != contract.get('source_authority'):
        raise ValueError('capture source authority differs from the labelled replay contract')
    records = manifest['records']
    names = [r['path'] for r in records]
    if len(set(names)) != len(names) or set(names) != {p.name for p in bank.glob('*.bin')}:
        raise ValueError('capture bank missing, extra or duplicate records')
    calls = {}
    for row in records:
        name = row['path']
        if Path(name).name != name:
            raise ValueError('capture path must be a single filename')
        path = bank / name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise ValueError('capture record authentication failed: ' + name)
        if '.pre.' in name:
            prefix = name.split('.pre.')[0]
            key = prefix.split('.s')[0]
            calls.setdefault(key, set()).add(prefix)
    counts = {key: len(value) for key, value in calls.items()}
    if counts != contract['calls']:
        raise ValueError('capture call coverage differs from the stated native case set')
    return calls


class Records:
    def __init__(self, library):
        self.lib = ct.CDLL(str(library.resolve()))
        self.lib.oracle_read.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_size_t)]
        self.lib.oracle_read.restype = ct.c_void_p
        self.lib.oracle_free.argtypes = [ct.c_void_p, ct.c_size_t]
        self.lib.oracle_write.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_void_p, ct.c_size_t]
        self.lib.oracle_write.restype = ct.c_int32

    def read(self, path):
        header = (ct.c_int32 * 21)()
        size = ct.c_size_t()
        pointer = self.lib.oracle_read(str(path).encode(), header, ct.byref(size))
        if not pointer:
            raise ValueError(f'Rust rejected raw native record {path}')
        kinds = {1: '<f4', 2: '<f8', 3: '<i4', 4: '<i4', 6: '<i8'}
        raw = (ct.c_uint8 * size.value).from_address(pointer)
        if header[0] == 5:
            value = bytes(raw)
        else:
            # Rust has validated type/extent. This view is host/device transport,
            # not file decoding, conversion, interpolation or numerical work.
            value = np.ctypeslib.as_array(raw).view(kinds[header[0]]).copy()
            value = value.reshape(tuple(header[3:3+header[2]]), order='F')
        self.lib.oracle_free(pointer, size.value)
        return value

    def write(self, template, dest, value):
        raw = value if isinstance(value, bytes) else np.asfortranarray(value).tobytes(order='F')
        buffer = ct.create_string_buffer(raw)
        if self.lib.oracle_write(str(template).encode(), str(dest).encode(), buffer, len(raw)):
            raise ValueError('Rust refused computed raw native record')


def replay(key, values, cp, cache):
    from hexcore.cuda_acoustic import CudaAcousticState, CudaAcousticForcing, CudaVerticalImplicitCoefficients
    from hexcore.cuda_acoustic_rrfs import compute_vertical_implicit_coefficients_cuda_rrfs, advance_acoustic_step_cuda_rrfs
    gpu = lambda value: cp.ascontiguousarray(cp.asarray(value))
    a = {name.removeprefix('argument.'): value for name, value in values.items() if name.startswith('argument.')}
    scalar = lambda name: a[name].item()
    if key.endswith('atm_advance_scalars_mono_work'):
        from hexcore.cuda_monotonic_rrfs import advance_monotonic_work_cuda_rrfs
        from hexcore.cuda_transport_rrfs import advance_scalar_transport_cuda_rrfs
        mapping={name: value.decode().strip(' \x00') if isinstance(value,bytes) else value.item() if value.ndim==0 else gpu(value) for name,value in a.items()}
        for name in ('cellsonedge','cellsoncell','edgesoncell','advcellsforedge'):
            mapping[name]=gpu(a[name].T)-1
        for name in ('edgesoncell_sign','adv_coefs','adv_coefs_3rd'):
            mapping[name]=gpu(a[name].T)
        for name in ('cellstart','cellsolvestart','edgestart'):
            mapping[name]-=1
        public={name:value.copy() if hasattr(value,'shape') else value for name,value in mapping.items()}
        mesh=SimpleNamespace(cells_on_edge=public['cellsonedge'],cells_on_cell=public['cellsoncell'],
            edges_on_cell=public['edgesoncell'],n_edges_on_cell=public['nedgesoncell'],
            edge_sign_on_cell=public['edgesoncell_sign'],dv_edge=public['dvedge'])
        coefficients=SimpleNamespace(adv_coefs=public['adv_coefs'],adv_coefs_3rd=public['adv_coefs_3rd'],
            n_adv_cells_for_edge=public['nadvcellsforedge'],adv_cells_for_edge=public['advcellsforedge'])
        result=advance_scalar_transport_cuda_rrfs(mesh,SimpleNamespace(inv_area_cell=public['invareacell']),
            public['scalars_old'],public['scalars_new'],public['rho_zz_old'],public['rho_zz_new'],
            public['uhavg'],public['wwavg'],public['dt'],coefficients=coefficients,
            fzm=public['fnm'],fzp=public['fnp'],rdzw=public['rdnw'],
            config_coef_3rd_order=public['coef_3rd_order'],scalar_tendency=public['scalar_tend'],
            rk_step=3,config_scalar_advection=True,config_monotonic=True,config_positive_definite=False,
            config_split_dynamics_transport=public.get('advance_density',True),config_time_integration_order=3,
            mixing_cell_solve_count=public['ncellssolve'],
            config_apply_lbcs=bool(values['module.config_apply_lbcs'].item()),
            bdy_mask_cell=public['bdymaskcell'],bdy_mask_edge=public['bdymaskedge'],
            garbage_cell=public['ncells'],native_work=public,kernel_cache=cache)
        advance_monotonic_work_cuda_rrfs(mapping,config_apply_lbcs=bool(values['module.config_apply_lbcs'].item()),kernel_cache=cache)
        from hexcore.cuda_backend.containers import require_resident_array
        require_resident_array('native monotonic scalars',result.scalars,dtype=np.float32)
        require_resident_array('native monotonic density',result.density,dtype=np.float32)
        density_name='rho_zz_int' if mapping.get('advance_density',True) else 'rho_zz_new'
        if not bool(cp.all(result.density.view(cp.uint32)==mapping[density_name].view(cp.uint32)).item()):
            raise ValueError('production monotonic density differs from complete work')
        for name,value in {'scalars_new':result.scalars,'scalars_old':public['scalars_old'],'scalar_tend':public['scalar_tend']}.items():
            if not bool(cp.all(value.view(cp.uint32)==mapping[name].view(cp.uint32)).item()):
                raise ValueError('production monotonic adapter differs from complete work: '+name)
        # The global forecast's resident carriers omit native dummy storage.
        # Also exercise that production ingress on the same actual PRE words.
        if not bool(values['module.config_apply_lbcs'].item()):
            pnc,pne=public['ncells'],public['nedges']
            compact=lambda name:gpu(a[name][...,:pnc])
            compact_mesh=SimpleNamespace(cells_on_edge=mesh.cells_on_edge[:pne],cells_on_cell=mesh.cells_on_cell[:pnc],
                edges_on_cell=mesh.edges_on_cell[:pnc],n_edges_on_cell=mesh.n_edges_on_cell[:pnc],
                edge_sign_on_cell=mesh.edge_sign_on_cell[:pnc],dv_edge=mesh.dv_edge[:pne])
            compact_coefficients=SimpleNamespace(adv_coefs=coefficients.adv_coefs[:pne],adv_coefs_3rd=coefficients.adv_coefs_3rd[:pne],
                n_adv_cells_for_edge=coefficients.n_adv_cells_for_edge[:pne],adv_cells_for_edge=coefficients.adv_cells_for_edge[:pne])
            old_entry=compact('scalars_old');tend_entry=compact('scalar_tend')
            compact_result=advance_scalar_transport_cuda_rrfs(compact_mesh,SimpleNamespace(inv_area_cell=public['invareacell'][:pnc]),
                old_entry,compact('scalars_new'),compact('rho_zz_old'),compact('rho_zz_new'),gpu(a['uhavg'][...,:pne]),compact('wwavg'),
                public['dt'],coefficients=compact_coefficients,fzm=public['fnm'],fzp=public['fnp'],rdzw=public['rdnw'],
                config_coef_3rd_order=public['coef_3rd_order'],scalar_tendency=tend_entry,
                rk_step=3,config_scalar_advection=True,config_monotonic=True,config_positive_definite=False,
                config_split_dynamics_transport=public.get('advance_density',True),config_time_integration_order=3,kernel_cache=cache)
            require_resident_array('compact monotonic scalars',compact_result.scalars,dtype=np.float32)
            require_resident_array('compact monotonic density',compact_result.density,dtype=np.float32)
            if not bool(cp.all(compact_result.density.view(cp.uint32)==mapping[density_name][...,:pnc].copy().view(cp.uint32)).item()):
                raise ValueError('compact global production density differs from native work')
            for name,value in {'scalars_new':compact_result.scalars,'scalars_old':old_entry,'scalar_tend':tend_entry}.items():
                if not bool(cp.all(value.view(cp.uint32)==mapping[name][...,:pnc].copy().view(cp.uint32)).item()):
                    raise ValueError('compact global production ingress differs from native work: '+name)
        outputs=('scalars_old','scalars_new','scalar_tend','scalar_old','scalar_new','s_min','s_max','wdtn','scale_arr','flux_arr','flux_upwind_tmp','flux_tmp')
        if mapping.get('advance_density',True):outputs+=('rho_zz_int',)
        return {'argument.'+name:cp.asnumpy(mapping[name]) for name in outputs}
    if key.endswith('atm_compute_dyn_tend_work'):
        return replay_dynamics(values,cp,cache)
    if key.endswith('atm_zero_gradient_w_bdy_work'):
        from hexcore.cuda_regional_rrfs import apply_w_boundary_cuda_rrfs
        w=gpu(a['w'])
        apply_w_boundary_cuda_rrfs(w,gpu(a['bdymaskcell']),gpu(a['nearestrelaxationcell'])-1,
            config_lbc_w=a['config_lbc_w'].decode().strip(' \x00'),
            cell_solve_start=scalar('cellsolvestart')-1,cell_solve_end=scalar('cellsolveend'),
            kernel_cache=cache)
        return {'argument.w':cp.asnumpy(w)}
    if scalar('cellsolvestart') != 1:
        raise ValueError('stated real case has a different native solve start; extend launcher before claiming it')
    changed = {}
    if key.endswith('atm_compute_vert_imp_coefs_work'):
        shape = a['zz'].shape
        entry = CudaVerticalImplicitCoefficients(**{
            f.name: gpu(a[f.name]) if f.name in a else cp.zeros(shape, dtype='f4')
            for f in fields(CudaVerticalImplicitCoefficients)})
        aliases = dict(zz='zz', cqw='cqw', pressure='p', theta='t', rho_base='rb', rho_theta_base='rtb',
                       pressure_base='pb', rho_theta_perturbation='rt', rdzw='rdzw', rdzu='rdzu', fzm='fzm', fzp='fzp')
        coefs = compute_vertical_implicit_coefficients_cuda_rrfs(
            dts=scalar('dts'), epssm=scalar('epssm'), cell_solve_count=scalar('cellsolveend'),
            initial_coefficients=entry, qtot=gpu(values['module.qtot']), kernel_cache=cache,
            **{name: gpu(a[alias]) for name, alias in aliases.items()})
        for name in ('cofwr','cofwz','coftz','cofwt','cofrz','a_tri','alpha_tri','gamma_tri'):
            changed['argument.'+name] = cp.asnumpy(getattr(coefs, name))
    elif key.endswith('atm_advance_acoustic_step_work'):
        state_aliases = {'ru_avg':'ruavg', 'ww_avg':'wwavg'}
        state = CudaAcousticState(**{f.name: gpu(a[state_aliases.get(f.name, f.name)])
                                     for f in fields(CudaAcousticState)})
        forcing = CudaAcousticForcing(**{f.name: gpu(a[f.name]) for f in fields(CudaAcousticForcing)})
        coefs = CudaVerticalImplicitCoefficients(**{
            f.name: gpu(a[f.name]) if f.name in a else cp.zeros(a['zz'].shape, dtype='f4')
            for f in fields(CudaVerticalImplicitCoefficients)})
        mesh = SimpleNamespace(cells_on_edge=gpu(a['cellsonedge'].T)-1,
                               edges_on_cell=gpu(a['edgesoncell'].T)-1,
                               n_edges_on_cell=gpu(a['nedgesoncell']),
                               edge_sign_on_cell=gpu(a['edgesoncell_sign'].T), dv_edge=gpu(a['dvedge']))
        context = SimpleNamespace(inv_dc_edge=gpu(a['invdcedge']), inv_area_cell=gpu(a['invareacell']))
        advance_acoustic_step_cuda_rrfs(mesh,state,forcing,coefs,context=context,
            dts=scalar('dts'),small_step=scalar('small_step'),epssm=scalar('epssm'),
            cell_solve_count=scalar('cellsolveend'), edge_start=scalar('edgestart')-1, edge_end=scalar('edgeend'),
            cell_start=scalar('cellstart')-1,cell_end=scalar('cellend'),
            spec_zone_mask_cell=gpu(a['speczonemaskcell']),spec_zone_mask_edge=gpu(a['speczonemaskedge']),
            fzm=gpu(a['fzm']),fzp=gpu(a['fzp']),rdzw=gpu(a['rdzw']),kernel_cache=cache)
        for f in fields(CudaAcousticState):
            changed['argument.'+state_aliases.get(f.name,f.name)] = cp.asnumpy(getattr(state,f.name))
    elif key.endswith('atm_advance_scalars_work'):
        from hexcore.cuda_transport_rrfs import apply_gsl_scalar_mixing_cuda_rrfs, advance_scalars_standard_cuda_rrfs
        mesh=SimpleNamespace(n_edges=a['uhavg'].shape[1],max_edges=a['edgesoncell'].shape[0],
            cells_on_edge=gpu(a['cellsonedge'].T)-1,edges_on_cell=gpu(a['edgesoncell'].T)-1,
            n_edges_on_cell=gpu(a['nedgesoncell']),edge_sign_on_cell=gpu(a['edgesoncell_sign'].T),
            dv_edge=gpu(a['dvedge']))
        context=SimpleNamespace(inv_area_cell=gpu(a['invareacell']),inv_dc_edge=gpu(a['invdcedge']))
        coefficients=SimpleNamespace(adv_coefs=gpu(a['adv_coefs'].T),adv_coefs_3rd=gpu(a['adv_coefs_3rd'].T),
            n_adv_cells_for_edge=gpu(a['nadvcellsforedge']),adv_cells_for_edge=gpu(a['advcellsforedge'].T)-1)
        old=gpu(a['scalar_old']); stage=gpu(a['scalar_new']); source=gpu(a['scalar_tend_save'])
        flux=gpu(a['horiz_flux_arr'])
        if scalar('rk_step')==1 and scalar('scalar_eddy_mix'):
            apply_gsl_scalar_mixing_cuda_rrfs(mesh,context,old,source,
                kdiff=gpu(a['kdiff']),rho_edge=gpu(a['rho_edge']),mesh_scaling_del2=gpu(a['meshscalingdel2']),
                mesh_scaling_del4=gpu(a['meshscalingdel4']) if scalar('h_scalar_filter4')>0 else cp.zeros(mesh.n_edges,'f4'),
                h_scalar_filter4=scalar('h_scalar_filter4'),cell_count=scalar('cellend'),cell_solve_count=scalar('cellsolveend'),kernel_cache=cache)
            stage[:, :, scalar('cellstart')-1:scalar('cellend')]=old[:, :, scalar('cellstart')-1:scalar('cellend')]
        result=advance_scalars_standard_cuda_rrfs(mesh,context,old,stage,gpu(a['rho_zz_old']),gpu(a['rho_zz_new']),
            gpu(a['uhavg']),gpu(a['wwavg']),scalar('dt'),coefficients=coefficients,
            fzm=gpu(a['fnm']),fzp=gpu(a['fnp']),rdzw=gpu(a['rdnw']),rk_step=scalar('rk_step'),
            config_time_integration_order=scalar('config_time_integration_order'),config_coef_3rd_order=scalar('coef_3rd_order'),
            scalar_tendency=source,advance_density=bool(scalar('advance_density')),config_apply_lbcs=bool(values['module.config_apply_lbcs'].item()),
            bdy_mask_cell=gpu(a['bdymaskcell']),bdy_mask_edge=gpu(a['bdymaskedge']),cell_solve_count=scalar('cellsolveend'),
            horiz_flux_array=flux,edge_start=scalar('edgestart')-1,edge_end=scalar('edgeend'),kernel_cache=cache)
        changed.update({'argument.scalar_new':cp.asnumpy(result.scalars),
                        'argument.scalar_tend_save':cp.asnumpy(source),
                        'argument.horiz_flux_arr':cp.asnumpy(flux)})
    else:
        raise ValueError('this replay does not claim other dycore boundaries')
    return changed


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
def replay_dynamics(values,cp,cache):
    from hexcore.cuda_dyn_tend_rrfs import compute_dyn_tend_work_cuda_rrfs
    from hexcore.cuda_backend.containers import DeviceMesh,TransferStats
    from hexcore.cuda_v841 import CudaV841Context
    a={k.removeprefix('argument.'):v for k,v in values.items() if k.startswith('argument.')}
    nc,ne,nv,nl=(int(a[k]) for k in ('ncells','nedges','nvertices','nvertlevels_dummy'))
    me,me2,vd=(int(a[k]) for k in ('maxedges_dummy','maxedges2_dummy','vertexdegree'))
    gpu=lambda v:cp.ascontiguousarray(cp.asarray(v))
    # Regional stencils read the actual native dummy cell at open edges.
    # Detect that connectivity on the device; retain its original pool words.
    padded=bool(cp.any(gpu(a['cellsonedge']) == nc+1).item())
    physical_nc,physical_ne,physical_nv=nc,ne,nv
    nc,ne,nv=(count+int(padded) for count in (nc,ne,nv))
    # These axes follow pinned declarations, never shape-based guessing.
    edge_names=set('fedge dvedge dcedge invdcedge invdvedge meshscalingdel2 meshscalingdel4 weightsonedge rho_edge ru u v tend_u pv_edge zxu cqu ru_save tend_u_euler deriv_two cellsonedge verticesonedge edgesonedge nedgesonedge latedge angleedge advcellsforedge nadvcellsforedge adv_coefs adv_coefs_3rd'.split())
    vertex_names=set('invareatriangle vorticity edgesonvertex edgesonvertex_sign'.split())
    cell_names=set('invareacell zgrid rho_zz divergence ke theta_m rw tend_rho rt_diabatic_tend tend_theta tend_w w cqw rb rr pp pressure_b zz h_divergence kdiff edgesoncell_sign rw_save theta_m_save exner rr_save scalars tend_w_euler tend_theta_euler edgesoncell cellsoncell nedgesoncell latcell t_init ur_cell vr_cell defc_a defc_b tend_w_pgf tend_w_buoy rthdynten'.split())
    resident={}
    for name,value in a.items():
        if isinstance(value,bytes):resident[name]=value.decode().strip(' \x00');continue
        if value.ndim==0:resident[name]=value.item();continue
        count=ne if name in edge_names else nv if name in vertex_names else nc if name in cell_names else None
        resident[name]=gpu(value if count is None else value[..., :count])
    indexes=dict(cells_on_edge='cellsonedge',vertices_on_edge='verticesonedge',edges_on_cell='edgesoncell',
        cells_on_cell='cellsoncell',edges_on_edge='edgesonedge',edges_on_vertex='edgesonvertex')
    mesh_arrays={k:gpu(resident[v].T-1) for k,v in indexes.items()}
    if padded:
        # Only the unused owner row receives safe device geometry. Native
        # loops never visit it (5148-6409); no original argument is changed.
        for name,sentinel in dict(cells_on_edge=nc-1,vertices_on_edge=nv-1,
                                  edges_on_cell=ne-1,cells_on_cell=nc-1,
                                  edges_on_edge=ne-1,edges_on_vertex=ne-1).items():
            mesh_arrays[name][-1,:]=sentinel
    mesh_arrays.update(n_edges_on_cell=resident['nedgesoncell'],n_edges_on_edge=resident['nedgesonedge'])
    for name,alias in dict(edge_sign_on_cell='edgesoncell_sign',weights_on_edge='weightsonedge',
        dc_edge='dcedge',dv_edge='dvedge',lat_cell='latcell',lat_edge='latedge',angle_edge='angleedge',
        defc_a='defc_a',defc_b='defc_b',f_edge='fedge').items():
        v=resident[alias];mesh_arrays[name]=gpu(v.T) if v.ndim==2 else v
    # Use the same native-input selection as both production constructors.
    # Rebuilding on a regional cut would discard inherited border weights.
    from types import SimpleNamespace
    from hexcore.deformation_rrfs import select_deformation_weights_rrfs
    selected=select_deformation_weights_rrfs(SimpleNamespace(
        nEdgesOnCell=resident['nedgesoncell'],edgesOnCell=mesh_arrays['edges_on_cell'],
        defc_a=mesh_arrays['defc_a'],defc_b=mesh_arrays['defc_b']),sphere_radius=1.0)
    mesh_arrays.update(selected)
    # Unused diagnostic-only mesh slots are never consumed by this work routine.
    mesh=DeviceMesh(**mesh_arrays,vertices_on_cell=cp.zeros((nc,me),'i4'),cells_on_vertex=cp.zeros((nv,vd),'i4'),
        area_cell=cp.reciprocal(resident['invareacell']),area_triangle=cp.reciprocal(resident['invareatriangle']),
        kite_areas_on_vertex=cp.ones((nv,vd),'f4'),lon_cell=cp.zeros(nc,'f4'),lon_edge=cp.zeros(ne,'f4'),
        mesh_density=cp.ones(nc,'f4'),f_vertex=cp.zeros(nv,'f4'),spec_zone_mask_edge=cp.zeros(ne,'f4'),
        n_cells=nc,n_edges=ne,n_vertices=nv,max_edges=me,max_edges2=me2,vertex_degree=vd,
        nominal_min_dc=float(resident['config_len_disp']),dtype=np.dtype('f4'),index_dtype=np.dtype('i4'),h2d=TransferStats(0,0))
    context=CudaV841Context(**{name:resident[alias] for name,alias in dict(inv_area_cell='invareacell',
        inv_area_triangle='invareatriangle',inv_dc_edge='invdcedge',inv_dv_edge='invdvedge',u_init='u_init',v_init='v_init').items()},
        etp=cp.ones(nl,'f4'),etm=cp.ones(nl,'f4'),ewp=cp.ones(nl+1,'f4'),ewm=cp.ones(nl+1,'f4'),h2d=TransferStats(0,0))
    private={k.removeprefix('module.'):gpu(v[..., :ne if k=='module.tend_ru_physics' or k=='module.delsq_u' else nv if k=='module.delsq_vorticity' else nc])
             for k,v in values.items() if k.startswith('module.') and not k.endswith('.present0') and not k.endswith('.present1') and getattr(v,'ndim',0)>0}
    trig={}
    if resident['config_v_mom_eddy_visc2']>0 or resident['config_v_theta_eddy_visc2']>0:
        from hexcore.deformation_rrfs import initialize_edge_trigonometry_rrfs
        trig['edge_cos'],trig['edge_sin']=(gpu(v) for v in initialize_edge_trigonometry_rrfs(a['angleedge'][:ne]))
    changed,work=compute_dyn_tend_work_cuda_rrfs(resident,private,mesh=mesh,context=context,kernel_cache=cache,**trig)
    output={}
    for kind,data in (('argument.',changed),('module.',work)):
        for name,value in data.items():
            label=kind+name;entry=gpu(values[label])
            count=physical_ne if name in edge_names or name in ('delsq_u',) else physical_nv if name=='delsq_vorticity' else physical_nc
            entry[..., :count]=value[..., :count]
            output[label]=cp.asnumpy(entry)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank','out','transport','words'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    args=parser.parse_args()
    start=time.monotonic()
    contract=json.loads(args.contract.read_text())
    required_calls=validate_capture_bank(args.bank,args.manifest,contract)
    import cupy as cp
    from hexcore.cuda_compile_rrfs import RRFSKernelCache
    io=Records(args.transport)
    args.out.mkdir(parents=True,exist_ok=False)
    cache=RRFSKernelCache()
    cases=[]
    for key in contract['calls']:
        prefixes=sorted(required_calls[key])
        for prefix in prefixes:
            inputs={p.name.split('.pre.')[1][:-4]:p for p in args.bank.glob(prefix+'.pre.*.bin')}
            wanted={p.name.split('.post.')[1][:-4]:p for p in args.bank.glob(prefix+'.post.*.bin')}
            if inputs.keys() != wanted.keys():
                raise ValueError('incomplete native PRE/POST capture: '+prefix)
            values={name:io.read(path) for name,path in inputs.items()}
            changed=replay(key,values,cp,cache)
            comparisons=[]
            for label,template in inputs.items():
                computed=args.out/(prefix+'.post.'+label+'.bin')
                io.write(template,computed,changed.get(label,values[label]))
                result=subprocess.run([str(args.words.resolve()),'compare',str(wanted[label]),str(computed)],capture_output=True,text=True)
                if not result.stdout:
                    raise ValueError('Rust comparison failed: '+result.stderr)
                comparison=json.loads(result.stdout)
                comparisons.append(dict(field=label,**comparison))
                computed.unlink()
            cases.append(dict(capture=prefix,comparisons=comparisons))
            print(prefix, sum(c.get('different_words',0) or 0 for c in comparisons),flush=True)
            (args.out/'replay.json').write_text(json.dumps(dict(
                scope=contract['scope'] + '; every captured POST field including untouched padding; capture passivity separately required',
                source_authority=contract.get('source_authority'),
                cases=cases,
                compared_words=sum(c['words'] for case in cases for c in case['comparisons']),
                different_words=sum(c.get('different_words',0) or 0 for case in cases for c in case['comparisons']),
                output_records=sum(len(case['comparisons']) for case in cases),
                elapsed_seconds=time.monotonic()-start,
                passed=bool(cases) and all(c['bitwise_equal'] for case in cases for c in case['comparisons']),
                compile=cache.compile_manifest()),indent=2)+'\n')
    actual_words=sum(c['words'] for case in cases for c in case['comparisons'])
    actual_records=sum(len(case['comparisons']) for case in cases)
    if actual_words!=contract['compared_words'] or actual_records!=contract['output_records']:
        raise ValueError('replayed output inventory differs from the authenticated case set')
    if not cases or any(not c['bitwise_equal'] for case in cases for c in case['comparisons']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
