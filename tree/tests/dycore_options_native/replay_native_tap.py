"""Complete CUDA module replay on authenticated f1 native mesh/state captures.

The real cases are output-diagnostics PRE snapshots at native steps 0,18,72,
MPI rank count one. All 256 module branch selections and all native output
words in the physical mesh storage are compared; no tolerance or exclusion.
"""
from pathlib import Path
import argparse, hashlib, json, os, platform, subprocess, struct, sys, time
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_case(folder):
    layout=json.loads((folder/'layout.json').read_text());raw=(folder/'tap-input.bin').read_bytes();offset=0;d={}
    for name, spec in layout['records'].items():
        if sha(folder/name) != spec['sha256']:
            raise ValueError('changed staged native capture: '+name)
    for row in layout['fields']:
        shape=tuple(row['shape']);count=int(np.prod(shape))
        # Rust transported the native record. This is only driver buffer mapping.
        d[row['name']]=np.frombuffer(raw,dtype=row['dtype'],offset=offset,count=count).reshape(shape,order='F')
        offset+=4*count
    assert offset==len(raw)
    return layout,d

def setup(layout,d,angle_engine):
    import cupy as cp
    from hexcore.cuda_backend import DeviceMesh,TransferStats,KernelCache
    from hexcore.cuda_v841 import CudaV841Context
    from hexcore.cuda_horizontal_v841 import CudaHorizontalV841
    nc,ne,nv,nl,me=(layout[k] for k in ['nc','ne','nv','nl','me'])
    def pack(a):return cp.asarray(np.ascontiguousarray(a))
    idx={'cells_on_edge':d['coe'][:,:ne].T-1,'vertices_on_edge':d['voe'][:,:ne].T-1,
         'edges_on_cell':d['eoc'][:,:nc].T-1,'edges_on_vertex':d['eov'][:,:nv].T-1,
         'n_edges_on_cell':d['count'][:nc], 'cells_on_cell':np.zeros((nc,me),np.int32),
         'edges_on_edge':np.zeros((ne,2*me),np.int32),'n_edges_on_edge':np.zeros(ne,np.int32),
         'vertices_on_cell':np.zeros((nc,me),np.int32),'cells_on_vertex':np.zeros((nv,3),np.int32)}
    floats={'edge_sign_on_cell':d['ecs'][:,:nc].T,'weights_on_edge':np.zeros((ne,2*me),np.float32),
        'dc_edge':d['dc'][:ne],'dv_edge':d['dv'][:ne], 'area_cell':d['actual_area'][:nc],
        'area_triangle':d['actual_at'][:nv], 'kite_areas_on_vertex':np.ones((nv,3),np.float32),
        'lat_cell':np.zeros(nc,np.float32),'lon_cell':np.zeros(nc,np.float32),
        'lat_edge':np.zeros(ne,np.float32),'lon_edge':np.zeros(ne,np.float32),
        'angle_edge':d['angle'][:ne],'mesh_density':d['density'][:nc],
        'defc_a':np.zeros((nc,me),np.float32),'defc_b':np.zeros((nc,me),np.float32),
        'f_vertex':np.zeros(nv,np.float32),'f_edge':np.zeros(ne,np.float32),'spec_zone_mask_edge':np.zeros(ne,np.float32)}
    mesh=DeviceMesh(**{k:pack(v) for k,v in {**idx,**floats}.items()},n_cells=nc,n_edges=ne,
        n_vertices=nv,max_edges=me,max_edges2=2*me,vertex_degree=3,nominal_min_dc=float(d['length']),
        dtype=np.dtype(np.float32),index_dtype=np.dtype(np.int32),h2d=TransferStats(0,0))
    context=CudaV841Context(inv_area_cell=pack(d['area'][:nc]),inv_area_triangle=pack(d['at'][:nv]),
        inv_dc_edge=pack(d['idc'][:ne]),inv_dv_edge=pack(d['idv'][:ne]),
        etp=cp.zeros(nl,cp.float32),etm=cp.zeros(nl,cp.float32),ewp=cp.zeros(nl+1,cp.float32),
        ewm=cp.zeros(nl+1,cp.float32),u_init=pack(d['ui']),v_init=pack(d['vi']),h2d=TransferStats(0,0))
    h=CudaHorizontalV841(mesh,nl,context,kernel_cache=KernelCache())
    h._deformation_v841={k:pack(d[n][:,:nc].T) for k,n in [('coef_c2','defc2'),('coef_s2','defs2'),('coef_cs','defcs')]}
    inputs={k:pack(d[k][:,:n]) for k,n in [('u',ne),('v',ne),('w',nc),('theta',nc),('re',ne),('div',nc),('vort',nv),('rho',nc),('z',nc),('initial',nc)]}
    inputs.update(q=pack(d['q'][:,:,:nc]),ui=pack(d['ui']),vi=pack(d['vi']),zw=pack(d['zw']),zu=pack(d['zu']))
    raw=subprocess.run([str(angle_engine),'--protocol','mpas-nonles-angles-v1'],input=b'NONLES01'+struct.pack('<Q',ne)+d['angle'][:ne].tobytes(),stdout=subprocess.PIPE,check=True).stdout
    assert len(raw)==ne*8
    inputs.update(cos=pack(np.frombuffer(raw[:ne*4],dtype='<f4')),sin=pack(np.frombuffer(raw[ne*4:],dtype='<f4')))
    return h,inputs

def outputs(folder,branch,l):
    nl,nc,ne,nv,ns=(l[k] for k in ['nl','nc','ne','nv','ns']);raw=(folder/f'output-{branch}.bin').read_bytes();offset=0;out={}
    for name,shape in [('tend_u_euler',(nl,ne+1)),('tend_w_euler',(nl+1,nc+1)),('tend_theta_euler',(nl,nc+1)),('tend_scalars',(ns,nl,nc+1)),('delsq_u',(nl,ne+1)),('delsq_divergence',(nl,nc+1)),('delsq_vorticity',(nl,nv+1)),('delsq_w',(nl,nc+1)),('delsq_theta',(nl,nc+1)),('kdiff',(nl,nc+1))]:
        size=int(np.prod(shape));out[name]=np.frombuffer(raw,dtype='<f4',offset=offset,count=size).reshape(shape,order='F')[...,:shape[-1]-1];offset+=size*4
    assert offset==len(raw)
    return out

def run(bank,out,angle_engine):
    started=time.monotonic()
    import cupy as cp
    from hexcore.mixing_v842 import V842MixingConfig,compute_nonles_mixing_tendencies_v842
    from hexcore.cuda_upper_damping_v842 import apply_cam_floor_cuda,apply_rayleigh_u_cuda,direct_cache,CUDA_OPTIONS
    import hexcore.mixing_v842 as port
    authority=json.loads((bank/'authority.json').read_text())
    for case in authority['cases']:
        if sha(bank/case['path']/'layout.json') != case['layout_sha256']:
            raise ValueError('changed staged native layout')
    rows=[];different=0;words=0
    def compare(label,actual,expected):
        nonlocal different,words
        a=cp.asnumpy(actual);assert a.shape==expected.shape,(label,a.shape,expected.shape)
        n=int(np.count_nonzero(a.view(np.uint32)!=expected.view(np.uint32)));different+=n;words+=expected.size
        rows.append({'case':label,'words':expected.size,'different_words':n,'cuda_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'native_sha256':hashlib.sha256(expected.tobytes()).hexdigest()})
        if n:print('DIFFERENCE',label,n,flush=True)
    for folder in sorted(bank.glob('step-*')):
        l,d=read_case(folder);h,x=setup(l,d,angle_engine);nc,ne,nl=(l[k] for k in ['nc','ne','nl'])
        for b in range(1,257):
            bits=b-1;cfg=V842MixingConfig(config_horiz_mixing='2d_smagorinsky' if bits&32 else '2d_fixed',
                config_h_theta_eddy_visc2=37.5,config_h_mom_eddy_visc4=0 if bits&8 else 9.e8,
                config_h_theta_eddy_visc4=0 if bits&8 else 2.e9,config_visc4_2dsmag=0 if bits&8 else .05,
                config_v_mom_eddy_visc2=0 if bits&16 else 3.7,config_v_theta_eddy_visc2=0 if bits&16 else 1.9,
                config_mix_full=bool(bits&1),config_mix_scalars=bool(bits&2),config_h_ScaleWithMesh=not bool(bits&64),config_mpas_cam_coef=2 if bits&128 else 0)
            tq=cp.zeros_like(x['q'])
            result=compute_nonles_mixing_tendencies_v842(h,x['u'],x['v'],x['w'],x['theta'],x['re'],x['div'],x['vort'],dt=1200,config=cfg,
                rho_zz=x['rho'],zgrid=x['z'],rdzw=x['zw'],rdzu=x['zu'],u_init=x['ui'],v_init=x['vi'],t_init=x['initial'],
                cos_angle=x['cos'],sin_angle=x['sin'],scalars=x['q'],tend_scalars=tq,dynamics_substep=2 if bits&4 else 1)
            for key,expected in outputs(folder,b,l).items():compare(f'{folder.name}:{b}:{key}',tq if key=='tend_scalars' else getattr(result,key),expected)
        for b,(coef,camlevels,raylevels,days,solved) in {1:(2,4,6,5,ne-1),2:(.00001,8,8,.03125,ne),3:(0,1,1,2.7,0)}.items():
            raw=(folder/f'upper-{b}.bin').read_bytes();size=nl*(nc+1)
            ca=np.frombuffer(raw,dtype='<f4',count=size).reshape((nl,nc+1),order='F')[:,:nc]
            ray=np.frombuffer(raw,dtype='<f4',offset=size*4).reshape((nl,ne+1),order='F')[:,:ne]
            viscosity=cp.full((nl,nc),37.5,cp.float32);tendency=x['u'].copy()
            apply_cam_floor_cuda(viscosity,length=float(d['length']),coefficient=coef,levels=camlevels)
            apply_rayleigh_u_cuda(tendency,x['re'],x['u'],levels=raylevels,days=days,solved=solved)
            compare(f'{folder.name}:CAM:{b}',viscosity,ca);compare(f'{folder.name}:Rayleigh:{b}',tendency,ray)
        print(folder.name,'complete',words,'words',different,'different',flush=True)
    receipt={'status':'PASS' if different==0 and len(rows)==7698 else 'FAIL','native_process_count':1,'hex_process_count':1,
        'comparisons':len(rows),'words':int(words),'different_words':different,'excluded_words':[],
        'scope':'f1 native real full mesh/state PRE snapshots at steps 0,18,72; all256 non-LES branch selections and CAM/Rayleigh; all physical-mesh output words',
        'native_authority':authority,'mixing_module_sha256':sha(port.__file__),'replay_source_sha256':sha(__file__),
        'native_binary_sha256':sha(bank/'deck'),'angle_engine_sha256':sha(angle_engine),'cuda_options':list(CUDA_OPTIONS),'compiler':'direct NVRTC','compile_inventory':direct_cache().inventory,
        'execution_host':platform.node(),'replay_seconds':time.monotonic()-started,
        'cuda_device':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
        'native_compiler':subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        'native_flags':'-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fPIC -cpp',
        'rows':rows}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n')
    return 0 if receipt['status']=='PASS' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--angle-engine',type=Path,required=True);a=p.parse_args();sys.exit(run(a.bank,a.receipt,a.angle_engine))
