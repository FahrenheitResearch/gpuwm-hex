"""Batched device physics of the member init: vertical interpolation, hydrostatic adjustment, winds, soil.

Results must pass the independent CPU identity gate (see :mod:`hexcore.init_gpu`).
"""
from pathlib import Path
import calendar
import hashlib
import importlib.util
import time
import numpy as np


def engine_kernel_root():
    """The installed gpuwm's root, whose ``gpuwm/core/kernels`` carries the binary32 libm headers."""
    spec = importlib.util.find_spec('gpuwm')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('physics_batch needs the gpuwm engine (its gpuwm/core/kernels float maths headers); '
                           'install gpuwm or pass engine_root=')
    return Path(list(spec.submodule_search_locations)[0]).parent


def load_geometry(static, capsule):
    from netCDF4 import Dataset
    result = {}
    with Dataset(static) as ds:
        ds.set_auto_maskandscale(False)
        for name in ('latCell','lonCell','latEdge','lonEdge','angleEdge','landmask',
                     'cellsOnEdge','nEdgesOnCell','edgesOnCell','soiltemp','albedo12m',
                     'greenfrac','snoalb','ivgtyp','isltyp','isice_lu'):
            integer=name in ('landmask','cellsOnEdge','nEdgesOnCell','edgesOnCell','ivgtyp','isltyp','isice_lu')
            result[name]=np.asarray(ds[name][:],dtype='i4' if integer else 'f4').squeeze()
    with Dataset(capsule) as ds:
        ds.set_auto_maskandscale(False)
        result['nz']=len(ds.dimensions['nVertLevels'])
        for name in ('ter','zgrid','zz','fzm','fzp','dzu','rdzw','zb','zb3'):
            result[name]=np.asarray(ds[name][:],dtype='f4').squeeze()
    return result


def monthly_weights(date):
    year,month,day=map(int,date[:10].split('-'))
    days=[calendar.monthrange(year,m)[1] for m in range(1,13)]
    middle=[year*1000+sum(days[:m])+15 for m in range(12)]
    middle=[middle[0]-31,*middle,middle[-1]+31]
    target=year*1000+sum(days[:month-1])+day
    for l in range(13):
        if middle[l]<target<=middle[l+1]:
            m1,m2=(12,1) if l in (0,12) else (l,l+1)
            return (np.int32(m1-1),np.int32(m2-1),np.float32(middle[l+1]-target),
                    np.float32(target-middle[l]),np.float32(middle[l+1]-middle[l]))
    raise ValueError(date)


def physics_batch(kernels, fg, mesh, *, start_time, engine_root=None,
                  extrap_airtemp='lapse-rate', use_spechumd=False,
                  virtual_factor='reproduce-fortran', deep_soil_moisture='reproduce-fortran',
                  theta_adv_order=3, coef_3rd_order=.25, frac_seaice=True, tsk_seaice_threshold=100.):
    if virtual_factor not in ('reproduce-fortran','consistent'):
        raise ValueError(f'unknown virtual_factor: {virtual_factor}')
    if deep_soil_moisture not in ('reproduce-fortran','corrected'):
        raise ValueError(f'unknown deep_soil_moisture: {deep_soil_moisture}')
    if theta_adv_order not in (2,3):raise ValueError('theta_adv_order must be 2 or 3')
    cp=kernels.cp;started=time.monotonic()
    nb,nc,stride=fg['t'].shape;ne=fg['u'].shape[1];nz=mesh['nz'];nf=len(fg['vert_level']);ns=fg['st'].shape[-1]
    if not (2<=nf<=128 and 2<=nz<=128 and 1<=ns<=128):raise ValueError('unsupported column extent')
    headers=[];hashes={}
    engine_root=engine_kernel_root() if engine_root is None else engine_root
    for name in ('glibc_flt32.cuh','glibc_trig_flt32.cuh'):
        p=Path(engine_root)/'gpuwm/core/kernels'/name
        content=p.read_bytes();headers.append(content.decode());hashes[name]=hashlib.sha256(content).hexdigest()
    source=Path(__file__).with_name('physics.cu').read_text()
    module=kernels.compile('\n'.join(headers)+'\n'+source)
    device={name:cp.asarray(value) for name,value in mesh.items() if name!='nz'}
    I=np.int32;F=np.float32
    err=cp.zeros(1,dtype='i4')
    def launch(name,count,args):
        module.get_function(name)(((count+127)//128,),(128,),tuple(args))
    shape=(nb,nc,nz)
    vertical={name:cp.empty(shape,dtype='f4') for name in ('t','rh','sh','p')}
    vertical['psfc']=cp.empty((nb,nc),dtype='f4')
    isobaric=cp.all(fg['p'][:,:,:nf]==0,axis=(1,2)).astype('i4')
    levels=cp.asarray(fg['vert_level']);sfc=int(np.flatnonzero(fg['vert_level']==np.float32(200100.))[0])
    launch('vertical_cells',nb*nc,[fg[n] for n in ('t','rh','sh','p','z','psfc','soilz')]+
           [levels,device['zgrid'],isobaric,I(nc),I(nb),I(stride),I(nf),I(nz),
            I({'constant':0,'linear':1,'lapse-rate':2}[extrap_airtemp])]+
           [vertical[n] for n in ('t','rh','sh','p','psfc')]+[err])
    computed={'u':cp.empty((nb,ne,nz),dtype='f4')}
    launch('vertical_edges',nb*ne,[fg['u'],fg['v'],fg['z'],levels,device['angleEdge'],
           device['cellsOnEdge'],device['zgrid'],I(nc),I(ne),I(nb),I(stride),I(nf),I(nz),computed['u'],err])
    if int(err.get()[0]):raise ValueError('temperature extrapolation above source top')
    for name in ('qv','theta','rho','rho_base','theta_base','rho_zz','relhum'):
        computed[name]=cp.empty(shape,dtype='f4')
    for name in ('precipw','surface_pressure'):
        computed[name]=cp.empty((nb,nc),dtype='f4')
    iterations=cp.empty(shape,dtype='i4')
    use_sh=(cp.any(vertical['sh']!=0,axis=(1,2)) & use_spechumd).astype('i4')
    late=F(1.61 if virtual_factor=='reproduce-fortran' else F(461.6)/F(287.0)-F(1))
    launch('dynamics_cells',nb*nc,[vertical[n] for n in ('t','p','rh','sh')]+
           [device[n] for n in ('zgrid','zz','fzm','fzp','dzu')]+
           [F(mesh['rdzw'][0]),use_sh,I(nc),I(nb),I(nz),late]+
           [computed[n] for n in ('qv','theta','rho','rho_base','theta_base','rho_zz','precipw','surface_pressure','relhum')]+[iterations])
    computed['w']=cp.empty((nb,nc,nz+1),dtype='f4')
    launch('w_cells',nb*nc*(nz+1),[computed['u'],computed['rho_zz']]+
           [device[n] for n in ('zz','fzm','fzp','zb','zb3','cellsOnEdge','nEdgesOnCell','edgesOnCell')]+
           [I(nc),I(ne),I(nb),I(nz),I(mesh['edgesOnCell'].shape[-1]),I(theta_adv_order),F(coef_3rd_order),computed['w']])
    for name in ('skintemp','tmn','xice','seaice','snow','snowc','snowh','xland','sfc_albbck','vegfra','snoalb','shdmin','shdmax','q2'):
        computed[name]=cp.empty((nb,nc),dtype='f4')
    for name in ('tslb','smois','sh2o','dzs','zs'):
        computed[name]=cp.empty((nb,nc,4),dtype='f4')
    for name in ('ivgtyp','isltyp'):computed[name]=cp.empty((nb,nc),dtype='i4')
    computed['t2m']=cp.ascontiguousarray(fg['t'][:,:,sfc]);computed['rh2']=cp.ascontiguousarray(fg['rh'][:,:,sfc])
    soil_errors=cp.zeros(3,dtype='i4');counts=cp.zeros(4,dtype='i4')
    launch('soil_cells',nb*nc,[device[n] for n in ('landmask','ter','soiltemp','albedo12m','greenfrac','snoalb','ivgtyp','isltyp')]+
           [I(mesh['isice_lu'])]+[fg[n] for n in ('soilz','skintemp','snow','xice','st','sm')]+
           [cp.asarray(fg['dzs_cm']),computed['t2m'],computed['rh2'],vertical['psfc'],I(nc),I(nb),I(ns),
            I(deep_soil_moisture=='corrected'),I(frac_seaice),F(tsk_seaice_threshold),*monthly_weights(start_time)]+
           [computed[n] for n in ('skintemp','tmn','tslb','smois','sh2o','xice','seaice','snow','snowc','snowh',
                                'xland','sfc_albbck','vegfra','snoalb','ivgtyp','isltyp','shdmin','shdmax','dzs','zs','q2')]+
           [soil_errors,counts])
    failures=soil_errors.get()
    if np.any(failures):raise ValueError(f'soil consistency failed: {failures.tolist()}')
    for name in ('qc','qr','tke'):computed[name]=cp.zeros(shape,dtype='f4')
    for name in ('u10','v10'):computed[name]=fg[name]
    computed['sst']=fg['skintemp']
    computed.pop('rho_zz')
    cp.cuda.get_current_stream().synchronize()
    return computed,vertical,dict(seconds=time.monotonic()-started,math_headers=hashes,
                                  soil_counts=counts.get().tolist(),hydrostatic_max=int(cp.max(iterations).get()))
