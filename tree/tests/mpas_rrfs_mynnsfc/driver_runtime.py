# Appended verbatim to the generated runtime module by assemble.py.
# RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-1052,1148-1463 @1d6e370e
_DIAG_INPUT = ('br','cpm','chs','chs2','cqs','cqs2','flhc','flqc','gz1oz0',
               'hfx','qfx','qgh','qsfc','lh','mol','psim','psih','rmol',
               'ust','ustm','wspd','znt','zol','mavail','ch')
_SURFACE = {'tsk':'skintemp','xland':'xland','xice':'xice','snowh':'snowh'}
_WRITE = tuple(n for n in OUTPUTS if n not in ('pblh','stress','hflx','qflx','cm','fm','fh','fm10','fh2'))
_DIAG_NAMES = {'t2':'t2m','th2':'th2m'}
_BLEND = ('br','gz1oz0','mol','psih','psim','rmol','ust','wspd','zol','q2',
          't2','th2','u10','v10','cd','cda','ck','cka','ustm','ch')
DRIVER_SOURCE = r'''
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:517-622 @1d6e370e
extern "C" __global__ void mynnsfc_from_mpas(float *f,float *sea,
    const float *density,const float *sst,const float *ice,
    float len_disp,float threshold,int fraction,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    // Strict GNU 15.2 emits powf(x,0.25); see native power probe.
    F(dx)=__fdiv_rn(len_disp,SL_POW(density[i],0.25f));
    if(fraction) {
        for(int j=0;j<FIELD_COUNT;++j) sea[j*n+i]=f[j*n+i];
        sea[IDX_qsfc*n+i]=0.0f;
        if(ice[i]>=threshold && ice[i]<1.0f) {
            sea[IDX_xland*n+i]=2.0f;
            F(xland)=1.0f;
            sea[IDX_mavail*n+i]=1.0f;
            sea[IDX_znt*n+i]=0.0001f;
            sea[IDX_tsk*n+i]=fmaxf(sst[i],271.4f);
        }
    }
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:903-927,973-991 @1d6e370e
extern "C" __global__ void mynnsfc_to_mpas(float *f,const float *sea,
    const float *ice,float threshold,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || ice[i]<threshold || ice[i]>=1.0f) return;
    float x=ice[i],y=__fsub_rn(1.0f,x);
    for(int j=0;j<FIELD_COUNT;++j)
        if(BLEND_FIELD(j)) f[j*n+i]=__fadd_rn(__fmul_rn(f[j*n+i],x),
                                            __fmul_rn(y,sea[j*n+i]));
}
'''


def _driver_source():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:517-622,903-991 @1d6e370e."""
    macros='\n'.join('#define IDX_'+name+' '+str(i) for i,name in enumerate(FIELDS))
    macros+='\n#define FIELD_COUNT '+str(len(FIELDS))
    macros+='\n#define F(name) f[IDX_##name*n+i]\n'
    macros+='#define BLEND_FIELD(j) ('+' || '.join('(j)==IDX_'+k for k in _BLEND)+')\n'
    return macros+DRIVER_SOURCE


def from_mpas(mesh, diagnostics, surface, profiles, *, len_disp,
              xice_threshold, config_frac_seaice=False):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-698 @1d6e370e.

    Profiles hold prepared MPAS atmosphere arrays, float32 (level, cell).
    psfcpa is the prepared surface-pressure view from MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:602.
    The enclosing adapter owns atmosphere preparation and solve-cell views.
    """
    import cupy as cp
    n=surface['xland'].size
    for name in ('u3d','v3d','t3d','qv3d','p3d','dz8w','th3d','rho3d'):
        value=profiles[name]
        if (not isinstance(value,cp.ndarray) or value.dtype!=cp.float32
                or value.ndim!=2 or value.shape[0]<2 or value.shape[1]!=n
                or not value.flags.c_contiguous):
            raise ValueError('MYNN-SFC profile '+name+' requires contiguous float32 (levels>=2, nCells)')
    packed=cp.zeros((len(FIELDS),n),cp.float32)
    for name in _DIAG_INPUT: packed[FIELDS.index(name)]=diagnostics[name]
    packed[FIELDS.index('pblh')]=diagnostics['hpbl']
    for name,key in _SURFACE.items(): packed[FIELDS.index(name)]=surface[key]
    packed[FIELDS.index('psfcpa')]=profiles['psfcpa']
    for name in ('u3d','v3d','t3d','qv3d','p3d','dz8w','th3d','rho3d'):
        packed[FIELDS.index(name)]=profiles[name][0]
    for name in ('u3d','v3d','dz8w'): packed[FIELDS.index(name+'2')]=profiles[name][1]
    sea=cp.empty_like(packed) if config_frac_seaice else packed
    sst=surface['sst'] if config_frac_seaice else packed[0]
    if n:
        module().get_function('mynnsfc_from_mpas')(((n+127)//128,),(128,),
            (packed,sea,mesh['meshDensity'],sst,surface['xice'],np.float32(len_disp),
             np.float32(xice_threshold),np.int32(config_frac_seaice),np.int32(n)))
    return packed,sea


def to_mpas(packed, diagnostics, surface, *, sea=None,xice_threshold):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:772-991 @1d6e370e.

    hpbl and regime are unchanged in these native copy-back branches.
    cqs, flhc, flqc and the fluxes retain the first call's values.
    """
    n=packed.shape[1]
    if sea is not None and n:
        module().get_function('mynnsfc_to_mpas')(((n+127)//128,),(128,),
            (packed,sea,surface['xice'],np.float32(xice_threshold),np.int32(n)))
    for name in _WRITE: diagnostics[_DIAG_NAMES.get(name,name)][...]=packed[FIELDS.index(name)]
    return diagnostics


def driver(mesh, diagnostics, surface, profiles, *, itimestep,len_disp,
           xice_threshold,config_frac_seaice=False,config_do_restart=False,
           config_do_DAcycling=False,config_lsm_scheme='sf_ruc',
           config_mynn_sfcflux_land=0,config_mynn_sfcflux_water=0,spp_pbl=0):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1148-1209,1390-1463 @1d6e370e."""
    import cupy as cp
    initflag=int(not (config_do_restart or config_do_DAcycling or itimestep>1))
    lsm={'sf_noaa':1,'sf_noahmp':2,'sf_ruc':3}.get(config_lsm_scheme,4)
    values=dict(ivegsrc=0,redrag=0,sfc_z0_type=0,spp_pbl=spp_pbl,
        sf_mynn_sfcflux_water=config_mynn_sfcflux_water,sf_mynn_sfcflux_land=config_mynn_sfcflux_land,
        shalwater_z0=0,isfflx=1,restart=int(config_do_restart),cycling=int(config_do_DAcycling),
        initflag=initflag,flag_iter=1,flagc_lsm=lsm,itimestep=itimestep)
    options=cp.asarray([values[name] for name in OPTIONS],cp.int32)
    packed,sea=from_mpas(mesh,diagnostics,surface,profiles,len_disp=len_disp,
                         xice_threshold=xice_threshold,config_frac_seaice=config_frac_seaice)
    errors=run_packed(packed,options,xice_threshold=xice_threshold)
    if config_frac_seaice:
        # hpbl_p is shared by both native calls; the first cold-start update
        # is visible to the sea call, without modifying diag_physics%hpbl.
        sea[FIELDS.index('pblh')]=packed[FIELDS.index('pblh')]
        errors=run_packed(sea,options,xice_threshold=xice_threshold,errors=errors)
    result=to_mpas(packed,diagnostics,surface,sea=sea if config_frac_seaice else None,
                   xice_threshold=xice_threshold)
    return result


def launch_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1148-1463 @1d6e370e."""
    if (call.slot,call.scheme,str(call.backend))!=('surface_layer',SCHEME,BACKEND):
        raise ValueError('MYNN-SFC launcher requires surface_layer/sf_mynnsfclay/hex-cuda')
    pools=call.pools
    mesh,diag,sfc,atm,configs,state=(pools[k] for k in
        ('mesh','diag_physics','sfc_input','atm_input','configs','scalar_state'))
    return driver(mesh,diag,sfc,atm,itimestep=call.itimestep,
        len_disp=configs['config_len_disp'],xice_threshold=state['xice_threshold'],
        config_do_restart=configs['config_do_restart'],config_do_DAcycling=configs['config_do_DAcycling'],
        config_frac_seaice=call.options['config_frac_seaice'],
        config_lsm_scheme=call.options['config_lsm_scheme'],
        config_mynn_sfcflux_land=call.options['config_mynn_sfcflux_land'],
        config_mynn_sfcflux_water=call.options['config_mynn_sfcflux_water'])


def register_backend(registry):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1390-1463 @1d6e370e."""
    registry.register('surface_layer',SCHEME,BACKEND,launch_slot)
