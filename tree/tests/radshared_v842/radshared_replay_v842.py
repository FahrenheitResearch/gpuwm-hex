"""Strict standalone MPAS full-module oracle replay (synthetic, not acceptance).

All NumPy field calculations here create test inputs; production data paths
are CUDA. The native module reads the pinned table files itself.
"""
import argparse
import ctypes as ct
import hashlib
import json
from itertools import product
import os
from pathlib import Path
import platform
import sys

import numpy as np

from hexcore.mpasphys import radshared_v842 as rad

P=ct.POINTER(ct.c_float)


def ptr(x):
    assert x.dtype==np.float32 and x.flags.c_contiguous
    return x.ctypes.data_as(P)


def summary(a,b, *, backend='cuda'):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    diff=a.view('u4')!=b.view('u4')
    first=np.argwhere(diff)[:4].tolist()
    return {'words':a.size,'differing_words':int(diff.sum()),'first':first,
            'native_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
            f'{backend}_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
            'examples':[{'index':i,'native':float(a[tuple(i)]),backend:float(b[tuple(i)]),
                         'native_word':int(a.view('u4')[tuple(i)]),f'{backend}_word':int(b.view('u4')[tuple(i)])}
                        for i in first]}


def decks():
    rng=np.random.default_rng(2123)
    nc,nz=512,55
    height=np.linspace(20,26000,nz,dtype='f4')
    pressure=np.broadcast_to(101325*np.exp(-height/7500),(nc,nz)).copy().astype('f4')
    temperature=(290-6.5*np.minimum(height,11000)/1000+np.maximum(height-11000,0)*.0007)
    t=np.broadcast_to(temperature,(nc,nz)).copy().astype('f4')
    t+=rng.uniform(-20,20,(nc,1)).astype('f4')
    t+=rng.uniform(-2,2,(nc,nz)).astype('f4')
    x=np.maximum(-80.,t-273.16)
    water=[.611583699E03,.444606896E02,.143177157E01,.264224321E-1,.299291081E-3,
           .203154182E-5,.702620698E-8,.379534310E-11,-.321582393E-13]
    e=np.full_like(t,water[-1])
    for c in water[-2::-1]: e=np.float32(c)+x*e
    e=np.minimum(e,pressure*np.float32(.15))
    sat=np.float32(.622)*e/(pressure-e)
    rh=rng.uniform(.001,1.6,(nc,nz)).astype('f4')
    for i in range(0,nc,16): rh[i:i+8]=np.linspace(.78,1.6,nz,dtype='f4')
    v=(sat*rh).astype('f4')
    cond=10**rng.uniform(-14,-3,(3,nc,nz)).astype('f4')
    cond[:,0:128]=0
    cond[:,128:256]*=np.float32(1e-4)
    cond[:,:,0]=np.float32(1e-6)
    qi=cond[1].copy();qi[128:256]=0
    qs=cond[2].copy();qs[128:256]=0
    # Profiles with distinct theta inversions, deep warm and cold clouds.
    t[0]=np.linspace(300,200,nz,dtype='f4')
    t[1]=np.linspace(270,190,nz,dtype='f4')
    v[0:2]=sat[0:2]*np.float32(1.05)
    qc=cond[0].copy()
    dens=rng.uniform(.02,100,nc).astype('f4')
    land=np.where(np.arange(nc)%2,1.,2.).astype('f4')
    rho=(pressure/(np.float32(287.)*t)).astype('f4')
    hyd=(pressure*np.float32(.997)).astype('f4')
    yield dict(qv=v,qc=qc,qi=qi,qs=qs,t=t,pres=pressure,pres_hyd=hyd,rho=rho,
               meshDensity=dens,xland=land,len_disp=np.float32(480000.))
    # Independent homogeneous columns cover pure phase, clipped RH,
    # fully cloudy upper levels, clear levels, warm and cold thresholds.
    nc,nz=64,8
    p=np.broadcast_to(np.linspace(98000,2000,nz,dtype='f4'),(nc,nz)).copy()
    t=np.broadcast_to(np.array([298.16,273.16,261.16,253.16,233.16,203.16,190,180],dtype='f4'),(nc,nz)).copy()
    v=np.broadcast_to(np.geomspace(.02,1e-6,nz).astype('f4'),(nc,nz)).copy()
    c=np.zeros_like(v);ice=c.copy();s=c.copy()
    c[::4]=1.e-6;ice[1::4]=1.e-7;s[2::4]=1.e-5
    c[3::4]=np.nextafter(np.float32(1.e-6),np.float32(np.inf))
    yield dict(qv=v,qc=c,qi=ice,qs=s,t=t,pres=p,pres_hyd=p.copy(),rho=np.ones_like(v),
               meshDensity=np.linspace(.001,1000,nc,dtype='f4'),xland=np.tile(np.array([1,2],dtype='f4'),nc//2),len_disp=np.float32(1000))

    # Deep warm/cold decks with smooth stable potential temperature,
    # resolved condensate combinations, and strong density to reach the
    # native integrated-path limiting and existing-path early returns.
    nc,nz=96,55
    height=np.linspace(0,18000,nz,dtype='f4')
    p=np.broadcast_to(101325*np.exp(-height/7500),(nc,nz)).copy().astype('f4')
    t=np.broadcast_to((310-height*.006),(nc,nz)).copy().astype('f4')
    t[48:]-=np.float32(30)
    v=np.broadcast_to(np.geomspace(.06,1e-5,nz).astype('f4'),(nc,nz)).copy()
    c=np.zeros_like(v);ice=c.copy();s=c.copy()
    for i in range(nc):
        if i%8==0: c[i]=np.float32(2e-6)
        if i%8==1: ice[i]=np.float32(1e-7)
        if i%8==2: s[i]=np.float32(2e-5)
        if i%8==3: c[i]=np.float32(1e-3)
        if i%8==4: ice[i]=np.float32(1e-3)
        if i%8==5: s[i]=np.float32(1e-3)
        if i%8==6: v[i]=np.geomspace(.01,1e-6,nz).astype('f4')
    density=np.broadcast_to(np.array([1,100,1000],dtype='f4')[:,None,None],(3,nc//3,nz)).reshape(nc,nz).copy()
    yield dict(qv=v,qc=c,qi=ice,qs=s,t=t,pres=p,pres_hyd=p.copy(),rho=density,
               meshDensity=np.ones(nc,dtype='f4'),xland=np.tile(np.array([1,2],dtype='f4'),nc//2),len_disp=np.float32(100000))


    # Compound-condition boundaries and unusual but finite columns:
    # potential-temperature plateaus above 19 km, no tropopause surrogate,
    # cloud ice below -70 C, and explicit clouds outside phase limits.
    nc,nz=64,55
    height=np.linspace(0,30000,nz,dtype='f4')
    p=np.broadcast_to(101325*np.exp(-height/7500),(nc,nz)).copy().astype('f4')
    t=np.empty_like(p)
    for i in range(nc):
        mode=i%8
        if mode==0: t[i]=320*(p[i]/np.float32(100000))**np.float32(287.05/1004.)
        elif mode==1:
            theta=300+np.maximum(height-3000,0)*np.float32(.03)
            t[i]=theta*(p[i]/np.float32(100000))**np.float32(287.05/1004.)
        elif mode==2: t[i]=np.float32(203.16)
        elif mode==3: t[i]=np.float32(202.16)
        elif mode==4: t[i]=np.float32(300)
        elif mode==5: t[i]=np.float32(253.16)
        elif mode==6:
            t[i]=np.linspace(290,240,nz,dtype='f4');t[i,8:12]=np.float32(230)
        else: t[i]=np.linspace(273.16,180,nz,dtype='f4')
    v=np.full_like(p,np.float32(.02))
    c=np.zeros_like(p);ice=c.copy();s=c.copy()
    for i in range(nc):
        if i//8%4==0: ice[i]=np.float32(1.e-7)
        elif i//8%4==1: c[i]=np.float32(2.e-6)
        elif i//8%4==2: v[i]=np.float32(.0001)
        else:
            c[i]=np.float32(1.e-6);ice[i]=np.float32(1.e-8)
    ice[24:32,10]=np.float32(1.e-7)
    density=np.ones_like(p);density[24:32]=np.float32(1000000)
    yield dict(qv=v,qc=c,qi=ice,qs=s,t=t,pres=p,pres_hyd=p.copy(),rho=density,
               meshDensity=np.ones(nc,dtype='f4'),xland=np.tile(np.array([1,2],dtype='f4'),nc//2),len_disp=np.float32(100000))


    # Empty solve extent is legal for the thin pool adapter. Native loops
    # and device launchers must preserve halos without touching a column.
    empty=np.empty((0,8),dtype='f4')
    yield dict(qv=empty.copy(),qc=empty.copy(),qi=empty.copy(),qs=empty.copy(),
               t=empty.copy(),pres=empty.copy(),pres_hyd=empty.copy(),rho=empty.copy(),
               meshDensity=np.empty(0,dtype='f4'),xland=np.empty(0,dtype='f4'),len_disp=np.float32(1000))


def vertical_decks(nc, levels):
    for name,pressure in [('bounds',np.array([1200,900,300,100,10,1,.01,0],dtype='f4')),
                          ('levels',levels[::-1].copy())]:
        yield name,np.broadcast_to(pressure,(nc,pressure.size)).copy()
    # Different column histories force the source's shared minimum search
    # bound. Pressure inversions retain the native previous-index behavior.
    p=np.broadcast_to(np.array([900,300,100,30,10,1,.1],dtype='f4'),(nc,7)).copy()
    p[::2]*=np.float32(.25)
    p[1::3,3]=np.float32(.1)
    p[2::3,2]=np.float32(.01)
    yield 'column_histories',p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    import cupy as cp
    lib=ct.CDLL(str(args.oracle.resolve()))
    lib.cloud.argtypes=[ct.c_int]*5+[ct.c_float]+[P]*12
    lib.ozone_init.argtypes=[ct.c_int]*2+[P]*3
    lib.ozone_time.argtypes=[ct.c_int]*2+[ct.c_float]+[P]*2
    lib.ozone_update.argtypes=[ct.c_int]*5+[P]*2
    lib.ozone_vertical.argtypes=[ct.c_int]*3+[P]*4
    result={'purpose':'standalone synthetic full-module oracle, not whole-model acceptance',
            'hostname':platform.node(),'libc':platform.libc_ver(),'oracle':str(args.oracle.resolve()),
            'cuda_device':str(cp.cuda.runtime.getDeviceProperties(0)['name']),
            'decks':[],'ozone':{},'radii':{},'adapter':{},'controls':{}}
    import gpuwm.core.kernels
    header=Path(gpuwm.core.kernels.__file__).with_name('glibc_flt32.cuh')
    result['artifact_sha256']={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (args.oracle,Path(rad.__file__),Path(__file__),header)}
    result['cuda_options']=list(rad.CUDA_OPTIONS)
    result['compiler']='direct NVRTC, loaded through cupy.cuda.function.Module'
    result['process_count']=1
    lib.radiation_controls.argtypes=[ct.c_int]*6+[ct.POINTER(ct.c_int32)]
    overlaps=tuple(rad.OVERLAP)
    decorrelations=('constant','latitude_varying','unknown')
    micros=(*rad.RADIUS_SCHEMES,'off','other')
    pairs=(('rrtmg_lw','rrtmg_sw'),('rrtmg_lw','off'),('off','rrtmg_sw'),('off','off'))
    for overlap,decor,ozone,radius,micro,pair in product(range(6),range(3),range(2),range(2),range(5),range(4)):
        native=np.empty(6,dtype='i4')
        lib.radiation_controls(overlap,decor,ozone,radius,micro,pair,
                              native.ctypes.data_as(ct.POINTER(ct.c_int32)))
        options=rad.radiation_options(dict(config_radt_cld_overlap=overlaps[overlap],
            config_radt_cld_dcorrlen=decorrelations[decor],config_o3climatology=bool(ozone),
            config_microp_re=bool(radius),config_microp_scheme=micros[micro],
            config_radt_lw_scheme=pairs[pair][0],config_radt_sw_scheme=pairs[pair][1]))
        actual=np.array((options.cldovrlp,options.idcor,options.o3input,
                         options.has_reqc,options.has_reqi,options.has_reqs),dtype='i4')
        key=f'{overlap}:{decor}:{ozone}:{radius}:{micro}:{pair}'
        result['controls'][key]=summary(native,actual,backend='port')
    for deck_no,deck in enumerate(decks()):
        nc,nz=deck['qv'].shape
        inputs={name:cp.asarray(x) if isinstance(x,np.ndarray) else x for name,x in deck.items()}
        for scheme_no,scheme in enumerate(rad.CLOUD_SCHEMES):
            flags=((False,False),(False,True),(True,False),(True,True)) if scheme_no==0 else ((True,True),)
            for fqc,fqi in flags:
                native=np.empty((5,nc,nz),dtype='f4');dx=np.empty(nc,dtype='f4')
                lib.cloud(nc,nz,scheme_no,fqc,fqi,deck['len_disp'],
                   *[ptr(deck[k]) for k in ('meshDensity','xland','qv','qc','qi','qs','t','pres','pres_hyd','rho')],
                   ptr(native),ptr(dx))
                out=rad.driver_cloudiness(scheme=scheme,f_qc=fqc,f_qi=fqi,**inputs)
                row={'deck':deck_no,'scheme':scheme,'f_qc':fqc,'f_qi':fqi,'arrays':{}}
                for n,name in enumerate(('cldfrac','qvrad','qcrad','qirad','qsrad')):
                    row['arrays'][name]=summary(native[n],cp.asnumpy(out[name]))
                row['arrays']['dx']=summary(dx,cp.asnumpy(out['dx']))
                row['arrays']['xland']=summary(deck['xland'],cp.asnumpy(out['xland']))
                result['decks'].append(row)
        # Exercise the actual f4 callback with public [level,cell] storage,
        # halo sentinels, preserved physical fields and saved ozone metadata.
        from types import SimpleNamespace
        sentinel=np.float32(-12345)
        profiles={name:cp.asarray(np.ascontiguousarray(np.concatenate((deck[name].T,
                    np.full((nz,3),sentinel,dtype='f4')),axis=1))) for name in
                  ('qv','qc','qi','qs','t','pres','pres_hyd','rho')}
        snapshots={name:x.copy() for name,x in profiles.items()}
        public=cp.full((nz,nc+3),sentinel,dtype=cp.float32)
        radiation_full={name:cp.full((nz+1,nc+3),sentinel,dtype=cp.float32)
                        for name in ('cldfrac','qvrad','qcrad','qirad','qsrad')}
        radiation={name:value[:nz] for name,value in radiation_full.items()}
        for name in ('dx','xland'):
            radiation_full[name]=cp.full(nc+3,sentinel,dtype=cp.float32)
            radiation[name]=radiation_full[name]
        calendar_saved=rad.OzoneCalendarState(29)
        pools={'prepared':profiles,'mesh':{'meshDensity':cp.asarray(deck['meshDensity'])},
               'radiation_prepared':radiation,
               'sfc_input':{'xland':cp.asarray(deck['xland'])},'diag_physics':{'cldfrac':public},
               'nCellsSolve':nc,'config_len_disp':deck['len_disp'],'f_qc':True,'f_qi':True,
               'radshared':{'calendar':calendar_saved}}
        out=rad.cloudiness_slot(SimpleNamespace(scheme='cld_fraction_thompson',pools=pools))
        expected=np.full((nz,nc+3),sentinel,dtype='f4');expected[:,:nc]=native[0].T
        result['adapter'][f'deck:{deck_no}:cloudiness']=summary(expected,cp.asnumpy(public))
        for n,name in enumerate(('cldfrac','qvrad','qcrad','qirad','qsrad')):
            expected=np.full((nz+1,nc+3),sentinel,dtype='f4')
            expected[:nz,:nc]=native[n].T
            result['adapter'][f'deck:{deck_no}:native_radiation:{name}']=summary(
                expected,cp.asnumpy(radiation_full[name]))
        for name,values in (('dx',dx),('xland',deck['xland'])):
            expected=np.full(nc+3,sentinel,dtype='f4');expected[:nc]=values
            result['adapter'][f'deck:{deck_no}:native_radiation:{name}']=summary(
                expected,cp.asnumpy(radiation_full[name]))
        for name,x in profiles.items():
            result['adapter'][f'deck:{deck_no}:unchanged:{name}']=summary(cp.asnumpy(snapshots[name]),cp.asnumpy(x))
        assert pools['radshared']['calendar'] is calendar_saved
        assert pools['radshared']['cldfrac'] is out['cldfrac']
    # Actual ozone table decode is Fortran in the oracle; test-only NumPy
    # creates the same decoded inputs for the device launcher.
    tables=args.tables.resolve()
    old=os.getcwd();os.chdir(tables)
    lat_oz=np.loadtxt('OZONE_LAT.TBL',dtype='f4')
    pin=np.loadtxt('OZONE_PLEV.TBL',dtype='f4')
    table=np.loadtxt('OZONE_DAT.TBL',dtype='f4').reshape(12,64,pin.size)
    lat=np.deg2rad(np.array([-90,-86,-60,-30,0,30,60,86,90],dtype='f4')).astype('f4')
    nc=lat.size;nl=pin.size
    monthly=np.empty((nc,nl,12),dtype='f4');levels=np.empty(nl,dtype='f4')
    lib.ozone_init(nc,nl,ptr(lat),ptr(levels),ptr(monthly));os.chdir(old)
    gpu_month=rad.init_o3climatology(latCell=cp.asarray(lat),lat_ozone=cp.asarray(lat_oz),ozmixin=cp.asarray(table))
    result['ozone']['latitude']=summary(monthly,cp.asnumpy(gpu_month))
    result['ozone']['pressure_levels']=summary(levels,pin)
    init_month=cp.full((12,nl,nc+3),sentinel,dtype=cp.float32)
    init_pin=cp.empty(nl,dtype=cp.float32)
    init_pools={'nCells':nc,'nCellsSolve':nc-3,'mesh':{'latCell':cp.asarray(lat)},
                'atm_input':{'ozmixm':init_month,'pin':init_pin},
                'radshared_tables':{'lat_ozone':cp.asarray(lat_oz),'ozmixin':cp.asarray(table),'pin':cp.asarray(pin)}}
    rad.init_ozone_slot(SimpleNamespace(pools=init_pools))
    expected=np.full((12,nl,nc+3),sentinel,dtype='f4');expected[:,:,:nc]=monthly.transpose(2,1,0)
    result['adapter']['ozone:init_all_cells']=summary(expected,cp.asnumpy(init_month))
    result['adapter']['ozone:init_pin']=summary(levels,cp.asnumpy(init_pin))
    clim=np.empty((nc,nl),dtype='f4')
    # Tiny monthly mixing ratios must survive native multiply/add interpolation.
    # A forced-FTZ compilation makes these nonzero native outputs zero.
    tiny_words=np.resize(np.array([1,0x400000,0x7fffff,0x80000001,
                                  0x80400000,0x807fffff],dtype='u4'),(nc,nl,12))
    tiny=tiny_words.view('f4').copy(); tiny_native=np.empty((nc,nl),dtype='f4')
    lib.ozone_time(nc,nl,np.float32(73.5),ptr(tiny),ptr(tiny_native))
    native_words=tiny_native.view('u4') & np.uint32(0x7fffffff)
    assert np.any((native_words>0)&(native_words<0x800000)), 'oracle must produce subnormal outputs'
    tiny_gpu=rad.o3climatology_from_MPAS(np.float32(73.5),cp.asarray(tiny))
    result['ozone']['subnormal_time']=summary(tiny_native,cp.asnumpy(tiny_gpu))
    result['ozone']['subnormal_time']['native_nonzero_subnormal_words']=int(
        np.count_nonzero((native_words>0)&(native_words<0x800000)))
    from cupy.cuda import compiler, function
    from unittest.mock import patch
    ftz_options=tuple('--ftz=true' if x=='--ftz=false' else x for x in rad.CUDA_OPTIONS)
    ftz_binary,_=compiler.compile_using_nvrtc(header.read_text()+'\n'+rad.CUDA_SOURCE,
                 options=ftz_options,arch=cp.cuda.Device().compute_capability)
    ftz_module=function.Module();ftz_module.load(ftz_binary)
    with patch.object(rad,'_module',return_value=ftz_module):
        ftz_actual=cp.asnumpy(rad.o3climatology_from_MPAS(np.float32(73.5),cp.asarray(tiny)))
    ftz_comparison=summary(tiny_native,ftz_actual)
    assert ftz_comparison['differing_words']>0, 'negative control must detect forced FTZ'
    result['subnormal_negative_control']={'options':list(ftz_options),'comparison':ftz_comparison,
                                        'expected_failure_detected':True}
    for day in (0.,.5,14.9999,15.,44.,73.5,89.5,349.,354.5,364.9,365.,730.5):
        lib.ozone_time(nc,nl,day,ptr(monthly),ptr(clim))
        out=rad.o3climatology_from_MPAS(day,gpu_month)
        result['ozone'][f'time:{day}']=summary(clim,cp.asnumpy(out))
    # The real ozone hook converts public monthly storage and preserves halos.
    pubmonth=np.full((12,nl,nc+3),sentinel,dtype='f4')
    pubmonth[:,:,:nc]=monthly.transpose(2,1,0)
    pubout=cp.full((nl,nc+3),sentinel,dtype=cp.float32)
    ozone_pools={'atm_input':{'ozmixm':cp.asarray(pubmonth)},
                 'diag_physics':{'o3clim':pubout},'nCellsSolve':nc,'julian':np.float32(730.5)}
    rad.ozone_slot(SimpleNamespace(pools=ozone_pools))
    expected=np.full((nl,nc+3),sentinel,dtype='f4');expected[:,:nc]=clim.T
    result['adapter']['ozone:output_halos']=summary(expected,cp.asnumpy(pubout))
    result['adapter']['ozone:unchanged_monthly']=summary(pubmonth,cp.asnumpy(ozone_pools['atm_input']['ozmixm']))
    calendar=rad.OzoneCalendarState()
    for date in ('2023-01-01','2023-01-15','2023-03-31','2022-12-22','2024-02-29','2025-03-01','2025-03-31','2000-02-29','2100-03-01'):
        year,mon,day=map(int,date.split('-'))
        lib.ozone_update(nc,nl,year,mon,day,ptr(monthly),ptr(clim))
        out=rad.update_o3climatology(date,gpu_month,calendar_state=calendar)
        result['ozone'][f'update:{date}']=summary(clim,cp.asnumpy(out))
    saved=rad.OzoneCalendarState(29)
    ozone_pools['radshared']['ozone_calendar']=saved
    ozone_pools['nCellsSolve']=nc-3;ozone_pools['current_date']='2100-03-01'
    pubout.fill(sentinel)
    rad.update_ozone_slot(SimpleNamespace(pools=ozone_pools))
    expected=np.full((nl,nc+3),sentinel,dtype='f4');expected[:,:nc-3]=clim[:nc-3].T
    result['adapter']['ozone:date_update_halos']=summary(expected,cp.asnumpy(pubout))
    assert ozone_pools['radshared']['ozone_calendar'] is saved
    for name,p in vertical_decks(nc,levels):
        out=np.empty_like(p)
        lib.ozone_vertical(nc,p.shape[1],nl,ptr(p),ptr(levels),ptr(clim),ptr(out))
        result['ozone'][f'vertical:{name}']=summary(out,cp.asnumpy(rad.vinterp_ozn(cp.asarray(p),cp.asarray(levels),cp.asarray(clim))))
        pascal=p*np.float32(100)
        hpa=pascal/np.float32(100)
        lib.ozone_vertical(nc,p.shape[1],nl,ptr(hpa),ptr(levels),ptr(clim),ptr(out))
        result['ozone'][f'vertical:{name}:pa']=summary(out,cp.asnumpy(rad.vinterp_ozn(cp.asarray(pascal),cp.asarray(levels),cp.asarray(clim),pressure_in_pa=True)))
    # Radius driver paths copy unchanged words, with exact zero when off.
    radius=np.linspace(1.e-6,200.e-6,16,dtype='f4').reshape(2,8)
    device=cp.asarray(radius)
    lib.radius_from.argtypes=[ct.c_int]*4+[P]*4
    lib.radius_to.argtypes=[ct.c_int]*4+[P]*4
    for enabled in (False,True):
        for mode,scheme in enumerate((*rad.RADIUS_SCHEMES,'other')):
            out=rad.radii_from_MPAS(microp_scheme=scheme,microp_re=enabled,re_cloud=device,re_ice=device,re_snow=device)
            back=rad.radii_to_MPAS(microp_scheme=scheme,microp_re=enabled,
                                  rre_cloud=device,rre_ice=device,rre_snow=device)
            for direction,arrays in (('from',out),('to',back)):
                expected=np.empty((3,*radius.shape),dtype='f4')
                getattr(lib,f'radius_{direction}')(2,8,mode,enabled,ptr(radius),ptr(radius),ptr(radius),ptr(expected))
                actual=[np.full_like(radius,np.float32(-12345))]*3 if arrays is None else [cp.asnumpy(x) for x in arrays]
                for n,x in enumerate(actual):
                    result['radii'][f'{scheme}:{enabled}:{direction}:{n}']=summary(expected[n],x)
    # Native defect policy: unstable out-of-bounds native words use the
    # intended in-bounds northern endpoint. Compare every endpoint word.
    degrad=np.float32(np.float32(3.1415926)/np.float32(180))
    final_latitude=cp.asarray(np.array([lat_oz[-1]*degrad],dtype='f4'))
    endpoint=rad.init_o3climatology(latCell=final_latitude,lat_ozone=cp.asarray(lat_oz),ozmixin=cp.asarray(table))
    result['ozone']['proposed_native_boundary_endpoint']=summary(table[:,63,:].T[None,:,:].copy(),cp.asnumpy(endpoint))
    result['guards']={'native_exact_final_latitude':'retired after full executable host/rank instability witness'}
    failures=sum(x['differing_words'] for d in result['decks'] for x in d['arrays'].values())
    failures+=sum(x['differing_words'] for x in result['ozone'].values())
    failures+=sum(x['differing_words'] for x in result['radii'].values())
    failures+=sum(x['differing_words'] for x in result['adapter'].values())
    failures+=sum(x['differing_words'] for x in result['controls'].values())
    result['differing_words']=failures
    args.receipt.resolve().write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'differing_words':failures,'receipt':str(args.receipt)}))
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
