"""Word comparison of pristine MPAS driver and production statement source.

Synthetic oracle data handling is test-only. Production table decode remains
Rust-owned. This host witness is not the CUDA or whole-forecast acceptance.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from hexcore.mpasphys import thompson_v842 as port


def _lookup_arrays(library,path,names,records,little):
    """Expose Rust-decoded native buffers; Python only marshals metadata."""
    counts=[int(np.prod(port.TABLE_SHAPES[name])) for name in names]
    widths=[4 if name=='tnccn_act' else 8 for name in names]
    handle=library.thompson_lookup_open(os.fsencode(path),(C.c_uint64*len(names))(*counts),
        (C.c_uint32*len(names))(*widths),len(names),records,little)
    if not handle:raise ValueError(library.thompson_lookup_error().decode())
    try:
        result={}
        for i,(name,count,width) in enumerate(zip(names,counts,widths)):
            scalar=C.c_float if width==4 else C.c_double
            pointer=library.thompson_lookup_data(handle,i)
            result[name]=np.ctypeslib.as_array(C.cast(pointer,C.POINTER(scalar)),shape=(count,)).copy()
        return result
    finally:library.thompson_lookup_close(handle)


def tables(directory):
    directory=directory.resolve()
    library=C.CDLL(str(directory/'lookup_decode.so'))
    library.thompson_lookup_open.argtypes=[C.c_char_p,C.POINTER(C.c_uint64),C.POINTER(C.c_uint32),C.c_size_t,C.c_bool,C.c_bool]
    library.thompson_lookup_open.restype=C.c_void_p
    library.thompson_lookup_data.argtypes=[C.c_void_p,C.c_size_t];library.thompson_lookup_data.restype=C.c_void_p
    library.thompson_lookup_close.argtypes=[C.c_void_p]
    library.thompson_lookup_error.restype=C.c_char_p
    filenames={
        'MP_THOMPSON_QRacrQG_DATA.DBL': list(port.TABLE_ARGUMENTS[:6]),
        'MP_THOMPSON_QRacrQS_DATA.DBL': list(port.TABLE_ARGUMENTS[6:18]),
        'MP_THOMPSON_freezeH2O_DATA.DBL': list(port.TABLE_ARGUMENTS[18:24]),
        'MP_THOMPSON_QIautQS_DATA.DBL': list(port.TABLE_ARGUMENTS[24:27]),
    }
    result={}
    for filename,names in filenames.items():
        with (directory/filename).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==port.TABLE_FILE_SHA256[filename],filename
        result.update(_lookup_arrays(library,directory/filename,names,True,False))
    auxorder=('t_efrw','t_efsw','tnr_rev','tpc_wev','tnc_wev','tnccn_act')
    result.update(_lookup_arrays(library,directory/'auxiliary.bin',auxorder,False,True))
    return result


def synthetic(nz=16,stress=False,profiles=False):
    # One-column carriers isolate complete drivers; mixed profiles add melting,
    # overlap, warm/cold sources and sedimentation without selecting case names.
    regimes=[]
    for t in (220.,240.,253.15,263.15,268.15,273.15,274.15,285.,300.):
        for rh in (0.5,1.,1.1):
            for species in ('clear','cloud','rain','ice','snow','graupel','all'):
                regimes.append((t,rh,species))
    ncol=len(regimes)
    f=np.zeros((len(port.FIELD_ARGUMENTS),nz,ncol),np.float32)
    s=np.zeros((len(port.SURFACE_ARGUMENTS),ncol),np.float32)
    z=np.arange(nz,dtype=np.float32)
    for c,(t,rh,species) in enumerate(regimes):
        temp=np.full(nz,t,np.float32)
        if species=='all': temp=np.float32(t)+np.float32(12)*np.cos(z*np.float32(0.35))
        pressure=np.float32(95000)*np.exp(-z*np.float32(0.08))
        pii=(pressure/np.float32(100000))**np.float32(0.2857143)
        f[8,:,c]=temp/pii;f[9,:,c]=pii;f[10,:,c]=pressure
        f[11,:,c]=0.;f[12,:,c]=np.float32(100)
        x=np.maximum(np.float32(-80),temp-np.float32(273.16))
        coefs=(609.868993,49.9320233,1.84672631,0.0402737184,0.000565392987,0.00000521693933,0.0000000307839583,0.000000000105785160,0.000000000000161444444) if t<273.15 else (611.583699,44.4606896,1.43177157,0.0264224321,0.000299291081,0.00000203154182,0.00000000702620698,0.00000000000379534310,-0.0000000000000321582393)
        es=np.full(nz,coefs[-1],np.float32)
        for coef in reversed(coefs[:-1]): es=np.float32(coef)+x*es
        es=np.minimum(es,pressure*np.float32(0.15))
        f[0,:,c]=np.float32(rh)*np.float32(0.622)*es/(pressure-es)
        for idx,label in ((1,'cloud'),(2,'ice'),(3,'rain'),(4,'snow'),(5,'graupel')):
            if species in (label,'all'):f[idx,:,c]=np.float32(2.0e-4)*(1+np.float32(0.1)*z)
        f[6,:,c]=np.float32(1e4);f[7,:,c]=np.float32(3e4)
        if species in ('clear','cloud'):f[6:8,:,c]=0
        s[0,c]=0.125;s[2,c]=0.25;s[4,c]=0.5
        s[7,c]=np.float32(100e6 if c%2 else 300e6)
        s[8,c]=np.minimum(np.float32(15),np.float32(1000e6)/s[7,c]+np.float32(2))
    if stress:
        extra=[];surfaces=[];base_regimes=tuple(regimes)
        for t in (220.,253.15,273.15,280.,300.):
            for rh in (0.01,1.5,3.):
                for mass in (1e-10,1e-6,1e-3,0.02):
                    for number in (0.,1.,1e2,1e6,1e12):
                        for species in ('cloud','rain','ice','snow','graupel','all'):
                            # Start with an exact existing carrier and vary
                            # the numerical extremes independently.
                            c=base_regimes.index((min(base_regimes,key=lambda x:abs(x[0]-t))[0],1.,species))
                            a=f[:,:,c].copy();b=s[:,c].copy()
                            a[8]=np.float32(t)/a[9];a[0]*=np.float32(rh)
                            for idx,label in ((1,'cloud'),(2,'ice'),(3,'rain'),(4,'snow'),(5,'graupel')):
                                a[idx]=np.float32(mass if species in (label,'all') else 0.)
                            a[6:8]=np.float32(number);a[11]=np.float32((-5.,0.,5.)[len(extra)%3])
                            b[7]=np.float32((20e6,300e6,2e9)[len(extra)%3]);b[8]=np.float32(2)
                            extra.append(a);surfaces.append(b);regimes.append((t,rh,species,mass,number))
        # Different collector/collected masses exercise Stokes-number and
        # conservation branches that equal-mass columns do not reach.
        for t in (220.,238.,250.,263.,272.,274.,283.):
            for qc in (1e-8,1e-5,1e-3,0.005):
                for frozen in (1e-5,1e-3,0.02):
                    for rh in (0.1,1.05):
                        for collector in ('snow','graupel','mixed'):
                            c=base_regimes.index((min(base_regimes,key=lambda x:abs(x[0]-t))[0],1.,'all'))
                            a=f[:,:,c].copy();b=s[:,c].copy()
                            a[8]=np.float32(t)/a[9];a[0]*=np.float32(rh)
                            a[1:8]=0.;a[1]=np.float32(qc)
                            a[4 if collector=='snow' else 5]=np.float32(frozen)
                            if collector=='mixed':
                                a[2]=np.float32(frozen/10);a[3]=np.float32(qc/10)
                                a[4]=np.float32(frozen/100)
                            a[6:8]=np.float32(10 if len(extra)%2 else 1e8)
                            b[7]=np.float32((100e6,300e6,2e9)[len(extra)%3])
                            b[8]=np.float32(2)
                            extra.append(a);surfaces.append(b)
                            regimes.append((t,rh,collector,qc,frozen))
        f=np.concatenate((f,np.stack(extra,axis=2)),axis=2)
        s=np.concatenate((s,np.stack(surfaces,axis=1)),axis=1)
    if profiles:
        # Empty layers and short total height exercise vertical masks and
        # cloud-sedimentation loop completion rather than its height exit.
        selected=[c for c,(_,rh,species) in enumerate(regimes[:189]) if rh==1. and species!='clear']
        extra=[];surfaces=[]
        for c in selected:
            for pattern in ('alternate','top','bottom'):
                a=f[:,:,c].copy();b=s[:,c].copy()
                mask=(z.astype(np.int32)%2==0) if pattern=='alternate' else (z>=nz//2 if pattern=='top' else z<nz//2)
                a[1:8,mask]=0.
                a[12]=np.float32(10.)
                b[7]=np.float32(300e6);b[8]=np.float32(2.)
                extra.append(a);surfaces.append(b)
                regimes.append((*regimes[c],pattern))
        f=np.concatenate((f,np.stack(extra,axis=2)),axis=2)
        s=np.concatenate((s,np.stack(surfaces,axis=1)),axis=1)
    return f,s,regimes


def pointer(a):return C.c_void_p(a.ctypes.data)


def sedimentation_limits():
    """Finite stress inputs reach the zero-substep ice path at F:3332.

    Large pressures deliberately reduce density-corrected fall speed. These
    inputs test the module's arithmetic domain, not atmospheric validity.
    """
    f,s,_=synthetic()
    fields=[];surfaces=[];regimes=[]
    for pressure in (1e5,1e8,1e10,1e12,1e14):
        for mass in (2e-12,1e-10,1e-7):
            a=f[:,:,49].copy();b=s[:,49].copy()
            a[1:8]=0.;a[2]=np.float32(mass);a[6]=np.float32(1e8)
            a[9]=1.;a[10]=np.float32(pressure);a[8]=np.float32(250.)
            a[0]=np.float32(0.001*1e5/pressure)
            fields.append(a);surfaces.append(b);regimes.append((pressure,mass))
    return np.stack(fields,axis=2),np.stack(surfaces,axis=1),regimes,None


def compare(reference,actual,names):
    rows={}
    for i,name in enumerate(names):
        mask=reference[i].view(np.uint32)!=actual[i].view(np.uint32)
        row=dict(words=int(mask.sum()),total=int(mask.size),reference_nan=int(np.isnan(reference[i]).sum()),actual_nan=int(np.isnan(actual[i]).sum()))
        if mask.any():
            at=tuple(int(x) for x in np.argwhere(mask)[0])
            row.update(first_index=at,reference=float(reference[i][at]),actual=float(actual[i][at]))
        rows[name]=row
    return rows


def shared_native_log10_source(directory):
    """Use the pinned engine's existing glibc 2.43 routine unchanged.

    Preserve its constants, explicit operations and MIT notice. This selects
    shared header text; it implements no Thompson-private transcendental.
    """
    text=(directory/'gpuwm/core/kernels/urban_bem.cuh').read_text()
    start=text.index('// log10f: CORE-MATH ')
    end=text.index('\n__device__ void bem_bem(',start)
    return text[start:end]


def engine_math_source(directory,promote_log10=False,native_log10=False):
    """Diagnostic use of the pinned engine's shared libm, never acceptance.

    Binary64 shared header targets glibc 2.43. Binary32 header provenance
    still names 2.39; atan/hypot entrypoints retain the engine CUDA path.
    These limits must be discharged by the shared MPAS maths owner.
    """
    if promote_log10 and native_log10:
        raise ValueError('select one shared log10 path')
    root=directory/'gpuwm/core/kernels'
    names=('glibc_flt32.cuh','glibc_flt64.cuh','thompson_aerosol_libm.cuh')
    source='\n'.join((root/n).read_text() for n in names)
    for n in ('expf','exp','logf','log','log10f','log10','powf','pow'):
        source+=f'\n#define mpas_{n} thompson_aa_{n}\n'
    for n in ('atan','atanf','atan2','hypot'):
        source+=f'\n#define mpas_{n} {n}\n'
    if promote_log10:
        # Diagnostic composition of an existing shared entrypoint. This is
        # not a replacement libm implementation or an accepted identity path.
        source+='\n#undef mpas_log10f\n#define mpas_log10f(x) ((float)thompson_aa_log10((double)(x)))\n'
    if native_log10:
        source+='\n'+shared_native_log10_source(directory)
        source+='\n#undef mpas_log10f\n#define mpas_log10f ubm_log10f\n'
    return source


def replay(directory,receipt,backend='host',engine=None,stress=False,levels=16,profiles=False,column_inputs=None):
    directory=directory.resolve();os.chdir(directory)
    native=C.CDLL(str(directory/'liboracle.so'))
    host_name={'host-shared':'host_shared_port.so','host-trace':'host_shared_trace_port.so','host-promoted':'host_shared_promoted.so','host-native-log10':'host_shared_native_log10.so'}.get(backend,'host_port.so')
    host=C.CDLL(str(directory/host_name)) if backend.startswith('host') else None
    native.init_oracle();native.export_aux()
    data=tables(directory)
    if host is not None:
        host.run_host.argtypes=[C.c_int,C.c_int,C.c_float,C.c_void_p,C.c_void_p]+[C.c_void_p]*len(data)
    native.run_oracle.argtypes=[C.c_int,C.c_int,C.c_float,C.c_void_p,C.c_void_p]
    native.oracle_radar.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_void_p]
    native.oracle_radar_diagnostics.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
    if host is not None:host.radar_host.argtypes=native.oracle_radar.argtypes
    else:
        import cupy as cp
        math_source=engine_math_source(engine.resolve(),backend=='engine-promoted-cuda',backend=='engine-native-log10-cuda')
        device_tables={n:cp.asarray(data[n].reshape(port.TABLE_SHAPES[n],order='F'),order='F') for n in port.TABLE_ARGUMENTS}
        device_tables['_file_sha256']=dict(port.TABLE_FILE_SHA256)
    if column_inputs is None:
        inputs,surfaces,regimes=synthetic(levels,stress,profiles)
        heights=None
    else:
        inputs,surfaces,regimes,heights=column_inputs
    ncol=inputs.shape[2];nz=inputs.shape[1]
    cases=[]
    for dt in ((1.,10.,100.,121.,1000.) if stress else (1.,10.,100.)):
        f=inputs.copy();s=surfaces.copy();g=f.copy();u=s.copy()
        native.run_oracle(nz,ncol,dt,pointer(f),pointer(s))
        if host is not None:
            host.run_host(nz,ncol,dt,pointer(g),pointer(u),*(pointer(data[n]) for n in port.TABLE_ARGUMENTS))
        else:
            fields={n:cp.asarray(g[i]) for i,n in enumerate(port.FIELD_ARGUMENTS)}
            surface={n:cp.asarray(u[i]) for i,n in enumerate(port.SURFACE_ARGUMENTS)}
            port.launch_driver(fields,surface,device_tables,dt=dt,device_math_source=math_source)
            g=np.stack([cp.asnumpy(fields[n]) for n in port.FIELD_ARGUMENTS]);u=np.stack([cp.asnumpy(surface[n]) for n in port.SURFACE_ARGUMENTS])
        a=np.zeros((nz,ncol),np.float32);b=a.copy()
        native.oracle_radar(nz,ncol,pointer(f),pointer(a))
        if host is not None:host.radar_host(nz,ncol,pointer(f),pointer(b))
        else:
            # Same native post-state isolates radar arithmetic from trajectory.
            fields={n:cp.asarray(f[i]) for i,n in enumerate(port.FIELD_ARGUMENTS)}
            fields['z']=cp.asarray(heights if heights is not None else np.broadcast_to(np.arange(nz,dtype=np.float32)[:,None]*np.float32(100),(nz,ncol)).copy())
            radar={'refl10cm':cp.asarray(np.zeros((nz,ncol),np.float32)),**{n:cp.asarray(np.full(ncol,-35,np.float32)) for n in ('refl10cm_max','refl10cm_1km','refl10cm_1km_max')}}
            port.launch_radar(fields,radar,device_math_source=math_source)
            b=cp.asnumpy(radar['refl10cm'])
        row=dict(dt=dt,fields=compare(f,g,port.FIELD_ARGUMENTS),surface=compare(s,u,port.SURFACE_ARGUMENTS),radar=compare(a[None],b[None],('reflectivity',)))
        # Radar's native kp+1 access requires a bracketing grid. It is
        # intentionally independent of the short-column sedimentation case.
        z=heights if heights is not None else np.broadcast_to(np.arange(nz,dtype=np.float32)[:,None]*np.float32(100),(nz,ncol)).copy()
        radar_ref=np.full((nz+3,ncol),-30,np.float32);radar_actual=radar_ref.copy()
        native.oracle_radar_diagnostics(nz,ncol,pointer(f),pointer(z),pointer(radar_ref))
        if host is not None:
            host.radar_diagnostics_host.argtypes=native.oracle_radar_diagnostics.argtypes
            host.radar_diagnostics_host(nz,ncol,pointer(f),pointer(z),pointer(radar_actual))
        else:
            radar_actual[:nz]=cp.asnumpy(radar['refl10cm'])
            for i,n in enumerate(('refl10cm_max','refl10cm_1km','refl10cm_1km_max')):
                radar_actual[nz+i]=cp.asnumpy(radar[n])
            # Running maximum begins at -30 in this test.
            radar_actual[-1]=np.maximum(np.float32(-30),radar_actual[-1])
        row['radar'].update(compare(radar_ref[nz:],radar_actual[nz:],('refl10cm_max','refl10cm_1km','refl10cm_1km_max')))
        row['differing_words']=sum(v['words'] for group in ('fields','surface','radar') for v in row[group].values())
        cases.append(row)
        print(f'dt={dt:g}: {row["differing_words"]} differing words',flush=True)
    if backend=='host-trace':host.report_math_trace()
    result=dict(acceptance=False,backend=backend,comparison=(('host-shared-engine-maths' if backend!='host' else 'host-glibc') if host is not None else 'diagnostic-engine-device-maths')+'-production-source-versus-pristine-MPAS',regimes=len(regimes),levels=nz,cases=cases)
    receipt.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host','host-shared','host-trace','host-promoted','host-native-log10','engine-cuda','engine-promoted-cuda','engine-native-log10-cuda'),default='host');p.add_argument('--engine',type=Path)
    p.add_argument('--stress',action='store_true');p.add_argument('--levels',type=int,default=16)
    p.add_argument('--profiles',action='store_true',help='add empty-layer and short-column sedimentation profiles')
    p.add_argument('--sedimentation-limits',action='store_true',help='finite stress columns reaching zero ice sedimentation substeps')
    a=p.parse_args();replay(a.directory,a.receipt.resolve(),a.backend,a.engine,a.stress,a.levels,a.profiles,
                          sedimentation_limits() if a.sedimentation_limits else None)
