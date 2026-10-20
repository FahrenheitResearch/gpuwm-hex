"""Word comparison of the full production boundary and pristine driver loops.

The explicit host witness and CUDA modes use the same complete native cases.
CUDA calls the production step launcher, including conversions and writeback.
"""
from pathlib import Path
import ctypes as ct
import json
import numpy as np
from hexcore.mpasphys.wsm6_v842 import FIELD_NAMES


def compare(native,host,receipt,backend='host-witness',header=None):
    if backend not in ('host-witness','hex-cuda','host-glibc-transport'):
        raise ValueError('select hex-cuda or an explicit host witness')
    if backend=='hex-cuda' and not header:
        raise ValueError('hex-cuda requires the complete shared device math header')
    fp=ct.POINTER(ct.c_float);ip=ct.POINTER(ct.c_int);vp=ct.POINTER(ct.c_void_p)
    funcs=[ct.CDLL(str(Path(native).resolve())).wsm6_driver_oracle,
           ct.CDLL(str(Path(host).resolve())).wsm6_driver_host]
    for f in funcs:f.argtypes=[ct.c_int]*4+[ct.c_float]*2+[ct.c_int,fp,vp,ip]+[ct.c_int]*2+[ct.c_float,ct.c_int]+[ct.c_float]*3
    c=np.array([9.80616,1004.5,1846.4,287,461.6,273.15,np.float32(461.6)/np.float32(287)-np.float32(1),np.float32(287)/np.float32(461.6),0,0,0,1e-15,2.85e6,2.5e6,3.5e5,1.28,1000,4190,2106,610.78,2.49e-6,4.99e-6,9.99e-6,50e-6,125e-6,999e-6],np.float32)
    ratio=np.float32(c[4]/c[3]);rcv=np.float32(c[3]/np.float32(c[1]-c[3]))
    base_constants=c.copy();rng=np.random.default_rng(921377);words=0
    surfaces=set(FIELD_NAMES[30:40])|set(FIELD_NAMES[41:]);rows=[]
    for case in range(68):
        c=base_constants.copy()
        nz=8 if case<48 else 55;nc=3;ns=2;top=(1,nz//2,nz-1,nz)[case%4]
        if case >= 64:
            # Cross multiple launch batches and retain four trailing halos.
            # Compare every driver field, rather than only the scheme buffer.
            nc=2053;ns=2049
        fields={n:np.full((nc,) if n in surfaces else (nz+1,nc) if n=='zgrid' else (nz,nc),.0123,np.float32) for n in FIELD_NAMES}
        p=np.repeat(np.linspace(96000,22000,nz,dtype=np.float32)[:,None],nc,axis=1)
        t=rng.uniform(210,299,(nz,nc)).astype(np.float32)
        q=rng.uniform(1e-6,.025,(nz,nc)).astype(np.float32)
        fields['qv']=q
        for n in ('qc','qi','qr','qs','qg'):
            fields[n]=rng.uniform(0,.004,(nz,nc)).astype(np.float32)
            if case%4==0:fields[n].fill(0)
            elif case%4==2:fields[n]*=np.float32(1e-7)
        pi=np.power(p/np.float32(100000),np.float32(c[3]/c[1]))
        fields['theta_m']=t/pi*(np.float32(1)+ratio*q)
        fields['rho_zz']=p/(c[3]*t);fields['zz'].fill(1)
        fields['exner']=pi.copy();fields['exner_base']=pi.copy()
        fields['pressure_base']=p;fields['pressure_p'].fill(0)
        fields['rtheta_base']=fields['rho_zz']*fields['theta_m'];fields['rtheta_p'].fill(0)
        fields['zgrid']=np.repeat(np.linspace(17,17000,nz+1,dtype=np.float32)[:,None],nc,axis=1)
        fields['rainnc'].fill([0,10,15,20][case%4]);fields['surface_pressure'].fill(98000)
        dt=np.float32([1.5,30,120,300][case%4]);dt_dyn=np.float32([1.5,60][case%2])
        hail=case%2;radii_on=(case//2)%2;radar_on=(case//4)%2;bucket=np.float32([0,10][case%2]);acrain=(case//8)%2
        if case >= 64:
            radar_on=1
        if case == 67:
            c[11] = np.finfo(np.float32).smallest_subnormal
            for name in ('qv','qc','qi','qr','qs','qg','rainnc','snownc','graupelnc'):
                fields[name].fill(np.array([0x007fffff], np.uint32).view(np.float32)[0])
            dt = np.array([0x007fffff], np.uint32).view(np.float32)[0]
        out=[]
        for candidate,f in enumerate(funcs):
            vv={n:a.copy() for n,a in fields.items()};counts=np.zeros(nc,np.int32)
            if candidate and backend in ('hex-cuda','host-glibc-transport'):
                import cupy as cp
                from hexcore.mpasphys.wsm6_v842 import step
                device={n:cp.asarray(a) for n,a in vv.items()}
                device['i_rainnc']=cp.asarray(counts)
                options=dict(dt_microp=dt,dt_dyn=dt_dyn,
                     hail_opt=hail,mp_top_level=top,solve_cells=ns,
                     rvord=ratio,rcv=rcv,p0=np.float32(100000),
                     bucket_rainnc=bucket,l_acrain=bool(acrain),
                     l_diags=bool(radar_on),has_reqc=radii_on,
                     has_reqi=radii_on,has_reqs=radii_on)
                if backend=='hex-cuda':
                    step(device,cp.asarray(c),device_math_header=header,**options)
                else:
                    from types import SimpleNamespace
                    from diagnostic_host_glibc_entry import invoke
                    invoke(SimpleNamespace(pools=dict(fields=device,constants=cp.asarray(c),options=options)))
                vv={n:cp.asnumpy(device[n]) for n in FIELD_NAMES}
                counts=cp.asnumpy(device['i_rainnc'])
            else:
                ptrs=(ct.c_void_p*len(FIELD_NAMES))(*[vv[n].ctypes.data for n in FIELD_NAMES])
                f(nz,nc,ns,top,dt,dt_dyn,hail,c.ctypes.data_as(fp),ptrs,counts.ctypes.data_as(ip),radii_on,radar_on,bucket,acrain,ratio,rcv,np.float32(100000))
            out.append((vv,counts))
        words+=sum(a.size for a in out[0][0].values())+out[0][1].size
        diff={n:int(np.count_nonzero(out[0][0][n].view(np.uint32)!=out[1][0][n].view(np.uint32))) for n in FIELD_NAMES}
        diff['i_rainnc']=int(np.count_nonzero(out[0][1]!=out[1][1]))
        # Scheme callback owns solve cells; untouched halo words are also compared.
        halos=sum(int(np.count_nonzero(out[1][0][n][...,ns:].view(np.uint32)!=a[...,ns:].view(np.uint32))) for n,a in fields.items())
        rows.append(dict(case=case,nz=nz,nc=nc,solve_cells=ns,top=top,dt=float(dt),dt_dyn=float(dt_dyn),hail=hail,radii_on=radii_on,radar_on=radar_on,bucket=float(bucket),acrain=acrain,subnormal_inputs=case==67,qmin_word=int(c[11:12].view(np.uint32)[0]),differences=diff,changed_halo_words=halos))
        if any(diff.values()):
            print(case,{n:d for n,d in diff.items() if d})
            if case==6:
                for n,d in diff.items():
                    if d and n!='i_rainnc':
                        a,b=out[0][0][n],out[1][0][n]
                        for ix in np.argwhere(a.view(np.uint32)!=b.view(np.uint32))[:5]:
                            idx=tuple(ix);print(n,idx,a[idx],b[idx],hex(a.view(np.uint32)[idx]),hex(b.view(np.uint32)[idx]))
    total=sum(sum(r['differences'].values())+r['changed_halo_words'] for r in rows)
    scope='complete production MPAS driver boundary versus strict pristine MMM/wrapper and verbatim driver loops; not whole-forecast acceptance'
    Path(receipt).write_text(json.dumps(dict(scope=scope,backend=backend,process_count=1,compared_words=words,rows=rows,total_differing_words=total),indent=2)+'\n')
    print('driver differing words',total)
    return total

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('native',type=Path);p.add_argument('host',type=Path)
    p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host-witness','hex-cuda','host-glibc-transport'),default='host-witness')
    p.add_argument('--device-math-header',type=Path)
    args=p.parse_args()
    raise SystemExit(bool(compare(args.native,args.host,args.receipt,args.backend,
        args.device_math_header.read_text() if args.device_math_header else None)))
