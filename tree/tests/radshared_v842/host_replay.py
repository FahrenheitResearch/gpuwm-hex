"""CPU/glibc witness against the pristine full MPAS cloudiness driver."""
import ctypes as ct
import json
from pathlib import Path
import sys

import numpy as np

from radshared_replay_v842 import P,ptr,decks,summary as compare,vertical_decks


def summary(a,b):
    return compare(a,b,backend='host_glibc')

native=ct.CDLL(str(Path(sys.argv[1]).resolve()))
host=ct.CDLL(str(Path(sys.argv[2]).resolve()))
native.cloud.argtypes=[ct.c_int]*5+[ct.c_float]+[P]*12
host.host_index.argtypes=[ct.c_int]
host.cloud_from.argtypes=[ct.c_int]*2+[ct.c_float]+[P]*13
host.cloud_simple.argtypes=[ct.c_int]*4+[P]*7
host.cloud_thompson.argtypes=[ct.c_int]*2+[P]*11
results=[]
for d,deck in enumerate(decks()):
    nc,nz=deck['qv'].shape
    for mode in range(3):
        flags=((False,False),(False,True),(True,False),(True,True)) if mode==0 else ((True,True),)
        for fqc,fqi in flags:
            n=np.empty((5,nc,nz),dtype='f4');ndx=np.empty(nc,dtype='f4')
            native.cloud(nc,nz,mode,fqc,fqi,deck['len_disp'],
                         *[ptr(deck[k]) for k in ('meshDensity','xland','qv','qc','qi','qs','t','pres','pres_hyd','rho')],ptr(n),ptr(ndx))
            cf=np.empty((nc,nz),dtype='f4');v=cf.copy();c=cf.copy();ice=cf.copy();snow=cf.copy()
            dx=np.empty(nc,dtype='f4');land=dx.copy()
            for i in range(nc*nz):
                host.host_index(i)
                host.cloud_from(nc,nz,deck['len_disp'],*[ptr(deck[k]) for k in ('meshDensity','xland','qv','qc','qi','qs')],
                                *[ptr(x) for x in (dx,land,v,c,ice,snow,cf)])
            if mode<2:
                for i in range(nc*nz):
                    host.host_index(i)
                    host.cloud_simple(nc*nz,mode,fqc,fqi,*[ptr(x) for x in (v,c,ice,snow,deck['t'],deck['pres'],cf)])
            else:
                scratch=np.empty((nc,nz,3),dtype='f4')
                for i in range(nc):
                    host.host_index(i)
                    host.cloud_thompson(nc,nz,*[ptr(x) for x in (v,c,ice,snow,deck['pres_hyd'],deck['t'],deck['rho'],land,dx,cf,scratch)])
            rows={name:summary(expected,got) for name,expected,got in zip(
                  ('cldfrac','qvrad','qcrad','qirad','qsrad','dx'),(*n,ndx),(cf,v,c,ice,snow,dx))}
            results.append({'deck':d,'mode':mode,'fqc':fqc,'fqi':fqi,'arrays':rows})

# Ozone full-module witness, including both distinct calendar routines.
from datetime import date
native.ozone_init.argtypes=[ct.c_int]*2+[P]*3
native.ozone_time.argtypes=[ct.c_int]*2+[ct.c_float]+[P]*2
native.ozone_update.argtypes=[ct.c_int]*5+[P]*2
native.ozone_vertical.argtypes=[ct.c_int]*3+[P]*4
host.ozone_latitude.argtypes=[ct.c_int]*2+[P]*4
host.ozone_time.argtypes=[ct.c_int,ct.c_float,P,P]
host.ozone_update.argtypes=[ct.c_int]*5+[P]*2
IP=ct.POINTER(ct.c_int)
host.ozone_min_upper.argtypes=[ct.c_int]*2+[IP]*2
host.ozone_vertical_level.argtypes=[ct.c_int]*5+[P]*3+[IP]*2+[P]
import os
base=Path(sys.argv[4]).resolve();old=os.getcwd();os.chdir(base)
lat_oz=np.loadtxt('OZONE_LAT.TBL',dtype='f4')
levels=np.loadtxt('OZONE_PLEV.TBL',dtype='f4')
table=np.loadtxt('OZONE_DAT.TBL',dtype='f4').reshape(12,64,levels.size)
lat=np.deg2rad(np.array([-90,-86,-60,-30,0,30,60,86,90],dtype='f4')).astype('f4')
nc=lat.size;nl=levels.size
monthly=np.empty((nc,nl,12),dtype='f4');pin=np.empty(nl,dtype='f4')
native.ozone_init(nc,nl,ptr(lat),ptr(pin),ptr(monthly));os.chdir(old)
hmonth=np.empty_like(monthly)
for i in range(hmonth.size):
    host.host_index(i);host.ozone_latitude(nc,nl,ptr(lat),ptr(lat_oz),ptr(table),ptr(hmonth))
ozone={'latitude':summary(monthly,hmonth),'pressure':summary(pin,levels)}
nclim=np.empty((nc,nl),dtype='f4');hclim=np.empty_like(nclim)
for day in (0.,.5,14.9999,15.,44.,73.5,89.5,349.,354.5,364.9,365.,730.5):
    native.ozone_time(nc,nl,day,ptr(monthly),ptr(nclim))
    for i in range(nclim.size):
        host.host_index(i);host.ozone_time(nclim.size,day,ptr(hmonth),ptr(hclim))
    ozone[f'time:{day}']=summary(nclim,hclim)
from hexcore.mpasphys.radshared_v842 import OzoneCalendarState,_monthly_date_weights
calendar=OzoneCalendarState()
for date_str in ('2023-01-01','2023-01-15','2023-03-31','2022-12-22','2024-02-29','2025-03-01','2025-03-31','2000-02-29','2100-03-01'):
    year,mon,day=map(int,date_str.split('-'))
    native.ozone_update(nc,nl,year,mon,day,ptr(monthly),ptr(nclim))
    m1,m2,d1,d2=_monthly_date_weights(date_str,calendar)
    for i in range(nclim.size):
        host.host_index(i);host.ozone_update(nclim.size,m1,m2,d1,d2,ptr(hmonth),ptr(hclim))
    ozone[f'update:{date_str}']=summary(nclim,hclim)
for name,p in vertical_decks(nc,pin):
    for pascal in (False,True):
        source=p*np.float32(100) if pascal else p
        native_pressure=source/np.float32(100) if pascal else source
        nout=np.empty_like(p);hout=np.empty_like(p)
        native.ozone_vertical(nc,p.shape[1],nl,ptr(native_pressure),ptr(pin),ptr(nclim),ptr(nout))
        upper=np.zeros(nc,dtype='i4');start=np.empty(1,dtype='i4')
        for k in range(p.shape[1]-1,-1,-1):
            host.host_index(0);host.ozone_min_upper(nc,nl,upper.ctypes.data_as(IP),start.ctypes.data_as(IP))
            for i in range(nc):
                host.host_index(i);host.ozone_vertical_level(nc,p.shape[1],nl,k,pascal,ptr(source),ptr(pin),ptr(hclim),
                    start.ctypes.data_as(IP),upper.ctypes.data_as(IP),ptr(hout))
        ozone[f'vertical:{name}:pascal={pascal}']=summary(nout,hout)
adapter={}
host.profile_from_mpas.argtypes=[ct.c_int]*3+[P]*2
host.profile_to_mpas.argtypes=[ct.c_int]*3+[P]*2
host.copy_triplet.argtypes=[ct.c_int]*2+[P]*6
nsolve,nz,ntotal=7,11,10
public=np.arange(nz*ntotal,dtype='f4').reshape(nz,ntotal)
local=np.empty((nsolve,nz),dtype='f4')
for i in range(local.size):
    host.host_index(i);host.profile_from_mpas(nsolve,nz,ntotal,ptr(public),ptr(local))
adapter['from_public']=summary(public[:,:nsolve].T,local)
back=np.full_like(public,np.float32(-12345))
for i in range(local.size):
    host.host_index(i);host.profile_to_mpas(nsolve,nz,ntotal,ptr(local),ptr(back))
expected=np.full_like(public,np.float32(-12345));expected[:,:nsolve]=local.T
adapter['to_public_with_halos']=summary(expected,back)
radii=[np.linspace(1e-6,2e-4,77,dtype='f4').reshape(7,11)*np.float32(n) for n in (1,2,3)]
native.radius_from.argtypes=[ct.c_int]*4+[P]*4
native.radius_to.argtypes=[ct.c_int]*4+[P]*4
for enabled in (False,True):
    outputs=[np.empty_like(local) for _ in range(3)]
    for i in range(local.size):
        host.host_index(i);host.copy_triplet(local.size,enabled,*[ptr(x) for x in (*radii,*outputs)])
    for mode in range(4):
        for direction in ('from','to'):
            native_out=np.empty((3,nsolve,nz),dtype='f4')
            getattr(native,f'radius_{direction}')(nsolve,nz,mode,enabled,
                                                   *[ptr(r) for r in radii],ptr(native_out))
            expected=outputs if mode<3 else [np.full_like(local,np.float32(-12345))]*3
            for n,out in enumerate(expected):
                adapter[f'radius:{mode}:{enabled}:{direction}:{n}']=summary(native_out[n],out)
count=sum(x['differing_words'] for d in results for x in d['arrays'].values())
count+=sum(x['differing_words'] for x in ozone.values())
count+=sum(x['differing_words'] for x in adapter.values())
Path(sys.argv[3]).write_text(json.dumps({'purpose':'host-glibc witness, not forecast acceptance','decks':results,
                                       'ozone':ozone,'adapter':adapter,'differing_words':count},indent=2)+'\n')
print('host differing words:',count)
raise SystemExit(1 if count else 0)
