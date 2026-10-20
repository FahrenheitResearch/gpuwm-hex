"""Exercise boundary outcomes that full land columns cannot ordinarily reach."""
from pathlib import Path
import argparse
import ctypes as C
import json
import numpy as np
from hexcore.mpasphys.noah_v842 import (COLUMN_LAYOUTS, table_dtype, _CORE_CUDA,
    _device_math_source, _compile_module)

parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
base=Path(__file__).resolve().parent
layouts=json.loads((base/'helper_layouts.json').read_text())
fixture=np.load('n1out/noah-synthetic.npz');tables=fixture['table_bytes'].copy()
soil=np.frombuffer(tables,table_dtype(),count=1)[0]
native=C.CDLL(str(base/'libnoah_oracle.so')) if not args.gpu else None
if native:
    import os
    old=Path.cwd();os.chdir(base);native.read_tables();os.chdir(old)
    host=C.CDLL(str(base/'libnoah_host.so'))
else:
    import cupy as cp
    module=_compile_module(_device_math_source()+_CORE_CUDA+(base/'helper_entry.cu').read_text())
defaults=dict(zip(COLUMN_LAYOUTS['noah']['float'],fixture['native'][0]))
defaults.update(SMCMAX=.45,SMCWLT=.1,SMCREF=.3,SMCDRY=.05,BEXP=4.,PSIS=.1,PSISAT=.1,
    SFCPRS=95000.,CH=.01,T2V=280.,TH2=280.,SFCTMP=280.,T1=280.,TBOT=280.,TEMP=255.,
    T24=np.float32(280.)**np.int32(4),FDOWN=400.,DT=60.,SLOPE=.1,KDT=3.,FRZFACT=.5,
    DKSAT=1.e-5,DWSAT=1.e-6,SHDFAC=.5,SBETA=-2.,PC=.5,CMCMAX=.0005,CFACTR=.5,
    ZBOT=-8.,F1=1.,DF1=1.,RCH=10.,RR=1.,EPSCA=1.,Q2=.005,Q2SAT=.005,DQSDT2=.0003,
    QUARTZ=.5,QZ=.5,CSOIL=2.e6,FXEXP=2.,EMISSI=.96,EMISSI_IN=.96,ALB=.2,SNOALB=.8,
    EMBRD=.96,TSNOW=270.,SHDMIN=.1,LVCOEF=.5,SNOTIME1=0.,SNCOVR=.5,AOASIS=0.,
    ALBEDO=.2,SOLDN=200.,FVB=.5,GAMA=.5,STC1=270.,ETP=.001,ETP1=.000001,CMC=0.,
    PRCP=0.,PRCPF=0.,SFHEAD1RT=0.,INFXS1RT=0.,ETPND1=0.,HCPCT_FASDAS=2.e6,
    IRRIGATION_CHANNEL=0.,XSDA_QFX=.001,XQNORM=1.,ESD=.1,SNDENS=.2,SNOWH=.5,
    NEWSN=.001,TKELV=265.,SMC=.3,SH2O=.2)
for k in range(4):
    defaults.update({f'SMC[{k}]':.3,f'SH2O[{k}]':.2,f'STC[{k}]':280.,f'ZSOIL[{k}]':(-.1,-.4,-1.,-2.)[k],f'RTDIS[{k}]':.25,f'ET[{k}]':0.})
idefaults=dict(NSOIL=4,NROOT=4,SOILTYP=3,VEGTYP=7,ISURBAN=1,OPT_THCND=1,FASDAS=0,SNOWNG=0,FRZGRA=0,UA_PHYS=0)
results=[]
for name,l in layouts.items():
    n=8192 if name in ('noah_frh2o','noah_alcalc','gl_alcalc') else 64
    f={nm:k for k,nm in enumerate(l['float'])};ii={nm:k for k,nm in enumerate(l['int'])}
    x=np.zeros((n,len(f)),np.float32);ix=np.zeros((n,max(1,len(ii))),np.int32)
    for nm,k in f.items():x[:,k]=defaults.get(nm,0.)
    for nm,k in ii.items():ix[:,k]=idefaults.get(nm,0)
    def put(nm,vals):
        if nm in f:x[:,f[nm]]=np.resize(np.asarray(vals,np.float32),n)
        elif nm in ii:ix[:,ii[nm]]=np.resize(np.asarray(vals,np.int32),n)
    put('SNOWNG',(0,1));put('UA_PHYS',(0,1));put('FASDAS',(0,1));put('OPT_THCND',(0,1,2,3));put('SOILTYP',(3,4,1,2))
    if name=='noah_fac2mit':put('SMCMAX',(.395,.434,.404,.465,.406,.476,.439,.200,.464,.45))
    if name=='noah_frh2o':
        rng=np.random.default_rng(1576)
        put('BEXP',10.**rng.uniform(-5,1,n));put('PSIS',10.**rng.uniform(-10,5,n))
        put('TKELV',rng.uniform(150,273.149,n));put('SMC',rng.uniform(.02,.5,n));put('SH2O',rng.uniform(0,.5,n))
        # Invalid negative B is accepted by this native routine; exercise its iteration fallback.
        x[::8,f['BEXP']]*=-1.
    if name in ('gl_snow_new','noah_snow_new'):
        put('SNOWH',(0.,1.e-8,.1,.5));put('NEWSN',(1.e-9,0.,.0001,.01));put('TEMP',(250.,270.,260.,280.))
    if name in ('noah_evapo','noah_nopac','noah_snopac'):
        put('ETP',(-.001,0.,.001,.01));put('ETP1',(-.000001,0.,.000001,.00001))
    if name=='noah_snopac':
        put('UA_PHYS',(0,1,1,0));put('ESD',(.001,.1,1.e-12,1.e-12));put('DT',(60.,60.,1.e6,1.e6));put('SNCOVR',(0.,1.,.5,1.));put('RIBB',(.5,.5,0.,0.));put('FDOWN',(400.,100.,400.,400.))
        # Warm complete-sublimation cases reach both UA outcomes at line 3296.
        put('ETP',(-.001,0.,.000001,.000001));put('FDOWN',(400.,100.,1000.,1000.))
    if name=='noah_devap_hydro':
        put('SMC',(.01,.3));put('SFHEAD1RT',(0.,10.));put('ZSOIL',(-.1,))
    if name=='noah_penman':
        put('SNEQV',(.1,.1,0.,0.));put('FDOWN',(500.,500.,500.,500.));put('SOLDN',(0.,0.,100.,100.))
    if name in ('noah_alcalc','gl_alcalc'):
        rng=np.random.default_rng(999)
        put('SNOALB',rng.uniform(0.,1.,n));put('ALB',rng.uniform(0.,1.,n))
        put('SNCOVR',(0.,.5,1.,1.));put('LVCOEF',rng.uniform(0.,1.,n))
    a=x.copy();ia=ix.copy();b=x.copy();ib=ix.copy()
    if not args.gpu:
        nf=getattr(native,'helper_'+name);nf.argtypes=[C.c_void_p,C.c_void_p,C.c_int];nf(a.ctypes.data,ia.ctypes.data,n)
        hf=getattr(host,'helper_'+name);hf.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int];hf(b.ctypes.data,ib.ctypes.data,tables.ctypes.data,n)
        np.savez(Path('n1out')/(name+'-helper.npz'),input=x,input_int=ix,native=a,native_int=ia,table_bytes=tables)
        errors=0
    else:
        z=np.load(Path('n1out')/(name+'-helper.npz'));a=z['native'];ia=z['native_int'];dx,di,dt=map(cp.asarray,(z['input'],z['input_int'],z['table_bytes']));status=cp.empty(n,cp.int32)
        module.get_function('helper_'+name)(((n+63)//64,),(64,),(dx,di,dt,status,np.int32(n)))
        b=cp.asnumpy(dx);ib=cp.asnumpy(di);errors=int(cp.count_nonzero(status).get())
    d=a.view(np.uint32)!=b.view(np.uint32)
    names=l['float']
    result={'routine':name,'columns':n,'differing_words':int(d.sum()),'integer_differing_words':int((ia!=ib).sum()),'error_columns':errors,'by_array':{nm:int(d[:,k].sum()) for nm,k in f.items() if d[:,k].any()},
        'examples':[{'column':int(c),'array':names[k],'native':float(a[c,k]),'port':float(b[c,k]),'native_word':hex(int(a.view(np.uint32)[c,k])),'port_word':hex(int(b.view(np.uint32)[c,k]))} for c,k in np.argwhere(d)[:10]]}
    result.update(process_count=1, native_execution='standalone, no MPI')
    results.append(result)
path=Path('receipts/helpers-cuda.json' if args.gpu else 'n1out/helpers-host.json');path.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if any(r['differing_words'] or r['integer_differing_words'] or r['error_columns'] for r in results):raise SystemExit(1)
