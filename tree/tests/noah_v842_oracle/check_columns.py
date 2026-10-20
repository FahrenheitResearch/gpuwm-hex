from pathlib import Path
import ctypes as C,json,os,argparse
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hexcore.mpasphys.noah_v842 import TABLE_SCHEMA, COLUMN_LAYOUTS
parser=argparse.ArgumentParser();parser.add_argument('--landuse',choices=('MODIFIED_IGBP_MODIS_NOAH','USGS'),default='MODIFIED_IGBP_MODIS_NOAH');parser.add_argument('--output',type=Path,default=Path('n1out'));args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
isurban,isice,iswater=(1,24,16) if args.landuse=='USGS' else (13,15,17)
schema={'globals':TABLE_SCHEMA};layouts=COLUMN_LAYOUTS
class Tables(C.Structure):pass
fields=[]
for nm,v in schema['globals'].items():
    typ={'float':C.c_float,'int':C.c_int,'bool':C.c_bool,'char':C.c_char}[v['type']]
    if v['type']=='char':typ=typ*256
    elif v['dims']:
        d=v['dims'][0].split(':')[-1];typ=typ*{'NLUS':50,'NSLTYPE':30,'NSLOPE':30}.get(d,int(d) if d.isdigit() else 0)
    fields.append((nm,typ))
Tables._fields_=fields+[('error',C.c_int)]
base=Path(__file__).resolve().parent;native=C.CDLL(str(base/'libnoah_oracle.so'));host=C.CDLL(str(base/'libnoah_host.so'))
old=Path.cwd();os.chdir(base);native.read_tables_for.argtypes=[C.c_char_p,C.c_char_p,C.c_int,C.c_int];veg=args.landuse.encode();native.read_tables_for(veg,b'STAS',len(veg),4);os.chdir(old)
t=Tables()
for nm,typ in fields:
    v=schema['globals'][nm]
    if v['parameter']:
        val=v['init'].replace('D','E');setattr(t,nm,int(val) if v['type']=='int' else float(val));continue
    try:var=typ.in_dll(native,'__module_sf_noahlsm_MOD_'+nm.lower())
    except ValueError:continue
    if v['dims']:C.memmove(C.addressof(getattr(t,nm)),C.addressof(var),C.sizeof(typ))
    elif v['type']=='char':setattr(t,nm,bytes(var))
    else:setattr(t,nm,var.value)
results=[]
for name,l in layouts.items():
    n={'noah':4096,'glacial':1024,'driver':1024,'init':512,'sfcdif':1024}[name];x=np.zeros((n,len(l['float'])),dtype=np.float32);ix=np.zeros((n,len(l['int'])),dtype=np.int32)
    f={nm:k for k,nm in enumerate(l['float'])};ii={nm:k for k,nm in enumerate(l['int'])}
    values={'FFROZP':0.,'DT':60.,'ZLVL':25.,'LWDN':300.,'SOLDN':500.,'SOLNET':400.,'SFCPRS':95000.,'PRCP':0.,'SFCTMP':290.,'Q2':.006,'SFCSPD':4.,'TH2':294.,'Q2SAT':.012,'DQSDT2':.0008,'SHDFAC':.4,'SHDMIN':.1,'SHDMAX':.8,'ALB':.2,'SNOALB':.8,'TBOT':285.,'Z0BRD':.1,'Z0':.1,'EMISSI':.96,'EMBRD':.96,'CMC':.0001,'T1':292.,'CH':.01,'CM':.01,'XLAI':2.,'RIBB':-.1,'AOASIS':1.,'XQNORM':.003}
    for nm,val in values.items():
        if nm in f:x[:,f[nm]]=val
    for nm,val in {'IILOC':1,'JJLOC':1,'NSOIL':4,'ISURBAN':isurban,'ISICE':isice,'VEGTYP':7,'SOILTYP':3,'SLOPETYP':1,'OPT_THCND':1}.items():
        if nm in ii:ix[:,ii[nm]]=val
    if name=='sfcdif':
        rng=np.random.default_rng(642)
        for c in range(n):
            for nm,val in {'ZLM':rng.uniform(2.,100.),'Z0':rng.uniform(.0001,1.),'THZ0':290.,'THLM':rng.choice((280.,290.,300.)),'SFCSPD':rng.choice((0.,.1,4.,20.)),'CZIL':.1,'AKMS':.01,'AKHS':.01}.items():x[c,f[nm]]=val
    elif name=='init':
        for c in range(n):
            ix[c,ii['IVGTYP']]=1+c%t.LUCATS;ix[c,ii['ISLTYP']]=1+c%19
            ix[c,ii['RESTART']]=(c>>4)&1;ix[c,ii['INPUT_SFC_ALBEDO']]=(c>>5)&1
            x[c,f['SNOALB']]=.25;x[c,f['SNOW']]=40.;x[c,f['SNOWH']]=.2
            for k in range(4):
                x[c,f[f'TSLB[{k}]']]=269.+k if c&1 else 280.+k
                x[c,f[f'SMOIS[{k}]']]=.2+.03*k;x[c,f[f'SH2O[{k}]']]=.01
    elif name=='driver':
        values={'DT':60.,'TSK':292.,'TMN':285.,'XLAND':1.,'VEGFRA':40.,'SHDMIN':10.,'SHDMAX':80.,'SNOALB':.8,'ALBEDO':.2,'ALBBCK':.2,'EMISS':.96,'EMBCK':.96,'Z0':.1,'ZNT':.1,'CANWAT':.1,'CHS':.01,'CQS2':.01,'LAI':2.,'RIB':-.1,'XICE_THRESHOLD':.55,'GLW':300.,'SWDOWN':500.,'QGH':.012}
        for nm,val in values.items():x[:,f[nm]]=val
        for nm,val in {'ITIMESTEP':2,'ISURBAN':isurban,'ISICE':isice,'IVGTYP':7,'ISLTYP':3,'OPT_THCND':1}.items():ix[:,ii[nm]]=val
        for k in range(2):
            x[:,f[f'P8W3D[{k}]']]=95000.-k*1000.;x[:,f[f'T3D[{k}]']]=290.;x[:,f[f'QV3D[{k}]']]=.006;x[:,f[f'DZ8W[{k}]']]=50.
        for k,dep in enumerate((.1,.3,.6,1.)):
            x[:,f[f'DZS[{k}]']]=dep;x[:,f[f'TSLB[{k}]']]=285.;x[:,f[f'SMOIS[{k}]']]=.3;x[:,f[f'SH2O[{k}]']]=.3
        for c in range(n):
            if c&1:
                x[c,f['TSK']]=268.;x[c,f['TMN']]=270.
                for k in range(2):x[c,f[f'T3D[{k}]']]=265.
                for k in range(4):x[c,f[f'TSLB[{k}]']]=269.+k;x[c,f[f'SH2O[{k}]']]=.1
            if c&2:x[c,f['SNOW']]=40.;x[c,f['SNOWH']]=.2;x[c,f['SNOWC']]=.7
            if c&4:x[c,f['RAINBL']]=.06
            if c&8:x[c,f['GLW']]=150.;x[c,f['SWDOWN']]=0.
            terrain=(c>>4)&3
            if terrain==1:x[c,f['XLAND']]=2.
            if terrain==2:x[c,f['XICE']]=.8
            if terrain==3:ix[c,ii['IVGTYP']]=isice
            if c&64:ix[c,ii['ITIMESTEP']]=1
    else:
        for k,dep in enumerate((.1,.3,.6,1.)):
            x[:,f[f'SLDPTH[{k}]']]=dep;x[:,f[f'STC[{k}]']]=285.
            if name=='noah':x[:,f[f'SMC[{k}]']]=.3;x[:,f[f'SH2O[{k}]']]=.3
        for c in range(n):
            if c&1:
                x[c,f['SFCTMP']]=265.;x[c,f['TH2']]=267.;x[c,f['T1']]=268.;x[c,f['TBOT']]=270.;x[c,f['FFROZP']]=1.
                for k in range(4):x[c,f[f'STC[{k}]']]=269.+k
                if name=='noah':
                    for k in range(4):x[c,f[f'SH2O[{k}]']]=.1
            if c&2:x[c,f['SNEQV']]=.04;x[c,f['SNOWH']]=.2
            if c&4:x[c,f['PRCP']]=.001
            if c&8:x[c,f['LWDN']]=150.;x[c,f['SOLNET']]=0.
            if name=='noah':
                if c&16:ix[c,ii['VEGTYP']]=t.BARE
                if c&32:ix[c,ii['UA_PHYS']]=1
            else:
                if c&16:x[c,f['SNEQV']]=.2;x[c,f['SNOWH']]=.9
                if c&32:ix[c,ii['ISICE']]=-1
    # Deterministic broad forcing set; retain the original branch fixtures first.
    rng=np.random.default_rng(642)
    first=128 if name=='driver' else 64
    for c in range(first,n):
        if name=='sfcdif':continue
        if name=='init':
            soil=1+c%19
            ix[c,ii['ISLTYP']]=soil
            for k in range(4):
                x[c,f[f'TSLB[{k}]']]=rng.uniform(245.,290.)
                x[c,f[f'SMOIS[{k}]']]=rng.uniform(.001,float(t.MAXSMC[soil-1]))
            continue
        if name=='driver':
            ix[c,ii['IVGTYP']]=1+c%t.LUCATS
            soil=(1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19)[c%18]
            ix[c,ii['ISLTYP']]=soil
            ix[c,ii['USEMONALB']]=c%2
            x[c,f['TSK']]=rng.uniform(245.,320.)
            x[c,f['RAINBL']]=rng.choice((0.,.01,1.,20.))
            x[c,f['VEGFRA']]=rng.choice((0.,5.,40.,80.,100.))
            x[c,f['CANWAT']]=rng.choice((0.,.1,1.))
            for k in range(4):
                x[c,f[f'TSLB[{k}]']]=rng.uniform(250.,310.)
                m=rng.choice((1.e-8,rng.uniform(.001,float(t.MAXSMC[soil-1]))))
                x[c,f[f'SMOIS[{k}]']]=m;x[c,f[f'SH2O[{k}]']]=m*rng.choice((.05,.5,1.))
            continue
        if name=='glacial':x[c,f['SNCOVR']]=rng.choice((0.,1.))
        for nm,limits in {'DT':(1.,1800.),'ZLVL':(2.,100.),'SFCTMP':(240.,320.),'TH2':(240.,325.),'T1':(240.,325.),'TBOT':(255.,305.),'Q2SAT':(.001,.04),'DQSDT2':(.00005,.003),'CH':(.001,.1),'CM':(.001,.1),'LWDN':(100.,500.),'SOLDN':(0.,1200.),'SFCPRS':(55000.,105000.)}.items():
            if nm in f:x[c,f[nm]]=rng.uniform(*limits)
        x[c,f['SOLNET']]=rng.uniform(0.,960.)
        x[c,f['Q2']]=x[c,f['Q2SAT']]*rng.uniform(.01,1.2)
        x[c,f['PRCP']]=rng.choice((0.,.001,.1,1.))
        x[c,f['FFROZP']]=rng.choice((0.,.5,1.))
        x[c,f['SNEQV']]=rng.choice((0.,1.e-9,1.e-5,.001,.05,.2,1.))
        x[c,f['SNOWH']]=x[c,f['SNEQV']]/rng.uniform(.05,.5)
        x[c,f['SNOTIME1']]=rng.uniform(0.,8.e6)
        x[c,f['RIBB']]=rng.choice((-1.,0.,.1,.5,10.))
        x[c,f['SNOALB']]=rng.choice((0.,.01,.4,.8))
        for k in range(4):x[c,f[f'STC[{k}]']]=rng.uniform(245.,310.)
        if name=='noah':
            soil=(1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19)[c%18]
            ix[c,ii['SOILTYP']]=soil;ix[c,ii['VEGTYP']]=1+c%t.LUCATS
            ix[c,ii['SLOPETYP']]=1+c%9
            for flag in ('RDLAI2D','USEMONALB','UA_PHYS','FASDAS'):ix[c,ii[flag]]=int(rng.integers(2))
            ix[c,ii['OPT_THCND']]=1+c%2
            x[c,f['SHDFAC']]=rng.choice((0.,.05,.1,.4,.8,1.))
            x[c,f['CMC']]=rng.choice((0.,.0001,.001))
            x[c,f['IRRIGATION_CHANNEL']]=rng.choice((0.,.00001,.001))
            x[c,f['XSDA_QFX']]=rng.uniform(-.002,.002)
            for k in range(4):
                m=rng.choice((1.e-8,rng.uniform(.001,float(t.MAXSMC[soil-1]))))
                x[c,f[f'SMC[{k}]']]=m;x[c,f[f'SH2O[{k}]']]=m*rng.choice((.05,.5,1.))
    if name=='driver':
        # MPAS repairs water-typed land soil and inconsistent snow depth.
        ix[0,ii['ISLTYP']]=14
        # Fractional ice below the driver threshold keeps water soil unchanged.
        x[4]=x[0];ix[4]=ix[0];x[4,f['XICE']]=.1
        x[2,f['SNOWH']]=0.
        x[3]=x[2];ix[3]=ix[2];x[3,f['SNOWC']]=0.
    if name=='glacial':
        # Long synthetic column steps reach complete sublimation of the minimum ice snowpack.
        x[::16,f['DT']]=1.e6
    if name=='noah':
        # Exact Penman balance reaches ETP == 0 with finite, ordinary temperatures.
        c=0;temp=np.float32(280.);ix[c,ii['UA_PHYS']]=0;ix[c,ii['VEGTYP']]=7
        ix[c,ii['USEMONALB']]=0;ix[c,ii['RDLAI2D']]=0;ix[c,ii['FASDAS']]=0
        for nm in ('SFCTMP','TH2','T1','TBOT'):x[c,f[nm]]=temp
        for k in range(4):x[c,f[f'STC[{k}]']]=temp
        for nm in ('SNEQV','SNOWH','PRCP','SHDFAC','SOLNET','SOLDN'):x[c,f[nm]]=0.
        x[c,f['Q2']]=x[c,f['Q2SAT']]
        t24=np.float32(np.float32(np.float32(temp*temp)*temp)*temp)
        # At SHDFAC < SHDMIN, SFLX selects EMISSMINTBL for vegetation 7.
        x[c,f['LWDN']]=np.float32(np.float32(t.EMISSMINTBL[6]*t.SIGMA)*t24)
        x[1]=x[0];ix[1]=ix[0]
        for k in range(4):x[1,f[f'SLDPTH[{k}]']]=1.e-7
        # NaN vegetation bounds reach SFLX's mean-parameter fallback, whose
        # inequalities exclude finite inputs (native lines 446-491).
        for c in range(4,8):
            x[c]=x[0];ix[c]=ix[0]
            x[c,f['SHDFAC']]=.4
            x[c,f['SHDMIN']]=np.nan;x[c,f['SHDMAX']]=np.nan
            ix[c,ii['RDLAI2D']]=c&1;ix[c,ii['USEMONALB']]=(c>>1)&1
    saved=x.copy();saved_ix=ix.copy();a=x.copy();ia=ix.copy();b=x.copy();ib=ix.copy()
    nf=getattr(native,'replay_'+name);nf.argtypes=[C.c_void_p,C.c_void_p,C.c_int];nf(a.ctypes.data,ia.ctypes.data,n)
    hf=getattr(host,'replay_'+name);hf.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int];hf(b.ctypes.data,ib.ctypes.data,C.byref(t),n)
    diff=a.view(np.uint32)!=b.view(np.uint32);rows=np.argwhere(diff)
    result={'scheme':name,'columns':n,'differing_words':int(diff.sum()),'integer_differing_words':int((ia!=ib).sum()),'by_array':{nm:int(diff[:,k].sum()) for k,nm in enumerate(l['float']) if diff[:,k].any()},'examples':[{'column':int(c),'array':l['float'][k],'native':float(a[c,k]),'host':float(b[c,k])} for c,k in rows[:20]]};results.append(result)
    np.savez(args.output/(name+'-synthetic.npz'),input=saved,input_int=saved_ix,native=a,native_int=ia,table_bytes=np.frombuffer(bytes(t),dtype=np.uint8))
print(json.dumps(results,indent=2))
(args.output/'host-comparison.json').write_text(json.dumps(results,indent=2))

if args.landuse=='USGS':
    if any(r['differing_words'] or r['integer_differing_words'] for r in results):raise SystemExit(1)
    raise SystemExit(0)

# Native LANDUSE.TBL decoder and original seasonal initialization slice.
from hexcore.mpasphys.noah_v842 import LANDUSE_FLOATS, LANDUSE_INTEGERS
n=256;f={nm:k for k,nm in enumerate(LANDUSE_FLOATS)};ii={nm:k for k,nm in enumerate(LANDUSE_INTEGERS)}
x=np.zeros((n,14),dtype=np.float32);ix=np.zeros((n,9),dtype=np.int32)
lt=np.zeros((7,12,100),dtype=np.float32)
os.chdir(base);native.read_landuse.argtypes=[C.c_void_p];native.read_landuse(lt.ctypes.data);os.chdir(old)
for c in range(n):
    x[c,f['LATCELL']]=-.5 if c&1 else .5
    x[c,f['SNOALB']]=.8;x[c,f['SNOWC']]=.6 if c&2 else .5
    x[c,f['XICE']]=.55 if c&4 else .54;x[c,f['XICE_THRESHOLD']]=.55
    x[c,f['ALBBCK']]=.17;x[c,f['MAVAIL']]=.2;x[c,f['UST']]=.3
    ix[c,ii['IVGTYP']]=c%21;ix[c,ii['JULDAY']]=(104,105,287,288)[(c>>3)&3]
    ix[c,ii['ISWATER']]=17;ix[c,ii['ISICE']]=15
    ix[c,ii['SFC_ALBEDO_INPUT']]=(c>>5)&1;ix[c,ii['FRAC_SEAICE']]=(c>>6)&1
    ix[c,ii['HAS_MAVAIL']]=(c>>7)&1;ix[c,ii['HAS_UST']]=1-(c>>7)&1
    ix[c,ii['RESTART']]=int(c>=224)
a=x.copy();b=x.copy();ia=ix.copy();ib=ix.copy()
for lib,arr,ints in [(native,a,ia),(host,b,ib)]:
    fn=lib.replay_landuse;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int];fn(arr.ctypes.data,ints.ctypes.data,lt.ctypes.data,n)
d=a.view(np.uint32)!=b.view(np.uint32)
result={'scheme':'landuse','columns':n,'differing_words':int(d.sum()),'integer_differing_words':int((ia!=ib).sum()),'by_array':{nm:int(d[:,k].sum()) for k,nm in enumerate(LANDUSE_FLOATS) if d[:,k].any()}}
results.append(result)
np.savez(args.output/'landuse-synthetic.npz',input=x,input_int=ix,native=a,native_int=ia,table_bytes=lt)
print(json.dumps(result,indent=2));(args.output/'host-comparison.json').write_text(json.dumps(results,indent=2))

if any(r['differing_words'] or r['integer_differing_words'] for r in results):
    raise SystemExit(1)
