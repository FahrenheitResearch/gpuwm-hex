from pathlib import Path
import ctypes as C,json,itertools
import numpy as np
def load_oracles(build_dir):
    root = Path(build_dir).resolve()
    native = C.CDLL(str(root / "libnative.so"))
    host = C.CDLL(str(root / "libtranslated.so"))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    bptr = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    native.kf_native_tables.argtypes = [ptr]
    native.kf_native_column.argtypes = [C.c_int,ptr,ptr,ptr,C.POINTER(C.c_int),ptr,iptr]
    native.kf_native_init.argtypes = [C.c_int,ptr,ptr,C.c_int,C.c_int,C.c_int,ptr]
    host.kf_host_tables.argtypes = [ptr,ptr]
    host.kf_host_column.argtypes = [C.c_int,C.c_int,ptr,ptr,ptr,bptr,ptr,iptr,ptr,ptr]
    native.kf_native_dx.argtypes = [C.c_int,C.c_float,ptr,ptr]
    native.kf_native_step1.argtypes = [C.c_int,C.c_float,ptr,ptr]
    native.kf_native_step2.argtypes = [C.c_int,C.c_float,C.c_float,C.c_int,ptr,ptr,iptr]
    native.kf_native_tables(PARAMS)
    table = np.empty(110423, np.float32)
    host.kf_host_tables(table, PARAMS)
    return native, host, table


PARAMS=np.array([60,5,-1,3.15e6,2370,2.905e6,259.532,1004.5,287,9.80616,np.float32(461.6)/np.float32(287)-np.float32(1),np.float32(287)/np.float32(461.6),.6112,17.67,29.65,273.15],np.float32)
def fixture(nz,lapse=7,rh=.8,ts=300,w=.5):
    z=np.arange(nz+1)*250+125
    t=np.maximum(ts-lapse*z/1000,190)
    p=100000*np.exp(-9.81*z/(287*(ts-lapse*z/2000)))
    es=611.2*np.exp(17.67*(t-273.15)/(t-29.65))
    q=rh*.622*es/(p-es)
    f=np.array([5+.0005*z,2+.0002*z,t/(p/1e5)**(287/1004.5),t,np.full(nz+1,w),np.full(nz+1,250),p,(p/1e5)**(287/1004.5),p/(287*t*(1+.608*q)),q],np.float32)
    state=np.zeros((7,nz+1),np.float32)
    surface=np.array([100000,1e10,0,0,-100,1,nz+1],np.float32)
    return f,state,surface


def cases():
    for lapse, rh, ts, w, mode in itertools.product(
            [3,6,7,8,9], [.3,.7,.95], [285,300,310], [0,.5,2], range(4)):
        yield ("smooth",lapse,rh,ts,w,mode), fixture(55,lapse,rh,ts,w), np.array(
            [5,1,0,mode==0,1,mode==3,mode>=2], np.int32)
    # Capped boundary layers force the shallow-cloud search/retry and closure.
    for cap, jump, rh, w in itertools.product([1000,1500,2000,3000], [2,5,10], [.7,.9,.99], [.5,2,5]):
        f,s,v=fixture(55,8,rh,300,w)
        z=np.arange(56)*250+125
        f[3] += (z>cap)*jump
        f[8]=f[6]/(PARAMS[8]*f[3]*(1+.608*f[9]))
        yield ("cap",cap,jump,rh,w), (f,s,v), np.array([5,1,0,0,1,1,1],np.int32)
    # Persisted rates are sentinels; hold and skipped cadence must retain them.
    for ktau, nca, adapt, cudt in itertools.product([1,2,5],[-100,0,29,30,300],[0,1],[0,5]):
        f,s,v=fixture(55,8,.9,300,1)
        s[:]=np.float32(.0125);v[2:4]=np.float32(.25);v[4]=nca
        yield ("schedule",ktau,nca,adapt,cudt),(f,s,v),np.array([5,ktau,adapt,0,1,1,1],np.int32)
    for fqr,fqi,fqs,warm in itertools.product([0,1],repeat=4):
        yield ("phase",fqr,fqi,fqs,warm),fixture(55,8,.95,300,2),np.array([5,1,0,warm,fqr,fqi,fqs],np.int32)
    for ts,lapse,rh,w in itertools.product([260,270,275,280],[6,8,9],[.7,.95],[.5,2,5]):
        yield ("cold",ts,lapse,rh,w),fixture(55,lapse,rh,ts,w),np.array([5,1,0,0,1,1,1],np.int32)
    for dt,curr in itertools.product([30,60,300],[-1,0,239,240,299,300,540]):
        yield ("adaptive",dt,curr),fixture(55,8,.95,300,2),np.array([5,2,1,0,1,1,1],np.int32)
    # Thin/deep columns reach insufficient 50-hPa source-layer depth, LCL
    # above the model top, and plume top closure; strong shear tests fallout.
    for nz,depth,lapse,rh,w,shear in itertools.product(
            [8,20,55],[400,1200,3000,8000,14000],[3,8,12],[.02,.5,.95],[0,5],[0,1]):
        z=(np.arange(nz+1)+.5)*depth/nz
        t=np.maximum(300-lapse*z/1000,190)
        p=1e5*np.exp(-9.81*z/(287*np.maximum(300-lapse*z/2000,220)))
        es=611.2*np.exp(17.67*(t-273.15)/(t-29.65))
        q=rh*.622*es/(p-es);pi=(p/1e5)**(287/1004.5)
        f=np.array([5+shear*.04*z,2+shear*.02*z,t/pi,t,
            np.full(nz+1,w),np.full(nz+1,depth/nz),p,pi,
            p/(287*t*(1+.608*q)),q],np.float32)
        s=np.zeros((7,nz+1),np.float32)
        v=np.array([100000,1e10,0,0,-100,1,nz+1],np.float32)
        yield ('depth',nz,depth,lapse,rh,w,shear),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    # Absolute detrainment threshold and very high, dry cloud bases.
    for dx,lapse,rh,ts in itertools.product([1,10,100,1000,10000,1e5,1e6],[6,8,12],[.5,.95],[280,300,310]):
        f,s,v=fixture(55,lapse,rh,ts,2*1e5/dx);v[:2]=[dx,dx*dx]
        yield ('geometry',dx,lapse,rh,ts),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    for nz,lapse,rh,ts,w in itertools.product([55,65],[10,12,15],[.0001,.001,.01,.1],[300,310,320],[5,50,500]):
        f,s,v=fixture(nz,lapse,rh,ts,w)
        f[6]=np.maximum(f[6],5500)
        f[7]=(f[6]/1e5)**np.float32(287/1004.5)
        f[2]=f[3]/f[7]
        f[8]=f[6]/(287*f[3]*(1+.608*f[9]))
        yield ('dry',nz,lapse,rh,ts,w),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    # Pressure-depth variation reaches natural downdraft-search exhaustion.
    for beta,lapse,rh in itertools.product([.001,.01,.1,.3,.7,1,3,10],[6,8,12],[.5,.95]):
        f,s,v=fixture(55,lapse,rh,300,2)
        z=(np.arange(56)+.5)*250
        pcap=np.interp(1500,z,f[6])
        f[6]=np.maximum(np.where(z>1500,pcap*np.exp(-beta*9.81*(z-1500)/(287*290)),f[6]),5500)
        f[7]=(f[6]/1e5)**np.float32(287/1004.5)
        f[2]=f[3]/f[7];f[8]=f[6]/(287*f[3]*(1+.608*f[9]))
        yield ('pressure',beta,lapse,rh),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    # Deterministic profile diversity: elevated/dry inflow, descending winds,
    # negative vertical motion, caps, column depth and variable cell geometry.
    rng=np.random.default_rng(842)
    for i in range(10000):
        nz=int(rng.choice([30,55,80]));dz=13750/nz
        z=(np.arange(nz+1)+.5)*dz
        ts=rng.uniform(280,310);lapse=rng.uniform(2.5,9.5)
        t=np.maximum(ts-lapse*z/1000,190)
        if i%2:t+=np.where(z>rng.uniform(500,3500),rng.uniform(0,12),0)
        p=rng.uniform(.7,1)*100000*np.exp(-9.81*z/(287*(ts-lapse*z/2000)))
        es=611.2*np.exp(17.67*(t-273.15)/(t-29.65))
        rh=rng.uniform(.1,.99,nz+1)
        # Three quarters of columns retain a moist source layer.
        if i%4:rh[z<2000]=rng.uniform(.6,.99)
        q=rh*.622*es/(p-es);pi=(p/1e5)**(287/1004.5)
        u=rng.uniform(-10,10)+rng.uniform(-.003,.003)*z
        v=rng.uniform(-5,5)+rng.uniform(-.001,.001)*z
        w=np.full(nz+1,rng.uniform(-2,8))
        f=np.array([u,v,t/pi,t,w,np.full(nz+1,dz),p,pi,p/(287*t*(1+.608*q)),q],np.float32)
        s=np.zeros((7,nz+1),np.float32)
        dx=rng.choice([10000,100000,500000])
        surface=np.array([dx,dx*dx*rng.uniform(.7,1.3),0,0,-100,1,nz+1],np.float32)
        yield ("diverse",i),(f,s,surface),np.array([5,1,0,0,1,1,1],np.int32)

    yield from closure_cases()
    # Layer-local density changes reach a capped initial mass flux and
    # the tenth deep-closure iteration and native mass-budget diagnostics.
    # Strict and bounds-checked builds define every selected fixture.
    for lapse,rh,low,width,factor in [(12,.5,500,2000,.01),
            (6,.95,2000,2000,10),(6,.95,4000,6000,.0001),
            (6,.95,6000,250,1e7)]:
        f,s,v=fixture(55,lapse,rh,300,2)
        z=(np.arange(56)+.5)*250
        f[8,(z>=low)&(z<low+width)]*=np.float32(factor)
        yield ('density-layer',lapse,rh,low,width,factor),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    # Elevated inflow also exercises diagnostic masking below the source layer.
    f,s,v=fixture(55,6,.95,300,2)
    z=(np.arange(56)+.5)*250
    f[8,(z>=6000)&(z<6250)]*=np.float32(1e7)
    f[9,z<1000]*=np.float32(.01)
    yield ('elevated-mass-budget',),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    # Bounds-defined native cycle at NCOUNT=10 exhausts the DO loop
    # (MPAS module_cu_kfeta.F:1675,1924-1928,1995). Input words are retained
    # from diagnostic search seed 8422207; no oracle outputs are stored.
    edge = np.load(Path(__file__).with_name("closure_tenth_cycle.npz"))
    yield ('tenth-cycle', float(edge['params'][0])), tuple(
        edge[name].copy() for name in ('fields','state','surface')), edge['flags'].copy()
    # Module-argument stress, outside the driver's fixed Earth gravity:
    # low G with unchanged DP reaches weak-CAPE/deep-cloud fallback and the
    # adjusted LCL search above the model top. These are not weather cases.
    for gravity in (.1,.001):
        f,s,v=fixture(55,6,1.1,300,2)
        f[8]*=PARAMS[9]/np.float32(gravity)
        yield ('parameter-gravity',gravity),(f,s,v),np.array([5,1,0,0,1,1,1],np.int32)
    # The real capture carries a noncanonical native warm_rain word. The
    # oracle transfers it verbatim into LOGICAL. Preserve the cold transport
    # fixture and add a phase-sensitive column: canonical warm_rain changes
    # seven tendency words, so matching raw values tests native semantics.
    for word in (0,1,2,3,32764,32765,-1):
        yield ('logical-word',word),fixture(55,6,.95,270,.5),np.array([5,1,0,word,1,0,1],np.int32)
        yield ('logical-phase-word',word),fixture(55,6,.95,285,.5),np.array([5,1,0,word,1,0,1],np.int32)

def closure_cases():
    """Seven bounds-defined extrema from a deterministic closure search.

    Pristine native comparison, not the diagnostic instrumented build, is
    the authority. Index 3848 reaches the native moisture-budget error.
    """
    rng=np.random.default_rng(84219)
    for i in range(11258):
        nz=int(rng.choice([20,30,55,80]));depth=rng.uniform(6000,18000)
        dz=np.full(nz+1,depth/nz);dz[:min(10,nz)]=rng.uniform(20,800)
        z=np.cumsum(dz)-.5*dz
        ts=rng.uniform(270,312);a=rng.uniform(2,14);b=rng.uniform(2,14);knee=rng.uniform(500,4000)
        t=np.maximum(ts-a*np.minimum(z,knee)/1000-b*np.maximum(z-knee,0)/1000,190)
        if i%3==0:t+=np.where(z>rng.uniform(500,3500),rng.uniform(0,15),0)
        p=rng.uniform(70000,105000)*np.exp(-9.81*z/(287*np.maximum(ts-a*z/2000,230)))
        p=np.maximum(p,5500);es=611.2*np.exp(17.67*(t-273.15)/(t-29.65))
        rh=np.full(nz+1,rng.uniform(.02,.999));rh[z>knee]*=rng.uniform(.1,1.2)
        q=np.clip(rh*.622*es/(p-es),1e-9,.08);pi=(p/1e5)**(287/1004.5)
        u=rng.uniform(-20,20)+rng.uniform(-.01,.01)*z;v=rng.uniform(-20,20)+rng.uniform(-.01,.01)*z
        f=np.array([u,v,t/pi,t,np.full(nz+1,rng.uniform(-1,10)),dz,p,pi,p/(287*t*(1+.608*q)),q],np.float32)
        s=np.zeros((7,nz+1),np.float32);dx=10**rng.uniform(3,6)
        surf=np.array([dx,dx*dx,0,0,-100,1,nz+1],np.float32)
        params=PARAMS.copy();params[0]=rng.choice([1,30,60,300,900,1800,3000])
        flags=np.array([5,1,0,0,1,1,1],np.int32)
        if i in {0,560,2993,3848,5027,8211,11257}:
            yield ("closure",i,float(params[0])),(f,s,surf),flags

def native_column(native, fields, state, surface, flags, params=PARAMS):
    state, surface = state.copy(), surface.copy()
    active = C.c_int(0)
    native.kf_native_column(fields.shape[1]-1, fields, state, surface, C.byref(active), params, flags)
    return state, surface, active.value


def word_diff(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    return int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
