#!/usr/bin/env python3
"""Conservation/positivity regression at full global120 column count.

This checks the chemistry operator. It is separate from whole-forecast proof.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
import numpy as np
import cupy as cp
from hexcore.chem_transport import configure, _CONFIGS, ChemistryTransport

class Cache:
    def __init__(self):
        self.modules = {}
    def raw_kernel(self, name, source, *, module_key):
        if module_key not in self.modules:
            self.modules[module_key] = cp.RawModule(code=source, options=('--std=c++17',))
        return self.modules[module_key].get_function(name)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
rows = [dict(name=n, dry_deposition_m_s=vd, settling_m_s=vs, mass_extinction_m2_kg=mee)
        for n,vd,vs,mee in [('smoke',0.001,0.0001,4500.),('dust_1',0.002,0.02,800.),
                            ('sea_salt_1',0.003,0.04,500.)]]
config = out/'operator-config.json'
config.write_text(json.dumps({'schema':'gpuwm-hex.chemistry/v1','species':rows}))
row = configure(config)
nz,nc,ns = 55,40962,len(rows)
r = ChemistryTransport(_CONFIGS[row.name], cache=Cache(), area=cp.ones(nc), nlev=nz,ncells=nc)
rho = cp.broadcast_to(cp.exp(-cp.arange(nz,dtype=cp.float32)[:,None]/20),(nz,nc)).copy()
z = cp.broadcast_to((cp.arange(nz+1,dtype=cp.float32)*100)[:,None],(nz+1,nc)).copy()
kh = cp.full((nz,nc),1000.,cp.float32)
q = cp.zeros((6+ns,nz,nc),cp.float32)
q[6:,20,:] = cp.arange(1,ns+1,dtype=cp.float32)[:,None]*1.e-6
state = SimpleNamespace(scalars=q,time_seconds=0.)
atmos = SimpleNamespace(state=state)
source_calls = []
r.boundary = lambda atmosphere, transport, elapsed, dt: source_calls.append((elapsed, dt))
weight = rho*(z[1:]-z[:-1])
mass = lambda: cp.sum(q[6:].astype(cp.float64)*weight,axis=1)
before = mass()
for i,s in enumerate(rows): r.surface_flux[s['name']].fill((i+1)*1.e-8)
dt,steps = 600.,24
for step in range(steps):
    state.time_seconds=(step+1)*dt
    r.step(atmos,rho=rho,z=z,kh=kh,dt=dt)
    r.commit()
assert source_calls == [(step*dt,dt) for step in range(steps)]
expected = before + cp.arange(1,ns+1,dtype=cp.float64)[:,None]*1.e-8*dt*steps
balance = mass()+r.dry+r.settled
error = float(cp.max(cp.abs(balance-expected)/expected))
assert error < 3.e-6, error
assert bool(cp.all(q[6:]>=0))
assert bool(cp.all(r.dry>0)) and bool(cp.all(r.settled>0))
products=r.products(atmos,rho,z)
assert bool(cp.allclose(products['aod550'],sum(products[s['name']+'_aod550'] for s in rows),rtol=1.e-6))
assert bool(cp.allclose(products['smoke_column_mass'],mass()[0],rtol=1.e-6))
# A diffusively mixed column spreads beyond its original injection level.
assert bool(cp.all(q[6:,10,:]>0)) and bool(cp.all(q[6:,30,:]>0))
# No-sink closed diffusion preserves an arbitrary constant mixing ratio.
r.vd.fill(0);r.vs.fill(0)
for v in r.surface_flux.values(): v.fill(0)
q[6:]=cp.arange(1,ns+1,dtype=cp.float32)[:,None,None]*1.e-7
initial=q[6:].copy()
r.step(atmos,rho=rho,z=z,kh=kh,dt=dt)
assert bool(cp.allclose(q[6:],initial,rtol=1.e-7,atol=0.))
# Refuse bad geometry rather than hiding it with mass clipping.
z[3,0]=z[2,0]
try: r.step(atmos,rho=rho,z=z,kh=kh,dt=dt)
except FloatingPointError: pass
else: raise AssertionError('nonpositive thickness accepted')
cp.cuda.Device().synchronize()
record={'operator_regression':'PASS','shape':[ns,nz,nc],'steps':steps,'dt_s':dt,
        'max_relative_budget_error':error,'wall_s':time.perf_counter()-start,
        'cupy_pool_peak_allocated_bytes':cp.get_default_memory_pool().total_bytes(),
        'device_name':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'scope':'chemistry operator regression; no whole forecast claim'}
(out/'operator-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
