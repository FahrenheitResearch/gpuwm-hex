"""Box-only initial/evolved dry donor shadow and exact native identity harness.

The dry run is an admission pilot, not a reference-physics timing or forecast
skill experiment. It writes compact hashes/receipts, never full model fields.
"""
import argparse
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]


def digest_arrays(obj):
    result = {}
    from dataclasses import fields
    for field in fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value,np.ndarray) or hasattr(value,'__cuda_array_interface__'):
            if isinstance(value,np.ndarray):
                array=value
            else:
                import cupy as cp
                array=cp.asnumpy(value)
            result[field.name] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('grid','static','init','out','cache'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--dt', type=float, default=18)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--max-rate', type=int, choices=(1,2,4,8,16), default=16)
    p.add_argument('--arm', choices=('native','off','shadow','uniform'), default='shadow')
    p.add_argument('--plot-rates',type=Path,default=None,help='computational mesh-policy analysis, not a weather map')
    p.add_argument('--hash-every',type=int,default=1,help='retain array hashes only at this step interval')
    p.add_argument('--stop-on-infeasible',action='store_true')
    a = p.parse_args()
    if a.steps < 0 or a.hash_every < 1:
        p.error('steps must be nonnegative and hash interval positive')
    from run_lts_dry_gates import prepare_host, build_config, dry_mass, qv_mass
    from hexcore.cuda_backend import require_cuda
    from hexcore.cuda_backend.runtime import KernelCache
    from hexcore.cuda_driver import CudaDryDycoreDriver
    config = replace(build_config('off', dt=a.dt, rates=(1,3), rings=2),
                     config_time_integration_order=2, config_number_of_sub_steps=2,
                     config_monotonic=True)
    config.validate()
    paths = {name:getattr(a,name).resolve(strict=True) for name in ('grid','static','init')}
    a.out.mkdir(parents=True,exist_ok=False)
    # Use the normal public mesh binder. The old dry gate alone assumes the
    # frozen x4 topology and cannot bind a newly initialized variable mesh.
    from hexcore import mesh_rows
    from hexcore.drivers import mpas_mesh_binding as binding
    import run_cuda_v841_forecast as door
    from netCDF4 import Dataset
    with Dataset(paths['static']) as dataset:
        nominal = float(dataset['nominalMinDc'][:].item())
    spec = a.out/'input-provenance.json'
    spec.write_text(json.dumps(dict(grid=str(paths['grid']),static=str(paths['static']),
                                   init=str(paths['init']),init_sha256=mesh_rows.sha256_file(paths['init']),
                                   scope='dry shadow pilot, original mesh coordinates')))
    row = replace(mesh_rows.describe_generated(name='fullstep-shadow-pilot',grid=paths['grid'],static=paths['static'],
        spec_path=spec,generator_receipt=spec,static_receipt=spec,point_deg=(0,0),fine_dx_m=nominal,
        core_radius_km=0,background_km=nominal/1000,dt_seconds=a.dt,n_levels=55,
        admission={'scope':'dry admission instrument, no forecast qualification'}),kind=mesh_rows.ARCHIVED_GLOBAL_KIND)
    rowfile = mesh_rows.write_rows(a.out/'mesh-row.json',[row])
    binding.MESH_BINDINGS = mesh_rows.apply_rows(binding.MESH_BINDINGS,binding.MeshBinding,rowfile)
    binding_receipt = binding.bind_mesh(door.proof,row.name,grid=paths['grid'],static=paths['static'],forecast=door,
        convection='off',pbl_cadence='auto',time_integration_order=2,acoustic_substeps=2,dt_seconds=a.dt)
    (a.out/'binding.json').write_text(json.dumps(binding_receipt,indent=2))
    host = prepare_host(paths,config,scalars='wsm6')
    capability = require_cuda(min_compute=(12,0))
    cache = KernelCache(capability=capability, cache_dir=str(a.cache))
    driver = CudaDryDycoreDriver.from_host(host['mesh'],host['state'],host['vertical'],host['reference'],config,
        saved_diagnostics=host['saved'],terrain_metrics=host['terrain_metrics'],kernel_cache=cache,
        reference_wind_profiles=host['profiles'])
    import cupy as cp
    initial_mass = dry_mass(np.asarray(host['mesh'].area_cell),host['vertical'].dzw,host['state'].rho)
    initial_qv = qv_mass(np.asarray(host['mesh'].area_cell),host['vertical'].dzw,host['state'].rho,host['state'].scalars[0])
    shadow = None
    with (a.out/'stages.jsonl').open('w') as stages:
        if a.arm == 'shadow':
            from hexcore.cuda_fullstep_lts_shadow import FullStepShadow
            def sink(row):
                stages.write(json.dumps(row)+'\n'); stages.flush()
            with Dataset(paths['grid']) as dataset:
                global_cell_ids=np.asarray(dataset['indexToCellID'][:],dtype=np.int64)
            shadow = FullStepShadow(driver,dt=a.dt,max_rate=a.max_rate,global_cell_ids=global_cell_ids,sink=sink)
            state = driver.atmosphere.state
            shadow.observe(state.rho,state.rho_u,state.rho_w,driver.atmosphere.vertical.rdzw,
                           stage=0,stage_dt=a.dt,time_seconds=state.time_seconds)
        elif a.arm in ('off','uniform'):
            # The actual public dispatcher is exercised, including its lazy
            # disabled path and rate-one shortcut; native bypasses it entirely.
            from hexcore.fullstep_lts_dispatch import attach_fullstep_lts
            from hexcore.config_fullstep_lts import FullStepLtsConfig
            enabled = a.arm == 'uniform'
            shadow = attach_fullstep_lts(driver, FullStepLtsConfig(enabled, fullstep_lts_max_rate=a.max_rate)) if enabled else attach_fullstep_lts(driver,FullStepLtsConfig())
            assert shadow is None
        times, hashes = [], []
        for step in range(a.steps):
            cp.cuda.get_current_stream().synchronize()
            began = time.perf_counter()
            result = driver.step_device()
            driver.atmosphere = result.atmosphere
            cp.cuda.get_current_stream().synchronize()
            times.append(time.perf_counter()-began)
            if (step+1)%a.hash_every==0 or step+1==a.steps:
                hashes.append(dict(step=step+1,state=digest_arrays(driver.atmosphere.state),
                                   saved=digest_arrays(driver.atmosphere.saved)))
            if a.stop_on_infeasible and shadow is not None and shadow.last['base_infeasible_cells']:
                break
        final = driver.atmosphere.state.to_host()
        final_mass = dry_mass(np.asarray(host['mesh'].area_cell),host['vertical'].dzw,final.rho)
        final_qv = qv_mass(np.asarray(host['mesh'].area_cell),host['vertical'].dzw,final.rho,final.scalars[0])
        receipt = dict(schema='gpuwm-hex.fullstep-lts-shadow-pilot/v1',arm=a.arm,dt=a.dt,steps=len(times),steps_requested=a.steps,
            numerical_configuration=asdict(config),fields_per_step=hashes,step_seconds=times,
            dry_mass_relative_drift=abs(final_mass/initial_mass-1),qv_mass_relative_drift=abs(final_qv/initial_qv-1),
            initial_dry_mass=initial_mass, final_dry_mass=final_mass,
            shadow=None if shadow is None else shadow.receipt(),
            identity_scope='dry prognostic and saved arrays, no physics sidecars or history file',
            fullstep_modules_loaded=sorted(k for k in sys.modules if 'fullstep_lts' in k),
            physics='off', qualified=False)
        (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if shadow is not None and a.plot_rates is not None:
            from hexcore.fullstep_lts import transport_ceiling,buffer_rates
            from plot_fullstep_lts_rates import plot_rates
            rates,infeasible=transport_ceiling(shadow.raw_geometric,cp.asnumpy(shadow.worst),a.dt,shadow.target)
            plot_rates(paths['grid'],buffer_rates(rates,shadow.coe),infeasible,a.plot_rates)
        print(json.dumps(receipt),flush=True)
        if shadow is not None:
            shadow.detach()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
