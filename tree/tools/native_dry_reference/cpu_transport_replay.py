#!/usr/bin/env python3
"""CPU replay of one native scalar step from native's own averaged fluxes.

Needs no GPU. A native one-step run that also writes ruAvg and wwAvg
(EXTRA_STREAM_VARS="ruAvg wwAvg") gives the exact mass fluxes native used for
its three scalar stages. The NumPy authority (hexcore.transport, v8.4.1 source
lane) replays those stages from the time-zero scalar and the result is compared
word for word with native's scalar after the step. This isolates the transport
arithmetic from the dynamics.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--init', type=Path, required=True)
p.add_argument('--native', type=Path, required=True)
p.add_argument('--scalar', default='qv')
p.add_argument('--dt', type=float, default=360.0)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cells', type=int, default=0, help='compare only the first N cells worth of output (0 = all)')
p.add_argument('--hex-tree', type=Path, default=None, help='replay with another checkout (before/after)')
a = p.parse_args()
tree = (a.hex_tree or Path(__file__).resolve().parents[2]).resolve()
sys.path[:0] = [str(tree / 'src'), str(Path(__file__).resolve().parent)]
from compare_dry_identity import field_report
from hexcore.config_v841 import V841DryDycoreConfig
from hexcore.driver import DryDycoreDriver, load_mpas_initial_state, load_mpas_vertical_grid
from hexcore.dynamics_v841 import load_v841_reference_wind_profiles, precomputed_mesh_inverse_v841
from hexcore.mesh import Mesh
from hexcore.transport import advance_scalar_transport

files = sorted(a.native.glob('tracer.*.nc'))
mesh = Mesh.from_netcdf(a.init)
native = load_mpas_vertical_grid(a.init, mesh, config_coef_3rd_order=.25)
state, reference, saved = load_mpas_initial_state(files[0], mesh, native.vertical_grid, scalar_names=(a.scalar,),
    reference_path=files[0], terrain_metrics=native.terrain_metrics, return_saved_diagnostics=True)
profiles = load_v841_reference_wind_profiles(a.init, n_vert_levels=native.vertical_grid.n_vert_levels)
config = V841DryDycoreConfig(config_dt=a.dt, config_time_integration_order=2, config_number_of_sub_steps=4,
    config_dynamics_split_steps=3, config_split_dynamics_transport=True, config_scalar_advection=True,
    config_monotonic=True, config_positive_definite=False, config_apvm_upwinding=0.5)
cpu = DryDycoreDriver(mesh, native.vertical_grid, reference, config, terrain_metrics=native.terrain_metrics,
                      reference_wind_profiles=profiles)

def arr(f, name):
    with Dataset(f) as d:
        d.set_auto_mask(False)
        return np.ascontiguousarray(np.asarray(d[name][0]).T)

q0 = arr(files[0], a.scalar)[None]; target = arr(files[1], a.scalar)[None]
rho0 = arr(files[0], 'rho_zz'); rho1 = arr(files[1], 'rho_zz')
hu = arr(files[1], 'ruAvg'); hw = arr(files[1], 'wwAvg')
inv_area = precomputed_mesh_inverse_v841(mesh, 'areaCell', np.dtype(np.float32))
q = q0.copy(); t = time.time()
for stage, dt in enumerate((a.dt / 2, a.dt / 2, a.dt), 1):
    q = advance_scalar_transport(mesh, q0, q, rho0, rho1, hu, hw, dt, rk_step=stage,
        config_time_integration_order=2, config_monotonic=True, config_positive_definite=False,
        config_split_dynamics_transport=True, coefficients=cpu.advection_coefficients,
        fzm=native.vertical_grid.fzm, fzp=native.vertical_grid.fzp, rdzw=native.vertical_grid.rdzw,
        inv_area_cell=inv_area).scalars
    print('stage', stage, round(time.time() - t), 's', flush=True)
q = np.asarray(q, np.float32)
# atm_srk3 F:2799-2800
q[q < 0] = 0
report = field_report(q, target)
report['changed_by_native_step'] = int(np.count_nonzero(q0 != target))
a.out.write_text(json.dumps({'schema': 'gpuwm-hex.cpu-transport-replay/v1', 'native': str(a.native),
    'scalar': a.scalar, 'hex_tree': str(tree), 'report': report}, indent=1) + '\n')
print(json.dumps({k: report.get(k) for k in ('identical', 'words', 'different_words', 'max_abs', 'max_ulp', 'changed_by_native_step')}))
