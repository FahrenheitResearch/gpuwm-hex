"""Exercise the MM5 storage binder against initialized f3 pools, without GPU.

This checks storage/compiled-constant ABI only. It is neither column identity
nor forecast proof. Native input decoding belongs to the f4 Rust provider.
"""
import argparse, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hexcore.mpasphys.driver_v842 import resolve_namelist, interval_seconds
from hexcore.mpasphys.forecast_v842 import read_streams, bind_stream_inputs
from hexcore.mpasphys.runtime_v842 import _configure_streams
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import decode_registry_pools
from hexcore.mpasphys.session_v842 import NativePhysicsSession, native_working_directory
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.mpasphys import sfclay_mm5_v842 as mm5

p = argparse.ArgumentParser(description=__doc__)
for name in ('library', 'reader', 'source', 'namelist', 'init', 'tables', 'work', 'receipt'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
a.work.mkdir(parents=True, exist_ok=False)
text = a.namelist.read_text()
config = resolve_namelist(text)
layout = decode_registry_pools(a.reader, a.source, config, text, [a.init], a.work/'pools')
streams = read_streams(a.namelist.with_name('streams.atmosphere'), layout, a.tables)
inputs = {'input':a.init}
oro = list(a.init.parent.glob('*.ugwp_oro_data.nc'))
if oro:
    inputs['ugwp_oro_data_in'] = oro[0]
streams = bind_stream_inputs(streams, {n:v for n,v in inputs.items()
    if n in {s.name for s in streams if 'input' in s.directions}})
stop = (datetime.strptime(layout.configs['config_start_time'], '%Y-%m-%d_%H:%M:%S') +
        timedelta(seconds=float(interval_seconds(layout.configs['config_run_duration'])))).strftime('%Y-%m-%d_%H:%M:%S')
for root in (a.source/'src/core_atmosphere/physics/physics_wrf/files', a.tables):
    for table in root.iterdir():
        if table.is_file() and (table.suffix in ('.TBL', '.DBL') or table.name.startswith(('RRTMG_', 'CCN_'))):
            target = a.work/table.name
            if not target.exists():
                target.symlink_to(table.resolve())
with NativePhysics(a.library, test_only=True) as native:
    session = NativePhysicsSession(native, layout, a.work/'pools', config,
        dt=layout.configs['config_dt'], start=layout.configs['config_start_time'],
        stop=stop, stream_intervals={s.name:(s.input_interval if s.name == 'surface'
            else s.output_interval) for s in streams},
        post_transport=lambda call: (_ for _ in ()).throw(RuntimeError('storage probe only')))
    call = SimpleNamespace(slot='surface_layer', options=config.execution_options,
                           scheme=config.selectors['surface_layer'], itimestep=1,
                           time_seconds=0.0,
                           dt=layout.configs['config_dt'])
    missing_fields = _configure_streams(native, streams, a.work, layout)
    with native_working_directory(a.work):
        session.initialize()
        for operation in ('allocate_forall_physics', 'MPAS_to_physics', 'allocate_sfclayer'):
            session.hooks.invoke(operation, call)
        provider = NativeDevicePoolViews(session, np)
        arguments = provider.begin(call, {'module':'sfclay_mm5_v842', 'bind':'bind_native_pools'})
        # Access every buffer the callback will consume, including diagnostic outputs.
        arrays = {name: value for name,value in arguments['atmosphere'].items()}
        for name in mm5.OUTPUTS:
            arrays['diag_physics.'+name] = arguments['diag_physics'][mm5._MPAS_NAMES.get(name,name)]
        for name in ('hpbl', 'mavail'):
            arrays['diag_physics.'+name] = arguments['diag_physics'][name]
        for name in ('skintemp', 'xland', 'sst', 'xice'):
            arrays['sfc_input.'+name] = arguments['sfc_input'][name]
        arrays['meshDensity'] = arguments['mesh_density']
        count = layout.dimensions['nCellsSolve']
        assert all(v.shape == (count,) and v.dtype == np.float32 and v.flags.c_contiguous
                   for v in arrays.values())
        assert arguments['constants'].shape == (13,)
        result = dict(status='PASS', scope='actual native initialized storage ABI only; no forecast or CUDA',
            scheme=call.scheme, process_count=1, mpi_ranks=None, nCellsSolve=count,
            library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
            reader_sha256=hashlib.sha256(a.reader.read_bytes()).hexdigest(),
            namelist_sha256=hashlib.sha256(a.namelist.read_bytes()).hexdigest(),
            fields={n:dict(shape=list(v.shape),dtype=str(v.dtype)) for n,v in arrays.items()},
            missing_stream_fields=missing_fields,
            constant_words=arguments['constants'].view(np.uint32).tolist(),
            config_len_disp_word=int(arguments['config_len_disp'].view(np.uint32)),
            xice_threshold_word=int(arguments['xice_threshold'].view(np.uint32)),
            excluded_words=[])
        for operation in ('deallocate_sfclayer', 'deallocate_forall_physics'):
            session.hooks.invoke(operation, call)
a.receipt.write_text(json.dumps(result, indent=2)+'\n')
print('PASS actual native MM5 storage ABI', call.scheme)
