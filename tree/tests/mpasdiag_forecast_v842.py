"""Real forecast door with process-local cloud/PV/sounding output callbacks.

The callbacks carry existing device state and held dycore tendencies into the
native pools. They never replace a physics slot or edit the driver seam.
Pristine native vector initialization supplies the framework geometry; all
three diagnostic modules run their CUDA port, including the PV budget.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    from hexcore.cli import main as door
    from hexcore.mpasphys import runtime_v842 as runtime
    from hexcore.mpasdiag.cloud_v842 import NativeCloudPVSoundingsOutput
    from hexcore.mpasdiag.pv_v842 import PV_FIELDS, BUDGET_FIELDS
    from hexcore.mpasdiag.cloud_v842 import FIELDS
    bindings, meshes, services, held = {}, {}, {}, {}
    configure, write = runtime._configure_streams, runtime.write_native_output
    synchronize = runtime.ResidentNativeCoupling.synchronize
    load = runtime.load_native_dynamics_inputs
    profile = sys.getprofile()

    def observe_euler(frame, event, arg):
        # mpas_atm_time_integration.F:6329-6347,6585-6588,6725-6728,6885-6891.
        # Observe the existing driver return, retaining its actual RK1 buffers.
        if event == 'return' and frame.f_code.co_name == '_advance_dynamics_subcycle_v841':
            driver = frame.f_locals['self']
            held[id(driver)] = {name: frame.f_locals[local] for name, local in
                (('u_euler', 'saved_euler_ru'), ('w_euler', 'saved_euler_rw'),
                 ('theta_euler', 'saved_euler_theta'))}
        if profile is not None:
            profile(frame, event, arg)

    def load_mesh(arguments, layout, payload):
        result = load(arguments, layout, payload)
        meshes[id(layout)] = result[0]
        return result

    def configure_diagnostics(native, streams, out, layout):
        missing = configure(native, streams, out, layout)
        bindings[native] = (streams, out, layout)
        produced = set(FIELDS + PV_FIELDS + BUDGET_FIELDS)
        # Requested observations expose the actual forecast carriers through
        # their Registry fields. They do not alter the diagnostic computation.
        produced.update(os.environ.get('HEX_MPASDIAG_CARRIER_FIELDS', '').split())
        for stream in streams:
            for name in tuple(missing[stream.name]):
                if name in produced:
                    native.stream_field(stream.name, name)
                    missing[stream.name].remove(name)
        return missing

    def bind_current_dynamics(coupling, atmosphere, *, level=1):
        result = synchronize(coupling, atmosphere, level=level)
        native = coupling.session.native
        if native not in services:
            streams, out, layout = bindings[native]
            services[native] = NativeCloudPVSoundingsOutput(
                coupling.session, streams, out, coupling.cp)
            initialize_geometry(native, meshes[id(layout)])
        services[native].dycore = coupling.dycore
        return result

    def initialize_geometry(native, mesh):
        # mpas_vector_operations.F:652-775; mpas_vector_reconstruction.F:61-191.
        # This is a framework native-cpu stand-in, authenticated against f3 and
        # the source tree, rather than Python geometry arithmetic.
        library = Path(os.environ['HEX_MPASDIAG_NATIVE_OPERATORS'])
        pin = json.loads(library.with_suffix('.json').read_text())
        if hashlib.sha256(library.read_bytes()).hexdigest() != pin['library_sha256']:
            raise ValueError('native operator library digest changed')
        if hashlib.sha256(Path(native.path).read_bytes()).hexdigest() != pin['provider_sha256']:
            raise ValueError('native operator framework provider changed')
        source = Path(os.environ['HEX_MPASDIAG_SOURCE'])
        for name in ('mpas_vector_operations.F', 'mpas_vector_reconstruction.F', 'mpas_rbf_interpolation.F'):
            relative = 'src/operators/' + name
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != pin['sources'][relative]:
                raise ValueError('native operator source digest changed: ' + relative)
        attrs = mesh.attrs
        for key in ('on_a_sphere', 'is_periodic'):
            native.config(key, str(attrs.get(key, 'NO')).strip().upper() == 'YES', pool='mesh')
        for key in ('x_period', 'y_period'):
            native.config(key, float(attrs.get(key, 0.0)), pool='mesh')
        operators = ct.CDLL(str(library.resolve()))
        initialize = operators.__mpas_vector_operations_MOD_mpas_initialize_vectors
        initialize.argtypes, initialize.restype = [ct.c_void_p], None
        initialize(native.pool('mesh'))
        reconstruct = operators.__mpas_vector_reconstruction_MOD_mpas_init_reconstruct
        reconstruct.argtypes, reconstruct.restype = [ct.c_void_p, ct.POINTER(ct.c_int)], None
        reconstruct(native.pool('mesh'), ct.byref(ct.c_int(1)))
        # mpas_block_decomp.F:633-685. Global pools retain stream order;
        # friction's vertex reduction must visit edges in native block order.
        services[native].pv.set_global_edge_order()
        # Keep the native shared object alive for the complete forecast.
        services[native].operators = operators
        services[native].operator_receipt = {key: pin[key] for key in
            ('library_sha256', 'provider_sha256', 'flags', 'sources')}

    def publish_output_inputs(service, step):
        # mpas_atm_time_integration.F:7159-7179; mpas_vector_reconstruction.F:270-302.
        dycore, cp, session = service.dycore, service.cp, service.session
        atmosphere, dims = dycore.atmosphere, service.layout.dimensions
        solved = dycore.horizontal.solve_diagnostics(atmosphere.state.rho,
            atmosphere.state.rho_u, dt=dycore.config.config_dt,
            apvm_upwinding=dycore.config.config_apvm_upwinding,
            normal_velocity=atmosphere.saved.normal_velocity, rk_step=3)
        session.put('diag', 'vorticity', solved.vorticity)
        # mpas_atm_time_integration.F:7159-7179. The PV friction budget reads
        # the current solve's edge density, including the initial output;
        # retaining only the RK source density leaves zero/stale denominators.
        session.put('diag', 'rho_edge', solved.h_edge)
        mesh = service.pools['mesh']
        ns, nz, me = (cp.int32(dims[name]) for name in ('nCellsSolve', 'nVertLevels', 'maxEdges'))
        xyz = tuple(cp.asarray(service.native.get_field('diag', 'uReconstruct'+axis), order='F')
                    for axis in ('X', 'Y', 'Z'))
        velocity = cp.asarray(atmosphere.saved.normal_velocity, order='F')
        service.pv._kernel('reconstruct', int(ns*nz),
            (ns, nz, me, mesh['nEdgesOnCell'], mesh['edgesOnCell'],
             mesh['coeffs_reconstruct'], velocity, *xyz))
        for axis, value in zip(('X', 'Y', 'Z'), xyz):
            session.put('diag', 'uReconstruct'+axis, value)
        if step:
            tendencies = held[id(dycore)]
            for name, value in tendencies.items():
                if value is not None:
                    session.put('tend', name, value)
        # Initial output uses pristine hydrated native Euler storage, exactly
        # before any dynamics call. No manufactured per-scheme tendencies.

    def write_diagnostics(native, streams, *, step, receipt):
        service = services[native]
        publish_output_inputs(service, step)
        service.compute(step)
        receipt['cloud_pv_soundings_calls'] = service.calls
        receipt['diagnostics_backend'] = 'hex-cuda'
        receipt['diagnostics_native_operators'] = service.operator_receipt
        receipt['diagnostics_euler_source'] = 'resident dycore RK1 buffers at subcycle return'
        return write(native, streams, step=step, receipt=receipt)

    runtime._configure_streams, runtime.write_native_output = configure_diagnostics, write_diagnostics
    runtime.ResidentNativeCoupling.synchronize = bind_current_dynamics
    runtime.load_native_dynamics_inputs = load_mesh
    sys.setprofile(observe_euler)
    try:
        return door(sys.argv[1:])
    finally:
        sys.setprofile(profile)
        for service in services.values():
            service.cleanup()
        runtime._configure_streams, runtime.write_native_output = configure, write
        runtime.ResidentNativeCoupling.synchronize = synchronize
        runtime.load_native_dynamics_inputs = load


if __name__ == '__main__':
    raise SystemExit(main())
