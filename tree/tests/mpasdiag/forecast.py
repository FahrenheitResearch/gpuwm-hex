"""Run the real forecast door with explicit isobaric output callbacks.

The temporary bindings affect only this process, never the integrator seam.
All other active physics slots retain the deck's native-cpu selection.
"""
import sys
import os


def main():
    from hexcore.mpasphys import runtime_v842 as runtime
    from hexcore.mpasdiag.native_output import NativeIsobaricOutput
    from hexcore.mpasdiag import native_output
    from hexcore.cli import main as door
    services = {}
    configure, write = runtime._configure_streams, runtime.write_native_output
    synchronize = runtime.ResidentNativeCoupling.synchronize
    output_helper = native_output.compute_output_diagnostics

    def observe_output_helper(mesh, state, diag):
        observers = [service.output_boundary for service in services.values()
                     if hasattr(service, 'output_boundary') and service.pools['mesh'] is mesh]
        for observer in observers:
            observer.observe('pre')
        result = output_helper(mesh, state, diag)
        for observer in observers:
            observer.observe('post')
        return result

    def bind_current_dynamics(coupling, atmosphere, *, level=1):
        result = synchronize(coupling, atmosphere, level=level)
        service = services.get(coupling.session.native)
        if service is not None:
            service.dycore = coupling.dycore
        return result

    def configure_isobaric(native, streams, out, layout):
        import cupy as cp
        missing = configure(native, streams, out, layout)
        service = NativeIsobaricOutput(native, layout, streams, cp)
        library = os.environ.get('HEX_MPASDIAG_NATIVE_COMPANIONS')
        if library:
            from native_companions import NativeCompanions
            companions = NativeCompanions(native, library)
            service.attach(missing, companions=companions.adapters(),
                           companion_fields=companions.receipt['fields'])
        else:
            service.attach(missing)
        referee = os.environ.get('HEX_MPASDIAG_BOUND_REFEREE')
        if referee:
            from bound_referee import BoundIsobaricReferee
            service.bound_referee = BoundIsobaricReferee(service, referee)
        boundary = os.environ.get('HEX_MPASDIAG_OUTPUT_BOUNDARY')
        if boundary:
            from output_boundary import OutputBoundaryObserver
            service.output_boundary = OutputBoundaryObserver(service, boundary,
                os.environ['HEX_MPASDIAG_OUTPUT_BOUNDARY_LIBRARY'])
        services[native] = service
        return missing

    def write_isobaric(native, streams, *, step, receipt):
        services[native].compute(step)
        receipt['isobaric_calls'] = services[native].calls
        receipt['diagnostics_backend'] = 'hex-cuda'
        receipt['diagnostics_manager_complete'] = services[native].manager is not None
        if hasattr(services[native], 'bound_referee'):
            receipt['isobaric_same_input_referee'] = services[native].bound_referee.receipt()
        if hasattr(services[native], 'output_boundary'):
            receipt['output_boundary_comparison'] = services[native].output_boundary.receipt()
        # Native resets accumulated diagnostics before clearing stream alarms.
        # Keep those alarms ringing through manager.reset after the real writer.
        reset, pending = native.reset_alarm, []
        native.reset_alarm = pending.append
        try:
            result = write(native, streams, step=step, receipt=receipt)
            services[native].reset()
        finally:
            native.reset_alarm = reset
        for alarm in pending:
            reset(alarm)
        return result

    runtime._configure_streams, runtime.write_native_output = configure_isobaric, write_isobaric
    runtime.ResidentNativeCoupling.synchronize = bind_current_dynamics
    native_output.compute_output_diagnostics = observe_output_helper
    try:
        return door(sys.argv[1:])
    finally:
        for service in services.values():
            service.cleanup()
        runtime._configure_streams, runtime.write_native_output = configure, write
        runtime.ResidentNativeCoupling.synchronize = synchronize
        native_output.compute_output_diagnostics = output_helper


if __name__ == '__main__':
    raise SystemExit(main())
