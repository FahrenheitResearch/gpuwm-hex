"""Real CLI forecast with the lane's CUDA diagnostics bound to native streams.

This process-local test launcher leaves the protected production seam intact.
It measures the complete forecast before the integrator lands that seam; it
does not make a production-hook or landing-acceptance claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def main(argv=None):
    from hexcore.cli import main as door
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.mpasdiag.convective_v842 import FIELDS, MAXIMA, NativeConvectiveDiagnostics
    from hexcore.mpasphys import runtime_v842 as runtime

    configure = runtime._configure_streams
    synchronize = runtime.ResidentNativeCoupling.synchronize
    write = runtime.write_native_output
    previous_profile = sys.getprofile()
    step_code = CudaDryDycoreDriver._step_device_v841.__code__
    bindings, couplings, services, primary = {}, {}, {}, {}

    def observe_primary(frame, event, result):
        """mpas_atm_time_integration.F:2556-2560,7159-7179; retain final RK words."""
        if event == 'return' and frame.f_code is step_code and result is not None:
            primary[id(frame.f_locals['self'])] = frame.f_locals['diagnostics']
        if previous_profile is not None:
            previous_profile(frame, event, result)

    def configure_convective(native, streams, out, layout):
        """mpas_convective_diagnostics.F:53-85; use actual stream inclusion."""
        missing = configure(native, streams, out, layout)
        bindings[native] = (streams, Path(out))
        for stream in streams:
            for name in tuple(missing[stream.name]):
                if name in FIELDS:
                    native.stream_field(stream.name, name)
                    missing[stream.name].remove(name)
        return missing

    def bind_coupling(coupling, atmosphere, *, level=1):
        """mpas_atm_time_integration.F:2770-2816; preserve resident publication."""
        result = synchronize(coupling, atmosphere, level=level)
        couplings[coupling.session.native] = coupling
        return result

    def maxima_hashes(service):
        # Byte transport/hashing only; all diagnostic maths remains on CUDA.
        return {name: hashlib.sha256(service.session.native.get_field(
            'diag', name).tobytes(order='F')).hexdigest() for name in MAXIMA}

    def write_convective(native, streams, *, step, receipt):
        """mpas_atm_core.F:692-717,855-906; update/compute/write/reset order."""
        coupling = couplings[native]
        dycore, session = coupling.dycore, coupling.session
        if native not in services:
            # Native initial primary solve: mpas_atm_core.F:519-538.
            atmosphere = dycore.atmosphere
            initial = dycore.horizontal.solve_diagnostics(
                atmosphere.state.rho, atmosphere.state.rho_u,
                dt=dycore.config.config_dt,
                apvm_upwinding=dycore.config.config_apvm_upwinding,
                normal_velocity=atmosphere.saved.normal_velocity, rk_step=3)
            session.put('diag', 'vorticity', initial.vorticity)
            services[native] = NativeConvectiveDiagnostics(session, streams)
            services[native].reset()
        elif step:
            # Output-alarm core.F:916-965 changes thermo, never vorticity.
            session.put('diag', 'vorticity', primary[id(dycore)].vorticity)
        service = services[native]
        ringing = [stream for stream in streams if 'output' in stream.directions
                   and stream.output_interval != 'none'
                   and native.alarm_ringing(stream.name + ':out')]
        service.update()
        service.compute()
        event = dict(step=step, process_count=1,
                     output_alarms=[stream.name for stream in ringing],
                     maxima_before_reset=maxima_hashes(service),
                     primary_source='initial primary solve' if not step else 'held final RK diagnostics',
                     diagnostics_backend='hex-cuda')
        # Keep alarms ringing through reset; native writes never reset them.
        for stream in ringing:
            native.transfer_stream(stream.name, read=False, level=1)
            receipt['streams'].append(dict(name=stream.name, step=step))
        if step and any(stream.name == 'diagnostics' for stream in ringing):
            runtime.reset_native_output_diagnostics(native)
        service.reset()
        event['maxima_after_reset'] = maxima_hashes(service)
        for stream in ringing:
            native.reset_alarm(stream.name + ':out')
        receipt['convective_diagnostics'] = dict(
            backend='hex-cuda', process_count=1, calls=step + 1,
            production_hook=False, test_launcher=__file__)
        with (bindings[native][1] / 'convective-lifecycle.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')

    runtime._configure_streams = configure_convective
    runtime.ResidentNativeCoupling.synchronize = bind_coupling
    runtime.write_native_output = write_convective
    sys.setprofile(observe_primary)
    try:
        return door(sys.argv[1:] if argv is None else argv)
    finally:
        sys.setprofile(previous_profile)
        runtime._configure_streams = configure
        runtime.ResidentNativeCoupling.synchronize = synchronize
        runtime.write_native_output = write


if __name__ == '__main__':
    raise SystemExit(main())
