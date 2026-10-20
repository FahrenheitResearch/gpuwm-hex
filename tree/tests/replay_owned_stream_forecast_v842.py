"""Computed whole forecast with the owned writer bound at the driver boundary.

MPAS mpas_atm_core.F:663-705,877-890. This supporting experiment calls the
actual CLI, CUDA dycore and native CPU physics. Only the output callback is
replaced with NativePoolStreams. It records this test binding explicitly and
does not qualify the unmodified production dispatch or restart initialization.
"""
from __future__ import annotations

import sys

from replay_native_physics_restart_v842 import digest
from replay_hex_restart_forecast_v842 import main as run_six_arms


def run_bound_cli(argv, *, resident_momenta=False):
    """Actual output driver binding; MPAS mpas_atm_core.F:675-705,877-890."""
    from hexcore import cli
    from hexcore.mpasphys import runtime_v842 as runtime
    from hexcore.restart_v842 import NativePoolStreams
    from hexcore.streams_v842 import parse_streams

    original_run = runtime.run_native_dynamics
    original_write = runtime.write_native_output
    original_sync = runtime.ResidentNativeCoupling.synchronize

    def synchronize(coupling, atmosphere, *, level=1):
        """Actual momentum handoff; MPAS Registry.xml:1957-1971;
        dynamics/mpas_atm_time_integration.F:4352-4458.
        """
        result = original_sync(coupling, atmosphere, level=level)
        if level == 1:
            coupling.session.put('diag', 'ru', atmosphere.state.rho_u)
            coupling.session.put('diag', 'rw', atmosphere.state.rho_w)
        return result

    def bound_run(arguments, config, layout, payload, streams, library,
                  start, stop, dt, steps, out):
        """Keep the whole driver intact; MPAS mpas_atm_core.F:589-705."""
        adapter = None
        declared = parse_streams(arguments.streams,
            list_directory=getattr(arguments, 'stream_list_directory', None))
        selected = {name:set(stream.fields(layout.packages, streams=declared))
                    for name,stream in declared.items()}

        def publish(native, selected_streams, *, step, receipt):
            """Publish synchronized computed storage; MPAS mpas_atm_core.F:675-705."""
            nonlocal adapter
            if adapter is None:
                # The foundation inventory also names package-inactive rows.
                # Native selects no such fields: Registry.xml:498-1349 and
                # mpas_stream_manager.F:4577-4666. Retain that input inventory
                # while masking only fields this native stream actually selects.
                missing = {name:(set(names) - ({'ru','rw'} if resident_momenta else set())) & selected[name]
                           for name,names in receipt['missing_fields'].items()}
                inactive = {name:sorted(set(names) - selected[name])
                            for name,names in receipt['missing_fields'].items()
                            if set(names) - selected[name]}
                adapter = NativePoolStreams(native, layout, arguments.streams, out,
                    list_directory=getattr(arguments, 'stream_list_directory', None),
                    final_time=stop,
                    unavailable_fields={name:names for name,names in missing.items()
                                        if names})
                adapter.read_input_attributes(arguments.init,
                                              scratch=out/'owned-stream-metadata')
                receipt['stream_test_binding'] = dict(
                    callback='NativePoolStreams.write', production_dispatch=False,
                    resident_momentum_binding=resident_momenta,
                    resident_momentum_authority=('CUDA state.rho_u/state.rho_w'
                        if resident_momenta else None),
                    launcher_sha256=digest(__file__),
                    input_header_source_sha256=digest(arguments.init),
                    driver_missing_fields={name:list(fields) for name,fields
                        in receipt['missing_fields'].items()},
                    driver_inventory_package_inactive_fields=inactive,
                    scope=('Actual computed forecast; output callback and computed momentum handoff'
                        if resident_momenta else 'Actual computed forecast; output callback only'))
            previous = len(receipt['streams'])
            adapter.write(step=step, dt=dt, receipt=receipt)
            written = receipt['streams'][previous:]
            # Preserve the native alarm and diagnostic reset effects of the
            # replaced output callback; mpas_atm_core.F:675-705,877-890.
            for row in written:
                native.reset_alarm(row['name']+':out')
            if step and any(row['name']=='diagnostics' for row in written):
                runtime.reset_native_output_diagnostics(native)

        runtime.write_native_output = publish
        try:
            return original_run(arguments, config, layout, payload, streams,
                                library, start, stop, dt, steps, out)
        finally:
            runtime.write_native_output = original_write

    runtime.run_native_dynamics = bound_run
    if resident_momenta:
        runtime.ResidentNativeCoupling.synchronize = synchronize
    try:
        return cli.main(argv)
    finally:
        runtime.run_native_dynamics = original_run
        runtime.ResidentNativeCoupling.synchronize = original_sync


def main(argv=None):
    """Six actual forecast arms; MPAS mpas_atm_core.F:589-705."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('--cli','--cli-resident-momenta'):
        return run_bound_cli(argv[1:], resident_momenta=argv[0]=='--cli-resident-momenta')
    return run_six_arms([*argv,'--stream-test-binding'])


if __name__=='__main__':
    raise SystemExit(main())
