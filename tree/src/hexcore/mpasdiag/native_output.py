"""Bind isobaric CUDA diagnostics to f3's real native pools and stream writer.

This adapter supplies the isobaric part of the output call. Explicit companion
adapters bind the complete DiagnosticsManager for whole-model replay.
No atmospheric formulas or file decoding run in Python.
"""
from .isobaric import FIELDS, IsobaricDiagnostics, compute_output_diagnostics
from .utils import DiagnosticsUtils, Direction, StreamView
from .manager import DiagnosticsManager


class NativeIsobaricOutput:
    def __init__(self, native, layout, streams, device):
        """MPAS v8.4.2 mpas_isobaric_diagnostics.F:55-71,313-1001."""
        self.native, self.layout, self.streams, self.cp = native, layout, streams, device
        self.pools = dict(mesh={}, state={}, diag={})
        self.utils = DiagnosticsUtils()
        self.utils.init(self.views)
        self.isobaric = IsobaricDiagnostics(exchange_pressure_halo=lambda p: None)
        self.manager = None
        self.calls = []
        self.dycore = None

    def views(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_utils.F:85-129."""
        for stream in self.streams:
            output = 'output' in stream.directions
            enabled = output and stream.output_interval != 'none'
            direction = Direction.INPUT_OUTPUT if len(stream.directions) == 2 else (
                Direction.OUTPUT if output else Direction.INPUT)
            yield StreamView(stream.name, direction, True,
                             enabled and self.native.alarm_ringing(stream.name+':out'),
                             dict.fromkeys(stream.fields, True),
                             stream.input_interval, stream.output_interval)

    def attach(self, missing, *, companions=None, companion_fields=()):
        """MPAS v8.4.2 mpas_atm_core.F:664-719; native stream field attachment."""
        produced = set(FIELDS) | {'t_iso_levels', 'z_iso_levels', 'theta', 'rho',
                                  'pressure', 'vorticity'} | set(companion_fields)
        for stream in self.streams:
            for name in tuple(missing[stream.name]):
                if name in produced:
                    self.native.stream_field(stream.name, name)
                    missing[stream.name].remove(name)
        if companions is None:
            self.isobaric.setup(self.pools, self.native, self.utils)
        else:
            # MPAS v8.4.2 mpas_atm_diagnostics_manager.F:31-63.
            self.manager = DiagnosticsManager(isobaric=self.isobaric, **companions)
            self.manager.setup(self.views, self.native, self.pools, self.native, self.native)
            self.utils = self.manager.utils

    def compute(self, step):
        """MPAS v8.4.2 mpas_atm_core.F:825-910,916-965; mpas_isobaric_diagnostics.F:85-309."""
        due = any(view.output_alarm for view in self.views())
        if not due and self.manager is None:
            return
        cp, native, dims = self.cp, self.native, self.layout.dimensions
        # MPAS atm_compute_solve_diagnostics_work publishes vertex vorticity
        # (mpas_atm_time_integration.F:7159-7179). Reuse that existing CUDA
        # operator on the current resident state; preserve native dummy words.
        if self.dycore is not None:
            dycore, atmosphere = self.dycore, self.dycore.atmosphere
            solved = dycore.horizontal.solve_diagnostics(
                atmosphere.state.rho, atmosphere.state.rho_u,
                dt=dycore.config.config_dt,
                apvm_upwinding=dycore.config.config_apvm_upwinding,
                normal_velocity=atmosphere.saved.normal_velocity, rk_step=3)
            vorticity = cp.ascontiguousarray(cp.asarray(native.get_field('diag', 'vorticity')))
            vorticity[..., :dims['nVertices']] = solved.vorticity
            dtype, shape, levels = native._fields['diag', 'vorticity']
            native.put_field('diag', 'vorticity', cp.asnumpy(vorticity), nlevels=levels)
        if not due:
            if step == 0:
                self.manager.reset()
            self.manager.update()
            self.manager.compute()
            return
        # Exchange native storage before copying it. The device halo callback
        # can then be a no-op: this one-process adapter already completed it.
        native.exchange_halo('pressure_p')
        def field(pool, name):
            return cp.ascontiguousarray(cp.asarray(native.get_field(pool, name, level=1)))
        mesh = {name:dims[name] for name in
                ('nCells', 'nVertices', 'nVertLevels', 'vertexDegree', 'nIsoLevelsT', 'nIsoLevelsZ')}
        mesh.update({name:field('mesh', name) for name in ('zgrid', 'zz')})
        for name in ('cellsOnVertex', 'kiteAreasOnVertex', 'areaTriangle'):
            mesh[name] = cp.ascontiguousarray(field('mesh', name)[..., :dims['nVertices']])
        state = {name:field('state', name) for name in ('theta_m', 'rho_zz', 'scalars', 'w')}
        state['index_qv'] = self.layout.indices['state', 'index_qv']
        inputs = ('pressure_p', 'pressure_base', 'exner', 'relhum',
                  'uReconstructZonal', 'uReconstructMeridional', 'vorticity',
                  'theta', 'rho', 'pressure')
        diag = {name:field('diag', name) for name in (*inputs, *FIELDS, 't_iso_levels', 'z_iso_levels')}
        self.pools.update(mesh=mesh, state=state, diag=diag)
        compute_output_diagnostics(mesh, state, diag)
        for name in ('theta', 'rho', 'pressure'):
            dtype, shape, levels = native._fields['diag', name]
            native.put_field('diag', name, cp.asnumpy(diag[name]), nlevels=levels)
        if self.manager is None:
            self.isobaric.compute()
        else:
            if step == 0:
                self.manager.reset()
            self.manager.update()
            self.manager.compute()
        # Preserve complete native allocations, including unmodified dummy
        # words and family side effects. The native writer owns the file ABI.
        for name in (*FIELDS, 't_iso_levels', 'z_iso_levels'):
            dtype, shape, levels = native._fields['diag', name]
            native.put_field('diag', name, cp.asnumpy(diag[name]), nlevels=levels)
        self.calls.append(dict(step=step, state_level=1, process_count=1,
                               fields=[name for name in FIELDS if self.utils.field_will_be_written(name)]))

    def reset(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:132-143."""
        if self.manager is not None:
            self.manager.reset()

    def cleanup(self):
        """MPAS v8.4.2 mpas_atm_diagnostics_manager.F:156-173."""
        if self.manager is not None:
            self.manager.cleanup()
