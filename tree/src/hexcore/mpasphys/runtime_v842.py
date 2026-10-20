"""Native forecast orchestration; mpas_atm_core.F:589-701,1036-1040.

CUDA owns dynamics and common conversion kernels. Rust owns stream output;
Python schedules immutable snapshots and the ported clock's alarms.
"""
from __future__ import annotations

from dataclasses import fields
from contextlib import ExitStack
from datetime import datetime
from fractions import Fraction
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..stage_timing import stage as timed_stage
from ..stochastic_timing import measure_stochastic_step
from .driver_v842 import interval_seconds
from .native_backend import NativePhysics
from .pool_v842 import NativePoolHydration
from .session_v842 import NativePhysicsSession, native_working_directory


class ResidentNativeCoupling:
    """Hold outer sources and hand transport to native; mpas_atm_time_integration.F:2140-2816."""

    # MPAS outer-step coupling ownership: mpas_atm_time_integration.F:2140-2816.
    def __init__(self, session, dycore, geometry, names):
        self.session, self.dycore = session, dycore
        self.geometry, self.names = geometry, tuple(names)
        self.cp = dycore.cp
        from .compile_v842 import MpasPhysicsKernelCache
        cache_type=MpasPhysicsKernelCache
        module=getattr(session.config,'driver_module',None)
        if module is not None:
            from importlib import import_module
            cache_type=import_module(module).StrictPhysicsKernelCache
        self.physics_cache = cache_type(
            capability=dycore.cache.capability,
            cache_dir=Path(dycore.cache.cache_directory)/'mpasphys')
        self.sources = None
        self.transport = None
        # IAU (SuiteIau): None without config_IAU_option; itimestep is the run's step (MPAS itimestep).
        self.iau = None
        self.itimestep = 0

    def synchronize(self, atmosphere, *, level=1):
        """Copy native carriers/diagnostics; mpas_atmphys_interface.F:284-305."""
        from ..cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        s, d, native = atmosphere.state, atmosphere.saved, self.session
        for name, value in (('rho_zz',s.rho),('theta_m',d.theta_m),
                            ('u',d.normal_velocity),('w',d.vertical_velocity),
                            ('scalars',s.scalars)):
            native.put('state',name,value,level=level)
        for name, value in (('rho_p',d.density_perturbation),
                            ('rtheta_p',d.rho_theta_perturbation),
                            ('pressure_p',d.pressure_perturbation),('exner',d.exner)):
            native.put('diag',name,value)
        columns=prepare_mpas_to_phys_cuda_v841(atmosphere,scalar_names=self.names,
                                              geometry=self.geometry,kernel_cache=self.physics_cache,
                                              registry_scalar_names=next(row['components'] for row in native.layout.fields
                                                if row['pool']=='state' and row['name']=='scalars'))
        from .diagnostics_v842 import output_diagnostics
        theta,rho,pressure=output_diagnostics(atmosphere,self.names,self.physics_cache,self.cp)
        for name,value in (('uReconstructZonal',columns.u_p),('uReconstructMeridional',columns.v_p),
                           ('rho',rho),('theta',theta),('pressure',pressure)):
            native.put('diag',name,value)
        if getattr(self.dycore,'les_dissipation',None) is not None:
            # Native reconstructs at the outer boundary (time_integration.F:2735),
            # then holds these cell winds throughout the next dynamics splits.
            self.dycore.les_dissipation.winds=columns
        return columns

    def stage(self, *, rk_step, dynamics_substep, state, saved, diagnostics):
        """Assemble outer sources; mpas_atm_time_integration.F:2140-2176; mpas_atm_time_integration.F:3088-3185."""
        if self.sources is None:
            from ..cuda_physics_v841 import CudaMpasStageSourcesV842
            # Native RK setup copies level1 mass to level2 before physics_get_tend
            # reads it (mpas_atmphys_todynamics.F:148-153). Initial Registry
            # hydration has not derived rho_zz; reusing it would zero first-step
            # physics sources and then retain stale mass on subsequent steps.
            self.session.put('state','rho_zz',state.rho,level=2)
            # Native's outer rho_edge is diagnostic h_edge (= edge density).
            self.session.put('diag','rho_edge',diagnostics.h_edge)
            sources=self.session.driver.physics_get_tend(rk_step=rk_step,
                              dynamics_substep=dynamics_substep,pools=self.session)
            # microphysics_to_MPAS leaves this lagged heating in the native
            # tendency pool for the next RK step (interface.F:912-926). Native
            # adds it after rthdynten, then removes it at the final recovery
            # (time_integration.F:6931-6948,4436-4437). Keep it separate from
            # physics_get_tend so density weighting/order remain unchanged.
            diabatic=self.cp.ascontiguousarray(self.cp.asarray(
                self.session.field('tend','rt_diabatic_tend')))
            self.sources=CudaMpasStageSourcesV842(*(self.cp.ascontiguousarray(self.cp.asarray(value))
                              for value in (sources.rho,sources.rho_u,sources.rho_theta,sources.scalars)),
                              diabatic_theta=diabatic)
            if getattr(self, 'stochastic', None) is not None:
                self.stochastic.apply(self.sources, state=state,
                    rho_edge=diagnostics.h_edge, normal_velocity=saved.normal_velocity,
                    time_seconds=float(self.dycore.atmosphere.state.time_seconds),
                    dt=self.stochastic_dt, names=self.names)
            if getattr(self, 'iau', None) is not None:
                # mpas_atm_time_integration.F:2220-2230: once per step, after physics_get_tend and before
                # DYNAMICS_SUBSTEPS, onto the held physics sources.
                self.sources=self.iau.apply(itimestep=self.itimestep,sources=self.sources,state=state,
                                            saved=saved,diagnostics=diagnostics,cache=self.dycore.cache)
        return self.sources

    def release_dynamics_sources(self):
        """Keep only the scalar source once the last RK stage has run.

        mem/state-levels.  Past the dynamics subcycles the dycore reads the
        scalar source alone (transport's scalar_tendency) and so does
        prepare_transport; the mass, momentum, theta and lagged-heating
        sources are dead until the next step rebuilds them.  Holding them
        through transport kept 1.3 KB per column on the ordinary-step peak.
        """
        if self.sources is not None:
            from ..cuda_physics_v841 import CudaMpasStageSourcesV842
            self.sources = CudaMpasStageSourcesV842(
                rho=None, rho_u=None, rho_theta=None,
                scalars=self.sources.scalars, diabatic_theta=None)
        return self.sources

    def post_transport(self, *, state, saved, outer_state, modified_theta_dynamics_rate):
        """Enter before any scalar clamp; mpas_atm_time_integration.F:2770-2816."""
        self.transport=(state,saved,outer_state,modified_theta_dynamics_rate)
        # This diagnostic is formed for every DO_PHYSICS build, including
        # suite none (mpas_atm_time_integration.F:6931-6941; Registry.xml:3535).
        if modified_theta_dynamics_rate is not None:
            self.session.put('tend_physics','rthdynten',modified_theta_dynamics_rate)
        self.session.driver.finish_step(pools=self.session)
        native=self.session.native
        if not self.session.config.active('microphysics'):
            self.transport=None
            return
        # MPAS microphysics_to_MPAS supplies its own Exner and perturbation
        # words. Do not recompute those from the product on the host.
        for target,pool,name in ((state.scalars,'state','scalars'),
                                 (saved.theta_m,'state','theta_m'),
                                 (saved.exner,'diag','exner'),
                                 (saved.rho_theta_perturbation,'diag','rtheta_p'),
                                 (saved.pressure_perturbation,'diag','pressure_p')):
            target[...] = self.cp.asarray(self.session.field(pool,name,level=2 if pool=='state' else 1))
        # Native keeps rtheta_p after the rounded product/subtraction, then
        # recovers full rtheta by adding the base (mpas_atmphys_interface.F:923-926).
        state.rho_theta[...] = saved.rho_theta_perturbation
        self.dycore._add(state.rho_theta,self.dycore.atmosphere.reference.rho_theta_base)
        self.transport=None

    def prepare_transport(self, call):
        """Prepare GF/NTiedtke heating before microphysics; mpas_atm_time_integration.F:2770-2802."""
        from ..cuda_backend import DeviceAtmosphere
        from ..cuda_physics_v841 import (prepare_mpas_post_transport_v842,
                                        prepare_mpas_transport_source_baseline_v842)
        if self.transport is None:
            raise RuntimeError('native post-transport callback was not bound to the current CUDA step')
        state,saved,outer,rate=self.transport
        rth=self.cp.zeros_like(state.rho) if rate is None else rate
        rqv=self.cp.empty_like(state.rho)
        convection=self.session.config.selectors['convection']
        if convection in ('cu_grell_freitas','cu_ntiedtke') and rate is None:
            raise RuntimeError('native convection needs the final RK modified-theta rate')
        before=outer.scalars
        if (self.dycore.config.config_scalar_advection
                and (self.dycore.config.config_monotonic
                     or self.dycore.config.config_positive_definite)):
            # Native F:5208 modifies scalars_1 before F:2783 computes rqvdynten.
            # Subtracting the untouched outer state folds physical vapor sources
            # into the resolved-advection rate and then convection sees them twice.
            before=prepare_mpas_transport_source_baseline_v842(before=before,
                source=self.sources.scalars,rho_old=outer.rho,
                dt=self.dycore.config.config_dt,n_cells_solve=self.dycore.ncells,
                kernel_cache=self.physics_cache)
            self.session.put('state','scalars',before,level=1)
        prepare_mpas_post_transport_v842(before=before,after=state.scalars,
            theta_m=saved.theta_m,scalar_names=self.names,rqvdynten=rqv,rthdynten=rth,
            convection_scheme=convection,monotonic=self.dycore.config.config_monotonic,
            dt=self.dycore.config.config_dt,n_cells_solve=self.dycore.ncells,
            kernel_cache=self.physics_cache)
        atmosphere=self.dycore.atmosphere
        candidate=DeviceAtmosphere(atmosphere.mesh,state,atmosphere.vertical,
                    atmosphere.reference,saved,atmosphere.terrain,atmosphere.h2d)
        self.synchronize(candidate,level=2)
        # mpas_atm_time_integration.F:2768-2802 updates this field only for
        # GF/NTiedtke. Original Tiedtke allocates it but leaves it untouched.
        if (convection in ('cu_grell_freitas','cu_ntiedtke')
                and ('tend_physics','rqvdynten') in self.session.native._fields):
            self.session.put('tend_physics','rqvdynten',rqv)
        self.session.put('tend_physics','rthdynten',rth)


class SuiteIau:
    """MPAS-A v8.4.2 incremental analysis update on the explicit suite route.

    The iau stream's increments (tend_iau: u_amb, rho_amb, theta_amb, scalars_amb), read at initialization
    (mpas_atm_core.F:222-233) by the pinned pool reader, are held on the device, and atm_add_tend_anal_incr
    (hexcore.iau_v842: mpas_atm_iau.F:152-292 as CUDA kernels) adds them to the physics sources once per step,
    after physics_get_tend and before the dynamics substeps (mpas_atm_time_integration.F:2220-2230).  The
    density source then enters tend_rho at rk_step 1 (with_stage_density).  Exact mode is MPAS word for word
    (NINT step count, IN1 and IN2 as native); correct mode applies IN1-IN3 (overlap weights, a window position).
    """

    def __init__(self, *, layout, payload, config, mesh, dycore, dt):
        import numpy as np
        from ..iau_v842 import IAUConfig, iau_window_position

        configs = layout.configs
        mode = getattr(config, 'mode', 'exact')
        self.config = IAUConfig(str(configs.get('config_IAU_option', 'off')).strip(),
                                float(configs.get('config_IAU_window_length_s', 21600.0)), mode)
        self.dt = float(dt)
        cp = dycore.cp
        # The four tend_iau words the pinned pool reader decoded (the same payload the hydration checked).
        manifest = json.loads((Path(payload) / 'manifest.json').read_text())
        records = {(r['pool'], r['name']): r for r in manifest['fields']}
        names = {'u': 'u_amb', 'rho': 'rho_amb', 'theta': 'theta_amb', 'scalars': 'scalars_amb'}
        read = [name for name in names.values()
                if (record := records.get(('tend_iau', name))) is not None and record.get('origins')]
        if 'u_amb' not in read or 'rho_amb' not in read or 'theta_amb' not in read:
            # THE BREAKAGE THIS PREVENTS: with no increment file read, the pool holds the Registry's zero
            # defaults, and the run would apply a zero increment while its receipt says IAU was on.
            raise ValueError('config_IAU_option is on but the iau stream read no increments (u_amb, rho_amb, '
                             'theta_amb); bind the increment file with --stream-input iau=FILE or --iau-increment')

        def held(name):
            record = records[('tend_iau', name)]
            raw = (Path(payload) / record['payload']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError(f'native payload digest differs at tend_iau.{name}')
            words = np.frombuffer(raw, dtype='<f4').reshape(record['shape'], order='F')[..., :-1]
            return cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(words, dtype=np.float32)))

        self.increments = {key: held(name) for key, name in names.items()}
        self.sources_read = sorted(read)
        n_cells = int(self.increments['rho'].shape[1])
        cells_on_edge = np.asarray(mesh.arrays['cellsOnEdge'], dtype=np.int64)
        if cells_on_edge.shape[0] == 2 and cells_on_edge.shape[1] != 2:
            cells_on_edge = cells_on_edge.T
        cells_on_edge = np.where((cells_on_edge >= 0) & (cells_on_edge < n_cells), cells_on_edge, -1)
        self.cells_on_edge = cp.asarray(np.ascontiguousarray(cells_on_edge, dtype=np.int32))
        self.zz = dycore.atmosphere.vertical.zz
        self.moist = {key: int(layout.indices['state', key]) for key in ('moist_start', 'moist_end', 'index_qv')}
        self.position = np.array(0.0, dtype=np.float32)
        self.rule = 'exact mode: MPAS-A v8.4.2 begins the window at the start of every run'
        if self.config.correct:
            start = iau_window_position(self.config, restart=bool(configs.get('config_do_restart', False)),
                                        stored=None, da_cycling=bool(configs.get('config_do_DAcycling', False)))
            self.position, self.rule = start.position, start.rule
        self.calls = self.active_calls = 0
        self.weight_seconds = 0.0

    def apply(self, *, itimestep, sources, state, saved, diagnostics, cache):
        import numpy as np
        from ..iau_v842 import atm_add_tend_anal_incr, atm_iau_coef, atm_iau_coef_window

        weight = (atm_iau_coef_window(self.config, float(self.position), self.dt) if self.config.correct
                  else atm_iau_coef(self.config, int(itimestep), self.dt))
        nl, nc = (int(n) for n in state.rho.shape)
        ne = int(diagnostics.h_edge.shape[1])
        structs = {
            'mesh': {'nVertLevels': nl, 'nCells': nc, 'nEdges': ne, 'nCellsSolve': nc, 'nEdgesSolve': ne,
                     'zz': self.zz, 'cellsOnEdge': self.cells_on_edge},
            'state': {'rho_zz': state.rho, 'theta_m': saved.theta_m, 'scalars': state.scalars,
                      'u': saved.normal_velocity, **self.moist},
            'diag': {'rho_edge': diagnostics.h_edge, 'iau_window_elapsed': self.position},
            'tend': {'scalars_tend': sources.scalars},
            'tend_iau': self.increments,
        }
        atm_add_tend_anal_incr(self.config, structs, int(itimestep), self.dt, sources.rho_u, sources.rho_theta,
                               sources.rho, cache=cache)
        self.calls += 1
        if weight > np.float32(1.0e-12):
            self.active_calls += 1
            self.weight_seconds += float(weight) * self.dt
        return sources

    def receipt(self):
        return {'route': 'mpas-suite (atm_add_tend_anal_incr, hexcore.iau_v842 on CUDA)', 'mode': self.config.config_physics_mode,
                'corrections': ['IN1', 'IN2', 'IN3'] if self.config.correct else [],
                'option': self.config.config_IAU_option, 'window_seconds': float(self.config.config_IAU_window_length_s),
                'window_start': self.rule, 'increments_read': self.sources_read, 'calls': self.calls,
                'active_calls': self.active_calls, 'fraction_admitted': self.weight_seconds,
                **({'window_position_seconds': float(self.position)} if self.config.correct else {})}


def with_stage_density(base):
    """The suite route's dycore class with MPAS's density physics source in tend_rho (an IAU run only).

    MPAS forms tend_rho at rk_step 1 with tend_rho_physics (mpas_atm_time_integration.F:6602), which only
    atm_add_tend_anal_incr writes.  The dycore asks the suite route's stage callback for its sources inside
    the RK loop, after tend_rho is formed, so the first dynamics subcycle of each step would drop the density
    increment (later subcycles already receive the held sources).  This subclass asks the callback first and
    hands its held sources to the subcycle; the callback holds them, so its calls in the loop return the same
    object.  Selected only when config_IAU_option is on, so every other run constructs the unchanged class.
    Named difference (exact mode): native's rthdynten reads that tend_rho too (F:7077); the dycore's rate
    keeps the dynamics-only density tendency, so with GF or New Tiedtke the convective forcing differs from
    native by theta_m*tend_rho_physics/rho_zz during the window.
    """

    class StageDensityDycore(base):
        def _advance_dynamics_subcycle_v841(self, state, **kwargs):
            callback = kwargs.get('physics_stage_callback')
            if kwargs.get('physics_tendencies') is None and callback is not None:
                kwargs['physics_tendencies'] = callback(
                    rk_step=kwargs['schedule'].stages[0].stage, dynamics_substep=kwargs.get('dynamics_substep', 1),
                    state=state, saved=kwargs['time_level_one'], diagnostics=kwargs['initial_diagnostics'])
            return super()._advance_dynamics_subcycle_v841(state, **kwargs)

    StageDensityDycore.__name__ = StageDensityDycore.__qualname__ = base.__name__ + 'StageDensity'
    return StageDensityDycore


def expand_input_filename(template, start):
    """Stage a start-time file with native tokens; mpas_timekeeping.F:2225-2324."""
    time = datetime.strptime(start,'%Y-%m-%d_%H:%M:%S')
    second = time.second
    values = dict(Y=f'{time.year:04d}',M=f'{time.month:02d}',D=f'{time.day:02d}',
                  d=f'{time.timetuple().tm_yday:03d}',h=f'{time.hour:02d}',m=f'{time.minute:02d}',B='all')
    result = ''
    expanding = False
    for character in template:
        if character == '$':
            expanding = True
        elif expanding:
            if character == 'S':
                second += 60*time.minute+3600*time.hour
                result += f'{second:05d}'
            elif character == 's':
                result += f'{second:02d}'
            elif character in values:
                result += values[character]
            else:
                raise ValueError(f'invalid native filename token ${character}')
            expanding = False
        else:
            result += character
    return result


def native_stream_filename(stream, out, start):
    """Read source files, stage writable copies; mpas_stream_manager.F:3187-3190,3695-3697."""
    source = getattr(stream,'input_file',None)
    if source and 'output' not in stream.directions:
        return str(Path(source).resolve())
    root = Path(out).resolve()
    template = root/stream.filename
    if not template.resolve().is_relative_to(root):
        raise ValueError(f'native writable stream {stream.name} escapes its output directory')
    if source:
        target = Path(expand_input_filename(str(template),start))
        if not target.resolve().is_relative_to(root):
            raise ValueError(f'native writable stream {stream.name} resolves outside its output directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            with target.open('rb') as actual, Path(source).open('rb') as expected:
                if hashlib.file_digest(actual,'sha256').digest()!=hashlib.file_digest(expected,'sha256').digest():
                    raise ValueError(f'native staged stream input already differs: {target}')
        else:
            shutil.copyfile(source,target)
    return str(template)


def _configure_streams(native, streams, out, layout):
    """Bind requested native streams and Registry names; mpas_atm_core.F:675-701."""
    active={r['name']:r for r in layout.fields}
    components={c:r for r in layout.fields for c in r['components']}
    supported_diag={'rho','theta','pressure','pressure_base','pressure_p','exner','exner_base',
                    'rho_base','rho_p','rtheta_base','rtheta_p',
                    'uReconstructZonal','uReconstructMeridional',
                    # The native diag pool holds MPAS's own surface_pressure at every write: read from the init,
                    # re-diagnosed by the native MPAS_to_physics glue (mpas_atmphys_interface.F:426-438) and by
                    # the microphysics slot on the GPU, which commits it back (thompson_v842 4778-4781,
                    # interface.F:948-962).  THE BREAKAGE ITS ABSENCE CAUSED: the suite route's output carried
                    # no surface_pressure, so soundings and meteograms had no PSFC (hexmx proof, 2026-10-16).
                    'surface_pressure'}
    # Lagged heating already belongs to the native tendency pool and is
    # restart state (Registry.xml:704,2041; time_integration.F:4436-4437).
    supported_tend={'rt_diabatic_tend'}
    missing={}
    for stream in streams:
        kwargs=dict(filename=native_stream_filename(stream,out,layout.configs['config_start_time']),direction=';'.join(stream.directions),
                    input_interval=stream.input_interval,output_interval=stream.output_interval)
        if 'input_interval' not in inspect.signature(native.stream).parameters:
            raise RuntimeError('native stream ABI lacks independent input/output intervals: restart and surface cannot be reproduced')
        filename_interval=getattr(stream,'filename_interval','none')
        if 'filename_interval' in inspect.signature(native.stream).parameters:
            kwargs['filename_interval']=filename_interval
        elif filename_interval!='none':
            raise RuntimeError(f'native stream {stream.name} lacks filename_interval ABI: expected {filename_interval}, default none would combine native output files')
        native.stream(stream.name,**kwargs)
        missing[stream.name]=[]
        for name in stream.fields:
            row=active.get(name,components.get(name))
            if row is None or (row['pool']=='diag' and name not in supported_diag) or (row['pool']=='tend' and name not in supported_tend):
                missing[stream.name].append(name)
            else:
                native.stream_field(stream.name,name)
    return missing



def initialize_native_output(native, streams, *, restarting=False, reset_stream_alarms=True):
    """Initial time and stream alarms; mpas_atm_core.F:264-277,667-672."""
    native.initialize_state_time()
    if not reset_stream_alarms:
        return
    for stream in streams:
        if ('output' in stream.directions and stream.output_interval != 'none'
                and (stream.name == 'restart' or restarting
                     and stream.name in ('output','diagnostics'))):
            native.reset_alarm(stream.name+':out')


def native_publication_required(native, streams, *, step, steps, moving, lazy, output=None, when=None):
    """Keep state resident between writes; outer physics refreshes its own pools.

    Initial/final boundaries and moving callbacks always consume current pools.
    Alarm inspection does not reset or transfer a stream; the writer still owns
    those actions and diagnostic accumulator resets.
    """
    if not lazy or step == 0 or step == steps or moving is not None:
        return True
    if output is not None:
        return output.required(when)
    return any('output' in stream.directions and stream.output_interval != 'none'
               and native.alarm_ringing(stream.name+':out') for stream in streams)


def write_native_output(native, streams, *, step, receipt, session=None, time_seconds=None,
                        output=None, dt=None):
    """Native stream alarm decisions; mpas_atm_core.F:675-705; mpas_timekeeping.F:925-985."""
    if output is not None:
        diagnostics_written = output.write(step=step, dt=dt, receipt=receipt)
    else:
        # Explicit native-cpu referee helpers retain native stream I/O. The
        # forecast door always supplies its Rust output owner below.
        diagnostics_written=False
        for stream in streams:
            if ('output' in stream.directions and stream.output_interval != 'none'
                    and native.alarm_ringing(stream.name+':out')):
                native.transfer_stream(stream.name,read=False,level=1)
                native.reset_alarm(stream.name+':out')
                receipt['streams'].append(dict(name=stream.name,step=step))
                diagnostics_written |= stream.name == 'diagnostics'
    if step and diagnostics_written:
        reset=getattr(getattr(session,'driver',None),'after_output',None)
        if reset is None:
            reset_native_output_diagnostics(native, session=session)
        else:
            first=len(session.driver.calls)
            reset(time_seconds=time_seconds,diagnostics_alarm=True,itimestep=step,pools=session)
            # The step call log is captured before output; retain successful
            # output-boundary resets separately for native alarm comparison.
            receipt.setdefault('physics_output_calls',[]).extend(session.driver.calls[first:])


def reset_native_output_diagnostics(native, *, session=None):
    """Clear the accumulated reflectivity after all writes; mpas_atm_core.F:877-890,968-990."""
    import numpy as np
    # Native's optional associated lookup leaves inactive microphysics alone.
    key=('diag_physics','refl10cm_1km_max')
    metadata=native._fields.get(key)
    if metadata is not None:
        views = getattr(session, 'device_pools', None)
        if views is not None and getattr(views, 'resident', False):
            views.field(*key).fill(0)
            return
        dtype,shape,levels=metadata
        native.put_field(*key,np.zeros(shape,dtype=dtype,order='F'),nlevels=levels)


def native_mesh_state_exclusions(layout):
    """State is loaded separately; retain every Registry mesh spelling.

    MPAS Registry.xml separates the mesh and state pools. Scalar arrays can
    appear under their component names in the init stream. Do not read those
    large payloads into Mesh only to read them again as atmospheric state.
    """
    names = {'state': set(), 'mesh': set()}
    for row in layout.fields:
        if row['pool'] in names:
            selected = names[row['pool']]
            selected.add(row['name'])
            selected.add(row['pool_name'])
            selected.update(row.get('components') or ())
    return frozenset(names['state'] - names['mesh'])


def load_native_dynamics_inputs(arguments, layout, payload, *, keep_initialized_deformation=False):
    """Complete input ownership before CUDA; mpas_atm_core.F:116-200,589-596."""
    import numpy as np
    from ..mesh import Mesh
    from ..vector import initialize_reconstruction_coefficients
    from ..driver import load_mpas_vertical_grid,load_mpas_initial_state
    from ..dynamics_v841 import load_v841_reference_wind_profiles
    init=Path(arguments.init)
    hydration=NativePoolHydration(None,layout,payload)
    # The native input stream overlays the grid with initialized mesh fields,
    # including fEdge/fVertex; Registry.xml:498-545; mpas_atm_core.F:116-151.
    mesh_options={'native_parent_chain':True} if keep_initialized_deformation else {}
    mesh_options['exclude_variables']=native_mesh_state_exclusions(layout)
    mesh=Mesh.from_netcdf(getattr(arguments,'grid',None) or init,init,**mesh_options)
    # Native MPAS (single precision) reads a double-precision coordinate into RKIND on input
    # (mpas_io_streams.F:925-1023). An init_atmosphere init stores them single already, so this is a
    # no-op there (x1.2562: no double array remains after the init overlay). THE BREAKAGE IT PREVENTS:
    # a hex-door init (every moving member) keeps the generator's double coordinates; the edge normals
    # derived from them were double and the native mesh pool refused the run before its first step
    # ("native mesh.edgeNormalVectors shape/dtype differs").
    for name,value in list(mesh.arrays.items()):
        if getattr(value,'dtype',None) is not None and value.dtype==np.float64:
            mesh.arrays[name]=np.ascontiguousarray(value,dtype=np.float32)
    mesh.arrays['nominalMinDc']=hydration.horizontal_length()
    mesh.arrays['coeffs_reconstruct']=initialize_reconstruction_coefficients(mesh,include_halos=True)
    # Native initializes these runtime vectors before tend_toEdges reads them;
    # mpas_vector_operations.F:652-771; mpas_atmphys_todynamics.F:510-550.
    from ..vector import initialize_vector_geometry
    mesh.arrays['edgeNormalVectors']=initialize_vector_geometry(mesh).edge_normal_vectors
    grid=load_mpas_vertical_grid(init,mesh,config_coef_3rd_order=layout.configs['config_coef_3rd_order'])
    profiles=load_v841_reference_wind_profiles(init,n_vert_levels=grid.vertical_grid.n_vert_levels)
    names=next(r['components'] for r in layout.fields if r['pool']=='state' and r['name']=='scalars')
    state,reference,saved=load_mpas_initial_state(init,mesh,grid.vertical_grid,
            scalar_names=(),reference_path=init,terrain_metrics=grid.terrain_metrics,
            return_saved_diagnostics=True,cold_start=True)
    # Native allocates every selected carrier, including absent input species;
    # Registry.xml:1701-1754; mpas_atmphys_init_microphysics.F:98-226. Rust already
    # applied their Registry defaults while decoding the complete scalar buffer.
    state.scalars=np.ascontiguousarray(hydration.field_buffer('state','scalars')[...,:-1])
    for name in ('defc_a','defc_b'):
        if keep_initialized_deformation:
            # MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:373-614:
            # retain native initialized deformation weights, including cut borders.
            mesh.arrays[name]=np.ascontiguousarray(hydration.field_buffer('mesh',name)[...,:mesh.n_cells].T)
        else:
            mesh.arrays[name]=np.zeros((mesh.n_cells,mesh.arrays['edgesOnCell'].shape[1]),np.float32)
    return mesh,grid,profiles,state,reference,saved,names


#: The lake class of the 21-class MODIS land use (modis_landuse_20class_30s_with_lakes, WPS index islake=21),
#: which ``rw_mpas_static`` writes; MPAS's VEGPARM.TBL row MODIFIED_IGBP_MODIS_NOAH has 20 classes.
MODIS_LAKE_CLASS = 21


def fold_lake_class(native, layout):
    """Lake cells become water when no lake model runs, as WRF real does (module_initialize_real.F, islake).

    THE BREAKAGE THIS PREVENTS: a hex-door static (every moving member, any ``rw_mpas_static`` mesh) takes its
    land use from the 21-class MODIS set with lakes. MPAS-A v8.4.2 has no lake model and its Noah table holds
    20 MODIS classes, so one lake cell stopped every MPAS-suite run in the land-surface init
    ("MPAS Noah native fatal condition at source line 2441": too many input landuse types). An
    init_atmosphere static has no class 21, so this changes nothing there. Returns the folded cell count.
    """
    import numpy as np
    fields = getattr(native, '_fields', {})
    # Registry names -> native pool names (iswater_lu is held as iswater).
    names = {row['name']: row['pool_name'] for row in layout.fields if row['pool'] == 'sfc_input'}
    water_name = names.get('iswater_lu')
    if ('sfc_input', names.get('ivgtyp')) not in fields or ('sfc_input', water_name) not in fields:
        return 0
    try:
        mminlu = str(native.get_text_field('sfc_input', names.get('mminlu', 'mminlu')))
    except Exception:
        return 0    # the class table is unknown: class 21 is a land class in USGS, so nothing is folded
    if 'MODIS' not in mminlu.upper():
        return 0
    water = int(np.asarray(native.get_field('sfc_input', water_name)).reshape(-1)[0])
    folded = 0
    for name in (names.get('ivgtyp'), names.get('lu_index')):
        if name is None or ('sfc_input', name) not in fields:
            continue
        values = native.get_field('sfc_input', name)
        count = layout.dimensions['nCells']
        lake = values[..., :count] == MODIS_LAKE_CLASS
        if lake.any():
            values[..., :count][lake] = water
            native.put_field('sfc_input', name, values, nlevels=fields['sfc_input', name][2])
            folded = max(folded, int(lake.sum()))
    return folded


def native_table_roots(source,config,extra=None):
    """Folders whose tables a native run stages in its run directory, in order.

    THE BREAKAGE THIS PREVENTS: Noah-MP reads NoahmpTable.TBL from the run directory, and MPAS keeps it in
    physics_noahmp/parameters, not in physics_wrf/files.  A door run that selected sf_noahmp (by the licence
    opt-in, hexcore.licence_gate) on native-cpu stopped in Noah-MP's table reader with a Fortran end-of-file
    error unless the user also named that folder.  The folder is added only when the run selects Noah-MP, so
    every other run stages the same tables, and records the same table digests, as before.
    """
    roots=[Path(source)/'src/core_atmosphere/physics/physics_wrf/files',*(extra or [])]
    noahmp=Path(source)/'src/core_atmosphere/physics/physics_noahmp/parameters'
    if config.selectors.get('land_surface')=='sf_noahmp' and noahmp.is_dir() and noahmp not in roots:
        roots.append(noahmp)
    return roots


def framework_only(config):
    """Choose the ported framework for the GPU route without an opt-in.

    Explicit native-cpu physics remains a referee. The RRFS framework is a
    separate source contract. MPAS mpas_atmphys_control.F:310-319 selects
    active schemes; disabled slots do not require native runtime storage.
    """
    if getattr(config, 'table_key', 'mpasphys_v842') != 'mpasphys_v842':
        return False
    if any(str(backend) == 'native-cpu' and config.active(slot)
           for slot, backend in config.backends.items()):
        return False
    return not (config.options.get('config_sst_update', False) and
                str(config.service_backends.get('surface_update', 'native-cpu')) == 'native-cpu')


def run_native_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out, *, moving=None, execution=None):
    """Retain startup timings on successful and failed sessions."""
    timings = []
    runtime_rows = []
    from .. import time_ledger
    try:
        with timed_stage(runtime_rows, 'native-runtime',
                         parallelism={'device':'GPU plus native CPU','cards':1},
                         timeline='post.forecast_process.native.runtime'):
            return _run_native_dynamics(arguments, config, layout, payload, streams, library,
                start, stop, dt, steps, out, moving=moving, execution=execution, timings=timings)
    finally:
        time_ledger.end_if_open('prep.door')
        time_ledger.end_if_open('forecast.construct')
        time_ledger.end_if_open('forecast.integrate')
        (out/'native-runtime-timings.json').write_text(json.dumps({'stages':runtime_rows + timings},indent=2)+'\n')


def _run_native_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out, *, moving=None, execution=None, timings=None):
    """Real-door native physics plus resident CUDA dynamics; mpas_atm_core.F:589-701,1036-1040.

    ``moving`` (a ``moving_v842.MovingRoute``, only when a GPUWM_HEX_MOVING_* switch is set) adds the moving
    region's hooks: remapped resume, physics carry, shock series, frames, stop and carrier export.
    ``execution`` supplies inputs, an optional driver factory and process-local hooks. It does not
    relax the native singleton or the resident glue's global-domain admission.
    """
    from .config_v842 import NativeMpasDycoreConfigV842
    from ..cuda_driver import CudaDryDycoreDriver
    from ..cuda_backend import KernelCache,require_cuda
    from ..cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841

    from .. import time_ledger
    card_attrs = {"card": time_ledger.card_index(), "cards": 1, "backend": "hex-cuda"}
    # Initialization loads scheme resources inside native_working_directory.
    # Resolve the caller's packet before that directory change, for both the
    # framework and the CUDA provider, without changing the caller's arguments.
    cuda_resources = getattr(arguments, 'physics_resources', None)
    if cuda_resources is not None:
        cuda_resources = Path(cuda_resources).resolve()
    # MPAS mpas_subdriver.F:364 copies configs into stream attributes before
    # core init. Hydration resolves config_len_disp later (mpas_atm_core.F:159-191);
    # that resolved runtime value must not replace the earlier output attribute.
    output_config_snapshot = dict(layout.configs)

    receipt_path = (out/'native-physics-receipt.json' if execution is None
                    else execution.receipt_path(out))
    resumed = moving is not None and moving.carrier is not None
    # RP6 (correct mode with config_pbl_interval). Breakage this refusal prevents: the surface precipitation
    # window (precip_window_v842) lives on the device and the pristine Registry restart stream cannot hold it, so
    # a native-file restart would silently drop the precipitation that fell since the last land-surface call.
    # Native correct mode keeps the window in its restart stream (package bl_precip_window_in).  A moving
    # resume is not a native-file restart: its carrier holds the window (moving_v842), so it is not refused.
    if getattr(config,'correct',None) is not None and config.correct('RP6') and config.active('land_surface') \
            and layout.configs['config_do_restart'] and not resumed:
        raise ValueError('a native-file restart of correct mode with config_pbl_interval (RP6) would drop the '
                         'precipitation added up since the last land-surface call: the pristine restart stream '
                         'cannot hold the window. Restart in exact mode, without config_pbl_interval, or with RP6 '
                         'in config_physics_correct_skip')
    # IN3 (correct mode with IAU on). Breakage this refusal prevents: the pristine Registry restart stream holds no
    # IAU window position (iau_window_elapsed), so a native-file restart inside a window would begin the window
    # again and admit the increment a second time. A DA-cycling restart (config_do_DAcycling) begins a new
    # window by rule and runs; exact mode runs MPAS-A v8.4.2's own behaviour (a new window every run).
    if (str(layout.configs.get('config_IAU_option','off')).strip()=='on' and layout.configs['config_do_restart']
            and getattr(config,'mode','exact')=='correct' and not layout.configs.get('config_do_DAcycling',False)):
        raise ValueError('a native-file restart with IAU on in correct mode would admit the window again: the '
                         'pristine restart stream holds no IAU window position. Restart with config_do_DAcycling '
                         '= true (a new DA cycle, a new window), in exact mode, or start the window on a cold start')
    signature=inspect.signature(CudaDryDycoreDriver._step_device_v841).parameters
    missing={'physics_post_transport_callback','physics_mass_loading_count'}-set(signature)
    if config.moist_physics and missing:
        raise RuntimeError('native physics dycore reconciliation is absent: '+', '.join(sorted(missing)))
    # The forecast door resolves admission and passes no library for the
    # ported route. Explicit proof tools can supply a native referee even
    # with GPU physics; repeating config selection here would ignore it.
    ported_framework = library is None
    if ported_framework:
        from .framework_backend_v842 import FrameworkPhysics
        backend_type = FrameworkPhysics
    else:
        backend_type = NativePhysics
    for method in ('initialize_state_time','update_state_time'):
        if not callable(getattr(backend_type,method,None)):
            raise RuntimeError(f'native output time API {method} is absent: stale xtime/CF Time would invalidate every output')
    from ..physics_suites import load_table
    row=load_table()[getattr(config,'table_key','mpasphys_v842')]
    with timed_stage(timings, 'dycore-inputs', parallelism={'device':'CPU','host_threads':1}, timeline='prep.native.dycore_inputs'):
        input_loader = getattr(execution, 'load_inputs', None)
        if input_loader is not None:
            inputs = input_loader(load_native_dynamics_inputs, arguments, layout, payload,
                                  keep_initialized_deformation=row.get('dycore') is not None)
        elif execution is not None:
            inputs = execution.inputs
        else:
            inputs = load_native_dynamics_inputs(arguments,layout,payload,
                keep_initialized_deformation=row.get('dycore') is not None)
    mesh,grid,profiles,state,reference,saved,names=inputs
    physics_map=None
    if moving is not None:
        moving.begin(layout.configs['config_start_time'] if not resumed else None)
    if resumed:
        # The carrier's atmosphere on this member (the remap the ArWen route uses), before the dycore exists.
        state,saved,physics_map=moving.resume_dycore(mesh=mesh,grid=grid,reference=reference,state=state,
                                                     saved=saved,names=names,
                                                     target_grid=getattr(arguments,'grid',None) or arguments.init)
    if execution is not None:
        state,saved=execution.couple_input(state,saved,mesh,grid,reference)
    options={f.name:layout.configs[f.name] for f in fields(NativeMpasDycoreConfigV842)
             if f.init and f.name in layout.configs}
    options.update(config_dt=dt,config_moist_physics=config.moist_physics)
    from ..physics_suites import load_table
    row=load_table()[getattr(config,'table_key','mpasphys_v842')]
    dycore_config=NativeMpasDycoreConfigV842(**options)
    driver_type=CudaDryDycoreDriver
    cache_type=KernelCache
    if row.get('dycore') is not None:
        from ..dycore_variants import resolve_dycore_variant,dycore_driver_type,dycore_kernel_cache_type
        from dataclasses import fields as config_fields
        # The Registry supplies variant-only namelist settings too.
        dycore_config=resolve_dycore_variant(dycore_config,row)
        fork_options={f.name:layout.configs[f.name] for f in config_fields(type(dycore_config))
                      if f.init and f.name in layout.configs}
        dycore_config=resolve_dycore_variant(NativeMpasDycoreConfigV842(**options),row,overrides=fork_options)
        driver_type=dycore_driver_type(dycore_config)
        cache_type=dycore_kernel_cache_type(dycore_config)
        selected_signature=inspect.signature(driver_type._step_device_v841).parameters
        absent={'physics_post_transport_callback','physics_mass_loading_count'}-set(selected_signature)
        if config.moist_physics and absent:
            raise RuntimeError('selected fork dycore lacks the post-transport physics boundary: '+', '.join(sorted(absent)))
    if str(layout.configs.get('config_IAU_option','off')).strip()!='off':
        driver_type=with_stage_density(driver_type)
    les=None
    if dycore_config.config_les_model != 'none' or dycore_config.config_mix_scalars:
        # The v8.4.2 LES dissipation module (mpas_atm_dissipation_models.F)
        # at the native RK boundary, mpas_atm_time_integration.F:6331-6390.
        # Default (LES off, no scalar mixing) runs never construct it.
        from ..cuda_les_v842 import ResidentLesV842
        hydration=NativePoolHydration(None,layout,payload)
        les=ResidentLesV842(mesh,grid.vertical_grid,profiles,dycore_config,names,
                            hydration.field_buffer('mesh','t_init')[...,:-1])
    # Select the factory only after variant and IAU wrappers are resolved.
    build_dycore = getattr(execution, 'build_dycore', None)
    factory = driver_type.from_host
    if build_dycore is not None:
        from functools import partial
        factory = partial(build_dycore, factory)
    with timed_stage(timings, 'dycore-build', parallelism={'device':'GPU','cards':1,'host_threads':1}, timeline='prep.native.dycore_build'):
        dycore=factory(mesh,state,grid.vertical_grid,reference,
                dycore_config,saved_diagnostics=saved,
                terrain_metrics=grid.terrain_metrics,reference_wind_profiles=profiles,
                **({'les_dissipation':les} if les is not None else {}),
                # Use the architecture/compiler-keyed persistent cache. A cache
                # under pool scratch forced every forecast to rebuild unchanged
                # dycore and physics kernels after disposable output was removed.
                kernel_cache=cache_type(capability=require_cuda(min_compute=(7,5))))
    if build_dycore is not None:
        # Validate the actual returned driver before entering the native context.
        # Binding accepts forwarding **kwargs wrappers as well as named keywords.
        try:
            inspect.signature(getattr(dycore, '_step_device_v841', None)).bind(
                physics_stage_callback=None, physics_post_transport_callback=None,
                physics_mass_loading_count=0)
        except (TypeError, ValueError) as error:
            raise RuntimeError('execution-built dycore lacks the required physics callback interface') from error
    if execution is not None:
        execution.bind_dycore(dycore)
    geometry=CudaMpasToPhysGeometryV841.from_host(mesh)
    receipt=dict(schema='mpas-native-forecast-v1',identity_checked=False,completed=False,
                 physics=config.receipt(),steps_requested=steps,steps_completed=0,
                 native_library_sha256=None if ported_framework else hashlib.sha256(library.read_bytes()).hexdigest(),
                 framework='ported' if ported_framework else 'native referee',streams=[],stage_timings=timings)
    table_roots=([] if ported_framework else
        native_table_roots(layout.source,config,getattr(arguments,'native_physics_tables',None)))
    receipt['table_sha256']={}
    selector=row.get('native_table_selector')
    if ported_framework:
        # Scheme resources are authenticated and loaded by the CUDA provider.
        # The installed Registry contains no native source/table directory.
        tables=()
    elif selector is None:
        tables=(table for root in table_roots for table in sorted(root.iterdir())
                if table.is_file() and (table.suffix in ('.TBL','.DBL')
                    or table.name.startswith(('RRTMG_','CCN_'))))
    else:
        from importlib import import_module
        module,operation=selector.split(':',1)
        callback=getattr(import_module(module),operation)
        keyword=row.get('native_table_override_keyword')
        tables=(callback(table_roots[:1], **{keyword: table_roots[1:]})
                if keyword else callback(table_roots))
    for table in tables:
        target=out/table.name
        if target.exists() and target.resolve()!=table.resolve():
            raise ValueError(f'native table destination already differs: {target}')
        if not target.exists():target.symlink_to(table.resolve())
        receipt['table_sha256'][table.name]=hashlib.sha256(table.read_bytes()).hexdigest()
    selector=row.get('native_initialization_namelist')
    if selector is not None:
        from importlib import import_module
        module, operation=selector.split(':',1)
        receipt['initialization_namelist_sha256']=getattr(import_module(module),operation)(
            arguments.physics_namelist,out)
    native_options={'pin_set':config.pin_set} if getattr(config,'pin_set',None) is not None else {}
    surface_record_reader = None
    if config.service_backends.get('surface_update') == 'hex-cuda' and config.options['config_sst_update']:
        from .surface_update_v842 import RustSurfaceRecordReader
        resource_path = cuda_resources
        resource = None
        if resource_path is not None:
            resource_path = Path(resource_path).resolve()
            from .resources_build import load_packet
            resource = load_packet(resource_path).get('surface_update', {}).get('reader_library')
        if resource is not None:
            reader_library = Path(resource['path'])
            if not reader_library.is_absolute():
                reader_library = resource_path.parent / reader_library
            reader_library = reader_library.resolve()
            if hashlib.sha256(reader_library.read_bytes()).hexdigest() != resource['sha256']:
                raise ValueError('surface update Rust reader_library SHA256 differs')
        else:
            # Default-on surface updating (s2s-surface-update): the packaged
            # source-built Rust reader serves runs without a resource packet.
            from ..surface_series_reader import reader_library as packaged_reader
            reader_library = packaged_reader()
        reader_sha256 = hashlib.sha256(reader_library.read_bytes()).hexdigest()
        surface = next(s for s in streams if s.name == 'surface' and 'input' in s.directions)
        surface_file = Path(surface.input_file or expand_input_filename(surface.filename, start)).resolve()
        surface_record_reader = (RustSurfaceRecordReader(reader_library, surface_file,
                                                        n_cells=layout.dimensions['nCells'])
            if execution is None else execution.surface_reader(
                RustSurfaceRecordReader, reader_library, surface_file, layout))
        receipt['surface_input'] = dict(path=str(surface_file), sha256=hashlib.sha256(surface_file.read_bytes()).hexdigest(),
                                       reader_library=str(reader_library), reader_sha256=reader_sha256, reads=[])
    backend = (backend_type(layout=layout, resources=cuda_resources)
               if ported_framework else backend_type(library,test_only=True,**native_options))
    with backend as native, ExitStack() as output_lifetime:
        coupling=None
        core_operations={}
        core_initialize=None
        module=row.get('native_core_module')
        if module is not None:
            from importlib import import_module
            core_provider=import_module(module)
            core_initialize=core_provider.initialize_native_core
            core_operations['atm_reset_diagnostics']=lambda call:core_provider.reset_native_diagnostics(native)
        with timed_stage(timings, 'physics-session', parallelism={'device':'CPU','host_threads':1}, timeline='prep.native.physics_session'):
            session=NativePhysicsSession(native,layout,payload,config,dt=dt,start=start,stop=stop,
                core_operations=core_operations,
                stream_intervals={s.name:(s.input_interval if s.name=='surface' else s.output_interval) for s in streams},
                post_transport=lambda call:coupling.prepare_transport(call),cuda_device=dycore.cp,
                cuda_resources=cuda_resources,
                surface_record_reader=surface_record_reader)
        session.output_config_snapshot=output_config_snapshot
        receipt['lake_cells_folded_to_water']=fold_lake_class(native,layout)
        from .streams_output_v842 import PhysicsStreams, output_inventory
        receipt['missing_fields']=output_inventory(streams,layout)
        coupling=ResidentNativeCoupling(session,dycore,geometry,names)
        from ..stochastic import resolve_options, StochasticPhysics
        stochastic_path = getattr(arguments, 'physics_namelist', None)
        stochastic_config = resolve_options(getattr(arguments, 'stochastic', None),
            None if stochastic_path is None else Path(stochastic_path).read_text())
        stochastic_restart = None if not resumed else moving.carrier.get('stochastic')
        if stochastic_config.enabled:
            coupling.stochastic = StochasticPhysics(mesh, grid.vertical_grid, stochastic_config)
            dycore.stochastic = coupling.stochastic
            coupling.stochastic_dt = dt
            if stochastic_restart is not None:
                coupling.stochastic.process.restore(stochastic_restart)
            elif resumed or stochastic_config.restart_required:
                raise ValueError('stochastic restart is absent from this native carrier')
            receipt['stochastic'] = coupling.stochastic.receipt()
        elif stochastic_restart is not None:
            raise ValueError('checkpoint stochastic physics must remain enabled on resume')

        if str(layout.configs.get('config_IAU_option','off')).strip()!='off':
            coupling.iau=SuiteIau(layout=layout,payload=payload,config=config,mesh=mesh,dycore=dycore,dt=dt)
            receipt['iau']=coupling.iau.receipt()
            print(json.dumps({'iau':receipt['iau']}),flush=True)
        if not layout.configs['config_do_restart']:
            from .diagnostics_v842 import initialize_thermodynamics
            initialize_thermodynamics(dycore.atmosphere,coupling.physics_cache)
        resident_reference=dycore.atmosphere.reference
        for name,value in (('rho_base',resident_reference.rho_base),('rtheta_base',resident_reference.rho_theta_base),
                           ('pressure_base',resident_reference.pressure_base),('exner_base',resident_reference.exner_base)):
            session.put('diag',name,value)
        session.put('mesh','zz',dycore.atmosphere.vertical.zz)
        session.put('mesh','dss',dycore.atmosphere.vertical.dss)
        session.put('mesh','edgeNormalVectors',mesh.arrays['edgeNormalVectors'].T)
        coupling.synchronize(dycore.atmosphere)
        if resumed:
            # Carried physics state into the pools before the physics init, which runs in MPAS's restart mode
            # (config_do_restart), as a native restart reads its restart stream before physics_init.
            receipt['moving_physics']=moving.restore_physics(session,layout,physics_map,mesh)
        stopped_at=steps
        with native_working_directory(out):
            with timed_stage(receipt['stage_timings'], 'physics-initialize', timeline='prep.native.physics_initialize', parallelism={'device':'GPU plus native CPU','cards':1,'host_threads':1}):
                session.initialize()
            if core_initialize is not None:
                core_initialize(session,mesh)
                bind_output_diagnostics=getattr(core_provider,'bind_native_output_diagnostics',None)
                if bind_output_diagnostics is not None:
                    bind_output_diagnostics(session,streams)
            initialize_native_output(native,streams,restarting=layout.configs['config_do_restart'],
                                     reset_stream_alarms=False)
            from .resident_v842 import ResidentPhysicsGlue, resident_requested
            resident = resident_requested(config, layout.dimensions)
            diagnostics = None
            # Output diagnostics belong to the dycore, including dry runs
            # without resident physics. The provider's halo operation is a
            # no-op only on the closed single-rank meshes it supports.
            closed_mesh = all(layout.dimensions.get(name) == layout.dimensions.get(name+'Solve')
                              for name in ('nCells', 'nEdges', 'nVertices'))
            if resident or (ported_framework and closed_mesh):
                from .resident_output_diagnostics_v842 import ResidentOutputDiagnostics
                diagnostics = ResidentOutputDiagnostics(session, dycore)
            output=output_lifetime.enter_context(PhysicsStreams(session,streams,out,final_time=stop,
                                                                diagnostics=diagnostics))
            receipt['missing_fields'] = output.missing
            session.output_streams=output
            receipt['output_metadata']=getattr(output,'metadata_receipt',None)
            dycore.atmosphere.state.scalars[...] = dycore.cp.asarray(native.get_field('state','scalars')[...,:-1])
            if resident:
                ResidentPhysicsGlue(session, coupling, mesh)
            receipt['resident_physics'] = dict(enabled=resident,
                scope='exact global convection_permitting',
                field_authority='GPU' if resident else 'native pools')
            lazy_publication = resident or os.environ.get('GPUWM_HEX_NATIVE_LAZY_PUBLICATION') == '1'
            publication = receipt['native_publication'] = dict(
                lazy=lazy_publication, published=0, deferred=0)
            if execution is not None:
                # Bind after initialization and optional resident glue so wrappers
                # see the final coupling callbacks and initialized field authority.
                execution.bind_coupling(coupling)
                execution.initialized(dycore)
            for step in range(steps+1):
                if step:
                    if step == 1:
                        time_ledger.end_if_open("forecast.construct")
                        time_ledger.emit("forecast.integrate", "begin", attrs=card_attrs)
                        time_ledger.emit("forecast.first_step", "begin", attrs=card_attrs)
                    with measure_stochastic_step(coupling, 'mpas-outer-step'):
                        if execution is not None:
                            execution.before_step(step)
                        coupling.sources=None
                        with timed_stage(receipt['stage_timings'], 'outer-physics', timeline='post.forecast_process.native.outer_physics', parallelism={'device':'GPU plus native CPU','cards':1,'host_threads':1}):
                            coupling.synchronize(dycore.atmosphere)
                            session.begin_step(step,(step-1)*dt)
                        coupling.itimestep=step
                        acoustic_output = {}
                        if diagnostics is not None:
                            callback = diagnostics.prepare_step(step=step, dt=dt)
                            if callback is not None:
                                acoustic_output['acoustic_output_callback'] = callback
                        if les is not None:
                            # les_surface='varying' reads the physics fluxes
                            # (mpas_atm_dissipation_models.F:919,1533-1534).
                            for name in ('ustm','hfx','qfx'):
                                if ('diag_physics',name) in native._fields:
                                    les.set(name,session.field('diag_physics',name))
                        # Registry's qv group exists in the dry suite too. Native
                        # always computes the moist coefficients from that active
                        # group (mpas_atm_time_integration.F:2125-2152).
                        with timed_stage(receipt['stage_timings'], 'dycore-and-post-transport', timeline='post.forecast_process.native.dycore', parallelism={'device':'GPU plus native CPU callbacks','cards':1,'host_threads':1}):
                            result=dycore._step_device_v841(physics_stage_callback=coupling.stage,
                                physics_post_transport_callback=coupling.post_transport,
                                physics_mass_loading_count=layout.indices['state','moist_end'],
                                **acoustic_output)
                        dycore.atmosphere=result.atmosphere
                        if execution is not None:
                            # Complete boundary exchange before committing the native clock.
                            execution.after_step(dycore)
                        # Resident state is published at level1; compute the scalar
                        # clock word using the same native RKIND operation on that level.
                        native.update_state_time(dt,source_level=1,target_level=1)
                        native.advance_clock()
                    if step == 1:
                        time_ledger.emit("forecast.first_step", "end", attrs=card_attrs)
                    receipt['steps_completed']=step
                    if getattr(coupling, 'stochastic', None) is not None:
                        receipt['stochastic'] = coupling.stochastic.receipt()
                    if coupling.iau is not None:
                        receipt['iau']=coupling.iau.receipt()
                    if les is not None:
                        receipt['les_execution']=les.receipt(step)
                    calls_path = (out/'physics-calls.jsonl' if execution is None
                                  else execution.calls_path(out))
                    with calls_path.open('a') as log:
                        log.write(json.dumps(dict(step=step,calls=session.driver.calls))+'\n')
                publish = native_publication_required(native, streams, step=step, steps=steps,
                    moving=moving, lazy=lazy_publication,output=output,
                    when=output.current_time(step=step,dt=dt))
                # Always evaluate this hook: it may contain a rank collective,
                # including when the native stream schedule already needs output.
                consumer_publication = (execution is not None and
                    execution.publication_required(step, steps, dt))
                if consumer_publication:
                    publish = True
                with timed_stage(receipt['stage_timings'], 'publish-and-streams', timeline='forecast.output.history.capture',
                        parallelism={'device':'GPU capture plus background Rust I/O' if publish else 'CPU',
                                     'cards':int(publish),'host_threads':1}) as timing:
                    timing['state_published'] = publish
                    if publish:
                        columns=coupling.synchronize(dycore.atmosphere)
                        # JEDI and moving carriers still consume host pool
                        # views. Ordinary stream capture reads device buffers
                        # directly and never needs this compatibility copy.
                        if consumer_publication or moving is not None:
                            session.publish()
                        write_native_output(native,streams,step=step,receipt=receipt,session=session,
                                            time_seconds=step*dt,output=output,dt=dt)
                    publication['published' if publish else 'deferred'] += 1
                if execution is not None:
                    execution.output(step=step,steps=steps,dt=dt,session=session,layout=layout,names=names,out=out)
                if session.surface_service is not None:
                    receipt['surface_service'] = session.surface_service.receipt()
                    if surface_record_reader is not None and surface_record_reader.last_record is not None:
                        record = surface_record_reader.last_record
                        reads = receipt['surface_input']['reads']
                        if not reads or reads[-1]['record'] != record:
                            reads.append(dict(record=record, timestamp=surface_record_reader.times[record].strftime('%Y-%m-%d_%H:%M:%S'),
                                              model_clock=session.arguments.calendar['current_date'], step=step,
                                              callback_execution=receipt['surface_service']['operations']['physics_update_sst']))
                if resident:
                    receipt['resident_physics']['transfers'] = dict(session.device_pools.transfers)
                stop=False
                if moving is not None:
                    stop=moving.after_step(step,steps=steps,dt=dt,columns=columns,session=session,dycore=dycore,
                                           mesh=mesh,vertical=grid.vertical_grid,names=names,layout=layout,out=out)
                    receipt.update(moving.receipt())
                receipt_path.write_text(json.dumps(receipt,indent=2,default=str)+'\n')
                if stop:
                    # The controller asked to stop at this frame: the carrier below seals this instant.
                    stopped_at=step
                    break
            time_ledger.end_if_open("forecast.integrate")
            time_ledger.emit("post.forecast", "begin", attrs=card_attrs)
            if moving is not None:
                moving.finish(step=stopped_at,session=session,dycore=dycore,mesh=mesh,vertical=grid.vertical_grid,
                              names=names,layout=layout,dt=dt)
                receipt.update(moving.receipt())
            output.close()
            session.driver.finalize(session)
    if coupling.iau is not None:
        receipt['iau']=coupling.iau.receipt()
        # Correct mode admits exactly one increment over a window the run covers (IN3); exact mode keeps
        # native's NINT step count, which IN3 names.
        if (coupling.iau.config.correct and coupling.iau.config.config_IAU_option=='on'
                and steps*dt>=coupling.iau.config.config_IAU_window_length_s
                and abs(receipt['iau']['fraction_admitted']-1.0)>1e-5):
            raise RuntimeError(f"IAU window not fully applied: {receipt['iau']}")
    if execution is not None:
        receipt['rank_execution']=execution.receipt()
    receipt['completed']=True
    receipt['physics_compile_manifest']=coupling.physics_cache.compile_manifest()
    from ..device_memory_release import process_release
    receipt['device_memory_release']=process_release().receipt()
    receipt_path.write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps(receipt,default=str))
    return 0
