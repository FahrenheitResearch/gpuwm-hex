"""Resident output diagnostics, using the existing MPAS CUDA implementations.

MPAS v8.4.2 mpas_atm_core.F:675-701,825-910 schedules update/compute/write/reset.
No stream-manager operation or mutable atmospheric field crosses the host.
"""
from datetime import timedelta

from ..mpasdiag.isobaric import FIELDS as ISOBARIC, IsobaricDiagnostics
from ..mpasdiag.convective_v842 import FIELDS as CONVECTIVE, MAXIMA, ConvectiveDiagnosticsModule
from ..mpasdiag.utils import DiagnosticsUtils, Direction, StreamView


PRIMARY = frozenset(('vorticity', 'divergence', 'ke'))
# Registry.xml:606 reads theta_base at initialization; :1853 circulation has
# no dynamics writer. Preserve those initialized words, including restart
# input, rather than manufacturing a zero or dividing rounded base products.
INITIALIZED = frozenset(('circulation', 'theta_base'))
MOMENTUM = {'ru': 'rho_u', 'rw': 'rho_w'}
ACOUSTIC = frozenset(('ru_p', 'rw_p'))
PRODUCED = (PRIMARY | frozenset(ISOBARIC) | frozenset(CONVECTIVE)
            | {'t_iso_levels', 'z_iso_levels'} | INITIALIZED | MOMENTUM.keys() | ACOUSTIC)


class _Pools:
    def __init__(self, owner, pool):
        self.owner, self.pool = owner, pool

    def __getitem__(self, name):
        owner, pool = self.owner, self.pool
        if pool == 'mesh' and name in owner.session.layout.dimensions:
            return owner.session.layout.dimensions[name]
        if pool == 'state' and name == 'index_qv':
            return owner.session.layout.indices[pool, name]
        if pool == 'diag' and name in PRODUCED:
            return owner.buffer(pool, name)
        return owner.current(pool, name)


class ResidentOutputDiagnostics:
    """Keep produced fields on the device until the writer takes a snapshot.

    Closed single-rank meshes need no pressure or state halo exchange.
    Temporary padded views preserve the input allocation's sentinel words;
    authoritative resident pool allocations keep their existing owned extent.
    """
    produced = PRODUCED

    def __init__(self, session, dycore, *, halo=None):
        self.session, self.dycore = session, dycore
        # None on a closed single-rank mesh. A same-host rank's owner fills
        # the halo cells the vertex pressure (isobaric.py) averages over.
        self.halo = halo
        self.cp = session.device_pools.cp
        self.pools = {name: _Pools(self, name) for name in ('mesh', 'state', 'diag')}
        self.buffers = {}
        self.writer = None
        self.when = None
        self.updates = 0
        self.last_step = None
        self.acoustic = {}
        self.acoustic_captures = 0
        self.pending = None
        self.deferred_updates = 0

    def buffer(self, pool, name):
        key = pool, name
        if key not in self.buffers:
            # Initial allocation/default/dummy words are read once. All later
            # changes come from resident views or the diagnostic CUDA kernels.
            self.buffers[key] = self.cp.ascontiguousarray(self.cp.asarray(
                self.session.native.get_field(pool, name, level=1)))
        return self.buffers[key]

    def current(self, pool, name):
        cp, p = self.cp, self.session.device_pools
        target = self.buffer(pool, name)
        old = p.level
        try:
            p.level = 1
            # Nonresident coupling updates host pools directly. Cached slot
            # views can predate those writes, including when no physics slot
            # runs to clear them. Fetch the current authority for each input.
            value = (p.field(pool, name) if p.resident else self.cp.asarray(
                     self.session.native.get_field(pool, name, level=1)))
        finally:
            p.level = old
        target[tuple(slice(0, size) for size in value.shape)] = value
        dims = self.session.layout.dimensions
        if pool == 'mesh' and name in ('cellsOnVertex', 'kiteAreasOnVertex'):
            return cp.ascontiguousarray(target[:, :dims['nVertices']])
        if pool == 'mesh' and name == 'areaTriangle':
            return cp.ascontiguousarray(target[:dims['nVertices']])
        return target

    def bind(self, writer):
        self.writer = writer
        self.when = writer.current_time(step=0, dt=0)
        self.utils = DiagnosticsUtils()
        self.utils.init(self.views)
        self.isobaric = IsobaricDiagnostics(exchange_pressure_halo=(lambda p: None) if self.halo is None
                                            else self.halo.exchange_pressure)
        self.isobaric.setup(self.pools, getattr(self.session.native, 'clock_state', None), self.utils)
        self.convective = ConvectiveDiagnosticsModule()
        self.convective.setup(self.pools, getattr(self.session.native, 'clock_state', None), self.utils)
        if self.halo is not None:
            self.convective.driver.vertex_order = self.halo.vertex_order()
        self.accumulates = any(self.utils.stream_inclusion_count(name, Direction.OUTPUT)
                               for name in MAXIMA)

    def views(self):
        for name, stream in self.writer.streams.items():
            directions = stream.options['type'].split(';')
            output = 'output' in directions
            yield StreamView(name,
                Direction.INPUT_OUTPUT if len(directions) == 2 else
                    Direction.OUTPUT if output else Direction.INPUT,
                bool(stream.fields(self.session.layout.packages, streams=self.writer.streams)),
                output and self.writer.alarms.due(name, self.when),
                dict.fromkeys(stream.fields(self.session.layout.packages, streams=self.writer.streams), True),
                stream.interval('input'), stream.interval('output'))

    def defer(self, *, step, when):
        """Hold step's running-maximum update for the next synchronize.

        mpas_atm_core.F:675-701 updates the maxima after every step from the
        state that step left. When no stream is due, that state is unchanged
        until the next step's outer physics synchronizes it into the pools
        (winds, w, cell-reconstructed winds); flush() runs the update there,
        on the same words, so no publication or host sync is forced per step.
        """
        if self.accumulates:
            if self.pending is not None:
                raise RuntimeError(f'maxima of step {self.pending[0]} were never folded')
            self.pending = (step, when)

    def flush(self):
        """Fold the deferred step's maxima; call right after synchronize."""
        if self.pending is None:
            return
        step, when = self.pending
        self.pending = None
        self.when = when
        # No stream is due on a deferred step, so only the maxima read the
        # primary diagnostics: vorticity alone (divergence/KE are formed
        # again by before_write whenever a stream needs them).
        a, horizontal = self.dycore.atmosphere, self.dycore.horizontal
        if hasattr(horizontal, 'solve_vorticity'):
            value = horizontal.solve_vorticity(a.state.rho, a.state.rho_u,
                                               normal_velocity=a.saved.normal_velocity)
            target = self.buffer('diag', 'vorticity')
            target[..., :value.shape[-1]] = value
        else:
            self._solve_primary()
        self.convective.update()
        self.updates += 1
        self.deferred_updates += 1
        self.last_step = step

    def _solve_primary(self):
        # Final solve vorticity/KE/divergence depend only on normal velocity,
        # unchanged by subsequent scalar transport and microphysics. Reuse the
        # existing dycore operator, including native reduction order.
        a, driver = self.dycore.atmosphere, self.dycore
        solved = driver.horizontal.solve_diagnostics(a.state.rho, a.state.rho_u,
            dt=driver.config.config_dt, apvm_upwinding=driver.config.config_apvm_upwinding,
            normal_velocity=a.saved.normal_velocity, rk_step=3)
        for name, value in (('vorticity', solved.vorticity),
                            ('divergence', solved.divergence), ('ke', solved.kinetic_energy)):
            target = self.buffer('diag', name)
            target[..., :value.shape[-1]] = value

    def before_write(self, *, step, when):
        if self.pending is not None:
            # A deferred update must be folded on its own state before this one.
            raise RuntimeError(f'maxima of step {self.pending[0]} were not folded before step {step}')
        self.when = when
        if step == 0:
            self.convective.reset()
        if self.accumulates or any(view.output_alarm for view in self.views()):
            self._solve_primary()
        self.convective.update()
        self.isobaric.compute()
        self.convective.compute()
        self.updates += 1
        self.last_step = step

    def prepare_step(self, *, step, dt):
        """Retain acoustic allocations only for a due output, not ordinary steps."""
        self.acoustic.clear()
        when = (self.writer.current_time(step=step-1, dt=dt)
                + timedelta(seconds=float(dt)))
        for name, stream in self.writer.streams.items():
            if ('output' in stream.options['type'].split(';') and
                    ACOUSTIC.intersection(stream.fields(self.session.layout.packages,
                                                       streams=self.writer.streams)) and
                    self.writer.alarms.due(name, when)):
                return self.capture_acoustic
        return None

    def capture_acoustic(self, *, ru_p, rw_p):
        """The last RK stage/subcycle supplies its original device words."""
        self.acoustic.update(ru_p=ru_p, rw_p=rw_p)
        self.acoustic_captures += 1

    def after_snapshot(self):
        self.convective.reset()
        self.acoustic.clear()

    def field(self, pool, name):
        # atm_recover_large_step_variables_work F:4418,4476 writes rw/ru.
        # The CUDA prognostic state already owns those exact recovery words;
        # no velocity conversion or host pool publication is needed here.
        if pool == 'diag' and name in MOMENTUM:
            return getattr(self.dycore.atmosphere.state, MOMENTUM[name])
        if pool == 'diag' and name in ACOUSTIC:
            if self.last_step == 0:
                return self.buffer(pool, name)
            if name not in self.acoustic:
                raise RuntimeError(f'{name}: final acoustic output was not retained')
            return self.acoustic[name]
        if pool == 'diag' and name in PRODUCED:
            return self.buffer(pool, name)
        return None

    def close(self):
        self.acoustic.clear()
        self.buffers.clear()

    def receipt(self):
        return dict(updates=self.updates, last_step=self.last_step,
                    deferred_updates=self.deferred_updates,
                    produced=sorted(PRODUCED),
                    device_buffer_bytes=sum(value.nbytes for value in self.buffers.values()),
                    acoustic_captures=self.acoustic_captures,
                    acoustic_buffer_bytes=sum(value.nbytes for value in self.acoustic.values()),
                    pressure_halo='closed single-rank mesh' if self.halo is None
                                  else 'same-host rank owner exchange')
