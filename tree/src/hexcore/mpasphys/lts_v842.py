"""Full-step local time stepping on the MPAS-physics route, pure hex, one card.

The ArWen route's active LTS (``hexcore.drivers.fullstep_lts_forecast_v841``) steps every rate class of a
variable mesh at its own whole step: one resident dycore per class on the class's local mesh (its own cells,
the scratch zone it carries and three halo rings), coordinated fine-first by
:class:`hexcore.cuda_fullstep_lts_active.FullStepLtsActive` (accepted-state mailboxes, matched interface flux
and reflux).  Its class physics is a column backend.  This module gives the MPAS-physics route the same
coordinator with the MPAS suite as each class's physics: every class runs the ordinary ported route
(``runtime_v842.run_native_dynamics`` with ``library=None``) on its own local mesh, at its own step
``rate * dt``, in its own thread, and the coordinator drives it one class step at a time through the route's
``execution`` seam (the one the same-host ranks use, ``samehost_v842``):

* ``before_step`` waits until the coordinator asks this class for its next step, after the coordinator has
  filled the step-boundary halo (round A); ``after_step`` hands the committed endpoint back, so the
  coordinator records the trajectory, refluxes and moves on.  Exactly one class integrates at a time, on one
  device, in the coordinator's fine-first order; every hand-off synchronizes the device, so no class reads
  another's words before they land.
* The class's MPAS physics runs on every local column (the class pools are projected with every local entity
  solved), as the ArWen route's class backends do: the halo columns' physics is never published, and the
  scratch zone's is this class's own.  Before the microphysics the halo of the unpublished endpoint takes the
  owners' accepted end state (round P), as on the ArWen route.
* Output is written at frame times only, which the class clock check makes macro boundaries (every class
  stands there).  At a macro boundary each class waits for the coordinator to finish the macro (reflux
  included) before it synchronizes its pools, deposits the words it OWNS, and class 0 assembles the global
  frame (``samehost_v842.assemble_snapshot``) and writes it, so the history and diag files are the global files
  a one-card fixed-step run writes.

Not identical to the fixed step, by design (a different time scheme); the dycore of each class is the pinned
MPAS numerics at its own step.  Refused by name, each with its breakage (``admission_refusal``): several cards,
restart / DA cycling / IAU / stochastic forcing / chemistry, the balance series and nest triggers, timed
surface updating, a limited-area mesh and a non-v8.4.2 dycore row.  A moving segment runs on the classes
(``ClassMoving``: the move hooks at macro boundaries, as on the ArWen route's class executor).
"""
from __future__ import annotations

from copy import copy
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from .samehost_v842 import _ENTITY_MAPS, assemble_snapshot
from .streams_output_v842 import PhysicsStreams

#: How long any party waits for another before the run stops (a peer died or deadlocked).
WAIT_SECONDS = float(os.environ.get('GPUWM_HEX_MPAS_LTS_WAIT_SECONDS', '1800'))


class LtsAborted(RuntimeError):
    """Another class thread or the coordinator failed; every wait stops."""


class _Board:
    """Shared rendezvous of the coordinator and the class threads."""

    def __init__(self, classes: int):
        self.lock = threading.Condition()
        self.classes = classes
        self.failure: str | None = None
        self.built: dict[int, object] = {}
        self.klass = None
        self.ready = False
        self.go = [0] * classes          # steps granted per class
        self.done = [0] * classes        # steps handed back per class
        self.parked = [False] * classes  # waiting in before_step, or finished
        self.macro_seconds = -1.0        # model seconds since the start of the last completed macro boundary
        self.deposits: dict[tuple, dict[int, dict]] = {}
        self.stop_requested: float | None = None   # model seconds a moving controller stopped this segment at

    def fail(self, reason: str) -> None:
        with self.lock:
            if self.failure is None:
                self.failure = reason
            self.lock.notify_all()

    def wait(self, predicate, what: str) -> None:
        deadline = time.monotonic() + WAIT_SECONDS
        with self.lock:
            while not predicate():
                if self.failure is not None:
                    raise LtsAborted(self.failure)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = f'LTS rendezvous timed out waiting for {what}'
                    self.lock.notify_all()
                    raise LtsAborted(self.failure)
                self.lock.wait(min(remaining, 5.0))
            if self.failure is not None:
                raise LtsAborted(self.failure)


def session_partition(layout):
    """The class mesh as the MPAS pools see it: every local entity solved (physics on every local column)."""
    return SimpleNamespace(
        cell_l2g=np.asarray(layout.cell_l2g), edge_l2g=np.asarray(layout.edge_l2g),
        vertex_l2g=np.asarray(layout.vertex_l2g),
        n_owned_cells=int(len(layout.cell_l2g)), n_owned_edges=int(len(layout.edge_l2g)),
        n_owned_vertices=int(len(layout.vertex_l2g)))


def _device_synchronize():
    import cupy as cp
    cp.cuda.Device().synchronize()


class LtsClassExecution:
    """The ``execution`` seam of ``runtime_v842.run_native_dynamics`` for one rate class."""

    def __init__(self, board: _Board, index: int, rate: int, top: int, partitions, inputs, global_layout):
        self.board = board
        self.index = index
        self.rate = int(rate)
        self.top = int(top)
        self.partitions = partitions          # the true class layouts: who owns which entity
        self.partition = partitions[index]
        self.inputs = inputs
        self.global_layout = global_layout
        self.klass = None                     # the coordinator's class record, bound once it exists
        self.dycore = None
        self.step_times: list[float] = []
        self.wait_times: list[float] = []
        self._began = None
        self.streams_type = self._streams_type()
        self.cycling = None
        self.current_step = 0                 # the class step in flight (0: the initial state)

    # -- construction --------------------------------------------------------------------------------
    def receipt_path(self, out):
        return Path(out) / f'native-physics-receipt.class{self.index}.json'

    def calls_path(self, out):
        return Path(out) / f'physics-calls.class{self.index}.jsonl'

    def couple_input(self, state, saved, mesh, grid, reference):
        return state, saved

    def surface_reader(self, factory, library, path, layout):
        # admission_refusal names this before any class exists; kept so a direct caller cannot slip past it.
        raise ValueError('timed surface updating (config_sst_update) with local time stepping on the MPAS suite '
                         'route is not built: each class mesh needs its own projection of the surface record')

    def bind_dycore(self, dycore):
        self.dycore = dycore

    def bind_coupling(self, coupling):
        finish = coupling.post_transport

        def post_transport(**values):
            # Round P: the unpublished endpoint's halo from the owners, before the microphysics reads every
            # local column (the ArWen route's class transaction does the same before its phase two).
            self.klass.exchanger.round_physics_candidate(SimpleNamespace(
                state=values['state'], saved=values['saved']))
            return finish(**values)

        coupling.post_transport = post_transport

    def initialized(self, dycore):
        board = self.board
        _device_synchronize()
        with board.lock:
            board.built[self.index] = dycore
            board.lock.notify_all()
        board.wait(lambda: board.ready, 'the LTS coordinator')
        self.klass = board.klass[self.index]

    # -- stepping --------------------------------------------------------------------------------------
    def before_step(self, step):
        board = self.board
        mark = time.perf_counter()
        with board.lock:
            board.parked[self.index] = True
            board.lock.notify_all()
        board.wait(lambda: board.go[self.index] >= step, f'class {self.index} step {step}')
        with board.lock:
            board.parked[self.index] = False
        self._began = time.perf_counter()
        self.wait_times.append(self._began - mark)
        self.current_step = int(step)

    def after_step(self, dycore):
        board = self.board
        # The coordinator reads this class's endpoint on its own stream next.
        _device_synchronize()
        self.step_times.append(time.perf_counter() - self._began)
        with board.lock:
            board.done[self.index] += 1
            board.lock.notify_all()

    def publication_required(self, step, steps, dt):
        # At a macro boundary the coordinator still refluxes this class (its finer interface) after its last
        # step of the macro.  Pools are synchronized from the dycore and frames captured right after this
        # call, so wait for the macro to finish first; a frame is always a macro boundary.
        if (int(step) * self.rate) % self.top == 0:
            board = self.board
            seconds = float(step) * float(dt)
            board.wait(lambda: board.macro_seconds >= seconds - 1e-6, f'the macro boundary at {seconds} s')
        return False

    def output(self, **values):
        return None

    def receipt(self):
        return dict(schema='mpas-lts-class-v1', klass=self.index, rate=self.rate,
                    owned_cells=int(self.partition.n_owned_cells),
                    local_cells=int(len(self.partition.cell_l2g)),
                    steps=len(self.step_times),
                    step_wall_seconds_total=float(sum(self.step_times)),
                    rendezvous_wait_seconds_total=float(sum(self.wait_times)),
                    parallelism='one class thread; one GPU shared by every class, one class integrating at a time')

    def close(self):
        return None

    # -- output ----------------------------------------------------------------------------------------
    def _streams_type(self):
        execution = self

        class ClassPhysicsStreams(PhysicsStreams):
            """Each class deposits the words it owns at a frame; class 0 writes the global frame."""

            def __init__(self, session, streams, output, **options):
                super().__init__(session, streams, output, **options)
                # The encoder sizes NetCDF dimensions from the mesh view.
                self.mesh.dimensions = dict(execution.global_layout.dimensions)
                owned = copy(self.layout)
                owned.dimensions = dict(self.layout.dimensions)
                for entity, (_, owned_name) in _ENTITY_MAPS.items():
                    owned.dimensions[entity] = int(getattr(execution.partition, owned_name))
                self.layout = owned

            def _snapshot(self, selected):
                local = super()._snapshot(selected)
                board = execution.board
                key = (str(self.current_time(step=0, dt=0)), tuple(s.name for s in selected))
                with board.lock:
                    slot = board.deposits.setdefault(key, {})
                    # Copies: this class's pinned writer storage is reused at its next frame.
                    slot[execution.index] = {name: (copy(item) if not hasattr(item, 'values') else
                                                    type(item)(np.array(item.values, copy=True), item.dimensions,
                                                               item.attrs))
                                             for name, item in local.items()}
                    board.lock.notify_all()
                if execution.index != 0:
                    return {}
                # Class 0 always writes, so per-file frame counts (append records) never move between classes.
                board.wait(lambda: len(board.deposits.get(key, {})) == board.classes, f'every class at frame {key[0]}')
                with board.lock:
                    slot = board.deposits.pop(key)
                snapshots = [slot[i] for i in range(board.classes)]
                # On all cores (samehost_v842.assemble_snapshot).  The serial guard that stood here cited the
                # first-step validation refusals, whose cause was the physics edge projection reading past its
                # tendency arrays (resident_v842.ResidentPhysicsGlue, dummy-cell geometry), not the assembly.
                return assemble_snapshot(snapshots, execution.partitions, self.rows, execution.global_layout)

            def _encode(self, jobs, snapshot, when, step, *extra, **options):
                if execution.index != 0:
                    return []
                return super()._encode(jobs, snapshot, when, step, *extra, **options)

            def _read_first_call(self, names):
                """The first physics call's f000 fields on the global mesh: every class's owned words.

                THE BREAKAGE THIS PREVENTS: correct mode (the door default) completes the held initial frame
                from the first physics call; class 0 read only its own columns against the global frame and
                every MPAS-suite LTS run in correct mode stopped after its first step ("f000 hfx: first call
                (73869,) != frame (98034,)", MPAS-ROUTE-SPEED-2026-10-24).  Each class deposits the words it owns
                right after its own first step (class 0 holds the top rate, so it steps once a macro and waits
                here only for classes that step inside the same macro); class 0 assembles them.
                """
                from ..output import HistoryField
                from .. import f000_first_call as first
                wanted = tuple(name for name in first.MPAS_FIELDS if name in self.rows)
                local = super()._read_first_call(wanted)
                words = {name: HistoryField(np.array(value, copy=True), tuple(self.rows[name][0]['dimensions']), {})
                         for name, value in local.items()}
                board, key = execution.board, ('f000-first-call',)
                with board.lock:
                    board.deposits.setdefault(key, {})[execution.index] = words
                    board.lock.notify_all()
                if execution.index != 0:
                    return {}
                board.wait(lambda: len(board.deposits.get(key, {})) == board.classes, 'every class first call (f000)')
                with board.lock:
                    slot = board.deposits.pop(key)
                assembled = assemble_snapshot([slot[i] for i in range(board.classes)], execution.partitions,
                                              self.rows, execution.global_layout)
                return {name: assembled[name].values for name in names if name in assembled}

        return ClassPhysicsStreams


def moving_admitted(moving) -> bool:
    """Whether a moving route rides the class executor: a plain moving region (``moving_v842.MovingRoute``).

    A native checkpoint or restart route (``checkpoint_v842.CheckpointRoute``: --checkpoint-hours,
    --restart-from, --da-analysis) is refused by ``admission_refusal`` on this executor already, so it keeps
    the fixed step and its own hooks.
    """
    from .moving_v842 import MovingRoute
    return moving is None or type(moving) is MovingRoute


class ClassMoving:
    """The moving hooks of one rate class: the same-host ranks' owner assembly, at macro boundaries only.

    Built on ``moving_mpi_v842.DistributedMoving`` (the MPAS route's moving hooks across same-host ranks): the
    global remap and physics restore run once before the class projection (``prepare_moving``), each class
    restores its projected words, and at a frame every class gathers the words it OWNS so class 0 runs the
    route's own frame, controller verdict, seal and carrier export on the assembled global state.

    Classes stand together only at a macro boundary, so every collective is taken there and nowhere else:
    the post-move shock series is one row per macro step (the surface-pressure tendency over the macro), as on
    the ArWen route's class executor (``drivers.fullstep_lts_forecast_v841._pressure_shock_row``).  Between
    frames a shock row gathers only the surface pressure and each class's largest |w|, not the pools.
    """

    def __init__(self, comm, prepared, partitions, execution, *, board, rate, top, base_dt, alarm_phase,
                 restored=None):
        from .moving_mpi_v842 import DistributedMoving
        self._inner = DistributedMoving(comm, prepared, partitions, execution)
        self.comm, self.prepared, self.partitions = comm, prepared, partitions
        self.execution, self.board = execution, board
        self.rate, self.top, self.base_dt = int(rate), int(top), float(base_dt)
        self.route = self._inner.route
        self.carrier = self._inner.carrier
        self._alarm_phase = alarm_phase
        self._receipt = self.route.receipt()
        self._restored = restored      # this class's projected carrier words, in memory (project_restored_words)
        self._maps = None              # the partitions' id maps, built once on class 0 (partition_maps)
        self._frame_snapshot = None    # (step, snapshot) of the last frame, reused by finish at the same step

    # The attributes runtime_v842's publication schedule reads (moving_publication_due, moving_columns_due).
    @property
    def control(self):
        return self.route.control

    @property
    def frame_seconds(self):
        return self.route.frame_seconds

    @property
    def offset_seconds(self):
        return self.route.offset_seconds

    @property
    def shock(self):
        # Columns are prepared for the shock row at macro boundaries only (the class step just completed).
        return bool(self.route.shock) and self._macro_boundary(self.execution.current_step)

    def _macro_boundary(self, step):
        return (int(step) * self.rate) % self.top == 0

    # The resume hooks: the global remap ran once before the class projection.
    def begin(self, start):
        self._inner.begin(start)

    def resume_dycore(self, **values):
        return self._inner.resume_dycore(**values)

    def restore_physics(self, session, layout, weights, mesh):
        if self._restored is None:
            return self._inner.restore_physics(session, layout, weights, mesh)
        from .moving_mpi_v842 import apply_restored
        restored, self._restored = self._restored, None
        apply_restored(session, restored)
        return self.prepared['restore_receipt']

    def restore_alarm_phase(self, session, config):
        """``MovingRoute.restore_alarm_phase`` on this class's session, from the carrier's alarm phase."""
        if self._alarm_phase is None:
            return None
        carried, elapsed = self._alarm_phase
        manager = getattr(getattr(session, 'driver', None), 'manager', None)
        if carried is None or manager is None or getattr(config, 'mode', 'exact') == 'exact':
            return None
        from fractions import Fraction
        elapsed = Fraction(elapsed)
        # Every class steps at its own rate * dt, and the class clock (require_class_clock) puts every physics
        # interval on a macro boundary, so the alarms ring at the same model seconds in every class: the carried
        # phase is compared by alarm and period, not by the exporting class's step.
        if set(carried['alarms']) != set(manager.alarms):
            raise ValueError("the checkpoint's physics alarms differ from this run's suite")
        for name, item in carried['alarms'].items():
            alarm = manager.alarms[name]
            if Fraction(item['period']) != alarm.period:
                raise ValueError(f"the checkpoint's {name} alarm interval differs from this run's")
            alarm.next_ring = Fraction(item['next_ring']) - elapsed
        return {name: str(alarm.next_ring) for name, alarm in manager.alarms.items()}

    # Owner assembly.
    def _owned_buffer(self, row, buffer):
        """A class pool buffer (every local entity solved) as its true class layout declares it."""
        partition = self.partitions[self.comm.Get_rank()]
        values = np.asarray(buffer)
        for axis, dim in enumerate(row['dimensions']):
            for entity, (_, owned_name) in _ENTITY_MAPS.items():
                if dim == entity + 'Solve':
                    values = np.take(values, np.arange(int(getattr(partition, owned_name))), axis=axis)
        return values

    def _snapshot(self, session, dycore, columns=None):
        from types import SimpleNamespace as NS
        from . import moving_v842 as mv
        from .moving_mpi_v842 import (HostArrays, ServiceWords, WindowWords, assemble_columns, gather_pool_words,
                                      snapshot_wanted)
        from .partition_v842 import partition_maps
        rank, layout = self.comm.Get_rank(), self.prepared['layout']
        if rank == 0 and self._maps is None:
            self._maps = partition_maps(layout, self.partitions)
        native = gather_pool_words(self.comm, layout, self.partitions, snapshot_wanted(self.route, layout),
            lambda row, level: self._owned_buffer(row, session.native.get_field(row['pool'], row['pool_name'],
                                                                                level=level)),
            maps=self._maps)
        state, saved = {}, {}
        for target, names, source in ((state, mv.STATE_FIELDS, dycore.atmosphere.state),
                                      (saved, mv.SAVED_FIELDS, dycore.atmosphere.saved)):
            for name in names:
                buffers = self.comm.gather(dycore.cp.asnumpy(getattr(source, name)), root=0)
                if rank == 0:
                    target[name] = assemble_columns(buffers, self.partitions,
                        entity='edge' if name in ('rho_u', 'normal_velocity') else 'cell')
        window = session.precipitation_window
        windows = self.comm.gather(None if window is None else window.snapshot(), root=0)
        services = self.comm.gather(None if session.surface_service is None else
                                    session.surface_service.restart_state(), root=0)
        psfc = self.comm.gather(None if columns is None else dycore.cp.asnumpy(columns.psfc_p), root=0)
        if rank != 0:
            return None
        if any(service != services[0] for service in services):
            raise ValueError('moving surface service calendars differ between rate classes')
        global_window = None
        if windows[0] is not None:
            global_window = {key: assemble_columns([w[key] for w in windows], self.partitions, entity='cell')
                             for key in windows[0]}
        # Class 0's physics alarms: their phase is in model seconds, the same in every class (restore_alarm_phase).
        proxy = NS(native=native, config=session.config, driver=NS(manager=getattr(
                       getattr(session, 'driver', None), 'manager', None)),
                   precipitation_window=None if global_window is None else WindowWords(global_window),
                   surface_service=None if services[0] is None else ServiceWords(services[0]))
        global_dycore = NS(cp=HostArrays, atmosphere=NS(state=NS(**state), saved=NS(**saved)))
        global_columns = None if psfc[0] is None else NS(psfc_p=assemble_columns(psfc, self.partitions, entity='cell'))
        return proxy, global_dycore, global_columns

    def _shock_snapshot(self, dycore, columns):
        """The shock row's inputs alone: the owned surface pressure and each class's largest owned |w|."""
        from types import SimpleNamespace as NS
        from .moving_mpi_v842 import HostArrays, assemble_columns
        rank = self.comm.Get_rank()
        owned = int(self.partitions[rank].n_owned_cells)
        w = dycore.atmosphere.saved.vertical_velocity
        peak = float(dycore.cp.abs(w[..., :owned]).max())
        rows = self.comm.gather((peak, None if columns is None else dycore.cp.asnumpy(columns.psfc_p)), root=0)
        if rank != 0:
            return None
        if any(row[1] is None for row in rows):
            raise RuntimeError('a rate class reached its macro boundary without prepared surface pressure')
        psfc = assemble_columns([row[1] for row in rows], self.partitions, entity='cell')
        return None, NS(cp=HostArrays, atmosphere=NS(saved=NS(vertical_velocity=np.asarray([r[0] for r in rows])))), \
            NS(psfc_p=psfc)

    def after_step(self, step, **values):
        """Frames, verdicts, seals and the shock row at macro boundaries; True stops every class."""
        if not self._macro_boundary(step):
            return False
        route = self.route
        dt = float(values['dt'])
        elapsed = int(step) * dt
        model_seconds = route.offset_seconds + elapsed
        frame = route.control is not None and not model_seconds % route.frame_seconds
        if not frame and not route.shock:
            return False
        macro_dt = self.top * self.base_dt
        macro, macros = round(elapsed / macro_dt), round(int(values['steps']) * dt / macro_dt)
        snapshot = (self._snapshot(values['session'], values['dycore'], values['columns']) if frame else
                    self._shock_snapshot(values['dycore'], values['columns']))
        self._frame_snapshot = (int(step), snapshot) if frame else None
        result = False
        if self.comm.Get_rank() == 0:
            session, dycore, columns = snapshot
            result = route.after_step(macro, **{**values, 'steps': macros, 'dt': macro_dt, 'session': session,
                                               'dycore': dycore, 'columns': columns, 'mesh': self.prepared['mesh'],
                                               'vertical': self.prepared['vertical'],
                                               'layout': self.prepared['layout']})
        result, self._receipt = self.comm.bcast((bool(result), route.receipt()), root=0)
        if result:
            with self.board.lock:
                self.board.stop_requested = model_seconds
                self.board.lock.notify_all()
        return result

    def finish(self, **values):
        if not self.route.export_path:
            return
        held, self._frame_snapshot = self._frame_snapshot, None
        if held is not None and held[0] == int(values['step']):
            # The run stopped on a frame: nothing touched the state since that frame's assembly (every class takes
            # this branch together, so no collective is skipped by one class alone).  A carrier export used to
            # gather the same words a second time (about 11 s on the storm-following 3 km region).
            snapshot = held[1]
        else:
            snapshot = self._snapshot(values['session'], values['dycore'])
        if self.comm.Get_rank() == 0:
            session, dycore, _ = snapshot
            self.route.finish(**{**values, 'session': session, 'dycore': dycore, 'mesh': self.prepared['mesh'],
                                 'vertical': self.prepared['vertical'], 'layout': self.prepared['layout']})
        self._receipt = self.comm.bcast(self.route.receipt(), root=0)

    def receipt(self):
        return self._receipt


def lts_requested(arguments) -> bool:
    """Active full-step LTS on this route (the shadow is an observer on the fixed step, runtime_v842)."""
    return (bool(getattr(arguments, 'fullstep_local_timestep', False))
            and getattr(arguments, 'fullstep_lts_mode', None) == 'active')


def shadow_requested(arguments) -> bool:
    return (bool(getattr(arguments, 'fullstep_local_timestep', False))
            and getattr(arguments, 'fullstep_lts_mode', None) in (None, 'shadow'))


def attach_shadow(arguments, dycore, out):
    """The ArWen route's full-step shadow on this route's dycore (the same driver class), or None.

    The shadow is the measuring instrument of ``--fullstep-local-timestep`` without ``--fullstep-lts-mode
    active``: the fixed step runs unchanged and an observer of its scalar transport records the rate census
    (``hexcore.cuda_fullstep_lts_shadow``).  Rows go to ``fullstep-lts-stages.jsonl``, the summary to the
    receipt, as on the ArWen route (drivers/run_cuda_v841_forecast).  A mesh whose geometry gives one class
    attaches nothing, as there.
    """
    from netCDF4 import Dataset
    from ..config_fullstep_lts import validate_fullstep_lts_request
    from ..fullstep_lts_dispatch import attach_fullstep_lts
    config = dycore.config
    policy = validate_fullstep_lts_request(SimpleNamespace(
        fullstep_local_timestep=True, fullstep_lts_mode='shadow',
        # The forecast door's default top rate (ForecastRequest.fullstep_lts_max_rate = 8), as on the ArWen route;
        # the observer's own dataclass default (16) would count a class neither route steps.
        fullstep_lts_max_rate=getattr(arguments, 'fullstep_lts_max_rate', None) or 8,
        fullstep_lts_buffer_rings=getattr(arguments, 'fullstep_lts_buffer_rings', None),
        time_integration_order=int(config.config_time_integration_order),
        acoustic_substeps=int(config.config_number_of_sub_steps),
        dynamics_splits=int(config.config_dynamics_split_steps),
        monotonic=bool(config.config_monotonic)))
    grid = Path(getattr(arguments, 'grid', None) or arguments.init)
    with Dataset(grid) as dataset:
        ids = np.asarray(dataset['indexToCellID'][:], dtype=np.int64)
    sink_path = Path(out) / 'fullstep-lts-stages.jsonl'

    def sink(row):
        with sink_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    shadow = attach_fullstep_lts(dycore, policy, sink=sink, global_cell_ids=ids)
    if shadow is not None:
        state = dycore.atmosphere.state
        shadow.observe(state.rho, state.rho_u, state.rho_w, dycore.atmosphere.vertical.rdzw, stage=0,
                       stage_dt=float(config.config_dt), time_seconds=float(state.time_seconds))
    else:
        print(json.dumps({'fullstep_lts': {'route': 'mpas-suite', 'mode': 'shadow',
                                           'dispatch': 'one geometric class: nothing to observe'}}), flush=True)
    return shadow


def admission_refusal(arguments, layout, config, moving=None) -> str | None:
    """What this executor cannot carry yet, each named with its breakage; None when it runs."""
    if getattr(arguments, 'samehost_devices', None):
        return ("active local time stepping on the MPAS suite route runs on one card: its classes rendezvous on one "
                "device; several cards need the ArWen route's class blocks across cards, which carry no MPAS "
                "physics pools yet.  Drop --samehost-devices (one card), or run the classes' route on several cards "
                "without LTS")
    if getattr(config, 'table_key', 'mpasphys_v842') != 'mpasphys_v842':
        return ('active local time stepping on the MPAS suite route covers the MPAS-A v8.4.2 dycore row: a forked '
                'dycore (RRFS) steps a different driver whose halo seams the class coordinator has not been checked on')
    if getattr(arguments, 'lbc_dir', None) or layout.dimensions.get('nCells') != layout.dimensions.get('nCellsSolve'):
        return ('active local time stepping on the MPAS suite route runs global and closed variable meshes: a '
                'limited-area mesh needs the regional garbage column replicated in every class and the boundary '
                'zone held at one rate, which this executor does not build yet')
    for name, why in (('restart_from', 'restart'), ('da_analysis', 'DA cycling'),
                      ('iau_window_seconds', 'IAU'), ('jedi_background', 'the JEDI background'),
                      ('chemistry_config', 'chemistry'), ('balance_series', 'the balance series'),
                      ('nest_triggers', 'nest triggers')):
        if getattr(arguments, name, None):
            return (f'{why} with active local time stepping on the MPAS suite route is not built: each class '
                    'session holds its own clock, pools and samplers, and no class carries it across classes yet')
    if getattr(arguments, 'stochastic', None) not in (None, '', 'off', 'none'):
        from ..stochastic import resolve_options
        if resolve_options(arguments.stochastic, None).enabled:
            return ('stochastic forcing with active local time stepping on the MPAS suite route is not built: the '
                    'pattern is drawn per step and the classes step at different rates (the ArWen route has a '
                    'class-aware generator, LtsStochasticPhysics, that this route does not call yet)')
    if str(layout.configs.get('config_IAU_option', 'off')).strip() != 'off':
        return 'IAU with active local time stepping on the MPAS suite route is not built'
    if layout.configs.get('config_do_restart') and not (moving is not None and moving.carrier is not None):
        # A resumed moving segment is a restart by its carrier (moving_v842.mark_restart): the remapped words come
        # from the carrier through prepare_moving, never from a native restart file.
        return 'a native-file restart with active local time stepping on the MPAS suite route is not built'
    if layout.configs.get('config_sst_update'):
        return ('timed surface updating (config_sst_update) with active local time stepping on the MPAS suite route '
                'is not built: each class mesh needs its own projection of the surface record')
    return None


def _event_seconds(layout, streams, steps, dt):
    """Every global event the class clock must stand on: run length, physics intervals, stream intervals."""
    from .driver_v842 import interval_seconds
    events = {'run length': float(steps) * float(dt)}
    for key, value in sorted(layout.configs.items()):
        if not key.endswith('_interval') or not isinstance(value, str):
            continue
        try:
            seconds = interval_seconds(value)
        except ValueError:
            continue    # not a fixed interval (a calendar month); the session refuses it by itself
        if seconds:
            events[key] = float(seconds)
    for stream in streams:
        if 'output' in stream.directions and stream.output_interval not in (None, 'none', 'initial_only'):
            seconds = interval_seconds(stream.output_interval)
            if seconds:
                events[f'{stream.name} output interval'] = float(seconds)
    return events


def run_lts_classes(arguments, config, layout, payload, streams, start, stop, dt, steps, out, *, moving=None):
    """The MPAS-physics route with active full-step local time stepping on one card.  Returns 0.

    ``moving`` (a plain ``moving_v842.MovingRoute``, ``moving_admitted``) runs a moving segment on the classes:
    the carrier is remapped and its physics restored once on the global mesh before the class projection, and
    the frames, verdicts, seals, export and shock series are taken at macro boundaries (``ClassMoving``).
    """
    import cupy as cp
    from . import runtime_v842 as runtime
    from .mpi_v842 import local_inputs
    from .partition_v842 import project_payloads
    from ..cuda_backend import KernelCache, require_cuda
    from ..cuda_fullstep_lts_active import FullStepLtsActive
    from ..drivers.fullstep_lts_forecast_v841 import _boundary_rate
    from ..fullstep_lts_admission import admit_mesh_rates, metric_lengths, require_class_clock
    from ..fullstep_lts_classes import build_class_plan
    from ..physics_suites import load_table

    refusal = admission_refusal(arguments, layout, config, moving)
    if refusal is not None:
        raise ValueError(refusal)
    out = Path(out).resolve()
    payload = Path(payload).resolve()
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    # Hydrating the global inputs resolves config_len_disp in place (pool_v842.horizontal_length); a one-card run
    # snapshots its output attributes before that, so each class starts from these words (samehost_v842).
    configs_before = dict(layout.configs)
    began = time.perf_counter()
    inputs = runtime.load_native_dynamics_inputs(arguments, layout, payload,
                                                 keep_initialized_deformation=row.get('dycore') is not None)
    mesh, grid, profiles, state, reference, saved, names = inputs
    declarations = dict(getattr(mesh, 'variable_dimensions', {}) or {})
    for name in ('defc_a', 'defc_b'):
        if name in mesh.arrays:
            declarations.setdefault(name, ('nCells', 'maxEdges'))
    mesh.variable_dimensions = declarations
    moving_state = alarm_phase = None
    if moving is not None:
        if not moving_admitted(moving):
            raise ValueError('a native checkpoint or restart route does not ride the LTS class executor')
        from .moving_mpi_v842 import prepare_moving
        if moving.carrier is not None:
            carried = moving.carrier.get('physics') or {}
            alarm_phase = (carried.get('manager'), carried.get('manager_elapsed', '0'))
        # The remapped resume (atmosphere and physics) on the global mesh, as the same-host ranks take it.
        inputs, moving_state = prepare_moving(moving, inputs, config, layout, payload, arguments)
        mesh, grid, profiles, state, reference, saved, names = inputs
    base_dt = float(dt)
    # The ArWen route's full-physics top-rate default and its reason (drivers/fullstep_lts_forecast_v841).
    max_rate = int(getattr(arguments, 'fullstep_lts_max_rate', None) or 8)
    transports = bool(layout.configs.get('config_scalar_advection', True))
    rates, rate_info = admit_mesh_rates(
        mesh, state, saved, grid.vertical_grid, dt=base_dt, max_rate=max_rate,
        acoustic_substeps=int(layout.configs['config_number_of_sub_steps']),
        target_transport=0.90 if transports else None, boundary_rate=_boundary_rate(max_rate))
    if int(rates.max()) == 1:
        # One class: the ordinary route, unchanged (a uniform mesh "on" is "off", as on the ArWen route).
        print(json.dumps({'fullstep_lts': {'route': 'mpas-suite', 'dispatch': 'native single-class step',
                                           'rate_assignment': rate_info}}, default=str), flush=True)
        return runtime.run_native_dynamics(arguments, config, layout, payload, streams, None,
                                           start, stop, dt, steps, out)
    top = int(rates.max())
    clock = require_class_clock(base_dt, [top], _event_seconds(layout, streams, steps, dt))
    if steps % top:
        raise ValueError(f'the run is {steps} base steps, not a whole number of {top}-step macros: give --hours a '
                         f'multiple of {top * base_dt} s')
    plan = build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays['cellsOnCell']),
        n_edges_on_cell=np.asarray(mesh.arrays['nEdgesOnCell']),
        cells_on_edge=np.asarray(mesh.arrays['cellsOnEdge']),
        cells_on_vertex=np.asarray(mesh.arrays['cellsOnVertex']),
        area_cell=np.asarray(mesh.area_cell, dtype=np.float64),
        dc_edge_m=metric_lengths(mesh)[0], base_dt=base_dt,
        edges_on_edge=mesh.arrays.get('edgesOnEdge'), n_edges_on_edge=mesh.arrays.get('nEdgesOnEdge'))
    partitions = [tables.layout for tables in plan.classes]
    print(json.dumps({'fullstep_lts': {'route': 'mpas-suite', 'rate_assignment': rate_info,
                                       'plan': plan.receipt(), 'clock': clock}}, default=str), flush=True)
    job = out / 'lts-classes'
    job.mkdir(parents=True, exist_ok=False)
    from concurrent.futures import ThreadPoolExecutor

    # Every class's payload in one pass, each field on its own worker (partition_v842.project_payloads), and the
    # restored carrier words in memory (moving_mpi_v842.project_restored_words): no per-class serial projection
    # and no pickle round trip through the class payload.
    class_partitions = [session_partition(partition) for partition in partitions]
    local_layouts = project_payloads(layout, payload, class_partitions,
                                     [job / f'pool.class{index}' for index in range(len(partitions))])
    restored_words = [None] * len(partitions)
    if moving_state is not None:
        from .moving_mpi_v842 import project_restored_words
        restored_words = project_restored_words(moving_state, layout, class_partitions)

    def prepare(index):
        local_layouts[index].configs = dict(configs_before)
        return local_layouts[index], job / f'pool.class{index}', local_inputs(inputs, partitions[index])

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        prepared = list(pool.map(prepare, range(len(partitions))))
    preparation_seconds = time.perf_counter() - began
    board = _Board(len(partitions))
    executions = [LtsClassExecution(board, index, plan.classes[index].rate, top, partitions, prepared[index][2],
                                    layout) for index in range(len(partitions))]
    results: list = [None] * len(partitions)
    comms = None
    if moving_state is not None:
        from .samehost_v842 import ThreadComm
        comms = ThreadComm.world(len(partitions))
    device = cp.cuda.Device()
    before = Path.cwd()
    os.chdir(out)   # native_working_directory is shared by the class threads

    def class_thread(index):
        local_layout, local_payload, _ = prepared[index]
        rate = plan.classes[index].rate
        try:
            class_moving = None if comms is None else ClassMoving(
                comms[index], moving_state, partitions, executions[index], board=board, rate=rate, top=top,
                base_dt=base_dt, alarm_phase=alarm_phase, restored=restored_words[index])
            with device:
                results[index] = runtime.run_native_dynamics(
                    arguments, config, local_layout, local_payload, streams, None, start, stop,
                    base_dt * rate, steps // rate, out, moving=class_moving, execution=executions[index])
            if results[index]:
                raise RuntimeError(f'class {index} returned {results[index]}')
        except BaseException as error:
            import traceback
            results[index] = error
            if comms is not None:
                comms[index].abort(f'class {index}: {type(error).__name__}: {error}')
            board.fail(f'class {index}: {type(error).__name__}: {error}\n' + ''.join(
                traceback.format_exception(error))[-4000:])
        finally:
            with board.lock:
                board.parked[index] = True
                board.lock.notify_all()

    threads = [threading.Thread(target=class_thread, args=(index,), name=f'mpas-lts-class{index}', daemon=True)
               for index in range(len(partitions))]
    receipt = dict(schema='mpas-lts-launch-v1', completed=False, classes=len(partitions),
                   rates=[int(tables.rate) for tables in plan.classes], rate_assignment=rate_info,
                   clock=clock, preparation_seconds=preparation_seconds,
                   preparation_parallelism=f'{len(partitions)} host projection threads; before class CUDA work',
                   owned_cells=[int(p.n_owned_cells) for p in partitions],
                   local_cells=[int(len(p.cell_l2g)) for p in partitions],
                   parallelism=f'{len(partitions)} class threads on one GPU; one class integrates at a time')
    macro_rows = []
    failure = None
    try:
        for thread in threads:
            thread.start()
        board.wait(lambda: len(board.built) == len(partitions), 'every class dycore')
        capability = require_cuda(min_compute=(7, 5))
        shared_cache = KernelCache(capability=capability)
        area = np.asarray(mesh.area_cell, dtype=np.float64)
        areas = [area[p.cell_l2g.astype(np.int64)] for p in partitions]
        dzw = np.asarray(grid.vertical_grid.dzw, dtype=np.float64)
        drivers = [board.built[index] for index in range(len(partitions))]
        active = FullStepLtsActive(plan, drivers, base_dt=base_dt, kernel_cache=shared_cache,
                                   area_cell=areas, dzw=dzw)

        def stepper(klass):
            index = klass.index
            # The class reads the halo the coordinator just filled (round A) on its own thread.
            _device_synchronize()
            with board.lock:
                target = board.done[index] + 1
                board.go[index] = target
                board.lock.notify_all()
            board.wait(lambda: board.done[index] >= target, f'class {index} to finish step {target}')

        active.stepper = stepper
        _device_synchronize()
        with board.lock:
            board.klass = active.classes
            board.ready = True
            board.macro_seconds = 0.0
            board.lock.notify_all()
        run_began = time.perf_counter()
        for macro in range(steps // top):
            board.wait(lambda: all(board.parked), 'every class at the macro boundary')
            if board.stop_requested is not None:
                # The moving controller stopped the segment at this frame; every class sealed and left its loop.
                break
            mark = time.perf_counter()
            active.advance_macro(top)
            _device_synchronize()
            macro_rows.append(dict(macro=macro + 1, base_steps=top, seconds=time.perf_counter() - mark))
            with board.lock:
                board.macro_seconds = float((macro + 1) * top) * base_dt
                board.lock.notify_all()
        for thread in threads:
            thread.join(timeout=WAIT_SECONDS)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError('an LTS class thread did not finish')
        warm = macro_rows[1:]
        receipt.update(completed=True, execution_seconds=time.perf_counter() - run_began,
                       moving=None if moving is None else dict(stopped_at_model_seconds=board.stop_requested,
                                                               hooks='class owner assembly at macro boundaries'),
                       macro_seconds_mean=(sum(r['seconds'] for r in warm) / len(warm) if warm else None),
                       macros=len(macro_rows), events=dict(active.events),
                       registers_empty=bool(active.registers_empty()),
                       warm_seconds_per_forecast_hour=(
                           sum(r['seconds'] for r in warm) / len(warm) * 3600.0 / (top * base_dt)
                           if warm else None),
                       active=active.receipt())
    except BaseException as error:
        failure = error
        board.fail(f'coordinator: {type(error).__name__}: {error}')
        receipt.update(error=f'{type(error).__name__}: {error}', macros=len(macro_rows))
        for thread in threads:
            thread.join(timeout=60)
    finally:
        os.chdir(before)
    (out / 'mpas-lts-receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    if failure is not None:
        # The first class that failed by itself is the cause; the rest stopped because of it.
        causes = [r for r in results if isinstance(r, BaseException) and not isinstance(r, LtsAborted)]
        if causes and isinstance(failure, LtsAborted):
            raise causes[0] from failure
        raise failure
    classes = []
    for index in range(len(partitions)):
        path = executions[index].receipt_path(out)
        classes.append(json.loads(path.read_text()) if path.is_file() else None)
    if classes[0] is not None:
        combined = dict(classes[0], fullstep_lts=receipt, class_receipts=[
            None if c is None else {k: c.get(k) for k in ('steps_completed', 'completed', 'rank_execution',
                                                          'stage_timings')} for c in classes],
            completed=all(c is not None and c.get('completed') for c in classes))
        (out / 'native-physics-receipt.json').write_text(json.dumps(combined, indent=2, default=str) + '\n')
    import shutil
    shutil.rmtree(job, ignore_errors=True)
    return 0
