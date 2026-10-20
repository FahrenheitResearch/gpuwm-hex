"""Native-route feature dispatch; leaves the shared scheme runtime in charge."""
from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

from .cycling_v842 import recouple_analysis, write_background


def background_due(arguments, step, steps, dt):
    if not getattr(arguments, 'jedi_background', False):
        return False
    interval = getattr(arguments, 'history_every_minutes', None)
    if interval is not None and float(interval) <= 0:
        raise ValueError('native JEDI history interval must be positive')
    return step == steps or (interval is not None and (step * dt) % (float(interval) * 60) == 0)


class CyclingExecution:
    """The serial runtime extension, with no halo or rank receipt changes."""

    def __init__(self, arguments, layout, inputs):
        self.arguments, self.layout, self.inputs = arguments, layout, inputs
        self.backgrounds = []
        self.analysis_receipt = None

    def couple_input(self, state, saved, mesh, grid, reference):
        values = (mesh, grid, self.inputs[2], state, reference, saved, self.inputs[6])
        inputs, self.analysis_receipt = recouple_analysis(self.arguments, self.layout, values,
            work_dir=Path(self.arguments.out) / 'native-da-cycling')
        return inputs[3], inputs[5]

    def bind_dycore(self, dycore):
        pass

    def surface_reader(self, factory, library, path, layout):
        return factory(library, path, n_cells=layout.dimensions['nCells'])

    def bind_coupling(self, coupling):
        pass

    def before_step(self, step):
        pass

    def initialized(self, dycore):
        pass

    def after_step(self, dycore):
        pass

    def receipt_path(self, out):
        return Path(out) / 'native-physics-receipt.json'

    def calls_path(self, out):
        return Path(out) / 'physics-calls.jsonl'

    def publication_required(self, step, steps, dt):
        return background_due(self.arguments, step, steps, dt)

    def output(self, *, step, steps, dt, session, layout, names, out):
        if background_due(self.arguments, step, steps, dt):
            began = time.perf_counter()
            record = write_background(self.arguments, session, layout, names,
                                       step=step, dt=dt, out=out)
            self.backgrounds.append(dict(step=step, path=record.path,
                receipt=record.receipt_path, sha256=record.sha256,
                seconds=time.perf_counter() - began, parallelism='Rust CDF-5 writer'))

    def receipt(self):
        return dict(schema='mpas-native-cycling-v1', analysis=self.analysis_receipt,
                    backgrounds=self.backgrounds)


class MpiCycling(CyclingExecution):
    """Owner-only assembly before a single global CUDA JEDI background write."""

    def __init__(self, arguments, layout, execution):
        super().__init__(arguments, layout, execution.inputs)
        self.execution = execution

    def output(self, *, step, steps, dt, session, layout, names, out):
        if not background_due(self.arguments, step, steps, dt):
            return
        import numpy as np
        from ..jedi_contract import BACKGROUND_FIELDS
        from .moving_mpi_v842 import PoolWords
        from .partition_v842 import gather_owned_field
        began = time.perf_counter()
        execution, global_layout = self.execution, self.layout
        native = PoolWords(global_layout)
        for row in global_layout.fields:
            if row['name'] not in BACKGROUND_FIELDS:
                continue
            values = session.native.get_field(row['pool'], row['pool_name'])
            buffers = execution.comm.gather(values, root=0)
            if execution.rank == 0:
                words = gather_owned_field(row, global_layout, execution.partitions, buffers)
                if row['dimensions'] and row['dimensions'][-1] in ('nCells', 'nEdges', 'nVertices'):
                    words = np.concatenate((words, np.zeros((*words.shape[:-1], 1), words.dtype)), axis=-1)
                native.put_field(row['pool'], row['pool_name'], words)
        record = None
        if execution.rank == 0:
            result = write_background(self.arguments, SimpleNamespace(native=native),
                global_layout, names, step=step, dt=dt, out=out)
            record = dict(step=step, path=result.path, receipt=result.receipt_path,
                sha256=result.sha256, seconds=time.perf_counter() - began,
                parallelism=f'{len(execution.partitions)} MPI owner gathers; Rust CDF-5 writer')
        self.backgrounds.append(execution.comm.bcast(record, root=0))


def run_native_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out, *, moving=None):
    """Dispatch only requested native features; ordinary native calls are unchanged."""
    if getattr(arguments, 'samehost_devices', None):
        from .launch_v842 import run_native_ranks
        return run_native_ranks(arguments, config, layout, payload, streams,
            library, start, stop, dt, steps, out, moving=moving)
    from . import runtime_v842 as runtime
    execution = None
    bind_inputs = getattr(moving, 'bind_inputs', None)
    if (getattr(arguments, 'jedi_background', False) or getattr(arguments, 'da_analysis', None)
            or bind_inputs is not None):
        import inspect
        if 'execution' not in inspect.signature(runtime.run_native_dynamics).parameters:
            raise RuntimeError('native runtime lacks the RankExecution extension seam')
        from ..physics_suites import load_table
        row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
        inputs = runtime.load_native_dynamics_inputs(arguments, layout, payload,
            keep_initialized_deformation=row.get('dycore') is not None)
        if bind_inputs is not None:
            bind_inputs(inputs, config)
        execution = CyclingExecution(arguments, layout, inputs)
    options = dict(moving=moving)
    if execution is not None:
        options['execution'] = execution
    return runtime.run_native_dynamics(arguments, config, layout, payload, streams,
        library, start, stop, dt, steps, out, **options)
