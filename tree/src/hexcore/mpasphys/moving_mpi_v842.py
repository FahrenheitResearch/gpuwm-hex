"""Owner assembly at MPAS moving boundaries, using the existing carrier.

Remapping remains in moving_v842. This adapter assembles the global carrier
at a frame, and projects its restored words before the next rank launch.
"""
from __future__ import annotations

import os
import pickle
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import moving_v842 as mv
from .partition_v842 import gather_owned_field, project_buffer
from .pool_v842 import NativePoolHydration


class PoolWords:
    """Host pool storage for the existing remapper and carrier writer."""

    def __init__(self, layout, hydration=None):
        self.layout, self.hydration = layout, hydration
        self.data = {}
        #: Keys put_field wrote: the carrier words a moving resume hands to the ranks/classes.
        self.written = set()
        self._fields = {(row['pool'], row['pool_name']):
            (np.dtype('<f4' if row['kind'] == 'real' else '<i4'),
             tuple(row['shape']), row['nlevels'])
            for row in layout.fields if row['kind'] != 'character'}

    def get_field(self, pool, name, *, level=1):
        key = (pool, name, level)
        if key not in self.data:
            if self.hydration is None:
                raise ValueError(f'boundary did not assemble {key}')
            row = next(row for row in self.layout.fields
                       if (row['pool'], row['pool_name']) == (pool, name))
            self.data[key] = self.hydration.field_buffer(pool, row['name']).copy(order='F')
        return self.data[key]

    def put_field(self, pool, name, value, *, level=1, nlevels=1):
        self.data[pool, name, level] = np.array(value, copy=True, order='F')
        self.written.add((pool, name, level))

    def written_words(self):
        """Only the words put_field wrote, never a field get_field merely read from the member's own inputs.

        THE BREAKAGE THIS PREVENTS (box-aws2x 2026-10-24, MPAS-route storm-following, member statics from a
        WPS_GEOG with MODIS lakes): restore_physics reads the target's land-use classes from the hydrated
        static, so ``data`` held its raw ivgtyp with class 21. Handed over as carrier words, it overwrote the
        ranks'/classes' lake fold (runtime_v842.fold_lake_class) and every moved segment died at its first
        Noah call: "MPAS Noah native fatal condition at source line 2441" (VEGTYP > LUCATS), three of three
        retries.
        """
        return {key: value for key, value in self.data.items() if key in self.written}


class WindowWords:
    def __init__(self, data=None):
        self.data = data

    def restore(self, data):
        self.data = {key: np.array(value, copy=True) for key, value in data.items()}

    def snapshot(self):
        return self.data


class ServiceWords:
    def __init__(self, data=None):
        self.data = data

    def restore_restart_state(self, data):
        self.data = dict(data)

    def restart_state(self):
        return self.data


class HostArrays:
    """Carrier/frame API over already assembled host arrays."""
    asnumpy = staticmethod(np.asarray)
    abs = staticmethod(np.abs)


def prepare_moving(route, inputs, config, layout, payload, arguments):
    mesh, grid, profiles, state, reference, saved, names = inputs
    bind_inputs = getattr(route, 'bind_inputs', None)
    if bind_inputs is not None:
        bind_inputs(inputs, config)
    route.begin(layout.configs['config_start_time'] if route.carrier is None else None)
    native = PoolWords(layout, NativePoolHydration(None, layout, payload))
    session = SimpleNamespace(native=native, config=config, precipitation_window=None,
                              surface_service=None)
    restored = None
    if route.carrier is not None:
        state, saved, weights = route.resume_dycore(mesh=mesh, grid=grid,
            reference=reference, state=state, saved=saved, names=names,
            target_grid=getattr(arguments, 'grid', None) or arguments.init)
        physics = route.carrier['physics']
        if physics.get('window') is not None:
            session.precipitation_window = WindowWords()
        if physics.get('surface_service') is not None:
            session.surface_service = ServiceWords()
        restored = route.restore_physics(session, layout, weights, mesh)
    result = (mesh, grid, profiles, state, reference, saved, names)
    # MovingControl holds a thread lock. Recreate it in each worker rather
    # than pickling that lock, or retaining the complete source carrier.
    control = (None if route.control is None else
        dict(directory=route.control.directory, seal_every_seconds=route.control.seal_every_seconds))
    worker_route = copy(route)
    worker_route.control = None
    worker_route.carrier = None if route.carrier is None else {}
    return result, dict(route=worker_route, control=control, layout=layout, mesh=mesh, vertical=grid.vertical_grid,
        restored=native.written_words() if route.carrier is not None else {},
        restore_receipt=restored,
        window=None if session.precipitation_window is None else session.precipitation_window.snapshot(),
        surface_service=None if session.surface_service is None else session.surface_service.restart_state())


def project_restored_words(moving, layout, partitions, *, workers=None):
    """The restored carrier words of every partition, in memory, every field on its own worker.

    THE COST THIS REMOVES: the MPAS LTS class executor projected the restored pools class after class, field
    after field (each call rebuilding the partition's sorted id maps), then pickled them to disk for the class
    to read back: 16 s of a resumed storm-following segment's preparation (MPAS-ROUTE-SPEED-2026-10-24).
    """
    from concurrent.futures import ThreadPoolExecutor
    from .partition_v842 import _maps, _project_buffer
    by_key = {(r['pool'], r['pool_name']): r for r in layout.fields}
    maps = [_maps(layout, partition) for partition in partitions]
    keys = list(moving['restored'])

    def project(key):
        return [_project_buffer(moving['restored'][key], by_key[key[:2]], layout, mapping) for mapping in maps]

    with ThreadPoolExecutor(max_workers=max(1, min(len(keys), workers or os.cpu_count() or 1))) as pool:
        projected = list(pool.map(project, keys))
    window = moving['window']
    result = []
    for which, partition in enumerate(partitions):
        owned = None if window is None else {
            key: np.ascontiguousarray(value[partition.cell_l2g[:partition.n_owned_cells]])
            for key, value in window.items()}
        result.append(dict(pools={key: words[which] for key, words in zip(keys, projected)}, window=owned,
                           service=moving['surface_service']))
    return result


def project_restored_pools(moving, layout, partition, local_layout, payload):
    restored = project_restored_words(moving, layout, [partition])[0]
    with (Path(payload) / 'moving-restore.pickle').open('wb') as handle:
        pickle.dump(restored, handle, protocol=5)


def apply_restored(session, restored):
    """Write a rank's projected carrier words into its session (DistributedMoving.restore_physics)."""
    for (pool, name, level), value in restored['pools'].items():
        session.native.put_field(pool, name, value, level=level,
                                 nlevels=session.native._fields[pool, name][2])
    window = session.precipitation_window
    if (window is None) != (restored['window'] is None):
        raise ValueError('moving precipitation window differs on native rank')
    if window is not None:
        window.restore(restored['window'])
    if session.surface_service is not None and restored['service'] is not None:
        session.surface_service.restore_restart_state(restored['service'])


def assemble_columns(buffers, partitions, *, entity):
    """Assemble a CUDA carrier field without ever consulting a halo value."""
    if not partitions or len(buffers) != len(partitions):
        raise ValueError('moving assembly needs one buffer per rank')
    ids_name, owned_name, total_name = {
        'cell': ('cell_l2g', 'n_owned_cells', '_global_cells'),
        'edge': ('edge_l2g', 'n_owned_edges', '_global_edges'),
    }[entity]
    first = np.asarray(buffers[0])
    total = getattr(partitions[0], total_name)
    result = np.empty((*first.shape[:-1], total), dtype=first.dtype)
    seen = np.zeros(total, dtype=bool)
    for buffer, partition in zip(buffers, partitions):
        ids, owned = getattr(partition, ids_name), getattr(partition, owned_name)
        if (buffer.shape[:-1] != first.shape[:-1] or buffer.dtype != first.dtype or
                buffer.shape[-1] not in (owned, len(ids), len(ids) + 1)):
            raise ValueError('moving rank field shape/dtype differs')
        ids = ids[:owned]
        if np.any(seen[ids]):
            raise ValueError('moving rank field has duplicate owners')
        result[..., ids] = buffer[..., :owned]
        seen[ids] = True
    if not np.all(seen):
        raise ValueError('moving rank field lacks global coverage')
    return result


def snapshot_wanted(route, layout):
    """The pool fields a moving frame, verdict, seal or carrier export reads (Registry (pool, name) pairs)."""
    wanted = {(r['pool'], r['name']) for r in route.carried_rows(layout)}
    wanted.update(('sfc_input', name) for name in (*mv.CATEGORY_FIELDS, 'isice_lu'))
    wanted.update(('diag', name) for name in
                  ('pressure', 'theta', 'uReconstructZonal', 'uReconstructMeridional'))
    wanted.update(('diag_physics', name) for name in ('u10', 'v10'))
    # The tracker frame's setup-target fields (moving_v842.write_tracker_frame: qv, refl10cm_max, t2m, q2).
    # THE BREAKAGE THIS PREVENTS: an assembled frame without them raised "boundary did not assemble" on the
    # same-host ranks and the LTS classes once the tracker frame started writing them (rel/hex-0.4.5 f2c58ed0).
    wanted.update(('diag_physics', name) for name in ('refl10cm_max', 't2m', 'q2'))
    wanted.add(('state', 'scalars'))
    return wanted


def gather_pool_words(comm, layout, partitions, wanted, local_field, *, maps=None):
    """Every rank's owned words of the wanted pool fields in ONE collective; rank 0 assembles them on all cores.

    ``local_field(row, level)`` is this rank's buffer as gather_owned_field takes it.  Returns the assembled
    PoolWords on rank 0 (carrier ABI: native dummy storage appended), None elsewhere.
    THE BREAKAGE THIS PREVENTS: the frame and the carrier export gathered field by field and level by level (two
    barriers each) and rebuilt every partition's sorted id maps per call, about 11 s per frame on the
    storm-following 3 km region (23 s on a free-threaded host), three times a moving segment on the MPAS route
    against nothing on the ArWen route (MPAS-ROUTE-SPEED-2026-10-24).  The words written are the same.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .partition_v842 import partition_maps
    keys = [(row, level) for row in layout.fields if (row['pool'], row['name']) in wanted
            for level in range(1, row['nlevels'] + 1)]
    gathered = comm.gather([local_field(row, level) for row, level in keys], root=0)
    if comm.Get_rank() != 0:
        return None
    maps = partition_maps(layout, partitions) if maps is None else maps

    def assemble(index):
        row, _ = keys[index]
        words = gather_owned_field(row, layout, partitions, [ranks[index] for ranks in gathered], maps=maps)
        # Carrier ABI includes native dummy storage. Its value is disposable and no solved column reads it.
        if row['dimensions'] and row['dimensions'][-1] in ('nCells', 'nEdges', 'nVertices'):
            words = np.concatenate((words, np.zeros((*words.shape[:-1], 1), words.dtype)), axis=-1)
        return words

    native = PoolWords(layout)
    with ThreadPoolExecutor(max_workers=max(1, min(len(keys), os.cpu_count() or 1))) as pool:
        for (row, level), words in zip(keys, pool.map(assemble, range(len(keys)))):
            native.data[row['pool'], row['pool_name'], level] = np.asfortranarray(words)
    return native


class DistributedMoving:
    """All ranks enter a frame; rank zero alone runs the moving controller."""

    def __init__(self, comm, prepared, partitions, execution):
        self.comm, self.prepared, self.partitions = comm, prepared, partitions
        self.execution = execution
        self.route = prepared['route']
        if prepared['control'] is not None:
            from ..moving_control import MovingControl
            self.route.control = MovingControl(**prepared['control'])
        self.carrier = self.route.carrier
        self._receipt = self.route.receipt()

    def begin(self, start):
        self.route.begin(start)

    def resume_dycore(self, **values):
        # Global remapping ran once before rank projection and CUDA upload.
        return values['state'], values['saved'], None

    def restore_physics(self, session, layout, weights, mesh):
        with (session.hydration.payload / 'moving-restore.pickle').open('rb') as handle:
            restored = pickle.load(handle)
        apply_restored(session, restored)
        return self.prepared['restore_receipt']

    def _snapshot(self, session, dycore, columns=None):
        rank, layout = self.comm.Get_rank(), self.prepared['layout']
        native = gather_pool_words(self.comm, layout, self.partitions, snapshot_wanted(self.route, layout),
            lambda row, level: session.native.get_field(row['pool'], row['pool_name'], level=level))
        state, saved = {}, {}
        for target, fields, source in ((state, mv.STATE_FIELDS, dycore.atmosphere.state),
                                       (saved, mv.SAVED_FIELDS, dycore.atmosphere.saved)):
            for name in fields:
                local = dycore.cp.asnumpy(getattr(source, name))
                buffers = self.comm.gather(local, root=0)
                if rank == 0:
                    target[name] = assemble_columns(buffers, self.partitions,
                        entity='edge' if name in ('rho_u', 'normal_velocity') else 'cell')
        window = session.precipitation_window
        window_values = None if window is None else window.snapshot()
        windows = self.comm.gather(window_values, root=0)
        services = self.comm.gather(None if session.surface_service is None else
                                    session.surface_service.restart_state(), root=0)
        psfc = self.comm.gather(None if columns is None else dycore.cp.asnumpy(columns.psfc_p), root=0)
        if rank != 0:
            return None
        if any(service != services[0] for service in services):
            raise ValueError('moving surface service calendars differ between ranks')
        global_window = None
        if windows[0] is not None:
            global_window = {key: assemble_columns([window[key] for window in windows],
                self.partitions, entity='cell') for key in windows[0]}
        proxy = SimpleNamespace(native=native, config=session.config,
            precipitation_window=None if global_window is None else WindowWords(global_window),
            surface_service=None if services[0] is None else ServiceWords(services[0]))
        global_dycore = SimpleNamespace(cp=HostArrays, atmosphere=SimpleNamespace(
            state=SimpleNamespace(**state), saved=SimpleNamespace(**saved)))
        global_columns = None if columns is None else SimpleNamespace(
            psfc_p=assemble_columns(psfc, self.partitions, entity='cell'))
        return proxy, global_dycore, global_columns

    def after_step(self, step, **values):
        model_seconds = self.route.offset_seconds + step * values['dt']
        frame = self.route.control is not None and not model_seconds % self.route.frame_seconds
        due = getattr(self.route, 'checkpoint_due', lambda step, dt: False)(step, values['dt'])
        if not frame and not self.route.shock and not due:
            return False
        snapshot = self._snapshot(values['session'], values['dycore'], values['columns'])
        result = False
        if self.comm.Get_rank() == 0:
            session, dycore, columns = snapshot
            result = self.route.after_step(step, **{**values, 'session': session,
                'dycore': dycore, 'columns': columns, 'mesh': self.prepared['mesh'],
                'vertical': self.prepared['vertical'], 'layout': self.prepared['layout']})
        result, self._receipt = self.comm.bcast((result, self.route.receipt()), root=0)
        return result

    def finish(self, **values):
        if not self.route.export_path:
            return
        snapshot = self._snapshot(values['session'], values['dycore'])
        if self.comm.Get_rank() == 0:
            session, dycore, _ = snapshot
            self.route.finish(**{**values, 'session': session, 'dycore': dycore,
                'mesh': self.prepared['mesh'], 'vertical': self.prepared['vertical'],
                'layout': self.prepared['layout']})
        self._receipt = self.comm.bcast(self.route.receipt(), root=0)

    def receipt(self):
        return self._receipt
