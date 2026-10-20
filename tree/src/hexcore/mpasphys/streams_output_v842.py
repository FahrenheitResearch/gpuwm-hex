"""Rust output for the MPAS physics route, without a Fortran stream manager.

MPAS v8.4.2 mpas_atm_core.F:663-705,877-902 owns alarm/reset order;
mpas_io_streams.F:1659-1857 owns constituent and owned-entity selection.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

from ..output import HistoryField
from ..restart_v842 import NativePoolStreams, _date, inspect_stream, write_restart, write_stream
from ..streams_v842 import Stream, StreamAlarms
from .stream_metadata_v842 import global_attributes, input_field_attributes, NativeFileIds


def output_inventory(streams, layout, *, produced=()):
    """Retain the route's computed field inventory, including scalar aliases.

    MPAS Registry.xml:650-925,1360-4020. Allocated diagnostics without a
    computed authority are not made valid by changing their writer.
    """
    active = {r['name']: r for r in layout.fields}
    active.update({c: r for r in layout.fields for c in r['components']})
    diagnostics = {'rho', 'theta', 'pressure', 'pressure_base', 'pressure_p',
                   'exner', 'exner_base', 'rho_base', 'rho_p', 'rtheta_base',
                   'rtheta_p', 'uReconstructZonal', 'uReconstructMeridional',
                   # The microphysics callbacks already own these words;
                   # thompson_v842.py:4778-4781,4835-4843 binds the native
                   # interface.F:948-962 and microphysics.F:798-850 writes.
                   'surface_pressure', 'relhum'}
    diagnostics.update(produced)
    result = {}
    for stream in streams:
        result[stream.name] = [name for name in stream.fields
            if name not in active or
            (active[name]['pool'] == 'diag' and name not in diagnostics) or
            (active[name]['pool'] == 'tend' and name not in
             ('rt_diabatic_tend', 'tend_sfc_pressure'))]
    return result


def output_definitions(streams, layout, unavailable):
    """Expand the already parsed door lists without changing their order."""
    arrays = {r['name']: tuple(r['components']) for r in layout.fields if r['components']}
    definitions = {}
    for stream in streams:
        options = dict(getattr(stream, 'options', None) or {})
        # MPAS mpas_io.F:419-422 omits fformat when creating SMIOL files;
        # smiolf.F90:429 consequently selects CDF5, including default streams.
        options.setdefault('io_type', 'pnetcdf,cdf5')
        options.update(type=';'.join(stream.directions),
                       filename_template=stream.filename,
                       input_interval=stream.input_interval,
                       output_interval=stream.output_interval,
                       filename_interval=stream.filename_interval)
        members = []
        for name in stream.fields:
            if name not in unavailable.get(stream.name, ()):
                members.extend(('var', n, '') for n in arrays.get(name, (name,)))
        definitions[stream.name] = Stream(stream.name, options, tuple(members))
    return definitions


class PhysicsStreams(NativePoolStreams):
    """Snapshot selected pool words, then encode away from the forecast thread.

    One outstanding frame bounds memory. A single worker preserves append
    order and uses independent Rust processes for NetCDF, so no libnetcdf
    thread-safety assumption crosses into the CUDA driver. Errors propagate
    before the next capture and on close. Restart timestamps follow durable
    file publication. No live device/host pool view reaches the worker.
    """

    def __init__(self, session, streams, output, *, final_time, alarms=None, diagnostics=None):
        self.session = session
        self.diagnostics = diagnostics
        produced = () if diagnostics is None else diagnostics.produced
        self.missing = output_inventory(streams, session.layout, produced=produced)
        # Native only registers fields allocated by active Registry packages.
        # An allocated field requested by an enabled output stream must also
        # have an actual writer. Dropping it would silently change the deck.
        active = {r['name'] for r in session.layout.fields}
        active.update(c for r in session.layout.fields for c in r['components'])
        unsupported = {s.name: [name for name in self.missing[s.name] if name in active]
                       for s in streams if 'output' in s.directions and s.output_interval != 'none'}
        unsupported = {name: fields for name, fields in unsupported.items() if fields}
        if unsupported:
            fields = '; '.join(f'{name}: {", ".join(fields)}' for name, fields in unsupported.items())
            raise ValueError(f'MPAS output fields lack computed authority: {fields}; '
                             'cannot omit active fields from an enabled output stream')
        definitions = output_definitions(streams, session.layout, self.missing)
        super().__init__(session.native, session.layout, None, output,
                         stream_definitions=definitions, final_time=final_time,
                         alarms=alarms)
        self.file_ids = NativeFileIds()
        self._file_attributes = {}
        self.metadata_receipt = dict(file_id_seed=self.file_ids.seed,
                                     file_id_algorithm='GNU/Linux rand, MPAS random_id.c')
        source = next((s.input_file for s in streams if s.name == 'input'), None)
        if source is not None:
            header = inspect_stream(source, scratch=self.output/'stream-scratch')
            self.attrs = global_attributes(session.layout, header,
                configs=getattr(session, 'output_config_snapshot', None))
            self.field_attrs = input_field_attributes(session.layout, header)
        # mpas_atm_core.F:264-277: restarts retain the original CF epoch.
        if 'initial_time' in self.rows:
            row, _ = self.rows['initial_time']
            self.initial = _date(self.native.get_text_field(row['pool'], row['pool_name'], level=1))
            if alarms is None:
                self.alarms = StreamAlarms(self.streams, self.start, self.initial, self.final,
                    do_restart=self.control.config_do_restart,
                    active_packages=frozenset(self.layout.packages))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mpas-output')
        self._pending = None
        self._pending_receipt = None
        self._closed = False
        self._host_storage = {}
        if diagnostics is not None:
            diagnostics.bind(self)

    def current_time(self, *, step, dt):
        """Use the ported clock's committed time at the stream boundary."""
        clock = getattr(self.native, 'clock_state', None)
        if clock is not None:
            return _date(clock.current_time.stamp())
        return self.start + timedelta(seconds=step * dt)

    def required(self, when):
        """MPAS mpas_atm_core.F:675-701: poll without resetting an alarm."""
        if self.diagnostics is not None and self.diagnostics.accumulates:
            return True
        return any('output' in s.options['type'].split(';') and
                   s.fields(self.layout.packages, streams=self.streams) and
                   self.alarms.due(name, when) for name, s in self.streams.items())

    def _snapshot(self, selected):
        """Resident buffers go directly to immutable writer-owned host storage.

        Transfer integer views to preserve every floating word, including NaN
        payloads. Never publish resident values through native.put_field.
        """
        storage, result = {}, {}
        views = getattr(self.session, 'device_pools', None)
        resident = views is not None and getattr(views, 'resident', False)
        for name in dict.fromkeys(n for s in selected for n in
                                  s.fields(self.layout.packages, streams=self.streams)):
            if name in ('xtime', 'initial_time'):
                continue
            row, component = self.rows[name]
            pool, field = row['pool'], row['pool_name']
            if row['kind'] == 'character':
                result[name] = self.native.get_text_field(pool, field, level=1)
                continue
            key = pool, field
            dims = tuple(row['dimensions'])
            expected = tuple(self.layout.dimensions[d] for d in dims)
            if key not in storage:
                diagnostic = (None if self.diagnostics is None or name == 'Time' else
                              self.diagnostics.field(pool, field))
                if (resident and name != 'Time') or diagnostic is not None:
                    # The resident provider caches owned buffers. Request its
                    # normal view, not a conflicting full/dummy allocation.
                    previous = views.level
                    try:
                        views.level = 1
                        device = diagnostic
                        if device is None:
                            device = views.field(pool, field)
                        slices = tuple(slice(0, size) for size in expected)
                        device = device[slices]
                        if hasattr(views.cp, 'cuda'):
                            # A pageable multi-GB get can stall the forecast
                            # for seconds. Reuse pinned writer-owned storage;
                            # _drain completed before any buffer is reused.
                            host = self._host_storage.get(key)
                            if host is None:
                                owner = views.cp.cuda.alloc_pinned_memory(device.nbytes)
                                host = np.ndarray(device.shape, dtype=device.dtype,
                                                  buffer=owner, order='C')
                                self._host_storage[key] = host
                            if host.shape != device.shape or host.dtype != device.dtype:
                                raise ValueError(f'{pool}.{field}: resident output layout changed')
                            wire = device.view(views.cp.uint32) if device.dtype.itemsize == 4 else device
                            target = host.view(np.uint32) if device.dtype.itemsize == 4 else host
                            # Complete before returning the snapshot so the
                            # forecast can immediately mutate device pools.
                            wire.get(out=target, blocking=True)
                        elif device.dtype.itemsize == 4:
                            host = device.view(views.cp.uint32).get(order='C').view(device.dtype)
                        else:
                            host = device.get(order='C')
                        views.transfers['publication_download_bytes'] += host.nbytes
                    finally:
                        views.level = previous
                else:
                    host = self.native.get_field(pool, field, level=1)
                    host = host[tuple(slice(0, size) for size in expected)].copy()
                if host.shape != expected:
                    raise ValueError(f'{pool}.{field}: output shape {host.shape} != {expected}')
                storage[key] = host
            value = storage[key]
            if component is not None:
                value, dims = value[component], dims[1:]
            attrs = dict(self.field_attrs.get(name, {}))
            if name == 'Time':
                attrs['units'] = f'seconds since {self.initial:%Y-%m-%d %H:%M:%S}'
            result[name] = HistoryField(value, dims, attrs)
        return result

    def _drain(self):
        if self._pending is not None:
            pending, self._pending = self._pending, None
            records = pending.result()
            self._pending_receipt.setdefault('streams', []).extend(records)
            self._pending_receipt = None

    def write(self, *, step, dt, receipt):
        if self._closed:
            raise RuntimeError('MPAS output writer is closed')
        when = self.current_time(step=step, dt=dt)
        if self.diagnostics is not None:
            self.diagnostics.before_write(step=step, when=when)
            receipt['output_diagnostics'] = self.diagnostics.receipt()
        selected = [s for name, s in self.streams.items()
                    if 'output' in s.options['type'].split(';') and
                    s.fields(self.layout.packages, streams=self.streams) and
                    self.alarms.due(name, when)]
        if not selected:
            return False
        waiting = perf_counter()
        self._drain()
        started = perf_counter()
        snapshot = self._snapshot(selected)
        if self.diagnostics is not None:
            # Reset only after immutable capture, while this event's stream
            # alarms still ring (mpas_atm_core.F:877-910).
            self.diagnostics.after_snapshot()
        jobs = []
        for stream in selected:
            target = (self.output / stream.filename(when, self.initial)).resolve()
            if not target.is_relative_to(self.output):
                raise ValueError(f'{stream.name}: filename escapes output directory')
            record = self.records.get(target, 0)
            # never_modify disallows preexisting files, but an open native
            # stream may append its own subsequent frames to the same file.
            publication = stream
            if record:
                publication = replace(stream, options={**stream.options, 'clobber_mode': 'append'})
            jobs.append((publication, target, record))
        self._pending_receipt = receipt
        self._pending = self._executor.submit(self._encode, jobs, snapshot, when, step)
        for stream, target, record in jobs:
            self.records[target] = record + 1
            self.alarms.reset(stream.name, when)
        receipt.setdefault('output_capture', []).append(dict(step=step,
            seconds=perf_counter()-started, encoding='background Rust process',
            wait_seconds=started-waiting,
            fields=len(snapshot), resident=bool(getattr(getattr(self.session, 'device_pools', None), 'resident', False))))
        return any(s.name == 'diagnostics' for s in selected)

    def _encode(self, jobs, snapshot, when, step):
        records = []
        for stream, target, record in jobs:
            started = perf_counter()
            if target not in self._file_attributes:
                self._file_attributes[target] = dict(self.attrs, file_id=self.file_ids.next())
            options = dict(initial_time=self.initial, active_packages=self.layout.packages,
                           streams=self.streams, global_attrs=self._file_attributes[target],
                           field_attrs=self.field_attrs, scratch=self.output/'stream-scratch',
                           record_index=record,
                           registry_attributes=True)
            if stream.name == 'restart':
                write_restart(target, self.mesh, snapshot, when, stream=stream,
                              timestamp_path=self.output/'restart_timestamp', **options)
            else:
                write_stream(target, stream, self.mesh, snapshot, when, **options)
            records.append(dict(name=stream.name, step=step, seconds=perf_counter()-started,
                writer='rw_mpas_stream', path=str(target), record_index=record))
        return records

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._drain()
            finally:
                self._executor.shutdown(wait=True)
                self._host_storage.clear()
                close_diagnostics = getattr(self.diagnostics, 'close', None)
                if close_diagnostics is not None:
                    close_diagnostics()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
