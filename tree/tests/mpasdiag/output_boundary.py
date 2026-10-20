"""Compare the real output helper's live pools against sealed f1 captures.

MPAS authority: mpas_atm_core.F:916-965, state level selected by its caller.
Rust decodes and compares physical-column words. Native dummy IDs remain in
the authenticated inventory and are not labelled physical columns.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys


class OutputBoundaryObserver:
    def __init__(self, service, selection, library):
        self.service = service
        selection, library = Path(selection), Path(library)
        self.selection = json.loads(selection.read_text())
        self.selection_sha256 = hashlib.sha256(selection.read_bytes()).hexdigest()
        manifest = json.loads(library.with_suffix('.json').read_text())
        if hashlib.sha256(library.read_bytes()).hexdigest() != manifest['library_sha256']:
            raise ValueError('output boundary comparator differs from its build receipt')
        if hashlib.sha256(Path(__file__).with_name('oracle_live_words.rs').read_bytes()).hexdigest() != manifest['source_sha256']:
            raise ValueError('output boundary comparator source differs from its build receipt')
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                               'tools/native_physics_reference/backend'))
        from oracle_replay import Capture, _Metadata
        self.Capture = Capture
        self.live = ct.CDLL(str(library.resolve()))
        self.live.oracle_open.argtypes, self.live.oracle_open.restype = [ct.c_char_p], ct.c_void_p
        self.live.oracle_metadata.argtypes = [ct.c_void_p, ct.POINTER(_Metadata)]
        self.live.oracle_close.argtypes = [ct.c_void_p]
        self.live.oracle_compare_live_owned.argtypes = [ct.c_void_p, ct.c_void_p,
            ct.c_size_t, *[ct.POINTER(ct.c_size_t)] * 3]
        self.live.oracle_compare_live_owned.restype = ct.c_int
        self.rows = []
        self.current_step = None
        root = Path(self.selection['root'])
        if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() != self.selection['seal_sha256']:
            raise ValueError('output boundary seal changed')
        self.paths = {}
        for name, metadata in self.selection['files'].items():
            path = root/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError('output boundary capture changed '+name)
            _, _, _, _, phase, pool, field, *tail = name.split('.')
            level = int(tail[0][1:]) if tail[0].startswith('t') else 1
            self.paths[phase, pool, field, level] = path
        original = service.compute

        def compute(step):
            self.current_step = step
            try:
                return original(step)
            finally:
                self.current_step = None

        service.compute = compute

    def observe(self, phase):
        """MPAS v8.4.2 mpas_atm_core.F:916-965; no atmospheric host maths."""
        if self.current_step != self.selection['step']:
            return
        selected = (('state', 'theta_m'), ('state', 'rho_zz'), ('state', 'scalars'),
                    ('diag', 'pressure_p'), ('diag', 'pressure_base'), ('mesh', 'zz'))
        if phase == 'post':
            selected = (('diag', 'theta'), ('diag', 'rho'), ('diag', 'pressure'))
        for pool, field in selected:
            path = self.paths[phase, pool, field, 1]
            row = dict(phase=phase, pool=pool, field=field, state_level=1,
                       step=self.current_step, capture=str(path))
            try:
                if phase == 'post':
                    value = self.service.cp.asnumpy(self.service.pools[pool][field])
                else:
                    value = self.service.native.get_field(pool, field, level=1)
                import numpy as np
                value = np.asfortranarray(value)
                with self.Capture(self.live, path) as capture:
                    if value.shape != capture.shape or value.nbytes != capture.meta.payload_bytes:
                        raise ValueError('boundary allocation shape differs')
                    cells, words, different = ct.c_size_t(), ct.c_size_t(), ct.c_size_t()
                    status = self.live.oracle_compare_live_owned(capture.handle,
                        value.ctypes.data, value.nbytes, ct.byref(cells),
                        ct.byref(words), ct.byref(different))
                    if status:
                        raise ValueError('Rust boundary comparison status '+str(status))
                row.update(status='compared', cells=cells.value,
                           words=words.value, different_words=different.value)
            except (ValueError, KeyError, TypeError) as error:
                row.update(status='uncompared', error=str(error))
            self.rows.append(row)

    def receipt(self):
        return dict(scope='f1 output helper boundary; no full forecast claim',
            selection_sha256=self.selection_sha256, process_count=1,
            compared_words=sum(row.get('words', 0) for row in self.rows),
            different_words=sum(row.get('different_words', 0) for row in self.rows),
            uncompared_arrays=sum(row['status'] != 'compared' for row in self.rows),
            rows=self.rows)
