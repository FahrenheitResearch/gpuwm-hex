"""Replay each complete isobaric call on its actual forecast inputs.

The strict pristine Fortran slice owns the reference calculation. This is
same-input debugging alongside the real forecast, not whole-model acceptance.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path


class BoundIsobaricReferee:
    def __init__(self, service, library):
        import numpy as np
        from hexcore.mpasdiag.isobaric import FAMILIES, FIELDS, LEVELS
        self.service, self.np = service, np
        self.families, self.fields, self.levels = FAMILIES, FIELDS, LEVELS
        self.path = Path(library).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        manifest = json.loads(self.path.with_suffix('.json').read_text())
        if self.sha256 != manifest['library_sha256']:
            raise ValueError('strict standalone referee differs from its build receipt')
        self.library = ct.CDLL(str(self.path))
        self.library.ref_full.argtypes = [ct.c_int]*4+[ct.c_void_p]*23
        self.library.ref_full.restype = None
        self.rows = []
        compute = service.isobaric.compute

        def replay_complete_call():
            mesh, state, diag = (service.pools[name] for name in ('mesh', 'state', 'diag'))
            if not diag:
                return compute()
            np, cp = self.np, service.cp
            nc, nl, nv, degree = (mesh[name] for name in
                ('nCells', 'nVertLevels', 'nVertices', 'vertexDegree'))
            host = lambda value: np.asfortranarray(cp.asnumpy(value))
            mask = np.asarray([service.utils.field_will_be_written(name)
                               for name in self.fields], np.int32)
            if not mask.any():
                return compute()
            inputs = [host(diag[name]) for name in ('pressure_p', 'pressure_base')]
            inputs += [host(state['theta_m']), host(diag['exner']),
                       host(state['scalars'][state['index_qv']-1:state['index_qv']]),
                       host(mesh['zgrid'])]
            inputs += [host(diag[name]) for name in
                ('relhum', 'uReconstructZonal', 'uReconstructMeridional')]
            inputs += [host(state['w']), host(diag['vorticity'])]
            inputs += [host(mesh[name]) for name in
                ('cellsOnVertex', 'kiteAreasOnVertex', 'areaTriangle')]
            cell = np.asfortranarray(np.stack([np.stack([
                cp.asnumpy(diag[f'{family}_{level}hPa'][:nc]) for level in self.levels],
                axis=1) for family in self.families[:7]], axis=2))
            vort = np.asfortranarray(np.stack([
                cp.asnumpy(diag[f'vorticity_{level}hPa'][:nv]) for level in self.levels], axis=1))
            extra_names = ('t_iso_levels', 'z_iso_levels', 't_isobaric',
                           'z_isobaric', 'meanT_500_300', 'mslp')
            extras = [host(diag[name] if name.endswith('_levels') else diag[name][..., :nc])
                      for name in extra_names]
            self.library.ref_full(nc, nl, nv, degree,
                *(ct.c_void_p(array.ctypes.data) for array in (mask, *inputs, cell, vort, *extras)))
            result = compute()
            expected = {f'{family}_{level}hPa': cell[:, li, fi]
                for fi, family in enumerate(self.families[:7])
                for li, level in enumerate(self.levels)}
            expected.update({f'vorticity_{level}hPa': vort[:, li]
                             for li, level in enumerate(self.levels)})
            expected.update(zip(extra_names, extras))
            fields = []
            for name, target in expected.items():
                actual = cp.asnumpy(diag[name])
                if actual.shape != target.shape:
                    actual = actual[..., :target.shape[-1]]
                a, b = (np.ascontiguousarray(value).view(np.uint32)
                        for value in (actual, target))
                fields.append(dict(field=name, compared_words=int(a.size),
                    differing_words=int(np.count_nonzero(a != b)),
                    input_selected=bool(service.utils.field_will_be_written(name))))
            self.rows.append(dict(call=len(self.rows), fields=fields,
                                  compared_words=sum(row['compared_words'] for row in fields),
                                  differing_words=sum(row['differing_words'] for row in fields)))
            return result

        service.isobaric.compute = replay_complete_call

    def receipt(self):
        return dict(scope='same-input strict standalone debug; not forecast identity',
                    library=str(self.path), library_sha256=self.sha256, calls=self.rows,
                    compared_words=sum(row['compared_words'] for row in self.rows),
                    differing_words=sum(row['differing_words'] for row in self.rows))
