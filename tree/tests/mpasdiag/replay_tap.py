"""Replay complete isobaric output on sealed, real f1 output-boundary pools.

MPAS authority: mpas_isobaric_diagnostics.F:55-1351; the real state is taken
after mpas_atm_core.F:916-965 at the initial output of both native suites.
Rust decodes captures; Python binds buffers; Fortran/CUDA compute fields.
"""
import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys


def replay(library, transport, directories, receipt):
    import numpy as np
    import cupy as cp
    from hexcore.mpasdiag.isobaric import IsobaricDiagnostics, FIELDS, FAMILIES, LEVELS
    from hexcore.mpasdiag.utils import DiagnosticsUtils, StreamView, Direction
    from hexcore.mpasdiag._artifact import artifact_receipt
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/native_physics_reference/backend'))
    from oracle_replay import Capture, _Metadata
    decoder = ct.CDLL(str(transport.resolve()))
    decoder.oracle_open.argtypes = [ct.c_char_p]
    decoder.oracle_open.restype = ct.c_void_p
    decoder.oracle_metadata.argtypes = [ct.c_void_p, ct.POINTER(_Metadata)]
    decoder.oracle_close.argtypes = [ct.c_void_p]
    native = ct.CDLL(str(library.resolve()))
    native.ref_full.argtypes = [ct.c_int] * 4 + [ct.c_void_p] * 23
    native.ref_full.restype = None
    rows, sources = [], []
    for directory in directories:
        authority = json.loads((directory / 'authority.json').read_text())
        if authority['process_count'] != 1 or authority['index_qv'] != 1:
            raise ValueError('Standalone ref_full declares one qv scalar; this capture must match it')
        arrays = {}
        with ExitStack() as stack:
            for key, pin in authority['files'].items():
                path = directory / pin['file']
                if path.stat().st_size != pin['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
                    raise ValueError('Native tap input changed: ' + str(path))
                arrays[key] = stack.enter_context(Capture(decoder, path)).array()
            nl, nc1 = arrays['theta'].shape
            degree, nv1 = arrays['cov'].shape
            nc, nv = nc1 - 1, nv1 - 1
            names = ('pp', 'pb', 'theta', 'exner', 'q', 'h', 'rh', 'u', 'v', 'w', 'vort', 'cov', 'kite', 'area')
            inputs = []
            for name in names:
                a = arrays[name]
                if name == 'q':
                    a = a[:1]
                elif name in ('cov', 'kite'):
                    a = a[:, :nv]
                elif name == 'area':
                    a = a[:nv]
                inputs.append(np.asfortranarray(a))
            pp, pb, theta, exner, q, h, rh, u, v, w, vort, cov, kite, area = inputs
            if any(a.shape != (nl, nc1) for a in (pp, pb, theta, exner, rh, u, v)) or q.shape != (1, nl, nc1):
                raise ValueError('Native cell pool layout differs from the standalone driver')
            if h.shape != (nl + 1, nc1) or w.shape != h.shape or vort.shape != (nl, nv1):
                raise ValueError('Native vertical/vertex pool layout differs')
            fill = lambda shape: np.full(shape, -9876.5, np.float32, order='F')
            outputs = (fill((nc, 8, 7)), fill((nv, 8)), fill(5), fill(13), fill((5, nc)), fill((13, nc)), fill(nc), fill(nc))
            mask = np.ones(68, np.int32)
            native.ref_full(nc, nl, nv, degree, *[ct.c_void_p(a.ctypes.data) for a in (mask, *inputs, *outputs)])
            mesh = dict(nCells=nc, nVertLevels=nl, nVertices=nv, vertexDegree=degree, nIsoLevelsT=5, nIsoLevelsZ=13,
                        zgrid=cp.asarray(h), cellsOnVertex=cp.asarray(cov), kiteAreasOnVertex=cp.asarray(kite), areaTriangle=cp.asarray(area))
            state = dict(theta_m=cp.asarray(theta), w=cp.asarray(w), scalars=cp.asarray(q), index_qv=1)
            diag = {name: cp.asarray(a) for name, a in zip(('pressure_p', 'pressure_base', 'exner', 'relhum', 'uReconstructZonal', 'uReconstructMeridional', 'vorticity'), (pp, pb, exner, rh, u, v, vort))}
            for family in FAMILIES:
                for level in LEVELS:
                    diag[f'{family}_{level}hPa'] = cp.full(nv1 if family == 'vorticity' else nc1, -9876.5, cp.float32)
            for name, shape in (('t_iso_levels', 5), ('z_iso_levels', 13), ('t_isobaric', (5, nc1)), ('z_isobaric', (13, nc1)), ('meanT_500_300', nc1), ('mslp', nc1)):
                diag[name] = cp.full(shape, -9876.5, cp.float32)
            utils = DiagnosticsUtils()
            utils.init(lambda: [StreamView('diagnostics', Direction.OUTPUT, True, True, dict.fromkeys(FIELDS, True))])
            port = IsobaricDiagnostics(exchange_pressure_halo=lambda p: None)
            port.setup(dict(mesh=mesh, state=state, diag=diag), object(), utils)
            port.compute()
            expected = {}
            for fi, family in enumerate(FAMILIES[:7]):
                for li, level in enumerate(LEVELS):
                    expected[f'{family}_{level}hPa'] = outputs[0][:, li, fi]
            for li, level in enumerate(LEVELS):
                expected[f'vorticity_{level}hPa'] = outputs[1][:, li]
            expected.update(zip(('t_iso_levels', 'z_iso_levels', 't_isobaric', 'z_isobaric', 'meanT_500_300', 'mslp'), outputs[2:]))
            for name, target in expected.items():
                actual = cp.asnumpy(diag[name])
                if actual.shape != target.shape:
                    actual = actual[..., :target.shape[-1]]
                e, a = np.ascontiguousarray(target).view(np.uint32), np.ascontiguousarray(actual).view(np.uint32)
                if e.shape != a.shape:
                    raise ValueError('Output extent differs: ' + name)
                rows.append(dict(case=directory.name, field=name, compared_words=int(e.size), different_words=int(np.count_nonzero(e != a)), native_sha256=hashlib.sha256(e.tobytes()).hexdigest(), port_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        sources.append(dict(case=directory.name, authority=authority, authority_sha256=hashlib.sha256((directory / 'authority.json').read_bytes()).hexdigest()))
    result = dict(schema='mpas-o2.real-native-tap-module-replay/v1', scope='stated real cases: initial output of both MPAS native physics suites; all 68 diagnostics and both level arrays', process_count=1,
                  sources=sources, cuda_artifact=artifact_receipt(), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), transport_sha256=hashlib.sha256(transport.read_bytes()).hexdigest(),
                  compared_words=sum(r['compared_words'] for r in rows), different_words=sum(r['different_words'] for r in rows), rows=rows)
    result['passed'] = result['different_words'] == 0 and len(rows) == 70 * len(directories)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'compared_words', 'different_words')}), flush=True)
    return not result['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('library', type=Path)
    p.add_argument('transport', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('directories', nargs='+', type=Path)
    a = p.parse_args()
    raise SystemExit(replay(a.library, a.transport, a.directories, a.receipt))
