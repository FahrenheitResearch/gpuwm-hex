"""Replay Rust-decoded native initial-output pools against pristine Fortran/CUDA.

The JSON bridge carries stored integer words, not interpreted weather values.
Python binds those words to the two driver interfaces; all math is Fortran/CUDA.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json


def load(path):
    import numpy as np
    record = json.loads(path.read_text())
    arrays = {}
    for name, field in record['fields'].items():
        words = np.asarray(field['words'], dtype=np.int32 if field['type'] == 'i32' else np.uint32)
        array = words if field['type'] == 'i32' else words.view(np.float32)
        arrays[name] = array.reshape(field['shape'])
    return arrays


def replay(library, directories, receipt, cpu_only=False):
    import numpy as np
    if not cpu_only:
        import cupy as cp
    from hexcore.mpasdiag.isobaric import IsobaricDiagnostics, FIELDS, FAMILIES, LEVELS
    from hexcore.mpasdiag.utils import DiagnosticsUtils, StreamView, Direction
    lib = ctypes.CDLL(str(library.resolve()))
    ptr = ctypes.c_void_p
    lib.ref_full.argtypes = [ctypes.c_int]*4+[ptr]*23
    lib.ref_full.restype = None
    rows, sources = [], []
    def compare(case, field, target, actual, referee):
        e = np.ascontiguousarray(target).view(np.uint32)
        a = np.ascontiguousarray(actual).view(np.uint32)
        bad = int(np.count_nonzero(e != a))
        rows.append(dict(case=case, field=field, referee=referee, different_words=bad,
                         compared_words=int(e.size), native_sha256=hashlib.sha256(e.tobytes()).hexdigest(),
                         port_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        if bad:
            indices = np.argwhere(e != a)[:5]
            print(case, field, referee, bad, [(i.tolist(), hex(int(e[tuple(i)])),
                                             hex(int(a[tuple(i)]))) for i in indices], flush=True)
    for directory in directories:
        data, grid = load(directory/'columns.json'), load(directory/'mesh.json')
        # Native stored record,entity,level -> explicit level,entity pools.
        nc, nl = data['theta_m'].shape[-2:]
        nv, degree = grid['cellsOnVertex'].shape
        def pool(name, source=data, vertices=False):
            a = source[name]
            if a.ndim == 3:
                if a.shape[0] != 1:
                    raise ValueError('initial-output replay expects one record')
                a = a[0]
            result = np.zeros((a.shape[1], a.shape[0]+1), np.float32, order='F')
            result[:, :-1] = a.T
            return result
        pp, pb, theta, exner = [pool(name) for name in ('pressure_p','pressure_base','theta_m','exner')]
        q = np.asfortranarray(pool('qv')[None])
        h, rh, u, v, w, vort = [pool(name, source) for name, source in (
            ('zgrid',grid), ('relhum',data), ('uReconstructZonal',data),
            ('uReconstructMeridional',data), ('w',data), ('vorticity',data))]
        cov = np.asfortranarray(grid['cellsOnVertex'].T, dtype=np.int32)
        kite = np.asfortranarray(grid['kiteAreasOnVertex'].T)
        area = np.asfortranarray(grid['areaTriangle'])
        arrays = (pp,pb,theta,exner,q,h,rh,u,v,w,vort,cov,kite,area)
        F = lambda shape: np.full(shape, -9876.5, np.float32, order='F')
        outputs = (F((nc,8,7)),F((nv,8)),F(5),F(13),F((5,nc)),F((13,nc)),F(nc),F(nc))
        mask = np.ones(68, np.int32)
        lib.ref_full(nc,nl,nv,degree,*[ptr(a.ctypes.data) for a in (mask,*arrays,*outputs)])
        expected = {}
        for fi,family in enumerate(FAMILIES[:7]):
            for li,level in enumerate(LEVELS):
                expected[f'{family}_{level}hPa'] = outputs[0][:,li,fi]
        for li,level in enumerate(LEVELS):
            expected[f'vorticity_{level}hPa'] = outputs[1][:,li]
        for name,output in zip(('t_iso_levels','z_iso_levels','t_isobaric','z_isobaric','meanT_500_300','mslp'),outputs[2:]):
            expected[name] = output
        if cpu_only:
            for name,target in expected.items():
                native = data[name]
                if native.ndim > target.ndim:
                    native = native[0]
                if native.shape != target.shape:
                    native = native.T
                compare(directory.name,name,native,target,'strict binary versus standalone native')
            run = json.loads((directory/'run.json').read_text())
            sources.append(dict(directory=str(directory),host=run['host'],case=run['case'],
                                case_publication_sha256=run['case_publication_sha256'],
                                native_sha256=run['native_sha256'],
                                columns_sha256=hashlib.sha256((directory/'columns.json').read_bytes()).hexdigest(),
                                mesh_sha256=hashlib.sha256((directory/'mesh.json').read_bytes()).hexdigest()))
            continue
        mesh = dict(nCells=nc,nVertLevels=nl,nVertices=nv,vertexDegree=degree,nIsoLevelsT=5,nIsoLevelsZ=13,
                    zgrid=cp.asarray(h),cellsOnVertex=cp.asarray(cov),kiteAreasOnVertex=cp.asarray(kite),
                    areaTriangle=cp.asarray(area))
        state = dict(theta_m=cp.asarray(theta),w=cp.asarray(w),scalars=cp.asarray(q),index_qv=1)
        diag = {name:cp.asarray(a) for name,a in zip(('pressure_p','pressure_base','exner','relhum',
                                                    'uReconstructZonal','uReconstructMeridional','vorticity'),
                                                   (pp,pb,exner,rh,u,v,vort))}
        for family in FAMILIES:
            for level in LEVELS:
                diag[f'{family}_{level}hPa'] = cp.full(nv+1 if family == 'vorticity' else nc+1,-9876.5,cp.float32)
        for name,shape in (('t_iso_levels',5),('z_iso_levels',13),('t_isobaric',(5,nc+1)),
                           ('z_isobaric',(13,nc+1)),('meanT_500_300',nc+1),('mslp',nc+1)):
            diag[name] = cp.full(shape,-9876.5,cp.float32)
        utils = DiagnosticsUtils()
        utils.init(lambda:[StreamView('diagnostics',Direction.OUTPUT,True,True,dict.fromkeys(FIELDS,True))])
        port = IsobaricDiagnostics(exchange_pressure_halo=lambda p:None)
        port.setup(dict(mesh=mesh,state=state,diag=diag),object(),utils)
        port.compute()
        expected = {}
        for fi,family in enumerate(FAMILIES[:7]):
            for li,level in enumerate(LEVELS):
                expected[f'{family}_{level}hPa'] = outputs[0][:,li,fi]
        for li,level in enumerate(LEVELS):
            expected[f'vorticity_{level}hPa'] = outputs[1][:,li]
        for name,output in zip(('t_iso_levels','z_iso_levels','t_isobaric','z_isobaric','meanT_500_300','mslp'),outputs[2:]):
            expected[name] = output
        for name,target in expected.items():
            actual = cp.asnumpy(diag[name])
            if actual.shape != target.shape:
                actual = actual[..., :target.shape[-1]]
            compare(directory.name,name,target,actual,'strict standalone native')
            native = data[name]
            if native.ndim > target.ndim:
                native = native[0]
            if native.shape != target.shape:
                native = native.T
            compare(directory.name,name,native,actual,'published strict binary initial-output call')
        run = json.loads((directory/'run.json').read_text())
        sources.append(dict(directory=str(directory),host=run['host'],case=run['case'],
                            case_publication_sha256=run['case_publication_sha256'],
                            native_sha256=run['native_sha256'],
                            columns_sha256=hashlib.sha256((directory/'columns.json').read_bytes()).hexdigest(),
                            mesh_sha256=hashlib.sha256((directory/'mesh.json').read_bytes()).hexdigest()))
    from hexcore.mpasdiag._artifact import artifact_receipt
    result = dict(tier='real-column debug; no forecast or output timing proof', sources=sources,
                  process_count=1,
                  cuda_artifact=artifact_receipt() if not cpu_only else None,
                  reader='rw_mpas_words Rust stored-word transport',
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  compared_words=sum(r['compared_words'] for r in rows),
                  different_words=sum(r['different_words'] for r in rows),cases=rows)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('tier','compared_words','different_words')}),flush=True)
    return bool(result['different_words'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('library',type=Path)
    p.add_argument('receipt',type=Path)
    p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--cpu-only',action='store_true')
    args = p.parse_args()
    raise SystemExit(replay(args.library,args.directories,args.receipt,args.cpu_only))
