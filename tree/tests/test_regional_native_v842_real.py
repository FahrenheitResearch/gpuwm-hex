"""Real initialized regional columns, with independent Rust/Fortran decode.

The framework stream admits each published initial state or LBC state twice. This
pool replay proves boundary arithmetic on real columns. Optional adjacent
LBC pairs exercise changing fields at a real 3h interval; this is not a forecast. No numerical outputs are prescribed by the stub.
"""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os

import numpy as np
import pytest

from hexcore import regional_v841 as regional
from hexcore.lbc import LbcFile
from test_regional_native_v842 import read_arrays, record, record_cuda_compile

START = datetime(2000, 1, 1)
END = START + timedelta(seconds=10800.25)
STEP = START + timedelta(seconds=7200.125)


def _real_case_ids():
    manifest = os.environ.get('MPAS_REGIONAL_REAL_MANIFEST')
    return range(len(json.loads(Path(manifest).read_text())['inputs'])) if manifest else range(3)


@pytest.fixture(scope='module', params=_real_case_ids())
def real_columns(request):
    manifest = os.environ.get('MPAS_REGIONAL_REAL_MANIFEST')
    oracle = os.environ.get('MPAS_REGIONAL_REAL_ORACLE')
    if not manifest or not oracle:
        pytest.skip('real published-column manifest and strict oracle are required')
    entry = json.loads(Path(manifest).read_text())['inputs'][request.param]
    source = Path(entry['copy'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
    path = Path(oracle)/f'real-{request.param:02}'
    nc, ne, nv, nl, me, ns = np.fromfile(path/'dimensions.bin', dtype='i4').tolist()
    assert ns == len(entry.get("scalar_names", ("lbc_qv", "lbc_qc", "lbc_qr") if entry.get("lbc_copy") else ("lbc_qv", "lbc_qc")))
    # Independent native authority reads through netCDF-Fortran. The port
    # reads the published bytes through the engine's Rust decoder, never netCDF4.
    from gpuwm.netcdf_bridge import open_dataset
    dataset = open_dataset(source)
    names = ('bdyMaskCell', 'bdyMaskEdge', 'bdyMaskVertex', 'nEdgesOnCell',
             'cellsOnCell', 'cellsOnEdge', 'xCell', 'yCell', 'zCell', 'zz',
             'u', 'w', 'theta', 'rho', 'qv', 'qc', 'dcEdge')
    for name in names:
        dataset.variables[name].set_auto_maskandscale(False)
    dataset.prefetch(names)
    decoded = {name: np.asarray(dataset.variables[name][:],
               dtype='i4' if name in names[:6] else 'f4') for name in names}
    mesh = SimpleNamespace(**{name: decoded[name] for name in names[:9]})
    mesh.dcEdge = decoded['dcEdge']
    mesh.cellsOnCell = mesh.cellsOnCell - 1
    mesh.cellsOnEdge = mesh.cellsOnEdge - 1
    masks = regional.derive_regional_masks(mesh, np.dtype('f4'))
    zz = np.ascontiguousarray(decoded['zz'].T)
    native_mesh = read_arrays(path/'mesh.bin', [
        ('bc', (nc+1,)), ('be', (ne+1,)), ('bv', (nv+1,)), ('counts', (nc+1,)),
        ('coc', (me, nc+1)), ('coe', (ne+1, 2)),
        ('x', (nc+1,)), ('y', (nc+1,)), ('z', (nc+1,)), ('zz', (nl, nc+1))])
    for stored, native in [('bdyMaskCell', 'bc'), ('bdyMaskEdge', 'be'),
                           ('bdyMaskVertex', 'bv'), ('nEdgesOnCell', 'counts'),
                           ('xCell', 'x'), ('yCell', 'y'), ('zCell', 'z')]:
        record(f'real-{request.param:02}-mesh-decode', native,
               decoded[stored], native_mesh[native][:-1])
    record(f'real-{request.param:02}-mesh-decode', 'coc',
           np.where(decoded['cellsOnCell'] == 0, nc+1, decoded['cellsOnCell']),
           native_mesh['coc'][:, :-1].T)
    record(f'real-{request.param:02}-mesh-decode', 'coe',
           np.where(decoded['cellsOnEdge'] == 0, nc+1, decoded['cellsOnEdge']),
           native_mesh['coe'][:-1])
    record(f'real-{request.param:02}-mesh-decode', 'zz', zz, native_mesh['zz'][:, :-1])
    scalar_names = ('lbc_qv', 'lbc_qc')
    if entry.get('lbc_copy'):
        from hexcore.lbc import read_lbc_file
        lbc_source = Path(entry['lbc_copy'])
        assert hashlib.sha256(lbc_source.read_bytes()).hexdigest() == entry['lbc_sha256']
        admitted = read_lbc_file(lbc_source)
        assert admitted.valid_time == datetime.strptime(entry['date'], '%Y-%m-%d_%H.%M.%S')
        fields = admitted.fields
        scalar_names = tuple(entry.get('scalar_names', ('lbc_qv','lbc_qc','lbc_qr')))
    else:
        fields = {f'lbc_{name}': np.ascontiguousarray(decoded[name][0])
                  for name in ('u', 'w', 'theta', 'rho', 'qv', 'qc')}
    primary = read_arrays(path/'primary.bin', [
        ('u', (nl, ne+1)), ('w', (nl+1, nc+1)), ('theta', (nl, nc+1)),
        ('rho', (nl, nc+1)), ('scalars', (ns, nl, nc+1))])
    for name in ('u', 'w', 'theta', 'rho'):
        record(f'real-{request.param:02}-decode', name, fields[f'lbc_{name}'].T,
               primary[name][:, :-1])
    for j, field in enumerate(scalar_names):
        name = field.removeprefix('lbc_')
        record(f'real-{request.param:02}-decode', name, fields.get(f'lbc_{name}', np.zeros_like(fields['lbc_qv'])).T,
               primary['scalars'][j, :, :-1])
    next_fields = fields
    if entry.get('next_lbc_copy'):
        from hexcore.lbc import read_lbc_file
        source = Path(entry['next_lbc_copy'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['next_lbc_sha256']
        admitted = read_lbc_file(source)
        assert admitted.valid_time == datetime.strptime(entry['next_date'], '%Y-%m-%d_%H.%M.%S')
        next_fields = admitted.fields
        following = read_arrays(path/'next-primary.bin', [
            ('u', (nl, ne+1)), ('w', (nl+1, nc+1)), ('theta', (nl, nc+1)),
            ('rho', (nl, nc+1)), ('scalars', (ns, nl, nc+1))])
        for name in ('u', 'w', 'theta', 'rho'):
            record(f'real-{request.param:02}-next-decode', name,
                   next_fields[f'lbc_{name}'].T, following[name][:, :-1])
        for j, name in enumerate(scalar_names):
            record(f'real-{request.param:02}-next-decode', name,
                   next_fields.get(name,np.zeros_like(next_fields['lbc_qv'])).T, following['scalars'][j, :, :-1])
    if entry.get('omit_next_scalars'):
        next_fields = {k:v for k,v in next_fields.items() if k not in entry['omit_next_scalars']}
    expected = read_arrays(path/'masks.bin' , [('smc', (nc+1,)), ('sme', (ne+1,)),
                           ('smv', (nv+1,)), ('nearest', (nc+1,))])
    for name, actual in [('smc', masks.spec_zone_mask_cell),
                         ('sme', masks.spec_zone_mask_edge),
                         ('smv', masks.spec_zone_mask_vertex),
                         ('nearest', masks.nearest_relaxation_cell.astype('i4')+1)]:
        record(f'real-{request.param:02}-masks', name, actual, expected[name][:-1])
    pool_fields = [(name, ((nl+1 if name == 'w' else nl),
                   (ne+1 if name in ('u', 'ru', 'rho_edge') else nc+1)))
                   for name in ('u', 'ru', 'rho_edge', 'w', 'theta', 'rtheta_m', 'rho_zz', 'rho')]
    pool_fields += [('scalars', (ns, nl, nc+1))]
    return SimpleNamespace(case=request.param, path=path, fields=fields,
        mesh=mesh, masks=masks, zz=zz, nl=nl, nc=nc, ns=ns,
        scalar_names=scalar_names, pool_fields=pool_fields, next_fields=next_fields,
        retrieved_scalar="qc" if "lbc_qc" in scalar_names else "qv",
        third=bool(entry.get('omit_next_scalars')),
        end=START+timedelta(seconds=10800 if entry.get('next_lbc_copy') else 10800.25))


def inventory(a, monkeypatch):
    from hexcore import lbc
    # Admission fixtures retain real decoded primary slabs, not oracle output.
    first = LbcFile(Path('real-initial-state-stationary-fixture'), START, a.fields)
    following = LbcFile(first.path, a.end, a.next_fields)
    monkeypatch.setattr(lbc, 'read_lbc_file', lambda admitted: admitted)
    third = LbcFile(first.path, START+timedelta(seconds=21600), a.next_fields)
    return SimpleNamespace(latest_before=lambda when: first,
        earliest_strictly_after=lambda when: following if when < following.valid_time else third)


def compare_pool(a, driving, backend, to_host, *, padded=False):
    phases = [('first', START), ('next', a.end)]
    if a.third: phases.append(('third', START+timedelta(seconds=21600)))
    for phase, when in phases:
        if phase == 'first':
            driving.start(when)
        else:
            driving.advance()
        expected = read_arrays(a.path/f'{phase}-derived.bin', a.pool_fields)
        actual = {**driving._state, 'scalars': driving._state_scalars}
        for name, value in actual.items():
            record(f'real-{a.case:02}-{backend}-{phase}', name,
                   to_host(value), expected[name] if padded else expected[name][..., :-1])
    expected = read_arrays(a.path/'tendencies.bin', a.pool_fields)
    for name, value in {**driving._tend, 'scalars': driving._tend_scalars}.items():
        record(f'real-{a.case:02}-{backend}-tend', name,
               to_host(value), expected[name] if padded else expected[name][..., :-1])
    expected = read_arrays(a.path/'retrieved.bin', [('rho', (a.nl, a.nc+1)),
                           ('scalar', (a.nl, a.nc+1))])
    for field, name in [('rho_zz', 'rho'), (a.retrieved_scalar, 'scalar')]:
        record(f'real-{a.case:02}-{backend}-get-tend', name,
               to_host(driving.tendency(field)), expected[name] if padded else expected[name][:, :-1])
    expected = read_arrays(a.path/'interpolated.bin', [('rho', (a.nl, a.nc+1)),
                           ('scalar', (a.nl, a.nc+1)), ('scalars', (a.ns, a.nl, a.nc+1))])
    for field, name in [('rho_zz', 'rho'), (a.retrieved_scalar, 'scalar'), ('scalars', 'scalars')]:
        record(f'real-{a.case:02}-{backend}-state', name,
               to_host(driving.state_at(field, STEP, np.float32(.375))),
               expected[name] if padded else expected[name][..., :-1])


def test_real_boundary_cpu(real_columns, monkeypatch):
    a = real_columns
    driving = regional.RegionalDrivingState(inventory(a, monkeypatch), a.mesh,
                                           a.zz, scalar_names=a.scalar_names)
    compare_pool(a, driving, 'cpu', np.asarray)


def test_real_boundary_cuda(real_columns, monkeypatch):
    if os.environ.get('MPAS_REGIONAL_GPU') != '1':
        pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_v841 import CudaRegionalKernels, DeviceRegionalMasks, DeviceRegionalDrivingState
    a = real_columns
    kernels = CudaRegionalKernels(KernelCache())
    driving = DeviceRegionalDrivingState(inventory(a, monkeypatch),
        DeviceRegionalMasks.from_host(a.masks), cells_on_edge=cp.asarray(a.mesh.cellsOnEdge),
        zz=cp.asarray(a.zz), n_vert_levels=a.nl, kernels=kernels,
        scalar_names=a.scalar_names)
    compare_pool(a, driving, 'cuda', cp.asnumpy)
    cp.cuda.Stream.null.synchronize()
    if a.case == 0:
        record_cuda_compile(kernels, 'real-module')


def test_real_boundary_padded_cuda(real_columns, monkeypatch):
    if os.environ.get('MPAS_REGIONAL_GPU') != '1':
        pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_v841 import CudaRegionalKernels
    from hexcore.cuda_regional_forecast_v841 import PaddedRegionalDrivingState
    a = real_columns
    ne = a.mesh.cellsOnEdge.shape[0]
    coe = np.full((ne+1,2), a.nc, dtype='i4')
    coe[:-1] = np.where(a.mesh.cellsOnEdge<0, a.nc, a.mesh.cellsOnEdge)
    driving = PaddedRegionalDrivingState(inventory(a,monkeypatch),
        SimpleNamespace(n_cells=a.nc+1,n_edges=ne+1),
        cells_on_edge=cp.asarray(coe), zz=cp.asarray(np.pad(a.zz,((0,0),(0,1)),constant_values=1)),
        n_vert_levels=a.nl,kernels=CudaRegionalKernels(KernelCache()),
        scalar_names=a.scalar_names,garbage_cell=a.nc,garbage_edge=ne)
    compare_pool(a,driving,'padded-cuda',cp.asnumpy,padded=True)
    cp.cuda.Stream.null.synchronize()
