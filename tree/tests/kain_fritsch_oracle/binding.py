"""Actual f4 native-view binder check, with f3 compiled native storage.

This CPU check verifies association and scalar words only, never forecast proof.
Numerical column execution is covered by full_driver.py and queued CUDA checks.
"""
import argparse, ctypes, hashlib, importlib.util, json, socket
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import hexcore.mpasphys
from hexcore.mpasphys import kain_fritsch_v842 as kf
from support import PARAMS, fixture, load_oracles, word_diff

p = argparse.ArgumentParser()
p.add_argument('--seam', type=Path, required=True)
p.add_argument('--loader', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--build-dir', type=Path, required=True)
p.add_argument('--constants-library', type=Path,
               help='standalone pristine compiled constants; does not certify f3 constants ABI')
p.add_argument('--reference-constants-library', type=Path,
               help='independent strict compiled constants to compare with the actual f3 export')
p.add_argument('--receipt', type=Path, required=True)
a = p.parse_args()
hexcore.mpasphys.__path__.append(str(a.seam.resolve()))
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
spec = importlib.util.spec_from_file_location('kf_binding_native', a.loader)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
_, _, tables = load_oracles(a.build_dir)
rows = []
with m.NativePhysics(a.library, test_only=True) as b:
    if a.reference_constants_library:
        reference = ctypes.CDLL(str(a.reference_constants_library.resolve()))
        values = np.empty(len(kf._NATIVE_CONSTANTS), np.float32)
        reference.kf_native_constants.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')]
        reference.kf_native_constants(values)
        exported = np.asarray([b.get_constant(name) for name in kf._NATIVE_CONSTANTS], np.float32)
        assert word_diff(exported, values) == 0
    if a.constants_library:
        constant_library = ctypes.CDLL(str(a.constants_library.resolve()))
        constant_words = np.empty(len(kf._NATIVE_CONSTANTS), np.float32)
        constant_library.kf_native_constants.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')]
        constant_library.kf_native_constants(constant_words)
        # Separate pristine Fortran copy bridge for the pre-f3 column contract.
        # Preserve the actual f3 storage API; no Python constant expressions.
        b.get_constant = lambda name: constant_words[kf._NATIVE_CONSTANTS.index(name)]
    nz, nc, owned = 55, 3, 2
    f, s, v = fixture(nz)
    b.put_variable('dt_dyn', np.asarray(60, np.float32))
    b.put_variable('n_cu', np.asarray(5, np.int32))
    b.config('config_len_disp', 0.)
    b.config('config_len_disp', 100000.)
    for name, value in zip(kf.FLAGS[4:], (True, False, True)):
        b.put_variable(name, np.asarray(value))
    for i, name in enumerate(kf._PREPARED_FIELDS):
        b.put_variable(name, np.repeat(f[i, :nz+(name=='w_p')][None,:,None], nc, axis=0).copy())
    b.put_field('mesh', 'meshDensity', np.ones(nc, np.float32))
    b.put_field('mesh', 'areaCell', np.repeat(v[1], nc))
    b.put_field('diag_physics', 'w0avg', np.repeat(s[0,:nz,None], nc, axis=1))
    for i, name in enumerate(kf._SURFACE_POOL):
        if i >= 2: b.put_field('diag_physics', name, np.repeat(v[i], nc))
    for i, name in enumerate(kf.STATE):
        if i >= 1: b.put_field('tend_physics', name, np.repeat(s[i,:nz,None], nc, axis=1))
    layout = SimpleNamespace(dimensions={'nCellsSolve':owned}, configs={'config_len_disp':0.},
        pools=('mesh','diag_physics','tend_physics'),
        fields=[{'pool':pool,'name':name,'pool_name':name,'dimensions':['nCells']}
                for pool,name in b._fields])
    provider = NativeDevicePoolViews(SimpleNamespace(native=b,layout=layout), np)
    provider.kain_fritsch_tables = tables
    provider.kain_fritsch_native_arguments = kf.load_native_arguments(
        {'warm_rain_word':32765,'witness_sha256':'a'*64}, np)
    call = SimpleNamespace(slot='convection',scheme=kf.SCHEME,backend='hex-cuda')
    bound = provider.begin(call, {'module':'kain_fritsch_v842','bind':'bind_native_pools'})
    config = bound['kain_fritsch']
    expected_constants = np.asarray([b.get_constant(name) for name in kf._NATIVE_CONSTANTS], np.float32)
    assert word_diff(config['params'][3:],expected_constants) == 0
    assert np.array_equal(config['flags'][3:],np.array([32765,1,0,1],np.int32))
    assert bound['dt_dyn'] == np.float32(60) and bound['n_cu'] == np.int32(5)
    assert word_diff(np.asarray(bound['len_disp']), np.asarray(b.get_real_config('config_len_disp'))) == 0
    assert bound['len_disp'] == np.float32(100000)
    assert config['tables'] is tables
    for name in kf._PREPARED_FIELDS:
        assert bound['prepared'][name].shape == (nz+(name=='w_p'),owned)
    for name in ('w0avg',*kf._SURFACE_POOL[2:]):
        view = bound['diag_physics'][name]
        view[...] = np.float32(.125)
    for name in kf.STATE[1:]:
        bound['tend_physics'][name][...] = np.float32(.25)
    provider.commit()
    # Subsequent calls must reread the native pointer, including a zero word.
    displacement_rows = []
    for value in (125000., 0., 75000.):
        b.config('config_len_disp', value)
        rebound = provider.begin(call, {'module':'kain_fritsch_v842','bind':'bind_native_pools'})
        actual = b.get_real_config('config_len_disp')
        assert word_diff(np.asarray(rebound['len_disp']), np.asarray(actual)) == 0
        displacement_rows.append({'native_word':int(np.asarray(actual).view(np.uint32)),
                                  'differing_words':0})
        provider.commit()
    for pool,name in b._fields:
        if pool == 'mesh': continue
        field = b.get_field(pool,name)
        expected = np.float32(.25 if pool=='tend_physics' else .125)
        assert np.all(field[...,:owned] == expected)
        initial = v[kf._SURFACE_POOL.index(name)] if name in kf._SURFACE_POOL else np.float32(0)
        assert np.all(field[...,owned:] == initial)
        rows.append({'pool':pool,'field':name,'solve_cells':owned,'halo_cells':1,'differing_words':0})
receipt = {'schema':'mpas-kf-native-pool-binding/v1','execution_host':socket.gethostname(),
    'process_count':1,'numerical_physics_execution':False,'forecast_proven':False,
    'f3_constant_export_proven':a.constants_library is None,
    'independent_native_constant_comparison':a.reference_constants_library is not None,
    'standalone_constants_library':str(a.constants_library) if a.constants_library else None,
    'constant_words_differing':0,'raw_warm_rain_word':32765,'rows':rows,
    'live_displacement':displacement_rows,'immutable_namelist_displacement':0.,
    'artifacts':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (a.library,a.loader,Path(__file__),a.seam/'device_pools_v842.py',
         Path(kf.__file__), *([a.constants_library] if a.constants_library else []))}}
if a.reference_constants_library:
    path = a.reference_constants_library
    receipt['artifacts'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('rows','artifacts')}))
