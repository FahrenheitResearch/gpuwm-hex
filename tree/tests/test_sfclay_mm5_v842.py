"""Bitwise tests against strict pristine MPAS, including previous-call state.

MPAS_MM5_ORACLE is the directory built with sfclay_mm5_oracle/prepare.py.
GPU tests require MPAS_MM5_GPU=1 and must run through the GPU queue/the build queue.
These are synthetic columns, not the f1 or whole-forecast acceptance proof.
"""
from pathlib import Path
import ctypes
import importlib.util
import os
import numpy as np
import pytest
from hexcore.mpasphys import sfclay_mm5_v842 as mm5

SUPPORT = Path(__file__).with_name('sfclay_mm5_oracle')
OPTIONS = ((1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (1, 2, 0, 0, 0),
           (1, 0, 1, 0, 0), (1, 0, 2, 0, 0), (0, 0, 0, 0, 0),
           (1, 0, 0, 1, 0), (1, 0, 0, 0, 1))
PTR = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


def test_live_native_displacement_config_words():
    """Read updated native pool storage, including signed zero, without MPI/GPU."""
    path = os.environ.get('MPAS_MM5_NATIVE_LIBRARY')
    if not path:
        pytest.skip('set MPAS_MM5_NATIVE_LIBRARY to the sealed f3 live-config ABI')
    from hexcore.mpasphys.native_backend import NativePhysics
    with NativePhysics(path, test_only=True) as native:
        for word in (0x3f800001, 0x80000000, 0x47abcdef):
            value = np.array(word, dtype=np.uint32).view(np.float32)[()]
            native.config('config_len_disp', value)
            actual = mm5._native_len_disp(native)
            assert actual.dtype == np.float32
            assert actual.view(np.uint32) == word


def test_compiled_artifact_cannot_hide_changed_source_or_kernel(tmp_path):
    """A stable artifact filename must not admit different source/kernel words."""
    import hashlib
    import json
    ptx = tmp_path / 'module.cubin'
    ptx.write_bytes(b'guard fixture only; never loaded on a device')
    seal = dict(schema='mpas.mm5-cuda-artifact/v1', cubin_file=ptx.name,
                cubin_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest(),
                cuda_source_sha256=hashlib.sha256(
                    (mm5._shared_math_source() + mm5.CUDA_SOURCE).encode()).hexdigest(),
                options=[*mm5.COMPILE_OPTIONS, '--gpu-architecture=sm_120'])
    path = tmp_path / 'module.json'
    path.write_text(json.dumps(seal))
    assert mm5._read_cuda_artifact(path, '120') == ptx
    path.write_text(json.dumps({**seal, 'cuda_source_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='source/options'):
        mm5._read_cuda_artifact(path, '120')
    path.write_text(json.dumps(seal))
    ptx.write_bytes(b'different compiled kernel at the same path')
    with pytest.raises(ValueError, match='CUBIN hash changed'):
        mm5._read_cuda_artifact(path, '120')


@pytest.fixture(scope='module')
def oracle():
    path = os.environ.get('MPAS_MM5_ORACLE')
    if not path:
        pytest.skip('set MPAS_MM5_ORACLE to a prepared strict Fortran build')
    native = ctypes.CDLL(str(Path(path) / 'liboracle.so'))
    host = ctypes.CDLL(str(Path(path) / 'libhost.so'))
    native.oracle_init()
    native.oracle_run.argtypes = [PTR, PTR] + [ctypes.c_int] * 7
    host.host_run.argtypes = [PTR, PTR, PTR] + [ctypes.c_int] * 7
    host.host_tables.argtypes = [PTR]
    for library, prefix in ((native, 'oracle'), (host, 'host')):
        getattr(library, prefix + '_driver_from').argtypes = (
            [PTR] * 5 + [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int])
        getattr(library, prefix + '_driver_to').argtypes = (
            [PTR] * 4 + [ctypes.c_int, ctypes.c_float])
        getattr(library, prefix + '_diagnostics').argtypes = (
            [PTR] * 5 + [ctypes.c_int, ctypes.c_int, ctypes.c_float])
        getattr(library, prefix + '_wrf_diagnostics').argtypes = (
            [PTR] * 2 + [ctypes.c_int, ctypes.c_int])
    yield native, host
    native.oracle_finalize()


@pytest.fixture(scope='module')
def columns():
    spec = importlib.util.spec_from_file_location('mm5_synthetic', SUPPORT / 'synthetic_columns.py')
    recipes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipes)
    cols = recipes.build_columns()
    # Strong stability with previous UST below .01 reaches classic's RMOL reset.
    cols.append(('stable-old-small-ust', {**cols[8][1], 'ust': np.float32(.001),
                                        'mol': np.float32(.5), 'u': np.float32(.01),
                                        'v': np.float32(0), 'tsk': np.float32(240)}))
    cols.append(('boundary-land-mask', {**cols[0][1], 'xland': np.float32(1.5)}))
    # MPAS sf_sfclayrev.F90:941: ten-iteration Richardson return.
    cols.append(('richardson-iteration-limit', {**cols[0][1],
        'znt': np.float32(.31157806515693665), 'ust': np.float32(.2),
        'mol': np.float32(.5), 'hfx': np.float32(0), 'qfx': np.float32(0),
        'u': np.float32(.10627780854701996), 't': np.float32(275.9173278808594),
        'p': np.float32(59910.45703125), 'psfc': np.float32(59910.45703125),
        'dz8w': np.float32(2.881484031677246), 'tsk': np.float32(242.8180389404297),
        'pblh': np.float32(1315.4410400390625), 'xland': np.float32(1)}))
    # MPAS sf_sfclayrev.F90:943: equal residuals after a defined solver update.
    # Extreme positive roughness remains normal float32; every output is finite.
    cols.append(('richardson-equal-residual', {**cols[0][1],
        'znt': np.float32(1.333521443139104e-26), 'mol': np.float32(.5),
        'hfx': np.float32(0), 'qfx': np.float32(0), 'qsfc': np.float32(1e-8),
        'u': np.float32(.1), 'v': np.float32(0),
        't': np.float32(287.2928466796875), 'qv': np.float32(0),
        'p': np.float32(1e5), 'psfc': np.float32(1e5), 'dz8w': np.float32(20),
        'tsk': np.float32(280), 'dx': np.float32(1000), 'xland': np.float32(1)}))
    n = len(cols)
    a = np.zeros((len(mm5.FIELDS), n), np.float32)
    names = dict(ux='u', vx='v', t1d='t', qv1d='qv', p1d='p', dz8w1d='dz8w', psfcpa='psfc')
    for i, (_, col) in enumerate(cols):
        for j, name in enumerate(mm5.FIELDS):
            a[j, i] = col.get(names.get(name, name), 0)
    a[mm5.FIELDS.index('water_depth')] = np.resize(
        np.array([0, 9, 10, 55, 100, 101, 150], np.float32), n)
    for name, value in (('chs', .125), ('chs2', .25), ('cqs2', .375), ('lh', 15.5)):
        a[mm5.FIELDS.index(name)] = value
    r, rv = np.float32(287), np.float32(461.6)
    cp = np.float32(np.float32(7) * r / np.float32(2))
    c = np.array([cp, 9.80616, r/cp, r, 2.5e6, .6112, 17.67, 29.65,
                  273.15, rv/r-1, r/rv, .4, 1e5], np.float32)
    return a, c


def identical(actual, expected):
    np.testing.assert_array_equal(actual.view('u4'), expected.view('u4'))


@pytest.fixture(scope='module', params=['host', 'cuda'])
def backend(request, oracle):
    if request.param == 'cuda':
        if os.environ.get('MPAS_MM5_GPU') != '1':
            pytest.skip('CUDA test requires the GPU slot and MPAS_MM5_GPU=1')
        import cupy as cp
        return 'cuda', cp, oracle[0], None
    return 'host', np, *oracle


@pytest.mark.parametrize('revised', [0, 1])
@pytest.mark.parametrize('options', OPTIONS)
def test_complete_columns(backend, columns, revised, options):
    kind, xp, native, host = backend
    a, c = columns
    n = a.shape[1]
    expected, actual = a.copy(), xp.array(a)
    constants = xp.array(c)
    if host:
        tables = np.empty(6006, np.float32)
        host.host_tables(tables)
    for _ in range(3):
        native.oracle_run(expected, c, n, revised, *options)
        if host:
            host.host_run(actual, c, tables, n, revised, *options)
        else:
            mm5.launch_columns(actual, constants, scheme=mm5.SCHEMES[0 if revised else 1],
                               isfflx=options[0], isftcflx=options[1], iz0tlnd=options[2],
                               scm_force_flux=options[3], shalwater_z0=options[4])
        identical(actual if host else xp.asnumpy(actual), expected)


@pytest.mark.parametrize('revised', [0, 1])
def test_subnormal_wind_reaches_ten_metre_outputs(backend, columns, revised):
    """Both MM5 routines must preserve subnormal winds through multiplication.

    MPAS sf_sfclayrev.F90:763-764; module_sf_sfclay.F:800-801.
    Comparing U10/V10 as well as the inputs catches compiler-forced FTZ.
    """
    kind, xp, native, host = backend
    original, c = columns
    a = np.repeat(original[:, :1], 4, axis=1)
    wind = np.array([0x007fffff, 0x00400000, 0x807fffff, 0x80400000], np.uint32).view(np.float32)
    a[mm5.FIELDS.index('ux')] = wind
    a[mm5.FIELDS.index('vx')] = wind[::-1]
    expected = a.copy()
    native.oracle_run(expected, c, 4, revised, 1, 0, 0, 0, 0)
    for name in ('u10', 'v10'):
        magnitude = expected[mm5.FIELDS.index(name)].view(np.uint32) & 0x7fffffff
        assert np.all((magnitude > 0) & (magnitude < 0x00800000))
    actual = xp.array(a)
    if host:
        tables = np.empty(6006, np.float32)
        host.host_tables(tables)
        host.host_run(actual, c, tables, 4, revised, 1, 0, 0, 0, 0)
    else:
        mm5.launch_columns(actual, xp.array(c), scheme=mm5.SCHEMES[0 if revised else 1])
    identical(actual if host else xp.asnumpy(actual), expected)


@pytest.mark.parametrize('iz0tlnd', [3, 9])
@pytest.mark.parametrize('revised', [0, 1])
def test_native_uninitialized_czil_continues(backend, columns, revised, iz0tlnd):
    """Never refuse native's unassigned source alternatives.

    MPAS sf_sfclayrev.F90:713-721; module_sf_sfclay.F:781-789.
    Native words depend on call history. Only immutable inputs are asserted;
    this is a continuation check, not a numerical identity claim for outputs.
    MPAS driver fixes IZ0TLND=0 (mpas_atmphys_driver_sfclayer.F:30-34).
    """
    kind, xp, native, host = backend
    a, c = columns
    actual = xp.array(a)
    if host:
        tables = np.empty(6006, np.float32)
        host.host_tables(tables)
        host.host_run(actual, c, tables, a.shape[1], revised, 1, 0, iz0tlnd, 0, 0)
    else:
        mm5.launch_columns(actual, xp.array(c), scheme=mm5.SCHEMES[0 if revised else 1],
                           iz0tlnd=iz0tlnd)
    result = actual if host else xp.asnumpy(actual)
    np.testing.assert_array_equal(result[34:48].view('u4'), a[34:48].view('u4'))


@pytest.mark.parametrize('frac', [0, 1])
@pytest.mark.parametrize('revised', [0, 1])
def test_full_driver(backend, columns, frac, revised):
    kind, xp, native, host = backend
    a, c = columns
    n = a.shape[1]
    md = np.linspace(.1, 12, n, dtype=np.float32)
    sst = np.linspace(260, 305, n, dtype=np.float32)
    ice = np.resize(np.array([0, .1, .100001, .499999, .5, .9, 1, 1.1, -.1], np.float32), n)
    held = a[mm5.FIELDS.index('regime')].copy()
    expected, sea = a.copy(), np.empty_like(a)
    native.oracle_driver_from(expected, sea, md, sst, ice, n, 3000, .1, frac)
    native.oracle_run(expected, c, n, revised, 1, 0, 0, 0, 0)
    if frac:
        native.oracle_run(sea, c, n, revised, 1, 0, 0, 0, 0)
        native.oracle_driver_to(expected, sea, held, ice, n, .1)
    if host:
        actual, other = a.copy(), np.empty_like(a)
        tables = np.empty(6006, np.float32); host.host_tables(tables)
        host.host_driver_from(actual, other, md, sst, ice, n, 3000, .1, frac)
        host.host_run(actual, c, tables, n, revised, 1, 0, 0, 0, 0)
        if frac:
            host.host_run(other, c, tables, n, revised, 1, 0, 0, 0, 0)
            host.host_driver_to(actual, other, held, ice, n, .1)
        identical(actual, expected)
    else:
        atmosphere = {k: xp.array(a[mm5.FIELDS.index(k)]) for k in mm5.INPUTS[:7]}
        diag = {mm5._MPAS_NAMES.get(k, k): xp.array(a[j]) for j, k in enumerate(mm5.OUTPUTS)}
        diag.update(hpbl=xp.array(a[42]), mavail=xp.array(a[43]))
        sfc = dict(skintemp=xp.array(a[41]), xland=xp.array(a[44]), sst=xp.array(sst), xice=xp.array(ice))
        sea_exchange = {name: xp.empty(n, dtype=xp.float32) for name in mm5.SEA_EXCHANGE}
        mm5.run_surface_layer(atmosphere, diag, sfc, xp.array(md), xp.array(c),
                              scheme=mm5.SCHEMES[0 if revised else 1], config_len_disp=3000,
                              config_frac_seaice=bool(frac), xice_threshold=.1,
                              sea_exchange=sea_exchange)
        for j, name in enumerate(mm5.OUTPUTS):
            identical(xp.asnumpy(diag[mm5._MPAS_NAMES.get(name, name)]), expected[j])
        if frac:
            # MPAS driver_seaice.F:390-399 needs unblended open-water arrays.
            for name in mm5.SEA_EXCHANGE:
                identical(xp.asnumpy(sea_exchange[name]), sea[mm5.FIELDS.index(name)])


@pytest.mark.parametrize('lsm', [0, 1, 2])
def test_mpas_diagnostics(backend, columns, lsm):
    kind, xp, native, host = backend
    a, c = columns
    a = a.copy(); n = a.shape[1]
    # MPAS sfc_diagnostics.F:100: unordered land mask takes the land arm.
    # Exercise that SELECT CASE branch without feeding NaN to MM5 lookups.
    a[mm5.FIELDS.index('xland'), 0] = np.float32(np.nan)
    for name in ('chs2', 'cqs2'):
        a[mm5.FIELDS.index(name)] = np.resize(np.array([0, 1.e-6, 1.e-5, .25, -.1], np.float32), n)
    ice = np.resize(np.array([0, .09, .1, .5, 1], np.float32), n)
    q, t = np.linspace(.001, .02, n, dtype=np.float32), np.linspace(250, 300, n, dtype=np.float32)
    expected, actual = a.copy(), xp.array(a)
    native.oracle_diagnostics(expected, c, ice, q, t, n, lsm, .1)
    if host:
        host.host_diagnostics(actual, c, ice, q, t, n, lsm, .1)
    else:
        diag = {mm5._MPAS_NAMES.get(name, name): actual[j] for j, name in enumerate(mm5.OUTPUTS)}
        sfc = dict(skintemp=actual[41], xland=actual[44], xice=xp.array(ice))
        mm5.run_surface_diagnostics({'psfcpa': actual[40]}, diag, sfc, xp.array(c),
                                    lsm_scheme=('other', 'sf_noah', 'sf_noahmp')[lsm],
                                    xice_threshold=.1, q2mxy=xp.array(q), t2mxy=xp.array(t))
    identical(actual if host else xp.asnumpy(actual), expected)


@pytest.mark.parametrize('ua', [0, 1])
def test_wrf_diagnostics(backend, columns, ua):
    kind, xp, native, host = backend
    a, c = columns
    a = a.copy(); n = a.shape[1]
    for name in ('chs2', 'cqs2'):
        a[mm5.FIELDS.index(name)] = np.resize(np.array([0, 1.e-6, 1.e-5, .25, -.1], np.float32), n)
    # SFCDIAGS:47-57 uses IF (<threshold)/ELSE, including unordered input.
    a[mm5.FIELDS.index('cqs2'), :4] = np.array(
        [0x7fc00000, 0x7fc01234, 0xffc05678, 0x7f800001], dtype='u4').view('f4')
    expected, actual = a.copy(), xp.array(a)
    native.oracle_wrf_diagnostics(expected, c, n, ua)
    if host:
        host.host_wrf_diagnostics(actual, c, n, ua)
    else:
        mm5.wrf_surface_diagnostics(actual, xp.array(c), ua_phys=bool(ua))
    identical(actual if host else xp.asnumpy(actual), expected)


@pytest.mark.parametrize('revised,mask,options',
    [(revised, mask, options)
     for revised, mask in [(0, 255)] + [(1, m) for m in (7, 15, 31, 63, 127, 255)]
     for options in OPTIONS if not (revised and (options[0] != 1 or options[3] == 1))])
def test_pristine_wrappers(backend, columns, revised, options, mask):
    """Reach MPAS wrappers/pre_run and revised optional output absence.

    Classic SFCLAY:251 dereferences each optional array unconditionally;
    absence is valid at SFCLAY1D, exercised separately, but crashes SFCLAY.
    """
    kind, xp, native, host = backend
    a, c = columns
    n = a.shape[1]
    native.oracle_run_wrapped.argtypes = [PTR, PTR] + [ctypes.c_int] * 8
    expected, actual = a.copy(), xp.array(a)
    if host:
        tables = np.empty(6006, np.float32); host.host_tables(tables)
    for _ in range(3):
        native.oracle_run_wrapped(expected, c, n, revised, *options, mask)
        if host:
            host.host_run_optional.argtypes = [PTR, PTR, PTR] + [ctypes.c_int] * 7 + [ctypes.c_uint]
            host.host_run_optional(actual, c, tables, n, revised, *options, mask)
        else:
            mm5.launch_columns(actual, xp.array(c), scheme=mm5.SCHEMES[0 if revised else 1],
                               isfflx=options[0], isftcflx=options[1], iz0tlnd=options[2],
                               scm_force_flux=options[3], shalwater_z0=options[4], optional_mask=mask)
        result = actual if host else xp.asnumpy(actual)
        # Every word in each selected wrapper call is asserted. The alternative
        # flags that read unassigned LH_HV (module_sf_sfclayrev.F:128,225) have
        # separate native variation witnesses; the actual driver never uses them.
        identical(result, expected)


@pytest.mark.parametrize('revised', [0, 1])
@pytest.mark.parametrize('mask', [0, 1, 2, 4, 8, 16, 32, 64, 128, 15, 31, 63, 127, 255])
def test_optional_absence(backend, columns, revised, mask):
    kind, xp, native, host = backend
    a, c = columns
    n = a.shape[1]
    native.oracle_run_optional.argtypes = [PTR, PTR] + [ctypes.c_int] * 8
    expected, actual = a.copy(), xp.array(a)
    if host:
        host.host_run_optional.argtypes = [PTR, PTR, PTR] + [ctypes.c_int] * 7 + [ctypes.c_uint]
        tables = np.empty(6006, np.float32); host.host_tables(tables)
    for _ in range(3):
        native.oracle_run_optional(expected, c, n, revised, 1, 2, 2, 1, 0, mask)
        if host:
            host.host_run_optional(actual, c, tables, n, revised, 1, 2, 2, 1, 0, mask)
        else:
            mm5.launch_columns(actual, xp.array(c), scheme=mm5.SCHEMES[0 if revised else 1],
                               isftcflx=2, iz0tlnd=2, scm_force_flux=1, optional_mask=mask)
        identical(actual if host else xp.asnumpy(actual), expected)
