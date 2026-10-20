"""RP6 surface precipitation window on the CUDA device (correct mode with config_pbl_interval).

The kernel must reproduce mpas_atmphys_lsm_shared.F precip_window_add/precip_window_reset word for word (single
precision, no contraction), close the water budget, hand a one-step window over unchanged, and leave the native
one-step fields exactly as they were once the window closes. The single-precision reference below repeats the
Fortran statement order; it is a test oracle, not model code.
"""
import numpy as np
import pytest

cp = pytest.importorskip('cupy')
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - CPU-only host
    pytest.skip('no CUDA device', allow_module_level=True)

from hexcore.mpasphys.precip_window_v842 import SurfacePrecipitationWindow

F = np.float32


class FakeNative:
    """Native diag_physics fields of nCells+1 words, as native_backend hands them out (copies)."""

    def __init__(self, cells):
        self.store = {name: np.zeros(cells + 1, F) for name in ('rainncv', 'raincv', 'snowncv', 'graupelncv', 'sr')}
        self._fields = {('diag_physics', name): (np.dtype(F), (cells + 1,), 1) for name in self.store}

    def get_field(self, pool, name, level=1):
        return self.store[name].copy()

    def put_field(self, pool, name, value, level=1, nlevels=1):
        self.store[name] = np.array(value, F)


def step_precipitation(n, cells):
    i = np.arange(1, cells + 1, dtype=F)
    rainncv = np.maximum(F(0), np.sin(F(0.37) * n + F(1.3) * i)).astype(F) * F(0.05) * i
    rainncv[(n + np.arange(1, cells + 1)) % 5 == 0] = 0
    rainncv[-1] = 0
    snow = (rainncv * F(0.5) * (F(1) + np.cos(F(0.21) * n * i)) * F(0.6)).astype(F)
    graupel = (rainncv * F(0.5) * (F(1) + np.sin(F(0.13) * n + i)) * F(0.4)).astype(F)
    sr = ((snow + graupel) / (rainncv + F(1e-12))).astype(F)
    raincv = (np.maximum(F(0), np.cos(F(0.29) * n - F(0.7) * i)) * F(0.02)).astype(F)
    if n % 3 == 0:
        # drizzle words below the smallest normal float, as the x1.2562 forecast produces (about 6e-39 mm)
        rainncv[1] = F(6.0536e-39) * F(n)
        snow[1] = F(6.0536e-39)
        graupel[1] = F(0)
        raincv[1] = F(1e-40)
    return dict(rainncv=rainncv, raincv=raincv, snowncv=snow, graupelncv=graupel, sr=sr)


def reference_add(ref, p):
    """precip_window_add, statement for statement, in float32."""
    ref['n'] += 1
    ref['nc'] = (ref['nc'] + p['rainncv']).astype(F)
    ref['frozen'] = (ref['frozen'] + (p['sr'] * p['rainncv']).astype(F)).astype(F)
    ref['c'] = (ref['c'] + p['raincv']).astype(F)
    ref['s'] = (ref['s'] + p['snowncv']).astype(F)
    ref['g'] = (ref['g'] + p['graupelncv']).astype(F)
    ratio = p['sr'].copy()
    use = (ref['n'] > 1) & (ref['nc'] > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        q = (ref['frozen'] / ref['nc']).astype(F)
    ratio[use] = np.minimum(F(1), q[use])
    ref['sr'] = ratio


@pytest.mark.parametrize('interval', [1, 2, 3, 15])
def test_RP6_precipitation_window_words(interval):
    cells, steps = 9, 47
    native = FakeNative(cells)
    selectors = {'microphysics': 'mp_thompson', 'convection': 'cu_ntiedtke'}
    window = SurfacePrecipitationWindow(native, cells, selectors, cp)
    zero = lambda: np.zeros(cells, F)
    ref = dict(n=np.zeros(cells, np.int32), nc=zero(), frozen=zero(), c=zero(), s=zero(), g=zero(), sr=zero())
    made = got = 0.0
    made_frozen = got_frozen = 0.0
    for n in range(1, steps + 1):
        p = step_precipitation(n, cells)
        for name, value in p.items():
            native.store[name][:cells] = value
            native.store[name][cells] = F(-999)          # the garbage cell is never touched
        made += float(np.sum(p['rainncv'], dtype=np.float64) + np.sum(p['raincv'], dtype=np.float64))
        made_frozen += float(np.sum(p['sr'].astype(np.float64) * p['rainncv'].astype(np.float64)))
        window.accumulate()
        reference_add(ref, p)
        if (n - 1) % interval == 0:
            window.open()
            seen = {name: native.store[name][:cells].copy() for name in p}
            # the land-surface and sea-ice slots read the native routine's window words
            assert np.array_equal(seen['rainncv'], ref['nc']) and np.array_equal(seen['raincv'], ref['c'])
            assert np.array_equal(seen['snowncv'], ref['s']) and np.array_equal(seen['graupelncv'], ref['g'])
            assert np.array_equal(seen['sr'], ref['sr'])
            if interval == 1:                              # a one-step window is the step's own words
                assert all(np.array_equal(seen[name], p[name]) for name in p)
            rainbl = (F(0) + seen['rainncv']).astype(F) + seen['raincv']
            got += float(np.sum(rainbl, dtype=np.float64))
            got_frozen += float(np.sum(seen['sr'].astype(np.float64) * seen['rainncv'].astype(np.float64)))
            window.close()
            # every later reader sees the step's own words again; the garbage cell is untouched
            assert all(np.array_equal(native.store[name][:cells], p[name]) for name in p)
            assert all(native.store[name][cells] == F(-999) for name in p)
            for key in ('nc', 'frozen', 'c', 's', 'g'):
                ref[key] = zero()
            ref['n'][:] = 0
    pending = float(np.sum(ref['nc'], dtype=np.float64) + np.sum(ref['c'], dtype=np.float64))
    assert abs(got + pending - made) <= 1e-5 * made
    pending_frozen = float(np.sum(ref['frozen'], dtype=np.float64))
    assert abs(got_frozen + pending_frozen - made_frozen) <= 1e-5 * made_frozen
    if interval > 1:
        # what the first RP6 fix handed over: only the call steps' own precipitation
        lost = 1 - sum(float(np.sum(step_precipitation(n, cells)['rainncv'], dtype=np.float64)
                             + np.sum(step_precipitation(n, cells)['raincv'], dtype=np.float64))
                       for n in range(1, steps + 1, interval)) / made
        assert lost > 0.3


def test_RP6_window_keeps_subnormal_words():
    """Fails with a cupy.RawModule build of the same kernel on the 5090: subnormal precipitation words were flushed
    to zero, so a one-step window differed from the step's own words (port vs plain, interval = dt)."""
    cells = 4
    native = FakeNative(cells)
    window = SurfacePrecipitationWindow(native, cells, {'microphysics': 'mp_thompson', 'convection': 'cu_ntiedtke'}, cp)
    words = np.array([6.0536e-39, 1e-40, 1.4e-45, 0.25], F)
    for name in ('rainncv', 'raincv', 'snowncv', 'graupelncv', 'sr'):
        native.store[name][:cells] = words
    window.accumulate()
    window.open()
    assert all(np.array_equal(native.store[name][:cells].view(np.uint32), words.view(np.uint32))
               for name in ('rainncv', 'raincv', 'snowncv', 'graupelncv', 'sr'))
    window.close()


def test_RP6_window_snapshot_round_trip_and_open_guard():
    cells = 5
    native = FakeNative(cells)
    window = SurfacePrecipitationWindow(native, cells, {'microphysics': 'mp_wsm6', 'convection': 'off'}, cp)
    assert window.present() == ('rainncv', 'sr', 'snowncv', 'graupelncv')
    for n in (1, 2):
        for name, value in step_precipitation(n, cells).items():
            native.store[name][:cells] = value
        window.accumulate()
    state = window.snapshot()
    other = SurfacePrecipitationWindow(native, cells, {'microphysics': 'mp_wsm6', 'convection': 'off'}, cp)
    other.restore(state)
    assert all(np.array_equal(state[k], v) for k, v in other.snapshot().items())
    window.open()
    with pytest.raises(RuntimeError, match='open'):
        window.snapshot()
    window.close()
    with pytest.raises(RuntimeError, match='not open'):
        window.close()


def test_resident_window_uses_current_device_fields_without_host_pool_reads():
    class NoHost:
        def get_field(self, *args, **kwargs):
            raise AssertionError('resident window read stale host pools')
        def put_field(self, *args, **kwargs):
            raise AssertionError('resident window wrote stale host pools')
    class Pools:
        level = 2
        def __init__(self):
            self.values = {name: cp.zeros(9, cp.float32) for name in
                           ('rainncv','raincv','snowncv','graupelncv','sr')}
        def field(self, pool, name):
            assert pool == 'diag_physics' and self.level == 1
            return self.values[name]
        def write_field(self, pool, name, value, level):
            assert pool == 'diag_physics' and level == 1
            self.values[name][...] = value
    pools = Pools()
    window = SurfacePrecipitationWindow(NoHost(), 9,
        {'microphysics':'mp_thompson','convection':'cu_grell_freitas'}, cp)
    window.device_pools = pools
    zero = lambda: np.zeros(9, F)
    ref = dict(n=np.zeros(9,np.int32),nc=zero(),frozen=zero(),c=zero(),s=zero(),g=zero(),sr=zero())
    for step in range(1,4):
        values = step_precipitation(step,9)
        for name,value in values.items():
            pools.values[name][...] = cp.asarray(value)
        window.accumulate()
        reference_add(ref,values)
    saved = {name:value.get().view(np.uint32).copy() for name,value in pools.values.items()}
    window.open()
    for name,key in (('rainncv','nc'),('raincv','c'),('snowncv','s'),('graupelncv','g'),('sr','sr')):
        np.testing.assert_array_equal(pools.values[name].get().view(np.uint32),ref[key].view(np.uint32))
    window.close()
    for name,words in saved.items():
        np.testing.assert_array_equal(pools.values[name].get().view(np.uint32),words)
    assert pools.level == 2
