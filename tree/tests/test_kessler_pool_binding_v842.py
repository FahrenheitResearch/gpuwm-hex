"""Storage contract for Kessler on the foundation's actual device pool provider."""
from types import SimpleNamespace
import numpy as np
import pytest
from hexcore.mpasphys import kessler_v842 as scheme


@pytest.mark.parametrize('diagnostics', [False, True])
def test_binding_keeps_scalar_planes_and_native_words(monkeypatch, diagnostics):
    # The native provider packs only solve cells, at state time level two.
    class Provider(dict):
        pass
    p = Provider()
    p.session = SimpleNamespace(layout=SimpleNamespace(indices={
        ('state', 'index_qv'): 2, ('state', 'index_qc'): 4,
        ('state', 'index_qr'): 1}))
    p.native = SimpleNamespace(get_constant=lambda name: np.float32({
        'ep_2': .62, 'rho_w': 1000., 'r_d': 287.}[name]
        if name in ('ep_2', 'rho_w', 'r_d') else 1.))
    for pool, names in (
        ('state', ('theta_m', 'rho_zz')),
        ('mesh', ('zz', 'zgrid')),
        ('diag', ('exner', 'surface_pressure', 'dtheta_dt_mp', 'exner_base',
                  'pressure_base', 'pressure_p', 'rtheta_base', 'rtheta_p', 'relhum')),
        ('tend', ('tend_sfc_pressure', 'rt_diabatic_tend')),
        ('diag_physics', ('rainnc', 'rainncv', 'precipw', 'i_rainnc'))):
        p[pool] = {name: np.zeros((5, 3), dtype=np.float32) for name in names}
    p['state']['scalars'] = np.arange(4*5*3, dtype=np.float32).reshape(4, 5, 3)
    p['tend_physics'] = {'mp_top_level': np.array(4, dtype=np.int32)}
    p['scalar_state'] = {'dt_dyn': np.float32(1200)}
    monkeypatch.setattr(scheme, 'shared_math_header', lambda: 'shared header')
    result = scheme.bind_native_pools(p, SimpleNamespace(flags=SimpleNamespace(
        diagnostics=diagnostics)))
    state = result['state']
    for name, index in [('qv', 1), ('qc', 3), ('qr', 0)]:
        assert np.shares_memory(state[name], p['state']['scalars'])
        assert state[name].flags.c_contiguous
        state[name][0, 0] = 777
        assert p['state']['scalars'][index, 0, 0] == 777
    assert state['pressure_b'] is p['diag']['pressure_base']
    assert state['rho_zz'] is p['state']['rho_zz']
    assert ('relhum' in state) == diagnostics
    assert result['dt_dyn'].tobytes() == p['scalar_state']['dt_dyn'].tobytes()
    assert result['mp_top_level'] == 4
    assert result['constants']['ep2'].tobytes() == np.float32(.62).tobytes()
    assert result['constants']['rd'].tobytes() == np.float32(287).tobytes()
