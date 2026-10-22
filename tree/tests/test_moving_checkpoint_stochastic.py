"""The member's spectral clock and coefficients survive moving carrier seals."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from hexcore import moving_checkpoint as checkpoint


@pytest.mark.parametrize('blocks', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_sealed_boundary_carries_member_stochastic_snapshot(tmp_path, monkeypatch, blocks, enabled):
    class Device:
        id = 0
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class DeviceArray(np.ndarray):
        device = Device()
    cp = SimpleNamespace(ndarray=DeviceArray, asnumpy=np.asarray,
        cuda=SimpleNamespace(Device=Device,
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    coefficients = np.arange(12, dtype=np.float64).reshape(2, 6).view(DeviceArray)
    spectral = dict(time_seconds=7200., draws=18, coefficients=coefficients,
        process=dict(scales=[dict(coefficients=coefficients)],
                     streams=(dict(coefficients=coefficients),)))
    calls = []
    def snapshot(*, device):
        calls.append(device)
        return spectral
    values = {name: np.ones((1, 2), dtype=np.float32) for name in
        ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars', 'theta_m', 'exner',
         'density_perturbation', 'rho_theta_perturbation', 'pressure_perturbation',
         'normal_velocity', 'vertical_velocity')}
    atmosphere = SimpleNamespace(state=SimpleNamespace(time_seconds=7200., **values),
                                 saved=SimpleNamespace(**values))
    stack = dict(driver=SimpleNamespace(atmosphere=atmosphere),
        backend=SimpleNamespace(restart_state=lambda: dict(identity={},
            seam=dict(identity={}, scalars={}, arrays={}))))
    layout = SimpleNamespace(cell_l2g=np.array([0, 1]), n_owned_cells=2)
    tables = SimpleNamespace(layout=layout)
    active = SimpleNamespace(registers_empty=lambda: True,
        gather=lambda names: {name: values[name] for name in names}, scalars=None,
        plan=SimpleNamespace(classes=[tables], rates=np.array([1]), cell_class=np.array([0, 0])),
        G_cell=np.zeros(2), G_face=np.zeros(2), G_edge=np.zeros(2),
        stochastic=SimpleNamespace(snapshot=snapshot) if enabled else None,
        ntr=0, blocks=[SimpleNamespace(gid=0, device=0, tables=tables, index=0)], lanes={})
    path = tmp_path/'boundary.pkl'
    export = checkpoint.export_boundary_blocks if blocks else checkpoint.export_boundary
    export(path, active=active, stacks=[stack], previous=[None], forcing=[None],
        host=dict(prepared=SimpleNamespace(vertical=None, mesh=None), scalar_names=[]),
        physics_scope=SimpleNamespace(rank=lambda *args: nullcontext()))
    payload = checkpoint.load_boundary(path)
    assert calls == ([True] if enabled else [])
    if enabled:
        assert payload['stochastic']['time_seconds'] == 7200.
        assert payload['stochastic']['draws'] == 18
        np.testing.assert_array_equal(payload['stochastic']['coefficients'], coefficients)
        assert type(payload['stochastic']['coefficients']) is np.ndarray
        for collection in ('scales', 'streams'):
            carried = payload['stochastic']['process'][collection][0]['coefficients']
            assert type(carried) is np.ndarray
            np.testing.assert_array_equal(carried, coefficients)
    else:
        assert payload['stochastic'] is None
