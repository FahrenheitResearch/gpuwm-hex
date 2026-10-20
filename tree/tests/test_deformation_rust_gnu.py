"""A three-table or widened-libm bridge must never change native dycore weights."""
import numpy as np
import pytest
from hexcore import mixing_v841 as mixing, rust_hostprep, native_geometry_math, numerics
from test_mesh_validation import graded_sphere_mesh
from test_rust_hostprep import assert_bits


@pytest.mark.parametrize('halo', [False, True])
def test_all_five_gnu_tables_match_python(monkeypatch, halo):
    # The Rust bridge implements fast host libm, never the strict authority.
    monkeypatch.setattr(numerics, "_mode", "fast")
    if rust_hostprep.deformation_engine_digest() == 'python':
        pytest.skip('GNU five-table hostprep bridge not installed')
    mesh = graded_sphere_mesh(level=2)
    if halo:
        mesh.arrays['cellsOnCell'] = mesh.cellsOnCell.copy()
        mesh.arrays['cellsOnCell'][::7, 0] = -1
    direct = mixing.initialize_deformation_weights_v841.__wrapped_prep_cache__
    got = direct(mesh)
    monkeypatch.setattr(mixing, 'deformation_weights', lambda **kw: None)
    expected = direct(mesh)
    for field in ('coef_c2','coef_s2','coef_cs','coef_c','coef_s'):
        assert_bits(getattr(got, field), getattr(expected, field))


def test_rebound_geometry_uses_its_authority(monkeypatch):
    monkeypatch.setattr(numerics, "_mode", "fast")
    monkeypatch.setattr(native_geometry_math, 'sin', np.sin)
    assert not native_geometry_math.unmodified_gnu_float32()
    def wrong(**kw):
        raise AssertionError('GNU bridge cannot replace rebound geometry')
    monkeypatch.setattr(mixing, 'deformation_weights', wrong)
    mixing.initialize_deformation_weights_v841.__wrapped_prep_cache__(graded_sphere_mesh(level=1))
