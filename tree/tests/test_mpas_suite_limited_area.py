"""Limited-area runs on the MPAS suite route (parity run group 1, #3 and #4).

Defect: ``--lbc-dir`` never reached the suite route, so every region, point, country, swath and cycle fine forecast
with ``--physics-suite convection_permitting`` built the global dycore on a cull and stopped in its terrain coupling
("cells_on_edge contains a missing or out-of-range cell").  These tests fail on rel/hex-0.4.4 222d6e2c.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.forecast_v842 import bind_limited_area, namelist_option


def _arguments(lbc_dir=None, **extra):
    return SimpleNamespace(lbc_dir=lbc_dir, **extra)


def test_lbc_dir_sets_mpas_config_apply_lbcs(tmp_path):
    text = bind_limited_area(_arguments(tmp_path), "&physics\nconfig_physics_suite='convection_permitting'\n/\n")
    assert namelist_option(text, 'limited_area', 'config_apply_lbcs') is True
    assert namelist_option(text, 'physics', 'config_physics_suite') == 'convection_permitting'


def test_no_lbc_dir_leaves_a_global_run_unchanged():
    text = "&physics\nconfig_physics_suite='convection_permitting'\n/\n"
    assert bind_limited_area(_arguments(), text) == text


def test_namelist_limited_area_without_series_is_refused():
    with pytest.raises(ValueError, match='needs its lateral-boundary series'):
        bind_limited_area(_arguments(), "&limited_area\nconfig_apply_lbcs=.true.\n/\n")


@pytest.mark.parametrize('extra, named', [
    (dict(samehost_devices='0,1'), 'more than one card'),
    (dict(iau_window_seconds=3600.0), 'IAU'),
    (dict(jedi_background=True), 'JEDI'),
    (dict(stochastic='{"sppt":true}'), 'stochastic'),
    (dict(chemistry_config='chem.json'), 'chemistry'),
])
def test_combinations_the_regional_dycore_does_not_carry_are_refused_by_name(tmp_path, extra, named):
    with pytest.raises(ValueError, match=named):
        bind_limited_area(_arguments(tmp_path, **extra), '')


def test_regional_suite_runs_load_every_moist_species():
    # The regional qv-only coefficients stay for the dycore's own one-species regional runs; a suite run (which
    # passes its Registry mass-loading count) takes the global atm_compute_moist_coefficients path.
    import inspect
    from hexcore.cuda_driver import CudaDryDycoreDriver
    source = inspect.getsource(CudaDryDycoreDriver._step_device_v841)
    assert 'regional_suite_loading' in source


def test_projection_geometry_names_the_garbage_cell():
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no CUDA device')
    from hexcore.mpasphys.resident_v842 import regional_physics_geometry
    n_cells, n_edges = 3, 4
    # Native one-based cellsOnEdge (2, nEdges); the last edge is a ring-7 edge whose second endpoint is the garbage
    # cell nCells+1.
    cells_on_edge = np.array([[1, 2, 3, 3], [2, 3, 1, n_cells + 1]], dtype=np.int32)
    vectors = np.arange(3 * n_cells, dtype=np.float32).reshape(3, n_cells)
    fields = {'cellsOnEdge': cells_on_edge, 'east': vectors, 'north': vectors + 1,
              'edgeNormalVectors': np.ones((3, n_edges), np.float32)}
    provider = SimpleNamespace(cp=cp, field=lambda pool, name: cp.asarray(fields[name]),
                               session=SimpleNamespace(layout=SimpleNamespace(
                                   dimensions={'nCellsSolve': n_cells, 'nEdgesSolve': n_edges})))
    geometry = regional_physics_geometry(provider)
    assert geometry.n_cells == n_cells + 1 and geometry.regional_garbage_cell == n_cells
    assert cp.asnumpy(geometry.cells_on_edge)[-1].tolist() == [2, n_cells]
    assert not cp.asnumpy(geometry.east_cell)[n_cells].any()


def test_lb1_boundary_forcing_columns_follow_the_mode(monkeypatch):
    from hexcore.mpasphys.runtime_v842 import ResidentNativeCoupling
    coupling = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    coupling.cp = np
    coupling.regional = SimpleNamespace(bdy_mask_cell=np.array([0, 0, 3, 7, 7], np.int32))
    coupling.session = SimpleNamespace(config=SimpleNamespace(mode='correct'))
    assert coupling.boundary_forcing_columns().tolist() == [False, False, True, True, True]
    for mode, knob in (('exact', '1'), ('correct', '0')):
        fresh = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
        fresh.cp, fresh.regional = np, coupling.regional
        fresh.session = SimpleNamespace(config=SimpleNamespace(mode=mode))
        monkeypatch.setenv('GPUWM_HEX_MPAS_LB1', knob)
        assert fresh.boundary_forcing_columns() is None
    monkeypatch.delenv('GPUWM_HEX_MPAS_LB1')
    closed = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    closed.session = coupling.session
    assert closed.boundary_forcing_columns() is None


def test_stored_zero_neighbours_become_the_garbage_element():
    from hexcore.mpasphys.runtime_v842 import remap_limited_area_sentinels
    store = {('mesh', 'cellsOnEdge'): np.array([[1, 2, 0], [2, 0, 0]], np.int32),
             ('mesh', 'edgesOnVertex'): np.array([[1, 0], [2, 3]], np.int32)}
    native = SimpleNamespace(_fields={k: (v.dtype, v.shape, 1) for k, v in store.items()},
                             get_field=lambda pool, name: store[pool, name].copy(),
                             put_field=lambda pool, name, value, nlevels=1: store.__setitem__((pool, name), value))
    layout = SimpleNamespace(dimensions={'nCells': 2, 'nEdges': 3, 'nVertices': 1},
                             fields=[{'pool': 'mesh', 'name': n, 'pool_name': n}
                                     for n in ('cellsOnEdge', 'edgesOnVertex')])
    counts = remap_limited_area_sentinels(native, layout)
    assert counts == {'cellsOnEdge': 3, 'edgesOnVertex': 1}
    assert store['mesh', 'cellsOnEdge'].tolist() == [[1, 2, 3], [2, 3, 3]]
    assert store['mesh', 'edgesOnVertex'].tolist() == [[1, 4], [2, 3]]
