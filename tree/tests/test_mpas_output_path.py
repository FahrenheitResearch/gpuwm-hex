"""G1 output path: maxima deferred off the publication path, and the gather/multi-block
diagnostic kernels give the same words as the native-order kernels they replace."""
import os
import types

import numpy as np
import pytest


def _deferring_diagnostics():
    from hexcore.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    calls = []
    diagnostics = ResidentOutputDiagnostics.__new__(ResidentOutputDiagnostics)
    diagnostics.accumulates = True
    diagnostics.pending = None
    diagnostics.updates = diagnostics.deferred_updates = 0
    diagnostics.last_step = None
    diagnostics._solve_primary = lambda: calls.append('solve')
    # A horizontal without solve_vorticity takes the full primary solve.
    diagnostics.dycore = types.SimpleNamespace(atmosphere=None, horizontal=object())

    class Convective:
        def update(self):
            calls.append('update')
    diagnostics.convective = Convective()
    return diagnostics, calls


def test_deferred_maxima_fold_once_on_the_next_synchronize():
    diagnostics, calls = _deferring_diagnostics()
    diagnostics.flush()
    assert calls == []
    diagnostics.defer(step=4, when='t4')
    assert calls == [] and diagnostics.pending == (4, 't4')
    diagnostics.flush()
    assert calls == ['solve', 'update'] and diagnostics.pending is None
    assert (diagnostics.updates, diagnostics.deferred_updates, diagnostics.last_step) == (1, 1, 4)
    diagnostics.flush()
    assert calls == ['solve', 'update']


def test_unfolded_maxima_refuse_to_be_skipped():
    # The breakage it prevents: a step's maxima silently lost, or folded on a later state.
    diagnostics, _ = _deferring_diagnostics()
    diagnostics.defer(step=1, when='t1')
    with pytest.raises(RuntimeError, match='step 1'):
        diagnostics.defer(step=2, when='t2')
    with pytest.raises(RuntimeError, match='step 1'):
        diagnostics.before_write(step=2, when='t2')


def test_no_accumulation_defers_nothing():
    diagnostics, calls = _deferring_diagnostics()
    diagnostics.accumulates = False
    diagnostics.defer(step=1, when='t1')
    diagnostics.flush()
    assert diagnostics.pending is None and calls == []


@pytest.fixture
def cp():
    if os.environ.get('MPAS_CONVECTIVE_GPU') != '1':
        pytest.skip('CUDA tests require a card and MPAS_CONVECTIVE_GPU=1')
    import cupy
    return cupy


@pytest.mark.gpu
@pytest.mark.parametrize('ordered', [False, True])
def test_helicity_gather_matches_native_scatter_bitwise(cp, ordered):
    from hexcore.mpasdiag import convective_v842 as cv, convective_rrfs as cr
    rng = np.random.default_rng(7)
    nc, nv, degree, nl = 2000, 4000, 3, 55
    cov = rng.integers(1, nc + 2, size=(nv, degree)).astype(np.int32)  # includes the nCells+1 sentinel
    kite = rng.random((nv, degree)).astype(np.float32) * 1e8
    vort = (rng.standard_normal((nv, nl)) * 1e-4).astype(np.float32)
    cov, kite, vort = cp.asarray(cov), cp.asarray(kite), cp.asarray(vort)
    order = cp.asarray(rng.permutation(nv)) if ordered else None
    for module in (cv._module(cp.cuda.Device().id, None), cr._module(cp.cuda.Device().id)):
        scatter = cp.empty((nc + 1, nl), cp.float32)
        gather = cp.empty_like(scatter)
        c, k, v = (cov, kite, vort) if order is None else (
            cp.ascontiguousarray(a[order]) for a in (cov, kite, vort))
        module.get_function('vertex_helicity')((1,), (64,), (
            np.int32(nc), np.int32(nl), np.int32(nv), np.int32(degree), c, k, v, scatter))
        offsets, entries = cv.helicity_plan(cov, nc, order)
        n = (nc + 1) * nl
        module.get_function('vertex_helicity_gather')(((n + 127) // 128,), (128,), (
            np.int32(nc), np.int32(nl), np.int32(degree), offsets, entries, kite, vort, gather))
        assert int(cp.count_nonzero(scatter.view(cp.uint32) != gather.view(cp.uint32))) == 0


@pytest.mark.gpu
@pytest.mark.parametrize('monotone', [False, True])
def test_multiblock_fixed_pressure_matches_one_block_bitwise(cp, monotone):
    from hexcore.mpasdiag import isobaric as iso
    rng = np.random.default_rng(3)
    nc, nl = 5000, 55
    z = np.linspace(20000, 30, nl, dtype=np.float32)
    p = (1013 * np.exp(-z / 7400))[None, :] * (1 + 0.03 * rng.standard_normal((nc, 1)))
    p = p + (0 if monotone else 0.2) * rng.standard_normal((nc, nl))
    if monotone:
        p = np.sort(p, axis=1); p[:, 0] = 10; p[:, -1] = 1050
    else:
        p[::97] *= 0.88
    p = cp.asarray(np.ascontiguousarray(p, dtype=np.float32))
    f = cp.asarray((rng.standard_normal((nc, nl)) * 10 + 250).astype(np.float32))
    po = cp.broadcast_to(cp.asarray(iso.LEVELS, dtype=cp.float32), (nc, 8)).copy()
    one = cp.empty_like(po)
    iso._kernel('fixed_pressure')((1,), (128,), (np.int32(nc), np.int32(nl), np.int32(8), p, f, po, one,
                                                 cp.empty(nc, cp.int32)))
    many = iso.interp_tofixed_pressure(p, f, po)
    assert int(cp.count_nonzero(one.view(cp.uint32) != many.view(cp.uint32))) == 0
