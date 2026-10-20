"""Word-exact launch remapping, including odd level tails and large polygons."""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('nlev', [1, 4, 5, 55])
def test_corrected_theta_chunk_matches_element(nlev):
    cp = pytest.importorskip('cupy')
    from hexcore.cuda_dynamics_v841 import _launch

    ncells, nedges, max_edges = 129, 389, 11
    rng = np.random.default_rng(203)

    def floats(shape):
        a = rng.uniform(-2.0, 2.0, size=shape).astype('f4')
        # Exercise FTZ and signed zero alongside cancellation in ordinary data.
        a.ravel()[:8] = np.array([0, 0x80000000, 1, 0x80000001,
                                 0x007fffff, 0x807fffff, 0x00800000,
                                 0x80800000], dtype='u4').view('f4')
        return cp.asarray(a)

    counts = np.resize(np.array([0, 5, 6, 10, 11], dtype='i4'), ncells)
    inputs = (np.int32(nlev), np.int32(ncells), np.int32(nedges), np.int32(max_edges),
              cp.asarray(counts),
              cp.asarray(rng.integers(0, nedges, (ncells, max_edges), dtype='i4')),
              floats((ncells, max_edges)), floats((ncells,)),
              cp.asarray(rng.uniform(.01, 1, nlev).astype('f4')),
              floats((nlev, nedges)), floats((nlev + 1, ncells)),
              floats((nlev, nedges)), floats((nlev, nedges)),
              floats((nlev, ncells)), floats((nedges,)),
              cp.asarray(rng.integers(0, ncells, (nedges, 2), dtype='i4')))
    element = cp.empty((nlev, ncells), dtype='f4')
    chunk = cp.empty_like(element)
    _launch('theta_finish_corrected_v841_f32', nlev * ncells,
            (*inputs, element), None)
    _launch('theta_finish_corrected_chunk_v841_f32', ncells * ((nlev + 3) // 4),
            (*inputs, chunk, np.int32(4)), None)
    np.testing.assert_array_equal(cp.asnumpy(chunk).view('u4'),
                                  cp.asnumpy(element).view('u4'))
