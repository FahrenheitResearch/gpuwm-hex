"""A bare moving map uses CUDA on visible cards; retired flags cannot disable it."""
import builtins
import sys
from types import SimpleNamespace

import pytest
from hexcore.moving_vertical import _select_backend


@pytest.mark.parametrize('retired_value', [None, '0', '1'])
def test_visible_cuda_is_default_without_opt_in(monkeypatch, retired_value):
    for name in ('GPUWM_HEX_NO_LOCAL_GPU', 'GPUWM_NO_LOCAL_GPU', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    for name in ('GPUWM_HEX_MOVING_HEIGHT_GPU', 'GPUWM_HEX_MOVING_RATE_GPU'):
        if retired_value is None:monkeypatch.delenv(name,raising=False)
        else:monkeypatch.setenv(name,retired_value)
    monkeypatch.setitem(sys.modules,'cupy', SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDeviceCount=lambda:1))))
    assert _select_backend('auto')=='cuda'
    assert _select_backend('cpu')=='cpu'


def test_hidden_devices_do_not_import_cupy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='cupy':pytest.fail('CPU-only route imported CuPy')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert _select_backend('auto')=='cpu'


def test_cuda_errors_propagate_without_serial_retry(monkeypatch):
    for name in ('GPUWM_HEX_NO_LOCAL_GPU','GPUWM_NO_LOCAL_GPU','CUDA_VISIBLE_DEVICES'):
        monkeypatch.delenv(name,raising=False)
    def broken():raise RuntimeError('CUDA device query failed')
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDeviceCount=broken))))
    with pytest.raises(RuntimeError,match='CUDA device query failed'):
        _select_backend('auto')
    with pytest.raises(ValueError,match='backend'):
        _select_backend('invalid')


def test_physical_area_scaling_keeps_float32_operator_promotion():
    import numpy as np
    from scipy.sparse import csr_matrix
    from hexcore.moving_remap import ConservativeMap
    operator=csr_matrix(np.array([[.75,.25],[.25,.75]],dtype=np.float32))
    weights=ConservativeMap(operator,np.ones(2),np.ones(2),{})
    result=weights.with_physical_areas(np.ones(2),np.ones(2))
    assert result.overlap.dtype==np.dtype(np.float64)
    np.testing.assert_array_equal(result.overlap.toarray(),operator.toarray().astype(np.float64))
    assert result.receipt['prepared_donor_relative_error']==0.


def test_parallel_area_marginals_preserve_ordered_words(monkeypatch):
    import numpy as np
    from scipy.sparse import csr_matrix
    from hexcore.moving_remap import ConservativeMap
    n=17000
    ids=np.arange(n)
    cols=np.sort(np.stack([(ids+d)%n for d in (0,1,2,3)],axis=1),axis=1)
    operator=csr_matrix((np.full(n*4,.25),cols.ravel(),np.arange(n+1)*4),shape=(n,n))
    weights=ConservativeMap(operator,np.ones(n),np.ones(n),{})
    area=np.ones(n);area[::3]=np.nextafter(np.float32(1),np.float32(0))
    monkeypatch.setenv('GPUWM_HEX_HOST_COPY_THREADS','1')
    serial=weights.with_physical_areas(area,np.roll(area,101))
    monkeypatch.setenv('GPUWM_HEX_HOST_COPY_THREADS','2')
    parallel=weights.with_physical_areas(area,np.roll(area,101))
    assert serial.receipt==parallel.receipt
    assert serial.receipt['prepared_reconciliation_iterations']>1
    for key in ('data','indices','indptr'):
        assert getattr(serial.overlap,key).tobytes()==getattr(parallel.overlap,key).tobytes()
