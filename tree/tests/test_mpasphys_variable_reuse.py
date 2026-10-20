"""Native module writes preserve storage ownership, words, and live bounds."""
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys.native_backend import NativePhysics


@pytest.fixture
def library():
    value = os.environ.get('MPAS_NATIVE_LIBRARY')
    if not value:
        pytest.skip('requires the actual native library')
    assert Path(value).is_file()
    return value


@pytest.fixture(params=[False, pytest.param(True, marks=pytest.mark.gpu)])
def device(request):
    if not request.param:
        return None
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires an explicit GPU grant')
    return pytest.importorskip('cupy')


@pytest.mark.parametrize('layout', ['c', 'f', 'strided', 'reversed', 'broadcast', 'readonly', 'unaligned', 'empty'])
def test_variable_words_and_caller_ownership(library, device, layout):
    # ABI authority: backend/generate.py, native_vars_real_3 operations 0/1/2/3;
    # MPAS source: mpas_atmphys_vars.F:68-955.
    words = np.resize(np.array([0x80000000, 1, 0x7fc12345, 0x7f812345,
                                0xff800000, 0x3f800000], dtype=np.uint32), (6, 4, 1))
    host = np.array(words.view(np.float32), order='F' if layout == 'f' else 'C')
    if layout == 'strided': host = host[::2]
    if layout == 'reversed': host = host[::-1, ::-1]
    if layout == 'broadcast': host = np.broadcast_to(host[:1], (6, 4, 1))
    if layout == 'readonly': host.flags.writeable = False
    if layout == 'unaligned':
        storage = bytearray(host.nbytes + 1)
        other = np.ndarray(host.shape, dtype=host.dtype, buffer=storage, offset=1, order='F')
        other[...] = host
        host = other
    if layout == 'empty': host = host[:0]
    value = host if device is None else device.asarray(host)
    expected = host.tobytes(order='F')
    with NativePhysics(library, test_only=True) as native:
        for _ in range(2):
            native.put_variable('qv_p', value)
            got, lower = native.get_variable('qv_p')
            assert got.shape == host.shape and lower == (1, 1, 1)
            assert got.tobytes(order='F') == expected
        if device is not None or value.flags.writeable:
            value[...] = 99
        assert native.get_variable('qv_p')[0].tobytes(order='F') == expected


@pytest.mark.parametrize('name,dtype,value,expected', [
    ('dt_pbl', 'float32', 0x7fc12345, 0x7fc12345),
    ('its', 'int32', -123, -123),
    ('f_qc', 'int32', -7, 1),
])
def test_variable_scalar_semantics(library, device, name, dtype, value, expected):
    host = np.asarray(value, dtype='uint32').view('float32') if dtype == 'float32' else np.asarray(value, dtype=dtype)
    with NativePhysics(library, test_only=True) as native:
        native.put_variable(name, host if device is None else device.asarray(host))
        got, lower = native.get_variable(name)
        assert lower == () and got.shape == ()
        assert (got.view('uint32').item() if dtype == 'float32' else got.item()) == expected


def test_variable_observes_native_allocation_and_bounds(library, device):
    with NativePhysics(library, test_only=True) as native:
        for shape, lower in [((4, 3, 1), (-1, 0, 1)), ((4, 3, 1), (2, 1, -4)), ((2, 5, 1), (0, -3, 2))]:
            host = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            value = host if device is None else device.asarray(host)
            native.put_variable('qv_p', value, lower=lower)
            got, actual = native.get_variable('qv_p')
            assert actual == lower and got.tobytes(order='F') == host.tobytes(order='F')
            # Native code can release module arrays between Python calls.
            native.lib.native_variables_release()
            assert native.get_variable('qv_p') is None
            native.put_variable('qv_p', value, lower=lower)
            got, actual = native.get_variable('qv_p')
            assert actual == lower and got.tobytes(order='F') == host.tobytes(order='F')
