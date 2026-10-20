"""Actual CUDA/native ABI transfer, including raw words and native halo ownership."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', ['float32', 'int32'])
@pytest.mark.parametrize('layout,padded', [
    (layout, padded) for layout in ('c', 'f', 'strided', 'reversed', 'broadcast', 'empty')
    for padded in (False, True)
] + [('scalar', False)])
def test_commit_device_transfer_preserves_words_and_halos(dtype, layout, padded):
    # Native ownership authority: mpas_atmphys_interface.F:304-305 and
    # src/framework/mpas_pool_routines.F:2108-3211,3436-4240.
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires an explicit GPU grant and actual native library')
    cp = pytest.importorskip('cupy')
    library = os.environ['MPAS_NATIVE_LIBRARY']
    words = np.resize(np.array([0x80000000, 1, 0x7fc12345, 0xff800000,
                                0x7f812345, 0x3f800000], dtype=np.uint32), (4, 6))
    host = np.array(words.view(dtype), order='F' if layout == 'f' else 'C')
    device = cp.asarray(host)
    if layout == 'strided': host, device = host[:, ::2], device[:, ::2]
    if layout == 'reversed': host, device = host[::-1, ::-1], device[::-1, ::-1]
    if layout == 'broadcast':
        host, device = np.broadcast_to(host[:1], (4, 6)), cp.broadcast_to(device[:1], (4, 6))
    if layout == 'empty': host, device = host[:, :0], device[:, :0]
    if layout == 'scalar':
        host = np.asarray(0x7fc12345, dtype=np.uint32).view(dtype)
        device = cp.asarray(host)
    shape = host.shape[:-1] + (host.shape[-1]+1,) if padded else host.shape
    initial = np.full(shape, -123, dtype=dtype, order='F')
    expected = initial.copy(order='F')
    if padded: expected[..., :-1] = host
    else: expected[...] = host
    with NativePhysics(library, test_only=True) as native:
        owner = NativeDevicePoolViews(SimpleNamespace(native=native), cp)
        native.put_field('state', 'word_pack_probe', initial, nlevels=2)
        native.put_field('state', 'word_pack_probe', initial, level=2, nlevels=2)
        # Update twice to exercise both first transfer and reused native storage.
        for _ in range(2):
            selection = (..., slice(None, -1)) if padded else (...,)
            owner.buffers = {('state', 'word_pack_probe', 2): (initial.copy(order='F'), device, selection)}
            owner.commit()
            assert not owner.buffers
        assert native.get_field('state', 'word_pack_probe', level=2).tobytes(order='F') == expected.tobytes(order='F')
        assert native.get_field('state', 'word_pack_probe').tobytes(order='F') == initial.tobytes(order='F')
        assert device.get().tobytes() == host.tobytes()
