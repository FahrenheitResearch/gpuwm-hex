"""Preserve the actual engine face coupler's bytes under column permutation."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
from hexcore.cuda_column_order_v841 import ColumnOrder, original_pack_scope


def test_invalid_ordinal_metadata_refuses_before_upload():
    # These errors are checked before CUDA transfer, even on a CPU-only host.
    for order in ([],[1,1],[0,2]):
        with pytest.raises(ValueError):ColumnOrder.from_metadata(np.asarray(order,dtype=np.int32))


@pytest.mark.gpu
def test_real_face_coupler_matches_original_pack_bytes():
    if os.environ.get('HEXCORE_FUSION_GPU_TEST')!='1':pytest.skip('requires a card grant')
    import cupy as cp
    from gpuwm.core import physics
    n=7;nz=4;order=np.array([3,1,6,0,5,4,2],dtype=np.int32)
    reference_u=cp.arange(nz*(n+1),dtype=cp.float32).reshape(nz,1,n+1)*np.float32(0.125)
    reference_v=reference_u*np.float32(1.25)
    state=SimpleNamespace(p=reference_u,c1h=cp.ones(nz,dtype=cp.float32),c2h=cp.zeros(nz,dtype=cp.float32),has_msf=False)
    cfg=SimpleNamespace(specified=False,nested=False,open_x=False,open_y=False)
    base_u,base_v=physics._couple_momentum_to_faces(state,cfg,reference_u,reference_v)
    permutation=ColumnOrder.from_metadata(order)
    packed_u=cp.take(reference_u,permutation.forward,axis=-1)
    packed_v=cp.take(reference_v,permutation.forward,axis=-1)
    with original_pack_scope(permutation):
        new_u,new_v=physics._couple_momentum_to_faces(state,cfg,packed_u,packed_v)
    assert new_u.get().tobytes()==base_u.get().tobytes()
    assert cp.take(new_v,permutation.inverse,axis=-1).get().tobytes()==base_v.get().tobytes()
    # The context restores the ordinary path in the same process.
    after_u,after_v=physics._couple_momentum_to_faces(state,cfg,reference_u,reference_v)
    assert after_u.get().tobytes()==base_u.get().tobytes()
    assert after_v.get().tobytes()==base_v.get().tobytes()


@pytest.mark.gpu
def test_global_pack_without_a_garbage_column_matches_original_pack_bytes():
    # A global run's pack is nCells wide: no garbage column.  The padded map
    # alone refused step 1 of every one-card global run on a reordered mesh
    # (x1.163842: "cell momentum shape (55, 1, 163842), ordered columns 163843").
    if os.environ.get('HEXCORE_FUSION_GPU_TEST')!='1':pytest.skip('requires a card grant')
    import cupy as cp
    from gpuwm.core import physics
    n=7;nz=4;order=np.array([3,1,6,0,5,4,2],dtype=np.int32)
    reference_u=cp.arange(nz*n,dtype=cp.float32).reshape(nz,1,n)*np.float32(0.125)
    reference_v=reference_u*np.float32(1.25)
    state=SimpleNamespace(p=reference_u,c1h=cp.ones(nz,dtype=cp.float32),c2h=cp.zeros(nz,dtype=cp.float32),has_msf=False)
    cfg=SimpleNamespace(specified=False,nested=False,open_x=False,open_y=False)
    base_u,base_v=physics._couple_momentum_to_faces(state,cfg,reference_u,reference_v)
    permutation=ColumnOrder.from_metadata(order)
    real=cp.asarray(order)
    assert permutation.maps(n)[0].get().tolist()==order.tolist()
    assert permutation.maps(n+1)[0].get().tolist()==[*order.tolist(),n]
    packed_u=cp.take(reference_u,real,axis=-1)
    packed_v=cp.take(reference_v,real,axis=-1)
    with original_pack_scope(permutation):
        new_u,new_v=physics._couple_momentum_to_faces(state,cfg,packed_u,packed_v)
        # Any other width is neither pack: it still refuses, with the numbers.
        with pytest.raises(ValueError,match='ordered cells 7: a global pack is 7 columns wide, a limited-area pack 8'):
            physics._couple_momentum_to_faces(state,cfg,packed_u[...,:n-1],packed_v[...,:n-1])
    assert new_u.get().tobytes()==base_u.get().tobytes()
    assert cp.take(new_v,cp.argsort(real),axis=-1).get().tobytes()==base_v.get().tobytes()
    # The pack order is the only thing that moved: without the scope the
    # face-u carrier of the reordered pack differs from the original's.
    plain_u,_=physics._couple_momentum_to_faces(state,cfg,packed_u,packed_v)
    assert plain_u.get().tobytes()!=base_u.get().tobytes()
