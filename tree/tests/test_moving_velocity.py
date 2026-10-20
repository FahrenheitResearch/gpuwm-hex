import numpy as np
import pytest
from types import SimpleNamespace
from hexcore.acoustic import edge_signs_on_cells
from hexcore.terrain import TerrainCoupling,recover_velocities
from hexcore.moving_velocity import recover_vertical_velocity


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_vectorized_w_is_bitwise_scalar_authority_with_terrain_and_bottom_diagnosis(dtype):
    rng=np.random.default_rng(4599)
    nc,nl=17,8
    cells=np.array([[c,(c+1)%nc] for c in range(nc)])
    edges=np.array([[(c-1)%nc,c] for c in range(nc)])
    counts=np.full(nc,2)
    counts[::3]=1
    arrays=dict(nEdgesOnCell=counts,edgesOnCell=edges,cellsOnEdge=cells)
    state=SimpleNamespace(rho=rng.uniform(.1,2,(nl,nc)).astype(dtype),
        rho_u=rng.normal(size=(nl,nc)).astype(dtype),rho_w=rng.normal(size=(nl+1,nc)).astype(dtype))
    vertical=SimpleNamespace(zz=rng.uniform(.5,1.5,(nl,nc)).astype(dtype),
        fzm=np.full(nl,.42,dtype),fzp=np.full(nl,.58,dtype),cf1=1.2,cf2=-.3,cf3=.1)
    terrain=SimpleNamespace(zb_cell=rng.normal(size=(nl+1,nc,2)).astype(dtype),
        zb3_cell=rng.normal(size=(nl+1,nc,2)).astype(dtype))
    coupling=TerrainCoupling(counts,edges,cells,edge_signs_on_cells(SimpleNamespace(arrays=arrays)).astype(dtype),
                            terrain.zb_cell,terrain.zb3_cell,.25)
    expected=recover_velocities(rho_zz=state.rho,ru=state.rho_u,rw=state.rho_w,
        zz=vertical.zz,fzm=vertical.fzm,fzp=vertical.fzp,cf1=vertical.cf1,cf2=vertical.cf2,
        cf3=vertical.cf3,coupling=coupling).w
    actual=recover_vertical_velocity(state,vertical,arrays,terrain)
    bits=np.uint32 if dtype==np.float32 else np.uint64
    np.testing.assert_array_equal(actual.view(bits),expected.view(bits))
