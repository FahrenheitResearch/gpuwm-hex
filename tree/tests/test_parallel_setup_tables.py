import numpy as np
import pytest
from hexcore.rust_hostprep import compress_advection,vertical_velocity_damping
from hexcore.transport import _compress_advection_coefficients_serial, _native_advection_order, _native_advection_compression
from hexcore.damping_v841 import _vertical_velocity_damping_serial

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('source_order',[False,True])
@pytest.mark.parametrize('regional',[False,True])
def test_parallel_rust_compression_keeps_scalar_words(dtype,source_order,regional):
    rng=np.random.default_rng(894);nc=7;ne=11;width=15
    counts=np.full(nc,2,np.int64);coc=np.array([[(i+1)%nc,(i+3)%nc] for i in range(nc)],np.int64)
    coe=np.array([[i%nc,(i+2)%nc] for i in range(ne)],np.int64)
    if regional:coe[-1,1]=nc;coc[-1,1]=nc
    ids=np.array([5,2,7,1,4,6,3],np.int64);deriv=rng.normal(size=(ne,2,width)).astype(dtype)
    dc=rng.uniform(1000,12000,ne).astype(dtype);dv=rng.uniform(500,13000,ne).astype(dtype)
    expected=_compress_advection_coefficients_serial(counts,coc,coe,ids,deriv,dc,dv,source_order_v841=source_order,allow_regional_sentinels=regional)
    adv,adv3,n_adv,cells=compress_advection(counts=counts,edges=np.zeros((nc,2),np.int64),cells_on_cell=coc,cells_on_edge=coe,ids=ids,deriv=deriv,dc=dc,dv=dv,source_order=source_order,garbage=nc if regional else -1)
    if source_order:
        adv,adv3,cells=_native_advection_order(counts,coc,coe,adv,adv3,n_adv,cells,garbage=nc if regional else -1)
        adv,adv3=_native_advection_compression(counts,coc,coe,deriv,dc,dv,n_adv,cells)
    actual=adv,adv3,n_adv,cells
    for a,b in zip(actual,expected):assert a.tobytes()==b.tobytes()
    mutated=actual[0].copy();mutated[0,0]=np.nextafter(mutated[0,0],dtype(np.inf));assert mutated.tobytes()!=expected[0].tobytes()

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_parallel_rust_damping_keeps_scalar_words(dtype):
    heights=(np.arange(60,dtype=dtype)[:,None]*dtype(550)+np.arange(17,dtype=dtype)[None,:]*dtype(.13))
    xnutr=dtype(.2);start=dtype(16000);pi=np.arccos(dtype(-1))
    expected=_vertical_velocity_damping_serial(heights,xnutr_r=xnutr,start=start)
    actual=vertical_velocity_damping(heights,xnutr=xnutr,start=start,pi=pi)
    assert actual.tobytes()==expected.tobytes()
