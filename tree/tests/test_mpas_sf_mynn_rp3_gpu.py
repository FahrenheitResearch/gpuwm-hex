"""RP3: native blended ice input fails the component-preservation regression."""
import numpy as np
import pytest
cp = pytest.importorskip('cupy')
pytestmark = pytest.mark.gpu
from hexcore.mpasphys import sf_mynn_v842 as port


@pytest.mark.parametrize('fraction',[False,True])
def test_ice_call_recovers_component_only_with_rp3(fraction):
    # Exactly representable blend: half 250 K ice + half 280 K water = 265 K.
    def convert(corrections):
        n=1
        packed=cp.zeros((len(port.FIELDS),n),cp.float32)
        packed[port.FIELDS.index('tsk')]=265.
        sea=cp.zeros_like(packed)
        port._module(corrections).get_function('sf_mynn_from_mpas')((1,),(1,),
            (packed,sea,cp.ones(n,cp.float32),cp.full(n,280,cp.float32),
             cp.full(n,.5,cp.float32),np.float32(480000),np.float32(.02),
             np.int32(fraction),np.int32(n)))
        return float(packed[port.FIELDS.index('tsk')].get()[0])
    exact=convert(frozenset());correct=convert({'RP3'})
    assert exact==265.
    if fraction:
        # Applying the component-preservation assertion to native behaviour fails.
        with pytest.raises(AssertionError):assert exact==250.
        assert correct==250.
    else:assert correct==exact


def test_correct_variant_keeps_subnormal_words():
    # MPAS driver_sfclayer.F:672-747; correct mode must retain direct-NVRTC FTZ fix.
    n=1;packed=cp.zeros((len(port.FIELDS),n),cp.float32);sea=cp.zeros_like(packed)
    packed[port.FIELDS.index('flhc')]=cp.asarray(np.array([2],np.uint32).view(np.float32))
    port._module({'RP3'}).get_function('sf_mynn_to_mpas')((1,),(1,),
        (packed,sea,cp.full(n,.5,cp.float32),cp.zeros(n,cp.float32),
         np.float32(.02),np.int32(n)))
    assert packed[port.FIELDS.index('flhc')].get().view(np.uint32)[0]==1
