"""MPAS LW driver metadata regression; forecast identity requires the real door."""
from itertools import product
import pytest

from hexcore.mpasphys.rrtmg_lw_v842 import _radiation_lw_options


def options(**changes):
    result = dict(config_radt_cld_overlap='maximum_random',
                  config_radt_cld_dcorrlen='constant', config_o3climatology=True,
                  config_microp_re=True, config_microp_scheme='mp_thompson',
                  config_radt_lw_scheme='rrtmg_lw', config_radt_sw_scheme='rrtmg_sw')
    result.update(changes)
    return result


@pytest.mark.parametrize('overlap,index', list(zip(
    ('none','random','maximum_random','maximum','exponential','exponential_random'),
    range(6))))
@pytest.mark.parametrize('length', ['constant','latitude_varying'])
@pytest.mark.parametrize('ozone', [False,True])
def test_native_overlap_decorrelation_and_ozone_metadata(overlap,index,length,ozone):
    # driver_radiation_lw.F:847-865: only exponential overlap uses idcor.
    result = _radiation_lw_options(options(config_radt_cld_overlap=overlap,
        config_radt_cld_dcorrlen=length, config_o3climatology=ozone))
    assert (result.cldovrlp,result.idcor,result.o3input) == (
        index,int(index>=4 and length=='latitude_varying'),2 if ozone else 0)


@pytest.mark.parametrize('scheme,re_enabled,lw,sw', list(product(
    ['mp_thompson','mp_thompson_aerosols','mp_wsm6','mp_kessler'],
    [False,True], ['rrtmg_lw','cam_lw'], ['rrtmg_sw','cam_sw'])))
def test_manager_radius_flags_require_both_rrtmg_schemes(scheme,re_enabled,lw,sw):
    # manager.F:747-759: a LW-only selection does not enable all radius flags.
    result = _radiation_lw_options(options(config_microp_scheme=scheme,
        config_microp_re=re_enabled,config_radt_lw_scheme=lw,config_radt_sw_scheme=sw))
    enabled = int(re_enabled and scheme!='mp_kessler' and lw=='rrtmg_lw' and sw=='rrtmg_sw')
    assert (result.has_reqc,result.has_reqi,result.has_reqs) == (enabled,)*3


def test_actual_native_radius_flags_are_forwarded_independently():
    result = _radiation_lw_options(options(), (0,1,0))
    assert (result.has_reqc,result.has_reqi,result.has_reqs) == (0,1,0)


def test_unknown_overlap_preserves_native_zero_initialization():
    # MPAS: mpas_atmphys_driver_radiation_lw.F:845-863. Native SELECT CASE
    # leaves cldovrlp=0 for unknown input; p21's two-host full forecasts agree.
    result = _radiation_lw_options(options(config_radt_cld_overlap='unrecognized_overlap',
                                         config_radt_cld_dcorrlen='latitude_varying'))
    assert (result.cldovrlp, result.idcor) == (0, 0)
