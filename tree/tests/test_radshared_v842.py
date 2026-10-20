"""Metadata and dispatch checks; numerical proof is the standalone replay."""
import pytest

from hexcore.mpasphys.radshared_v842 import radiation_options


def resolved(**changes):
    config=dict(config_radt_cld_overlap='maximum_random',
                config_radt_cld_dcorrlen='constant',config_o3climatology=False,
                config_microp_re=True,config_microp_scheme='mp_thompson',
                config_radt_lw_scheme='rrtmg_lw',config_radt_sw_scheme='rrtmg_sw')
    config.update(changes)
    return config


@pytest.mark.parametrize('overlap,code',[('none',0),('random',1),('maximum_random',2),
                                        ('maximum',3),('exponential',4),('exponential_random',5)])
@pytest.mark.parametrize('dcorr',['constant','latitude_varying'])
def test_overlap_mapping(overlap,code,dcorr):
    options=radiation_options(resolved(config_radt_cld_overlap=overlap,
                                      config_radt_cld_dcorrlen=dcorr))
    assert options.cldovrlp==code
    assert options.idcor==int(code>=4 and dcorr=='latitude_varying')


@pytest.mark.parametrize('scheme',['mp_thompson','mp_thompson_aerosols','mp_wsm6','off','other'])
@pytest.mark.parametrize('microp_re',[False,True])
@pytest.mark.parametrize('pair',[('rrtmg_lw','rrtmg_sw'),('rrtmg_lw','off'),('off','rrtmg_sw')])
def test_radius_flags_require_both_spectra(scheme,microp_re,pair):
    options=radiation_options(resolved(config_microp_scheme=scheme,
                                      config_microp_re=microp_re,
                                      config_radt_lw_scheme=pair[0],
                                      config_radt_sw_scheme=pair[1]))
    expected=int(microp_re and scheme in ('mp_thompson','mp_thompson_aerosols','mp_wsm6')
                 and pair==('rrtmg_lw','rrtmg_sw'))
    assert (options.has_reqc,options.has_reqi,options.has_reqs)==(expected,)*3


@pytest.mark.parametrize('enabled',[False,True])
def test_ozone_flag(enabled):
    options=radiation_options(resolved(config_o3climatology=enabled))
    assert options.o3input==(2 if enabled else 0)


@pytest.mark.parametrize('overlap', ['unknown', 'invalid', ''])
def test_measured_native_unknown_overlap_reproduces_none(overlap):
    # Full native witness: both hosts, repeated 1/2-rank runs, both RRTMG.
    options = radiation_options(resolved(config_radt_cld_overlap=overlap,
                                        config_radt_cld_dcorrlen='latitude_varying'))
    assert options.cldovrlp == options.idcor == 0


@pytest.mark.parametrize('overlap', ['exponential', 'exponential_random'])
def test_native_initialized_decorrelation_survives_unknown_string(overlap):
    # MPAS LW:858-862 and SW:911-916 initialize zero before string tests.
    assert radiation_options(resolved(config_radt_cld_overlap=overlap,
                                     config_radt_cld_dcorrlen='unknown')).idcor == 0
