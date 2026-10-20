"""The native session keeps suite ownership at the non-LES boundary."""
from dataclasses import replace

import pytest

from hexcore.config_v842 import NativeNonlesDycoreConfigV842
from hexcore.errors import ConfigurationRefusal


def test_native_suite_retains_complete_nonles_configuration():
    config = NativeNonlesDycoreConfigV842(
        config_physics_suite="mesoscale_reference", config_moist_physics=True,
        config_horiz_mixing="2d_fixed", config_h_theta_eddy_visc2=37.5,
        config_mix_full=False, config_mix_scalars=True,
        config_v_theta_eddy_visc2=0.5, config_w_adv_order=4,
        config_time_integration_order=3)
    config.validate()
    assert config.config_physics_suite == "mesoscale_reference"
    assert config.config_moist_physics is True
    assert config.requires_nonles_mixing()
    mixing = config.mixing_config()
    assert mixing.config_mix_full is False
    assert mixing.config_mix_scalars is True
    assert mixing.config_h_theta_eddy_visc2 == 37.5
    assert config.config_w_adv_order == 4
    assert config.config_time_integration_order == 3
    with pytest.raises(ConfigurationRefusal):
        replace(config, config_v_theta_eddy_visc2=-1).validate()


def test_native_default_keeps_existing_mixing_dispatch():
    config = NativeNonlesDycoreConfigV842(config_horiz_mixing="2d_smagorinsky")
    config.validate()
    assert not config.requires_nonles_mixing()
