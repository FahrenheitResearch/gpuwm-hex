"""The MPAS suite route applies the door's dycore schedule flags (2026-10-27: --acoustic-substeps was dropped)."""
import argparse

import pytest

from hexcore import forecast_door
from hexcore.mpasphys.forecast_v842 import namelist_option

TEXT = "&nhyd_model\nconfig_dt=24.0\n/\n&physics\nconfig_physics_suite='convection_permitting'\n/\n"


def args(**kw):
    base = dict(acoustic_substeps=None, time_integration_order=None, monotonic=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_substeps_flag_reaches_the_namelist():
    text = forecast_door.bind_native_dynamics(args(acoustic_substeps=6), TEXT)
    assert int(namelist_option(text, "nhyd_model", "config_number_of_sub_steps")) == 6
    assert float(namelist_option(text, "nhyd_model", "config_dt")) == 24.0
    assert namelist_option(text, "physics", "config_physics_suite") == "convection_permitting"


def test_no_flag_leaves_the_text_alone():
    assert forecast_door.bind_native_dynamics(args(), TEXT) == TEXT


def test_order_and_monotonic_reach_the_namelist():
    text = forecast_door.bind_native_dynamics(args(time_integration_order=3, monotonic=False), TEXT)
    assert int(namelist_option(text, "nhyd_model", "config_time_integration_order")) == 3
    assert forecast_door._namelist_value(namelist_option(text, "nhyd_model", "config_monotonic")) is False


def test_namelist_disagreeing_with_a_flag_is_refused():
    named = TEXT.replace("config_dt=24.0", "config_dt=24.0\nconfig_number_of_sub_steps=4")
    with pytest.raises(ValueError, match="two values"):
        forecast_door.bind_native_dynamics(args(acoustic_substeps=6), named)
    assert forecast_door.bind_native_dynamics(args(acoustic_substeps=4), named) == named
