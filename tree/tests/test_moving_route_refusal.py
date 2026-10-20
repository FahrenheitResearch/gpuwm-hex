"""A moving launch on a route without the move hooks is refused by name, and the controller stops on it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hexcore.fullstep_lts_admission import (MOVING_LAUNCH_VARIABLES, MOVING_ROUTE_REFUSAL,
                                            moving_launch_switches, moving_route_refusal)

# The MPAS suite route has its own move hooks (hexcore.mpasphys.moving_v842; tests/test_moving_mpas_route.py).
ROUTES = ("same-host", "fixed-step")


def test_knobs_alone_are_not_a_moving_launch():
    knobs = {"GPUWM_HEX_MOVING_BALANCE": "0", "GPUWM_HEX_MOVING_ZONE_HEIGHT": "absolute",
             "GPUWM_HEX_MOVING_SHOCK": ""}
    assert moving_launch_switches(knobs) == ()
    assert all(moving_route_refusal(route, knobs) is None for route in ROUTES)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name", MOVING_LAUNCH_VARIABLES)
def test_every_launch_switch_refuses_each_route_without_hooks(route, name):
    text = moving_route_refusal(route, {name: "x"})
    assert text.startswith(MOVING_ROUTE_REFUSAL) and name in text and "never move" in text


def test_the_mpas_suite_route_is_no_longer_refused():
    with pytest.raises(KeyError):
        moving_route_refusal("mpas-suite", {"GPUWM_HEX_MOVING_CONTROL": "/c"})


def test_the_moving_door_sets_only_launch_switches():
    from hexcore.moving_run import RUN_STATE_VARIABLES

    assert set(RUN_STATE_VARIABLES) <= set(MOVING_LAUNCH_VARIABLES)


def test_the_controller_reads_the_refusal_sentence(tmp_path):
    from hexcore.moving_run import route_refusal

    sentence = moving_route_refusal("same-host", {"GPUWM_HEX_MOVING_CONTROL": "/c"})
    (tmp_path / "forecast.log").write_text("ADMISSION ...\ngpuwm-hex: " + sentence + "\n")
    assert route_refusal(tmp_path) == sentence
    (tmp_path / "forecast.log").write_text("gpuwm-hex: the forecast driver stopped with CUDADriverError\n")
    assert route_refusal(tmp_path) is None


@pytest.mark.parametrize("lts,mode,rate,multi,route", [
    (True, "active", 8, False, None),             # one card, active LTS: the class executor's hooks
    (True, "active", 8, True, None),              # several cards, active LTS: the multi-card class blocks' hooks
    (True, "active", None, True, None),           # the door's default maximum rate is above one
    (True, "active", 1, True, "same-host"),       # rate one runs the same-host ranks of the fixed step
    (False, None, None, True, "same-host"),       # the same-host ranks of the fixed step
    (False, None, None, False, "fixed-step"),     # the fixed-step executor
])
def test_the_door_refuses_only_routes_without_hooks(lts, mode, rate, multi, route):
    from hexcore.fullstep_lts_admission import forecast_moving_route

    assert forecast_moving_route(fullstep_local_timestep=lts, fullstep_lts_mode=mode, fullstep_lts_max_rate=rate,
                                 multi_card=multi) == route


def test_multi_card_moving_is_no_longer_named_a_gap():
    text = moving_route_refusal("same-host", {"GPUWM_HEX_MOVING_CONTROL": "/c"})
    assert "named gap" not in text and "several cards" in MOVING_ROUTE_REFUSAL
    assert "--fullstep-lts-mode active" in text


def test_cards_split_every_segment_and_refuse_a_second_card_choice():
    from hexcore.moving_run import card_arguments, segment_command

    case = {"physics_args": ["--convection", "off"], "start_time": "2025-11-08_12:00:00"}
    assert card_arguments(1, case) == []
    assert card_arguments(3, case) == ["--samehost-devices", "0,1,2"]
    member = {"mesh": "m", "grid": "g", "static": "s", "init": "i", "dt_seconds": 225.0}
    one = segment_command(case, member, Path("seg"), 2.0, 60.0, init_source="x")
    two = segment_command(case, member, Path("seg"), 2.0, 60.0, init_source="x", cards=2)
    assert "--samehost-devices" not in one and two[-2:] == ["--samehost-devices", "0,1"]
    assert two[:len(one)] == one
    with pytest.raises(SystemExit, match="--cards alone"):
        card_arguments(2, dict(case, physics_args=["--samehost-devices", "0,1"]))
    with pytest.raises(SystemExit, match="--cards alone"):
        card_arguments(2, case, ["--samehost-layouts", "x.npz"])
