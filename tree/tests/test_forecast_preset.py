"""Forecast presets are rows, and a row resolves into knobs the door has.

``gpuwm-hex forecast --preset NAME`` selects a row of
:mod:`hexcore.forecast_preset`.  The door turns the row into the surface/PBL
cadence at the bound mesh's timestep and hands that to the same admission
and the same driver argument an explicit ``--pbl-cadence`` takes, so a
preset is never a second code path.  THE BREAKAGES THESE TESTS HOLD SHUT:

* a preset holding the stack for a number of seconds that is not a whole
  number of the mesh's steps would be refused on every mesh whose dt does
  not divide it -- the row names a longest hold instead, and resolves to the
  largest whole number of steps inside it;
* a row declaring a radiation, land-surface, surface-layer or PBL scheme the
  pinned engine seam does not take as an argument would run the engine's
  suite under the row's name -- it is refused by name;
* an explicit ``--pbl-cadence`` is the user's own selection and must win.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import argparse

import pytest

from hexcore import dt_admission, forecast_door, forecast_preset
from hexcore.errors import ConfigurationRefusal


def test_the_table_names_the_default_first_and_every_row_resolves():
    names = forecast_preset.preset_names()
    assert names[0] == forecast_preset.DEFAULT_PRESET
    assert set(names) == set(forecast_preset.PRESETS)
    for name in names:
        row = forecast_preset.resolve_preset(name)
        assert row.name == name
        assert dict(row.suite) == dict(forecast_preset.ENGINE_COLUMN_SUITE)


def test_no_name_is_the_default_row():
    assert (
        forecast_preset.resolve_preset(None).name
        == forecast_preset.DEFAULT_PRESET
    )


def test_an_unknown_preset_is_refused_naming_the_rows():
    with pytest.raises(ConfigurationRefusal) as caught:
        forecast_preset.resolve_preset("quickest")
    message = str(caught.value)
    for name in forecast_preset.PRESETS:
        assert name in message


def test_the_reference_row_is_the_weld_on_every_timestep():
    row = forecast_preset.REFERENCE_PRESET
    for dt in (5.0, 20.0, 75.0, 120.0):
        assert forecast_preset.pbl_cadence_request(row, dt) == "auto"


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (5.0, "30"),  # six steps
        (10.0, "30"),  # three steps
        (12.0, "24"),  # two steps; three would be 36 s, past the hold
        (20.0, "auto"),  # one step fits: the weld
        (75.0, "auto"),
        (120.0, "auto"),
    ],
)
def test_the_fast_row_holds_the_largest_whole_number_of_steps_inside_30_s(dt, expected):
    assert forecast_preset.pbl_cadence_request(forecast_preset.FAST_PRESET, dt) == expected


def test_the_fast_cadence_is_admitted_where_the_welded_timestep_is():
    # The sub-kilometre point cull: dt 5 s, convection off by resolution.
    cadence = forecast_preset.pbl_cadence_request(forecast_preset.FAST_PRESET, 5.0)
    admitted = forecast_door.admit_timestep(
        "point-cull", 5.0, nominal_dx_m=937.5, pbl_cadence=cadence
    )
    assert admitted["surface_pbl_seconds"] == 30.0
    assert admitted["pbl_cadence"]["steps_between_calls"] == 6
    assert admitted["surface_pbl_anchor_derived_from"] == dt_admission.dt_key(5.0, None)


def test_a_row_declaring_a_suite_the_engine_cannot_run_is_refused(monkeypatch):
    moved = forecast_preset.ForecastPreset(
        name="rrtmgp",
        summary="synthetic row",
        suite={**forecast_preset.ENGINE_COLUMN_SUITE, "radiation": "rte_rrtmgp"},
        surface_pbl_hold_seconds=None,
        evidence="none",
    )
    monkeypatch.setattr(
        forecast_preset,
        "PRESETS",
        {**forecast_preset.PRESETS, moved.name: moved},
    )
    with pytest.raises(ConfigurationRefusal) as caught:
        forecast_preset.resolve_preset("rrtmgp")
    message = str(caught.value)
    assert "rte_rrtmgp" in message
    assert "rrtmg_legacy" in message
    assert "mpas_column_batch.py" in message


def test_a_row_must_fill_every_suite_slot():
    with pytest.raises(ValueError, match="suite slots"):
        forecast_preset.ForecastPreset(
            name="partial",
            summary="synthetic row",
            suite={"radiation": "rrtmg_legacy"},
            surface_pbl_hold_seconds=None,
            evidence="none",
        )


def test_the_door_offers_every_row_and_leaves_the_cadence_to_the_row():
    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    actions = {action.dest: action for action in parser._actions}
    assert tuple(actions["preset"].choices) == forecast_preset.preset_names()
    assert actions["preset"].default == forecast_preset.DEFAULT_PRESET
    # None, not "auto": an absent flag must be distinguishable from the user
    # asking for the weld, or the row could never decide the cadence.
    assert actions["pbl_cadence"].default is None


def test_the_door_resolves_the_row_into_the_one_cadence_knob():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    door = (root / "src" / "hexcore" / "forecast_door.py").read_text(
        encoding="utf-8"
    )
    # The row decides only when the flag is absent, the explicit flag wins,
    # and both reach the driver through the existing --pbl-cadence argument.
    assert "if explicit_pbl_cadence is None:" in door
    assert "forecast_preset.pbl_cadence_request(preset, row.dt_seconds)" in door
    assert '"--pbl-cadence", request.pbl_cadence,' in door
    assert '"pbl_cadence_source": request.pbl_cadence_source,' in door
