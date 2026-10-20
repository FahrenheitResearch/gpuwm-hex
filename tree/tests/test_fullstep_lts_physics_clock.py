"""Class clocks of the active full-step route with physics: CPU bookkeeping only.

What is checked here is the arithmetic the forecast executor relies on: which
events a class map can keep, that every class calls radiation at the same
model times, that a class's cadences follow its own step, and that the
physics halo round reads the planes it says it reads.  Nothing here advances
weather; the forecast itself is measured on a box.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from hexcore.fullstep_lts_admission import require_class_clock, suggest_base_steps
from hexcore.fullstep_lts_classes import macro_schedule


def test_events_must_be_coarsest_class_step_boundaries():
    events = {"radiation interval": 1800.0, "history frame": 3600.0, "run length": 86400.0}
    clock = require_class_clock(18.75, [8, 4, 2, 1], events)
    assert clock["coarsest_class_step_seconds"] == 150.0
    assert clock["events_in_class_steps"] == {"radiation interval": 12, "history frame": 24, "run length": 576}


def test_event_inside_a_class_step_is_refused_by_name():
    events = {"radiation interval": 1800.0, "run length": 86400.0}
    with pytest.raises(ValueError, match="different model times in different classes") as error:
        require_class_clock(18.0, [8, 4, 2, 1], events)
    # 8 x 18 = 144 s does not divide 1800 s; the refusal offers a base step that does.
    assert "radiation interval" in str(error.value)
    assert "18.75" in str(error.value)


def test_suggested_base_steps_keep_every_event():
    events = {"radiation interval": 1800.0, "history frame": 3600.0}
    for candidate in suggest_base_steps(18.0, 8, events):
        for seconds in events.values():
            assert (seconds / (candidate * 8)) == int(seconds / (candidate * 8))
    # with top rate 4 the 18 s base step already keeps both events
    assert suggest_base_steps(18.0, 4, events)[0] == 18.0


def test_every_class_calls_radiation_at_the_same_model_times():
    """Each backend counts its own steps; the counts land on the same instants."""

    base_dt, rates, radiation = 18.75, (8, 4, 2, 1), 1800.0
    macros = int(2 * radiation / (base_dt * max(rates)))
    radiation_times = {index: [] for index in range(len(rates))}
    physics_seconds = {index: 0.0 for index in range(len(rates))}
    steps_done = {index: 0 for index in range(len(rates))}
    for macro in range(macros):
        for kind, index, begin, length in macro_schedule(rates, max(rates)):
            if kind != "step":
                continue
            class_dt = length * base_dt
            every = int(round(radiation / class_dt))
            if steps_done[index] % every == 0:
                radiation_times[index].append((macro * max(rates) + begin) * base_dt)
            steps_done[index] += 1
            physics_seconds[index] += class_dt
    assert all(times == [0.0, 1800.0] for times in radiation_times.values())
    # every column's physics integrates exactly the elapsed time, once
    assert all(seconds == macros * max(rates) * base_dt for seconds in physics_seconds.values())


def test_last_class_step_before_a_frame_is_the_one_that_ends_on_it():
    """Reflectivity is asked of the class step whose endpoint is the history time."""

    rates, top = (8, 4, 2, 1), 8
    capture = 24                      # base steps
    due = {index: [] for index in range(len(rates))}
    for macro in range(capture // top):
        for kind, index, begin, length in macro_schedule(rates, top):
            if kind == "step" and macro * top + begin + length == capture:
                due[index].append(macro * top + begin)
    assert due == {0: [16], 1: [20], 2: [22], 3: [23]}


@dataclass
class _Config:
    config_dt: float = 18.75
    config_radt_seconds: float = 1800.0
    config_bldt_seconds: float = 18.75
    config_cudt_seconds: float | None = 18.75

    def validate(self):
        for value in (self.config_radt_seconds, self.config_bldt_seconds):
            assert (value / self.config_dt) == int(value / self.config_dt)


def test_class_host_welds_step_cadences_to_the_class_step(monkeypatch):
    from hexcore.drivers import fullstep_lts_forecast_v841 as lts
    import hexcore.partition_local_mesh_v841 as local_mesh
    import hexcore.drivers.samehost_forecast_v841 as samehost

    monkeypatch.setattr(local_mesh, "slice_prepared_host", lambda item, layout: item)
    monkeypatch.setattr(samehost, "_prepared_attribute_view", lambda item: item)

    class Layout:
        n_local_cells = 123

    host = {"config": _Config(), "prepared": object(), "gwdo_host": None, "f000_surface_diagnostics": {},
            "constructor_values": {"dt": 18.75, "radiation_seconds": 1800.0, "surface_pbl_seconds": 18.75,
                                   "cumulus_seconds": 18.75, "n_columns": 999}}
    local = lts.class_physics_host(host, Layout(), base_dt=18.75, rate=8)
    values = local["constructor_values"]
    assert (values["dt"], values["surface_pbl_seconds"], values["cumulus_seconds"]) == (150.0, 150.0, 150.0)
    assert values["radiation_seconds"] == 1800.0 and values["n_columns"] == 123
    assert (local["config"].config_dt, local["config"].config_bldt_seconds,
            local["config"].config_cudt_seconds, local["config"].config_radt_seconds) == (150.0, 150.0, 150.0, 1800.0)
    # the shared host is untouched
    assert host["constructor_values"]["dt"] == 18.75 and host["config"].config_dt == 18.75
    # no cumulus scheme: the cadence stays absent
    dry = dict(host, config=replace(host["config"], config_cudt_seconds=None),
               constructor_values=dict(host["constructor_values"], cumulus_seconds=None))
    local = lts.class_physics_host(dry, Layout(), base_dt=18.75, rate=4)
    assert local["constructor_values"]["cumulus_seconds"] is None
    assert local["config"].config_cudt_seconds is None and local["config"].config_dt == 75.0


def test_step_length_constant_is_bound_for_the_class_and_restored():
    from hexcore import cuda_gwdo_v841 as gwdo
    from hexcore.drivers.fullstep_lts_forecast_v841 import _class_step_length

    before = gwdo.X4_DT_SECONDS_F32
    with _class_step_length(150.0):
        assert gwdo.X4_DT_SECONDS_F32 == np.float32(150.0)
    assert gwdo.X4_DT_SECONDS_F32 is before
    with pytest.raises(RuntimeError):
        with _class_step_length(75.0):
            raise RuntimeError("a refused class step")
    assert gwdo.X4_DT_SECONDS_F32 is before


def test_unsupported_selections_name_what_breaks():
    from hexcore.drivers.fullstep_lts_forecast_v841 import _refuse_unsupported

    @dataclass
    class Config:
        config_local_timestep: bool = False

    base = dict(host={"config": Config()}, cycling=None, park_physics_tier=False,
                local_timestep_classing_path=None)
    _refuse_unsupported(**base)
    # Retired 2026-10-12: checkpoint/restart/DA cycling and limited-area meshes
    # were refused here; the class checkpoint and the per-class boundary
    # runtime carry them now (fullstep_lts_restart_v841, the admission's
    # boundary zone).  The JEDI background file is still refused, by name.
    _refuse_unsupported(**{**base, "cycling": {"restart_from": "x", "checkpoint_steps": {8}}})
    _refuse_unsupported(**{**base, "host": {"config": Config(), "regional": {}}})
    _refuse_unsupported(**{**base, "cycling": {"jedi_background": True}})
    with pytest.raises(ValueError, match="another class is about to read"):
        _refuse_unsupported(**{**base, "park_physics_tier": True})
    with pytest.raises(ValueError, match="same accepted interval"):
        _refuse_unsupported(**{**base, "host": {"config": Config(config_local_timestep=True)}})
