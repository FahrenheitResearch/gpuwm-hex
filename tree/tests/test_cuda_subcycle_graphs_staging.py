"""CPU checks of what the subcycle graph stages and keys (F1, 2026-10-15).

Breakage prevented: graph replay silently running without an input the eager
subcycle reads. F1 was one such input: SG2's lagged heating
``physics_tendencies.diabatic_theta`` was left out of the staged carrier, read
as None through ``getattr(..., None)``, and the graph key never changed when it
appeared. numpy stands in for cupy: staging and keying only need its array
type.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hexcore.cuda_physics_v841 import CudaHeldMpasPhysicsTendenciesV841
from hexcore.cuda_subcycle_graphs import (
    WithheldTendencyRead,
    _layout,
    _stage,
    _subcycle_tendencies,
)

cp = SimpleNamespace(ndarray=np.ndarray, empty_like=np.empty_like, copyto=np.copyto)


@dataclass(frozen=True, slots=True)
class _Held:
    rho: Any
    rho_u: Any
    rho_theta: Any
    scalars: Any
    time_seconds: float
    diabatic_theta: Any = None


def _held(heating=None, time_seconds=0.0):
    plane = np.zeros((2, 3), dtype=np.float32)
    return _Held(plane, plane + 1, plane + 2, np.zeros((1, 2, 3), np.float32),
                 time_seconds, heating)


def test_lagged_heating_is_staged_and_its_appearance_changes_the_key():
    absent = _subcycle_tendencies(_held(), cp)
    present = _subcycle_tendencies(_held(np.ones((2, 3), np.float32)), cp)
    assert getattr(absent, "diabatic_theta", "missing") is None
    assert isinstance(present.diabatic_theta, np.ndarray)
    assert _layout({"physics_tendencies": absent}, cp) != _layout({"physics_tendencies": present}, cp)
    staged = _stage({"physics_tendencies": present}, cp)["physics_tendencies"]
    assert type(staged) is type(present)
    assert staged.diabatic_theta is not present.diabatic_theta
    assert staged.diabatic_theta.tobytes() == present.diabatic_theta.tobytes()


def test_every_device_array_but_the_scalar_block_is_staged():
    staged = _subcycle_tendencies(_held(np.ones((2, 3), np.float32)), cp)
    assert {"rho", "rho_u", "rho_theta", "diabatic_theta"} <= set(vars(staged))
    assert "scalars" not in vars(staged) and "time_seconds" not in vars(staged)


@pytest.mark.parametrize("name", ["scalars", "time_seconds"])
def test_reading_a_withheld_field_raises_instead_of_reading_none(name):
    staged = _subcycle_tendencies(_held(), cp)
    with pytest.raises(WithheldTendencyRead, match=name):
        getattr(staged, name, None)


def test_a_field_the_carrier_never_had_still_reads_as_absent():
    carrier = SimpleNamespace(rho=np.zeros(2, np.float32))
    staged = _subcycle_tendencies(carrier, cp)
    assert getattr(staged, "diabatic_theta", None) is None


def test_the_real_carrier_has_no_field_outside_the_staging_rules():
    """A new device array field is staged; anything else must be withheld."""
    names = {f.name for f in fields(CudaHeldMpasPhysicsTendenciesV841)}
    assert "diabatic_theta" in names
    carrier = SimpleNamespace(**{name: (np.zeros(2, np.float32) if name in
                                        {"rho", "rho_u", "rho_theta", "scalars", "diabatic_theta"}
                                        else object()) for name in names})
    staged = _subcycle_tendencies(carrier, cp)
    assert set(vars(staged)) - {"_withheld"} | set(staged._withheld) == names


def test_host_values_are_keyed_by_value_except_the_unkeyed_ones():
    a = _layout({"outer_dt": 600.0, "flag": 0, "dynamics_substep": 1, "time_seconds": 0.0}, cp)
    assert a != _layout({"outer_dt": 600.0, "flag": 1, "dynamics_substep": 1, "time_seconds": 0.0}, cp)
    assert a == _layout({"outer_dt": 600.0, "flag": 0, "dynamics_substep": 3, "time_seconds": 9.0}, cp)
