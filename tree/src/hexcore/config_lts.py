"""Opt-in local-timestep configuration, default OFF.

This project's standing position keeps the **dycore** byte-identical to native
MPAS-A v8.4.1 while retiring physics parity.  Native v8.4.1 has no local time
stepping: ``Registry.xml:64-68`` offers SRK3 only, and ``dt_dynamics``,
``rk_timestep(1:3)``, ``rk_sub_timestep(1:3)`` and ``number_sub_steps(1:3)`` are
scalars at ``mpas_atm_time_integration.F:2053-2092``.  There is therefore no
byte-identical local-timestep implementation and there never can be.

``config_local_timestep`` defaults to ``False``, and with it off the solver
executes the pinned path unchanged, so the dycore pin is untouched by the
option existing.  A user who turns it on takes a declared divergence from
native.  This is a PERFORMANCE feature and opt-in is the correct shape for it;
"fixed means default" governs correctness remedies, and nothing here fixes a
defect.

``config_v841.py`` is one of the exact-byte pinned execution sources
(``tools/run_cuda_v841_full_physics_x4.py::EXECUTION_SOURCE_PINS``), so these
fields live in a sibling subclass in this unpinned module instead.  The
parent's ``validate()`` runs unchanged through a reconstructed parent
instance, which keeps every pinned knob exactly as declared -- including
``config_dt``, which since 2026-08-26 is admitted from the earned-anchor
registry ``hexcore.dt_admission`` and holds exactly one anchor, 120.0 s.
Local time stepping does not move the model time step: it changes how many
acoustic sub-steps a column executes inside the step it already had.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .config_v841 import (
    V841DryDycoreConfig,
    V841MpasColumnPhysicsGwdoConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)
from .errors import ConfigurationRefusal

#: Rate ladder for the released ``(1, 3, 6)`` acoustic schedule.  A rate must
#: divide every stage's sub-step count to be able to skip whole sub-steps, and
#: 2 and 4 do not divide 3.  Two classes, per the measured finding that a third
#: class launches a below-threshold kernel twice per macro step for nothing.
DEFAULT_LOCAL_TIMESTEP_RATES: tuple[int, ...] = (1, 3)


def _refuse(knob: str, value: Any, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


def _validate_local_timestep_block(config: Any) -> None:
    enabled = config.config_local_timestep
    if not isinstance(enabled, bool):
        _refuse(
            "config_local_timestep",
            enabled,
            "the local-timestep switch selects a declared divergence from "
            "native v8.4.1 and must be an explicit boolean",
            "config_local_timestep=False",
        )
    rates = config.config_local_timestep_rates
    if (
        not isinstance(rates, tuple)
        or not rates
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in rates)
        or list(rates) != sorted(set(rates))
        or rates[0] != 1
    ):
        _refuse(
            "config_local_timestep_rates",
            rates,
            "the ladder must be a strictly increasing tuple of positive "
            "integers starting at 1, so the finest cells keep the schedule's "
            "own acoustic sub-step",
            "config_local_timestep_rates=(1, 3)",
        )
    rings = config.config_local_timestep_buffer_rings
    if not isinstance(rings, int) or isinstance(rings, bool) or rings < 1:
        _refuse(
            "config_local_timestep_buffer_rings",
            rings,
            "a class boundary with no buffer puts the rate jump directly on "
            "the cells the refinement was placed for",
            "config_local_timestep_buffer_rings=1",
        )
    factor = config.config_local_timestep_safety_factor
    if not isinstance(factor, float) or not (0.0 < factor <= 1.0):
        _refuse(
            "config_local_timestep_safety_factor",
            factor,
            "the classing ratio may be made more conservative but never less; "
            "a factor above one would hand a cell a sub-step its own spacing "
            "does not permit",
            "config_local_timestep_safety_factor=1.0",
        )
    if not enabled and rates != DEFAULT_LOCAL_TIMESTEP_RATES:
        # Not an error, but a run whose ladder is set with the switch off has
        # almost certainly forgotten the switch.
        pass


@dataclass(frozen=True, slots=True)
class V841LocalTimestepDryConfig(V841DryDycoreConfig):
    """Dry v8.4.1 dycore, passive qv, plus the opt-in local-timestep block.

    This is the lane the conservation gate is defined in.  The 2.0e-8 mass and
    qv bound comes from ``tools/run_cuda_x1_163842_stabilized_products.py`` --
    ``:983-984`` refuses unless the run is no-physics, and ``:1632-1635``
    computes the passive-qv drift the bound is applied to.  In a full-physics
    run qv has sources and sinks, so its budget measures the microphysics, not
    the transport, and cannot decide whether local time stepping conserves.

    ``config_dt``, ``config_number_of_sub_steps`` and
    ``config_dynamics_split_steps`` are validated for sanity here but not
    pinned to one value the way the column-physics lane pins them, which is
    what lets the referee arm run the same mesh at a globally smaller acoustic
    step.
    """

    config_local_timestep: bool = False
    config_local_timestep_rates: tuple[int, ...] = DEFAULT_LOCAL_TIMESTEP_RATES
    config_local_timestep_buffer_rings: int = 1
    config_local_timestep_safety_factor: float = 1.0

    def validate(self) -> None:
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841DryDycoreConfig)
        }
        V841DryDycoreConfig(**parent_values).validate()
        _validate_local_timestep_block(self)


@dataclass(frozen=True, slots=True)
class V841LocalTimestepGwdoConfig(V841MpasColumnPhysicsGwdoConfig):
    """Released GWDO lane, mixing off, plus the opt-in local-timestep block."""

    config_local_timestep: bool = False
    config_local_timestep_rates: tuple[int, ...] = DEFAULT_LOCAL_TIMESTEP_RATES
    config_local_timestep_buffer_rings: int = 1
    config_local_timestep_safety_factor: float = 1.0

    def validate(self) -> None:
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841MpasColumnPhysicsGwdoConfig)
        }
        V841MpasColumnPhysicsGwdoConfig(**parent_values).validate()
        _validate_local_timestep_block(self)


@dataclass(frozen=True, slots=True)
class V841LocalTimestepSmagorinskyGwdoConfig(
    V841MpasColumnPhysicsSmagorinskyGwdoConfig
):
    """Released GWDO + 2-D Smagorinsky lane plus the opt-in local-timestep block."""

    config_local_timestep: bool = False
    config_local_timestep_rates: tuple[int, ...] = DEFAULT_LOCAL_TIMESTEP_RATES
    config_local_timestep_buffer_rings: int = 1
    config_local_timestep_safety_factor: float = 1.0

    def validate(self) -> None:
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841MpasColumnPhysicsSmagorinskyGwdoConfig)
        }
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(**parent_values).validate()
        _validate_local_timestep_block(self)


def local_timestep_enabled(config: Any) -> bool:
    """True only when a config carries the block and has it switched on."""

    return bool(getattr(config, "config_local_timestep", False))


__all__ = [
    "DEFAULT_LOCAL_TIMESTEP_RATES",
    "V841LocalTimestepDryConfig",
    "V841LocalTimestepGwdoConfig",
    "V841LocalTimestepSmagorinskyGwdoConfig",
    "local_timestep_enabled",
]
