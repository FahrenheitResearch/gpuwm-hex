"""Opt-in non-LES options from pinned MPAS-A v8.4.2.

This extends the existing v8.4.1 execution schema; release dispatch and all
existing configuration defaults remain unchanged. The integrator must wire
this subtype to mixing_v842 before admitting it at the forecast door.
"""
from dataclasses import dataclass, fields, replace
import numpy as np

from .config_v841 import V841DryDycoreConfig, V841UpperDampingDryConfig
from .errors import ConfigurationRefusal
from .mixing_v842 import V842MixingConfig


def nonles_dycore_config_from_namelist(text: str) -> "V842NonlesDycoreConfig":
    """Opt-in src/core_atmosphere/Registry.xml:194-291 namelist projection.

    Reuse the canonical scalar parser, binding types and unrelated refusals.
    The forecast door must explicitly select this boundary before dispatching
    the new configuration to CUDA; legacy namelist entry points stay unchanged.
    """
    from .config import (_V841_DYCORE_BINDINGS, _bindings, _coerce_binding,
                         _KNOWN_UNSUPPORTED, parse_fortran_namelist)
    bindings = dict(_V841_DYCORE_BINDINGS) | _bindings("nhyd_model",
        "config_mix_full", "config_w_adv_order", "config_theta_adv_order",
        "config_u_vadv_order", "config_w_vadv_order", "config_theta_vadv_order")
    groups = {binding.group for binding in bindings.values()}
    values = {}
    for group, assignments in parse_fortran_namelist(text).items():
        if group not in groups and not assignments:
            raise ConfigurationRefusal(group, "empty namelist group",
                "this group has no admitted dycore binding", "an admitted group")
        for name, value in assignments.items():
            binding = bindings.get(name.casefold())
            if binding is None:
                reason = _KNOWN_UNSUPPORTED.get(name.casefold(),
                    (name, "this MPAS option has no admitted dycore binding"))[1]
                raise ConfigurationRefusal(name, value, reason, "an admitted dycore option")
            if binding.group != group:
                raise ConfigurationRefusal(name, value,
                    f"the MPAS Registry places this knob in &{binding.group}",
                    f"&{binding.group} {name}=... /")
            if binding.attribute in values:
                raise ConfigurationRefusal(name, value,
                    "multiple assignments map to the same knob", "one assignment per knob")
            values[binding.attribute] = _coerce_binding(name, value, binding.attribute,
                config_type=V842NonlesDycoreConfig)
    if values.get("config_horiz_mixing") == "2d_smagorinsky":
        values.setdefault("config_visc4_2dsmag", 0.05)
        values.setdefault("config_smagorinsky_coef", 0.125)
    if "config_smdiv" in values:
        values["config_divergence_damping"] = values["config_smdiv"] != 0.0
    values.setdefault("config_positive_definite", False)
    values.setdefault("config_apvm_upwinding", 0.5)
    return V842NonlesDycoreConfig.from_mapping(values)


@dataclass(frozen=True, slots=True)
class V842NonlesDycoreConfig(V841UpperDampingDryConfig):
    """Registry.xml:194-223,249-291; mpas_atm_time_integration.F:6319-6410.

    The five independent order options have no Fortran reader in the pinned
    v8.4.2 source. They are retained as native Registry no-ops rather than
    selecting new numerical branches. Existing scalar order knobs retain
    their separate, consumed semantics.
    """

    config_w_adv_order: int = 3
    config_theta_adv_order: int = 3
    config_u_vadv_order: int = 3
    config_w_vadv_order: int = 3
    config_theta_vadv_order: int = 3
    config_mix_full: bool = True

    def requires_nonles_mixing(self):
        """mpas_atm_time_integration.F:6354-6369,6585-6598,6880-6906.

        Native inert order options and mix_full alone do not alter the
        established Smagorinsky dispatch or default forecast arithmetic.
        """
        return (self.config_horiz_mixing == "2d_fixed"
                or self.config_mix_scalars
                or (self.config_horiz_mixing != "off" and self.config_mpas_cam_coef > 0)
                or self.config_v_mom_eddy_visc2 > 0
                or self.config_v_theta_eddy_visc2 > 0)

    def mixing_config(self):
        """mpas_atm_time_integration.F:6241-6256,6330-6369."""
        return V842MixingConfig(**{
            field.name: getattr(self, field.name) for field in fields(V842MixingConfig)
        })

    def validate(self):
        """Registry.xml:194-223,249-291; mpas_atm_dissipation_models.F:579-1606."""
        for knob in ("config_w_adv_order", "config_theta_adv_order",
                     "config_u_vadv_order", "config_w_vadv_order",
                     "config_theta_vadv_order"):
            value = getattr(self, knob)
            if type(value) is not int or value not in (2, 3, 4):
                raise ConfigurationRefusal(knob, value,
                    "the native Registry declares integer orders 2, 3 and 4", "2, 3 or 4")
        for knob in ("config_mix_full", "config_mix_scalars"):
            if not isinstance(getattr(self, knob), (bool, np.bool_)):
                raise ConfigurationRefusal(knob, getattr(self, knob),
                    "the native Registry option is logical", "True or False")
        if self.config_horiz_mixing != "off":
            self.mixing_config().validate()
        elif self.config_mix_scalars or any(getattr(self, knob) != 0 for knob in (
            "config_v_mom_eddy_visc2", "config_v_theta_eddy_visc2",
            "config_h_theta_eddy_visc2", "config_h_mom_eddy_visc4",
            "config_h_theta_eddy_visc4")):
            raise ConfigurationRefusal("config_horiz_mixing", "off",
                "the non-LES saved dissipation boundary requires a native mixing selector",
                "2d_fixed or 2d_smagorinsky")
        value = self.config_h_mom_eddy_visc2
        if not np.isfinite(value) or value < 0 or not np.isfinite(np.float32(value)):
            raise ConfigurationRefusal("config_h_mom_eddy_visc2", value,
                "Registry viscosity must be finite and nonnegative in RKIND", ">=0")
        # Retain all unrelated refusals. These neutral fields are checked by
        # the complete non-LES authority above, rather than the old dispatcher.
        neutral = {field.name: getattr(self, field.name)
                   for field in fields(V841DryDycoreConfig)}
        neutral.update(config_horiz_mixing="off", config_mix_scalars=False,
                       config_len_disp=0.0, config_visc4_2dsmag=0.0,
                       config_smagorinsky_coef=0.0, config_mpas_cam_coef=0.0)
        for knob in ("config_h_theta_eddy_visc2", "config_h_mom_eddy_visc2",
                     "config_h_theta_eddy_visc4", "config_h_mom_eddy_visc4",
                     "config_v_mom_eddy_visc2", "config_v_theta_eddy_visc2"):
            neutral[knob] = 0.0
        V841DryDycoreConfig(**neutral).validate()
        for knob in ("config_number_cam_damping_levels", "config_number_rayleigh_damp_u_levels"):
            value = getattr(self, knob)
            if type(value) is not int or value < 1:
                raise ConfigurationRefusal(knob, value,
                    "the native damping loop requires a positive integer level count", ">=1")
        days = self.config_rayleigh_damp_u_timescale_days
        if not np.isfinite(days) or days <= 0:
            raise ConfigurationRefusal("config_rayleigh_damp_u_timescale_days", days,
                "the native Rayleigh coefficient divides by a positive timescale", ">0")


@dataclass(frozen=True, slots=True)
class NativeNonlesDycoreConfigV842(V842NonlesDycoreConfig):
    """Native physics owns suite selection; mpas_atm_core.F:1036-1040.

    Mirror mpasphys.config_v842's validation boundary while retaining every
    non-LES option. The native session resolves the actual suite and supplies
    its tendencies; the temporary view validates dynamics alone.
    """

    def validate(self):
        """mpas_atm_time_integration.F:2140-2176,6319-6410,6880-6906."""
        V842NonlesDycoreConfig.validate(replace(self, config_physics_suite="none"))
