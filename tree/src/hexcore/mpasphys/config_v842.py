"""Native physics configuration at the reused CUDA dynamics boundary.

MPAS source: mpas_atm_core.F:1036-1040; mpas_atm_time_integration.F:2140-2176.
Physics selection belongs to the native session; dynamics retain their checks.
"""
from dataclasses import dataclass, fields

from ..config_v841 import V841DryDycoreConfig


@dataclass(frozen=True, slots=True)
class NativeMpasDycoreConfigV842(V841DryDycoreConfig):
    """Preserve native suite metadata; mpas_atm_core.F:1036-1040."""

    # Read only by the LES dissipation frame (cuda_les_v842): u, w and scalar
    # dissipation pass it to mpas_atm_dissipation_models.F (Registry default
    # true). Carried here so an LES namelist's value reaches the kernels.
    config_mix_full: bool = True

    def validate(self, *, les_dissipation=None) -> None:
        """Validate dynamics separately; mpas_atm_time_integration.F:2140-2176."""
        values = {field.name: getattr(self, field.name)
                  for field in fields(V841DryDycoreConfig)}
        # Physics has already been resolved/checked by mpasphys control and is
        # supplied through the staged callbacks, independently of ArWen's
        # held-column adapter. Only the temporary validation view is dry-suite.
        values['config_physics_suite'] = 'none'
        V841DryDycoreConfig(**values).validate(les_dissipation=les_dissipation)
