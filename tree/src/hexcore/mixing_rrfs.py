"""Fork saved-Euler mixing options.

RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5393-6409
at 1d6e370e. The fork keeps fixed/vertical diffusion in this complete work routine.
"""
from dataclasses import dataclass
from .mixing_v841 import V841MixingConfig

@dataclass(frozen=True,slots=True)
class RRFSMixingConfig(V841MixingConfig):
    config_h_mom_eddy_visc2: float=0.0
    config_h_theta_eddy_visc2: float=0.0
    config_h_mom_eddy_visc4: float=0.0
    config_h_theta_eddy_visc4: float=0.0
    config_v_mom_eddy_visc2: float=0.0
    config_v_theta_eddy_visc2: float=0.0
    config_mix_full: bool=True

    # MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:148-321 at 1d6e370e.
    def validate(self):
        """Fork Registry.xml at 1d6e370e: native selectors, real coefficients."""
        if self.config_horiz_mixing not in ('off','2d_fixed','2d_smagorinsky'):
            raise ValueError('unsupported native horizontal mixing selector')
        if self.config_horiz_mixing=='2d_smagorinsky':
            from dataclasses import fields
            V841MixingConfig(**{f.name:getattr(self,f.name) for f in fields(V841MixingConfig)}).validate()
