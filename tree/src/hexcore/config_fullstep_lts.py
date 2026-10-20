"""Separate full-step policy. No imports or allocations on the fixed route."""
from dataclasses import dataclass
import math

from .errors import ConfigurationRefusal


@dataclass(frozen=True)
class FullStepLtsConfig:
    fullstep_local_timestep: bool = False
    fullstep_lts_mode: str = 'shadow'
    # The census ladder of the shadow.  The forecast door and the active
    # executor default the top rate to 8 (drivers/fullstep_lts_forecast_v841.py
    # names the breakage: a full-physics run at top rate 16 went non-finite at
    # 6 to 11 h, cause not found).  The earlier reason for 8, the rate-16/8
    # interface instability of the dry dynamics, is retired: the matched
    # interface flux of cuda_fullstep_lts_active removed it (2026-10-12).
    fullstep_lts_max_rate: int = 16
    fullstep_lts_buffer_rings: int = 2
    fullstep_lts_target_transport: float = 0.90
    fullstep_lts_target_advection: float = 0.95
    fullstep_lts_target_acoustic: float = 0.70
    fullstep_lts_target_mixing: float = 0.50
    fullstep_lts_max_retries: int = 4

    @property
    def rates(self):
        return tuple(r for r in (1, 2, 4, 8, 16) if r <= self.fullstep_lts_max_rate)


def validate_fullstep_lts_request(request):
    """Validate sibling policy and refuse unimplemented active execution."""
    policy = FullStepLtsConfig(**{
        name: field.default if getattr(request, name, field.default) is None else getattr(request, name, field.default)
        for name, field in FullStepLtsConfig.__dataclass_fields__.items()
    })
    defaults = FullStepLtsConfig()
    def refuse(knob, reason):
        raise ConfigurationRefusal(knob, getattr(policy, knob), reason,
                                   '--no-fullstep-local-timestep')
    if type(policy.fullstep_local_timestep) is not bool:
        refuse('fullstep_local_timestep', 'the enable switch must be Boolean')
    if not policy.fullstep_local_timestep:
        if policy != defaults:
            refuse('fullstep_local_timestep', 'the requested LTS policy would be ignored')
        return policy
    if type(policy.fullstep_lts_max_rate) is not int or policy.fullstep_lts_max_rate not in (1, 2, 4, 8, 16, 32):
        refuse('fullstep_lts_max_rate', 'only dyadic rates 1,2,4,8,16,32 have nested interval coverage')
    if policy.fullstep_lts_max_rate == 32 and policy.fullstep_lts_mode != 'active':
        refuse('fullstep_lts_max_rate', 'the shadow census ladder and its macro plan stop at rate 16; a '
               'rate-32 column would be counted in no class')
    if type(policy.fullstep_lts_buffer_rings) is not int or policy.fullstep_lts_buffer_rings != 2:
        refuse('fullstep_lts_buffer_rings', 'this admission policy requires two simultaneous neighbor-minimum passes')
    for name, cap in [('transport', .90), ('advection', .95), ('acoustic', .70), ('mixing', .50)]:
        knob = 'fullstep_lts_target_' + name
        value = getattr(policy, knob)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= cap:
            refuse(knob, f'a finite positive target no larger than {cap} is required')
    if type(policy.fullstep_lts_max_retries) is not int or policy.fullstep_lts_max_retries != 4:
        refuse('fullstep_lts_max_retries', 'the bounded macro recovery contract permits four retries')
    if policy.fullstep_lts_mode not in ('shadow', 'active'):
        refuse('fullstep_lts_mode', 'mode must be shadow or active')
    if getattr(request, 'local_timestep', False) or getattr(request, 'adaptive_timestep', False):
        refuse('fullstep_local_timestep', 'the acoustic attachment/global adaptive controller owns a different flux and clock schedule; the two schedules cannot describe the same accepted interval')
    from . import mpas_defaults
    from .fullstep_lts_classes import lts_acoustic_substeps
    order = getattr(request, 'time_integration_order', None)
    order = mpas_defaults.DEFAULT_TIME_INTEGRATION_ORDER if order is None else order
    substeps = getattr(request, 'acoustic_substeps', None)
    substeps = mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS if substeps is None else substeps
    splits = getattr(request, 'dynamics_splits', None)
    splits = 3 if splits is None else splits
    if (order, splits) != (2, 3) or not getattr(request, 'monotonic', True):
        refuse('fullstep_local_timestep', 'this full-step interface is specified for RK2/split3 with an even number of acoustic sub-steps and monotonic FCT; the requested schedule has different accepted flux weights or bounds')
    try:
        lts_acoustic_substeps(substeps)
    except ValueError as error:
        refuse('fullstep_local_timestep', f'acoustic_substeps={substeps!r}: {error}')
    # Retired 2026-10-11: active mode refused here because column physics had
    # no per-class clock.  Each class now owns a column backend sealed at its
    # own step (hexcore.drivers.fullstep_lts_forecast_v841), which carries
    # the refusals that remain (events off the coarsest class step).
    # Each class has its own boundary runtime and checkpoint state.
    # Several cards of one host are carried since the
    # multi-card stage (hexcore.fullstep_lts_blocks, cuda_fullstep_lts_multi).
    return policy
