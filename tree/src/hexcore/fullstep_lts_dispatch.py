"""Lazy dispatch. Active temporal interfaces remain unavailable pending gates."""


def attach_fullstep_lts(driver, policy, *, sink=None, global_cell_ids=None):
    # This check precedes all diagnostic imports and geometry/device work.
    if not policy.fullstep_local_timestep:
        return None
    from .config_fullstep_lts import validate_fullstep_lts_request
    validate_fullstep_lts_request(policy)
    if policy.fullstep_lts_max_rate == 1:
        return None
    from .fullstep_lts import geometric_rates
    import numpy as np
    cp = driver.cp
    mesh = driver.atmosphere.mesh
    def host(value):
        return cp.asnumpy(value) if isinstance(value,cp.ndarray) else np.asarray(value)
    rates = geometric_rates(host(mesh.dc_edge),host(mesh.cells_on_edge),int(driver.ncells),policy.fullstep_lts_max_rate)
    if np.all(rates == 1):
        return None
    from .cuda_fullstep_lts_shadow import FullStepShadow
    return FullStepShadow(driver,dt=float(driver.config.config_dt),max_rate=policy.fullstep_lts_max_rate,
                          target=policy.fullstep_lts_target_transport,global_cell_ids=global_cell_ids,sink=sink)
