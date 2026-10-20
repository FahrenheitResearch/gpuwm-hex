"""The moving remap hands the target no cloud water the microphysics would evaporate at once."""

import numpy as np

from hexcore.moving_balance import (DRY_CP, LATENT_HEAT, evaporation, liquid_saturation_mixing_ratio,
                                    saturation_adjustment)

RGAS, P0 = 287.0, 1.0e5


def _cell(temperature, pressure, vapor):
    """One cell at ``temperature`` and ``pressure``: (dry density, theta_m mass, zz=1)."""
    exner = (pressure / P0) ** (RGAS / DRY_CP)
    theta_m = temperature / exner * (1.0 + 461.6 / RGAS * vapor)
    # p = p0 (zz Rd rho theta_m / p0)^(cp/cv) with zz = 1
    rho = (pressure / P0) ** ((DRY_CP - RGAS) / DRY_CP) * P0 / (RGAS * theta_m)
    return np.array([[rho]]), np.array([[rho * theta_m]]), np.ones((1, 1))


def test_mixed_fog_is_evaporated_with_its_cooling_and_no_more():
    # A 48.0N 126.3E-like cell after the 2025-11-08 move: cloud water at 91 % relative humidity, more
    # cloud than the deficit, so some is left at saturation.
    temperature, pressure = 270.9, 97580.0
    vapor = 0.91 * liquid_saturation_mixing_ratio(temperature, pressure)
    rho, heat, zz = _cell(temperature, pressure, vapor)
    cloud = 1.0e-3
    total = lambda value: float(np.sum(value))
    new_vapor, new_cloud, new_heat, receipt = saturation_adjustment(
        rho * vapor, rho * cloud, rho, heat, zz, np.zeros_like(rho), total)
    # Each cell's total water is unchanged; the cloud the air cannot hold is gone, with its latent cooling.
    np.testing.assert_allclose(new_vapor + new_cloud, rho * (vapor + cloud), rtol=1e-14)
    moved = float((new_vapor - rho * vapor)[0, 0] / rho[0, 0])
    assert 0 < moved < cloud and receipt["cloud_to_vapor_kg"] > 0
    assert float(np.sum(new_heat)) < float(np.sum(heat))
    assert abs(receipt["largest_cooling_k"] - LATENT_HEAT / DRY_CP * moved) < 1e-9
    # The adjusted cell is saturated and Thompson has nothing left to evaporate there.
    left, _, _ = evaporation(rho, new_heat, zz, new_vapor / rho, new_cloud / rho)
    assert float(left.max()) < 1e-8


def test_what_the_donors_evaporate_themselves_is_carried_untouched():
    temperature, pressure = 285.0, 90000.0
    vapor = 0.97 * liquid_saturation_mixing_ratio(temperature, pressure)
    rho, heat, zz = _cell(temperature, pressure, vapor)
    cloud = 5.0e-5
    donor, _, _ = evaporation(rho, heat, zz, np.full_like(rho, vapor), np.full_like(rho, cloud))
    new_vapor, new_cloud, new_heat, receipt = saturation_adjustment(
        rho * vapor, rho * cloud, rho, heat, zz, rho * donor, lambda value: float(np.sum(value)))
    assert receipt["cells"] == 0
    np.testing.assert_array_equal(new_heat, heat)
    np.testing.assert_array_equal(new_cloud, rho * cloud)


def test_saturated_cloud_is_left_alone():
    temperature, pressure = 280.0, 85000.0
    vapor = liquid_saturation_mixing_ratio(temperature, pressure)
    rho, heat, zz = _cell(temperature, pressure, vapor)
    _, new_cloud, new_heat, receipt = saturation_adjustment(
        rho * vapor, rho * 2e-4, rho, heat, zz, np.zeros_like(rho), lambda value: float(np.sum(value)))
    assert receipt["cells"] == 0
    np.testing.assert_array_equal(new_heat, heat)
