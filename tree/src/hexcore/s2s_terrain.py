"""Put a 2 m temperature normal on the model's terrain before subtraction.

The fixed 6.5 K/km adjustment removes the elevation comparison term. It is
an explicit product convention, not a correction of forecast physics or a
claim to diagnose local inversions. All array arithmetic uses the caller's
GPU array namespace on the production route.
"""
import math

LAPSE_RATE_K_PER_M = 0.0065
GRAVITY_M_S2 = 9.80665


def temperature_on_model_terrain(xp, normal, model_height, normal_height,
                                 *, lapse_rate=LAPSE_RATE_K_PER_M):
    if not math.isfinite(lapse_rate) or lapse_rate < 0:
        raise ValueError("temperature terrain lapse rate must be finite and nonnegative")
    if normal.shape != model_height.shape or normal.shape != normal_height.shape:
        raise ValueError("temperature normal and both terrain fields must have identical grids")
    # Missing terrain must never silently become an uncorrected anomaly.
    if not all(bool(xp.isfinite(a).all()) for a in (normal, model_height, normal_height)):
        raise ValueError("temperature normal and both terrain fields must be finite")
    return normal - lapse_rate * (model_height - normal_height)


def normal_orography(xp, dataset, shape):
    """Read the ERA5 source terrain, in metres, on the product grid."""
    if "orography" not in dataset.variables:
        raise ValueError("ERA5 temperature anomaly requires source orography in metres")
    variable = dataset.variables["orography"]
    if variable.getncattr("units") != "m":
        raise ValueError("ERA5 source orography units must be m")
    terrain = xp.asarray(variable[:])
    if terrain.shape != shape or not bool(xp.isfinite(terrain).all()):
        raise ValueError("ERA5 source orography must be finite on the product grid")
    return terrain
