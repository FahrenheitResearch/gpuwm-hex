"""Area-weighted applied-versus-forcing diagnostics, reduced on the GPU.

Latitude is in radians and area in square metres. The ocean mask is the
immutable native landmask, including ocean cells currently classified as ice.
The reference is the producer's remapped CFS series, not a second GRIB decode.
"""
import math


def metrics(xp, area, latitude, ocean, sst, ice, source_sst, source_ice, t2,
            *, threshold=.02, xland=None, ivgtyp=None, isltyp=None,
            albedo=None, isice=15, iswater=17):
    arrays = (area, latitude, ocean, sst, ice, source_sst, source_ice, t2)
    if any(a.shape != area.shape for a in arrays):
        raise ValueError("surface audit fields must share the native cell order and shape")
    if not all(bool(xp.isfinite(a).all()) for a in arrays):
        raise ValueError("surface audit requires finite values, including the written model T2")
    if bool((area <= 0).any()) or not 4e14 < float(area.sum()) < 6e14:
        raise ValueError("surface audit requires dimensional global cell areas, not unit-sphere areas")
    if bool(((ice < 0) | (ice > 1) | (source_ice < 0) | (source_ice > 1)).any()):
        raise ValueError("surface audit ice fraction outside [0,1]")
    if any(a.dtype != xp.float32 for a in (sst, ice, source_sst, source_ice)):
        raise ValueError("surface word audit requires float32 SST and ice")
    for optional in (xland, ivgtyp, isltyp, albedo):
        if optional is not None and optional.shape != area.shape:
            raise ValueError("surface classification shape differs from the native grid")
    expected = xp.where(source_ice < threshold, xp.float32(0), source_ice)
    labels, reductions = [], []

    def add(name, value):
        labels.append(name)
        reductions.append(value)

    def mean(values, weights):
        total = weights.sum(dtype=xp.float64)
        return xp.where(total > 0, (values * weights).sum(dtype=xp.float64)
                        / xp.maximum(total, 1), xp.nan)

    add("applied_global_ice_area_km2", (area * ocean * ice).sum(dtype=xp.float64) / 1e6)
    add("source_raw_global_ice_area_km2", (area * ocean * source_ice).sum(dtype=xp.float64) / 1e6)
    add("source_cutoff_global_ice_area_km2", (area * ocean * expected).sum(dtype=xp.float64) / 1e6)
    add("sst_differing_words", ((sst.view(xp.uint32) != source_sst.view(xp.uint32)) & ocean).sum())
    add("ice_differing_words_after_cutoff", ((ice.view(xp.uint32) != expected.view(xp.uint32)) & ocean).sum())
    for name, polar in (("arctic", latitude >= math.pi / 3),
                        ("antarctic", latitude <= -math.pi / 3)):
        weights = area * ocean * polar
        for prefix, sea, fraction in (("applied", sst, ice), ("source", source_sst, source_ice)):
            add(name + "_" + prefix + "_sst_K", mean(sea, weights))
            add(name + "_" + prefix + "_xice", mean(fraction, weights))
        add(name + "_model_t2_over_ice_K", mean(t2, weights * ice))
        add(name + "_ice_area_km2", (weights * ice).sum(dtype=xp.float64) / 1e6)
    if xland is not None and ivgtyp is not None and isltyp is not None:
        active = ice >= threshold
        bad = ocean & ((xland != xp.where(active, 1, 2))
            | (ivgtyp != xp.where(active, isice, iswater))
            | (isltyp != xp.where(active, 16, 14)))
        add("water_ice_classification_mismatches", bad.sum())
    if albedo is not None:
        add("ice_albedo_outside_0_1", (ocean & (ice >= threshold)
            & (~xp.isfinite(albedo) | (albedo < 0) | (albedo > 1))).sum())
    return labels, xp.stack(reductions)
