"""The forecast admits ice columns as native init makes them (hexcore.surface_signature).

The defect: RRFS v2's conus3km init (NOAA MPAS fork, GFS 2026-10-03 06Z) was
refused at host preparation with "native xland/ice category signature is
inconsistent" over two frozen mountain-lake cells that native's
skin-temperature branch had converted to ice -- ivgtyp 15, xland 1, landmask
0, xice 0.  The signature demanded landmask 1 under a glacier, which native
never promises and nothing downstream routes on.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.surface_signature import (
    ISICE_CATEGORY,
    SurfaceSignatureRefusal,
    classify_native_surface,
)

THRESHOLD = 0.02


def _columns(rows):
    """Build the four native fields from (xland, xice, ivgtyp, landmask) rows."""

    xland = np.asarray([row[0] for row in rows], dtype=np.float32)
    xice = np.asarray([row[1] for row in rows], dtype=np.float32)
    ivgtyp = np.asarray([row[2] for row in rows], dtype=np.int32)
    landmask = np.asarray([row[3] for row in rows], dtype=np.int32)
    return dict(xland=xland, xice=xice, ivgtyp=ivgtyp, landmask=landmask)


# The x4 proof's own shapes: vegetated land, open water, sea ice on a water
# cell, a land glacier.
X4_ROWS = [
    (1.0, 0.0, 10, 1),  # grassland
    (2.0, 0.0, 17, 0),  # ocean
    (1.0, 0.9, ISICE_CATEGORY, 0),  # sea ice, converted by native
    (1.0, 0.0, ISICE_CATEGORY, 1),  # Antarctic glacier
]

# Native init's skin-temperature branch (physics_init_seaice): a water cell
# colder than config_tsk_seaice_threshold becomes ivgtyp=isice, xland=1;
# landmask is a static field and stays 0; xice is left as found (0 here).
# This is the conus3km shape, as measured: two cells near 52.2N 117.4W.
CONUS3KM_FROZEN_LAKE = (1.0, 0.0, ISICE_CATEGORY, 0)


def test_x4_shapes_partition_and_count_as_before():
    counts, masks = classify_native_surface(**_columns(X4_ROWS), xice_threshold=THRESHOLD)
    assert counts["sflx_land_columns"] == 1
    assert counts["open_water_columns"] == 1
    assert counts["sea_ice_columns"] == 1
    assert counts["glacier_columns"] == 1
    assert counts["xland_land_columns"] == 3
    assert counts["xland_water_columns"] == 1
    assert counts["glacier_columns_on_landmask_0"] == 0
    assert counts["sea_ice_columns_on_landmask_1"] == 0
    assert counts["glacier_columns_with_subthreshold_xice"] == 0
    assert masks["glacier"].tolist() == [False, False, False, True]
    assert masks["sea_ice"].tolist() == [False, False, True, False]


def test_conus3km_frozen_lake_cells_are_admitted_as_glacier_and_counted():
    rows = X4_ROWS + [CONUS3KM_FROZEN_LAKE, CONUS3KM_FROZEN_LAKE]
    counts, masks = classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    # They are glacier columns (xland 1, below the sea-ice threshold, isice),
    # exactly what the engine's native-xland law routes them to.
    assert counts["glacier_columns"] == 3
    assert masks["glacier"].tolist() == [False, False, False, True, True, True]
    # And the receipt says native's skin-temperature branch made two of them.
    assert counts["glacier_columns_on_landmask_0"] == 2
    assert counts["landmask_source"] == "native static, measured not refused"
    # The partition still closes.
    assert (
        counts["sflx_land_columns"]
        + counts["open_water_columns"]
        + counts["sea_ice_columns"]
        + counts["glacier_columns"]
        == len(rows)
    )


def test_native_frac_seaice_leaves_subthreshold_xice_on_a_converted_column():
    # physics_init_seaice does not rewrite xice on the skin-temperature
    # branch, so under config_frac_seaice a cold water cell with a small ice
    # fraction keeps it.  The old signature refused xice != 0 under a glacier.
    rows = X4_ROWS + [(1.0, 0.01, ISICE_CATEGORY, 0)]
    counts, masks = classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    assert masks["glacier"][-1]
    assert not masks["sea_ice"][-1]
    assert counts["glacier_columns_with_subthreshold_xice"] == 1
    assert counts["glacier_columns_on_landmask_0"] == 1


def test_sea_ice_on_a_land_cell_is_measured_not_refused():
    # The met data's ice fraction reaching a coastal land cell: native
    # converts it (ivgtyp isice, xland 1) and leaves landmask 1.  The old
    # signature demanded landmask 0 under sea ice.
    rows = X4_ROWS + [(1.0, 0.6, ISICE_CATEGORY, 1)]
    counts, masks = classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    assert masks["sea_ice"][-1]
    assert counts["sea_ice_columns"] == 2
    assert counts["sea_ice_columns_on_landmask_1"] == 1


def test_sea_ice_with_water_xland_refuses_naming_the_cells_and_the_branch():
    rows = X4_ROWS + [(2.0, 0.7, ISICE_CATEGORY, 0), (2.0, 0.8, ISICE_CATEGORY, 0)]
    with pytest.raises(SurfaceSignatureRefusal) as refusal:
        classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    message = str(refusal.value)
    assert "2 sea-ice column(s)" in message
    assert "cells [4, 5]" in message
    assert "open-water branch over the ice" in message


def test_sea_ice_still_vegetated_refuses_naming_the_category():
    rows = X4_ROWS + [(1.0, 0.7, 10, 0)]
    with pytest.raises(SurfaceSignatureRefusal) as refusal:
        classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    message = str(refusal.value)
    assert "1 sea-ice column(s)" in message
    assert f"isice ({ISICE_CATEGORY})" in message
    assert "cells [4]" in message


def test_refusal_names_at_most_eight_cells_then_the_remainder():
    rows = X4_ROWS + [(2.0, 0.7, ISICE_CATEGORY, 0)] * 11
    with pytest.raises(SurfaceSignatureRefusal) as refusal:
        classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    message = str(refusal.value)
    assert "11 sea-ice column(s)" in message
    assert "cells [4, 5, 6, 7, 8, 9, 10, 11] and 3 more" in message


def test_all_ocean_domain_has_no_land_and_is_admitted():
    rows = [(2.0, 0.0, 17, 0)] * 5
    counts, _ = classify_native_surface(**_columns(rows), xice_threshold=THRESHOLD)
    assert counts["open_water_columns"] == 5
    assert counts["xland_land_columns"] == 0


def test_mismatched_column_counts_are_a_value_error():
    fields = _columns(X4_ROWS)
    fields["xice"] = fields["xice"][:-1]
    with pytest.raises(ValueError, match="disagree on the column count"):
        classify_native_surface(**fields, xice_threshold=THRESHOLD)


def test_the_old_signature_text_no_longer_exists_in_the_forecast_driver():
    # Guard retirement: the refusal that named nothing is gone from the door.
    from pathlib import Path

    import hexcore.drivers.run_cuda_v841_forecast as driver

    source = Path(driver.__file__).read_text(encoding="utf-8")
    assert "native xland/ice category signature is inconsistent" not in source
    assert "classify_native_surface" in source
