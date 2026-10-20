"""The native surface-category signature a forecast admits, as native made it.

WHY THIS MODULE EXISTS.  The forecast driver partitions every column of an
init into sea ice, open water, glacier and vegetated land from three native
fields -- ``xland``, ``xice`` and ``ivgtyp`` -- the same three the engine's
native-xland law routes on, and then checked that the ice columns carried the
"category signature" the x4 proof had measured.  That signature demanded
``landmask == 0`` under sea ice, ``landmask == 1`` under a glacier and
``xice == 0`` under a glacier.  Native MPAS-A v8.4.1 does not promise any of
the three.

WHAT NATIVE ACTUALLY DOES (``mpas_atmphys_initialize_real.F``,
``physics_init_seaice``): a column becomes ice when ``xice >= xice_threshold``
OR when it is a water column (``landmask == 0``) whose skin temperature is
below ``config_tsk_seaice_threshold`` (271 K by default).  On every such
column native writes ``ivgtyp = isice``, ``isltyp = 16``, ``vegfra = 0`` and
``xland = 1``, and recomputes the soil.  It never writes ``landmask``: that is
a static mesh field, and the skin-temperature branch leaves it at 0.  It also
leaves ``xice`` as it found it, so under ``config_frac_seaice`` a cold-water
column may carry ``0 < xice < 0.02`` into the forecast.  The ``seaice`` flag
native writes last is ``xice > 0`` and nothing else, so it does not witness
the skin-temperature branch either.

THE DEFECT THIS RETIRES (RRFS v2 conus3km, NOAA fork 1d6e370e, GFS 2026-10-03
06Z; measured 2026-10-05): two frozen
mountain-lake cells near 52.2N 117.4W, skin 270.8-271.0 K, came out of native
init exactly as above -- ``ivgtyp 15, xland 1, landmask 0, xice 0`` -- and the
landmask demand refused the whole 2,341,818-cell forecast with "native
xland/ice category signature is inconsistent".  The hex fixer group of
2026-10-05 saw the same refusal on base and lane alike.

WHY THE LANDMASK CONDITIONS PREVENTED NOTHING.  Nothing downstream routes a
column on ``landmask``: the engine's native-xland law (gpuwm
``core/physics.py``) takes ``xland`` verbatim and names this exact column
shape -- native xland 1 on landmask 0 with VEGTYP=ISICE -- as the case that
law exists for, routing it to the glacier surface; and the sealed
constructor's own ``expected_surface_classification`` partitions on ``xland``,
``xice`` and ``ivgtyp`` only.  A refusal that names no breakage does not
exist (gate law), so landmask is measured into the receipt here and refuses
nothing.

WHAT STILL REFUSES, AND WHAT EACH REFUSAL PREVENTS.  A sea-ice column
(``xice >= xice_threshold``) must carry native's conversion:

* ``xland == 1``.  With ``xland 2`` the surface layer runs its water branch
  over the ice -- Charnock roughness and full moisture availability on a
  frozen surface -- while the sea-ice path reads the same column as ice.
* ``ivgtyp == isice``.  A sea-ice column native did not convert still carries
  its vegetated category and the unconverted soil state that goes with it
  (native rewrites ``smois`` to 1 and ``tslb`` toward 271.4 K on
  conversion), and the fractional sea-ice blend runs the vegetated land path
  on the ice-free fraction of a column whose soil is open water.

Each refusal names the condition, the count of offending columns and the
first few cell indices, so the next such init is diagnosed from the message
rather than from a box.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

#: MODIS/IGBP ice category in the modified-IGBP tables native runs with.
ISICE_CATEGORY = 15

#: How many offending cells a refusal names before it says "and N more".
_NAMED_CELLS = 8


class SurfaceSignatureRefusal(RuntimeError):
    """A sea-ice column does not carry native's ice conversion."""


def _name_cells(indices: np.ndarray) -> str:
    shown = ", ".join(str(int(index)) for index in indices[:_NAMED_CELLS])
    more = int(indices.size) - min(int(indices.size), _NAMED_CELLS)
    return f"cells [{shown}]" + (f" and {more} more" if more > 0 else "")


def classify_native_surface(
    *,
    xland: np.ndarray,
    xice: np.ndarray,
    ivgtyp: np.ndarray,
    landmask: np.ndarray,
    xice_threshold: float,
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    """Partition the columns as the engine will, and admit them as native made them.

    Returns ``(counts, masks)``.  ``counts`` is the surface-classification
    receipt block: the four-way partition the engine runs, plus the measured
    landmask and sub-threshold-xice facts the old signature used to refuse
    on.  ``masks`` carries ``sea_ice``, ``open_water``, ``land``, ``glacier``
    and ``sflx_land`` boolean arrays.

    Raises :class:`SurfaceSignatureRefusal` only for a sea-ice column that
    native did not convert (``xland != 1`` or ``ivgtyp != isice``), naming
    the condition, the count and the first cells.
    """

    xland = np.asarray(xland, dtype=np.float32).reshape(-1)
    xice = np.asarray(xice, dtype=np.float32).reshape(-1)
    ivgtyp = np.asarray(ivgtyp, dtype=np.int32).reshape(-1)
    landmask = np.asarray(landmask, dtype=np.int32).reshape(-1)
    n_cells = int(xland.size)
    if not (xice.size == ivgtyp.size == landmask.size == n_cells):
        raise ValueError(
            "surface fields disagree on the column count: "
            f"xland {xland.size}, xice {xice.size}, ivgtyp {ivgtyp.size}, "
            f"landmask {landmask.size}"
        )
    threshold = np.float32(xice_threshold)

    sea_ice = np.ascontiguousarray(xice >= threshold)
    open_water = np.ascontiguousarray((xland >= np.float32(1.5)) & ~sea_ice)
    land = np.ascontiguousarray(~(sea_ice | open_water))
    glacier = np.ascontiguousarray(land & (ivgtyp == np.int32(ISICE_CATEGORY)))
    sflx_land = np.ascontiguousarray(land & ~glacier)

    # Native's conversion, checked where it decides a surface branch.
    water_under_ice = np.flatnonzero(sea_ice & (xland != np.float32(1.0)))
    if water_under_ice.size:
        raise SurfaceSignatureRefusal(
            f"{int(water_under_ice.size)} sea-ice column(s) (xice >= "
            f"{float(threshold):g}) carry xland 2 instead of native's 1: "
            f"{_name_cells(water_under_ice)}. Native init writes xland 1 on "
            "every column it converts to ice; with xland 2 the surface layer "
            "runs its open-water branch over the ice while the sea-ice path "
            "treats the same column as ice"
        )
    unconverted_under_ice = np.flatnonzero(
        sea_ice & (ivgtyp != np.int32(ISICE_CATEGORY))
    )
    if unconverted_under_ice.size:
        raise SurfaceSignatureRefusal(
            f"{int(unconverted_under_ice.size)} sea-ice column(s) (xice >= "
            f"{float(threshold):g}) carry a vegetation category other than "
            f"isice ({ISICE_CATEGORY}): {_name_cells(unconverted_under_ice)}. "
            "Native init rewrites ivgtyp, isltyp, vegfra and the soil on every "
            "column it converts to ice; a column still vegetated under sea ice "
            "was not converted, and the fractional sea-ice blend would run the "
            "vegetated land path on soil that is open water"
        )

    counts: dict[str, Any] = {
        "xland_source": "native",
        "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
        "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        "xice_threshold": float(threshold),
        "sea_ice_columns": int(np.count_nonzero(sea_ice)),
        "open_water_columns": int(np.count_nonzero(open_water)),
        "sflx_land_columns": int(np.count_nonzero(sflx_land)),
        "glacier_columns": int(np.count_nonzero(glacier)),
        # MEASURED, never refused: landmask is native's static mesh field and
        # routes nothing.  Ice on landmask 0 is native's skin-temperature
        # conversion of a cold water column (the conus3km case); sea ice on
        # landmask 1 is a land cell the met data's ice fraction reached.
        "landmask_source": "native static, measured not refused",
        "glacier_columns_on_landmask_0": int(
            np.count_nonzero(glacier & (landmask == np.int32(0)))
        ),
        "sea_ice_columns_on_landmask_1": int(
            np.count_nonzero(sea_ice & (landmask == np.int32(1)))
        ),
        # Native leaves xice as it found it on a skin-temperature conversion,
        # so a glacier column may carry 0 < xice < threshold.  The engine
        # treats it as glacier (below threshold); counted, not refused.
        "glacier_columns_with_subthreshold_xice": int(
            np.count_nonzero(glacier & (xice > np.float32(0.0)))
        ),
    }
    if (
        counts["sflx_land_columns"]
        + counts["open_water_columns"]
        + counts["sea_ice_columns"]
        + counts["glacier_columns"]
        != n_cells
    ):
        raise RuntimeError("surface classification does not partition the mesh")
    masks = {
        "sea_ice": sea_ice,
        "open_water": open_water,
        "land": land,
        "glacier": glacier,
        "sflx_land": sflx_land,
    }
    return MappingProxyType(counts), MappingProxyType(masks)
