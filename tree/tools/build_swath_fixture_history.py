"""Write a history file for the swath placement layer to be tested against.

WHAT THIS IS, AND WHAT IT IS NOT.  The MESH is real when ``--grid`` names a
real MPAS grid file, and the WRITER is always the port's own shipped
``hexcore.output.write_history`` -- so the detector is read against the
real writer's on-disk conventions (one-based ``cellsOnCell`` with zero
padding, radians, MPAS time strings), never against a file this tool
wrote to its own taste.  That is deliberate: a reader tested only against
its own fixture agrees with itself and with nothing else.

The FIELDS are analytic.  A moving Gaussian pressure well with a
tangential circulation around it is not a tropical cyclone, and this tool
does not claim it is.  What it exercises is the mechanism: does a
mesh-native extremum search find a moving minimum, does the association
join it into one track, does the flared ring land on the projected path,
does the ranking put the deeper one first, does the delayed start fire on
the hour the field first exceeds.  The meteorology is proven by running
the same layer on a real coarse forecast, which needs a card; the receipt
in ``evidence/swath-following-20260826/`` states which legs are which.

Storms are ROWS.  Adding one to a scenario is a dict in a list, never a
branch -- the same discipline the layer under test is held to.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexcore.output import write_history  # noqa: E402

EARTH_RADIUS_M = 6371229.0
NEIGHBOURS_PER_CELL = 6


# ---------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------
@dataclass
class FixtureMesh:
    """A duck-typed mesh the shipped history writer accepts.

    ``arrays`` holds the port's AUTHORITY form -- zero-based connectivity
    with ``-1`` for absent -- because that is what ``write_history``
    converts to MPAS's one-based disk convention on the way out.  Handing
    it disk-form arrays would double-convert and every neighbour index
    would be one too high, which is exactly the class of defect this
    round trip exists to catch.
    """

    arrays: dict[str, Any]
    dimensions: dict[str, int]
    variable_dimensions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variable_attrs: dict[str, Any] = field(default_factory=dict)


def fibonacci_mesh(cells: int) -> FixtureMesh:
    """A quasi-uniform global cell graph: golden-angle centres, k-nearest
    neighbours, equal areas.

    Not a Voronoi tessellation and not claimed as one.  The detector needs
    coordinates, areas and an adjacency graph; this supplies all three,
    deterministically, at any size, with no external file.
    """

    from scipy.spatial import cKDTree

    index = np.arange(cells, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / cells
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = index * math.pi * (3.0 - math.sqrt(5.0))
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    points = np.column_stack((x, y, z))

    tree = cKDTree(points)
    _, neighbour = tree.query(points, k=NEIGHBOURS_PER_CELL + 1)
    neighbour = np.asarray(neighbour[:, 1:], dtype=np.int64)

    latitude = np.arctan2(z, np.hypot(x, y))
    longitude = np.arctan2(y, x) % (2.0 * math.pi)
    area = np.full(cells, 4.0 * math.pi * EARTH_RADIUS_M**2 / cells, dtype=np.float64)

    arrays: dict[str, Any] = {
        "latCell": latitude,
        "lonCell": longitude,
        "xCell": x * EARTH_RADIUS_M,
        "yCell": y * EARTH_RADIUS_M,
        "zCell": z * EARTH_RADIUS_M,
        "indexToCellID": np.arange(1, cells + 1, dtype=np.int32),
        "areaCell": area,
        "cellsOnCell": neighbour,
        "nEdgesOnCell": np.full(cells, NEIGHBOURS_PER_CELL, dtype=np.int32),
        "meshDensity": np.ones(cells, dtype=np.float64),
    }
    return FixtureMesh(
        arrays=arrays,
        dimensions={"nCells": cells, "maxEdges": NEIGHBOURS_PER_CELL},
        variable_dimensions={
            "cellsOnCell": ("nCells", "maxEdges"),
        },
    )


def mesh_from_grid(path: Path) -> FixtureMesh:
    """Take the real cell graph out of a real MPAS grid file."""

    from netCDF4 import Dataset

    with Dataset(str(path), "r") as source:
        cells = int(source.dimensions["nCells"].size)
        max_edges = int(source.dimensions["maxEdges"].size)
        latitude = np.asarray(source.variables["latCell"][:], dtype=np.float64)
        longitude = np.asarray(source.variables["lonCell"][:], dtype=np.float64)
        area = np.asarray(source.variables["areaCell"][:], dtype=np.float64)
        valency = np.asarray(source.variables["nEdgesOnCell"][:], dtype=np.int32)
        disk = np.asarray(source.variables["cellsOnCell"][:], dtype=np.int64)
        xyz = [
            np.asarray(source.variables[name][:], dtype=np.float64)
            for name in ("xCell", "yCell", "zCell")
        ]
    neighbour = disk - 1
    columns = np.arange(max_edges)[None, :]
    neighbour[columns >= valency[:, None]] = -1
    neighbour[disk == 0] = -1
    arrays: dict[str, Any] = {
        "latCell": latitude,
        "lonCell": longitude,
        "xCell": xyz[0],
        "yCell": xyz[1],
        "zCell": xyz[2],
        "indexToCellID": np.arange(1, cells + 1, dtype=np.int32),
        "areaCell": area,
        "cellsOnCell": neighbour,
        "nEdgesOnCell": valency,
        "meshDensity": np.ones(cells, dtype=np.float64),
    }
    return FixtureMesh(
        arrays=arrays,
        dimensions={"nCells": cells, "maxEdges": max_edges},
        variable_dimensions={"cellsOnCell": ("nCells", "maxEdges")},
    )


# ---------------------------------------------------------------------------
# the weather, as rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StormRow:
    """One moving feature, entirely described by data.

    THREE KINDS, AND ONLY ONE OF THEM IS A PHENOMENON.  ``low`` and
    ``convection`` build COUPLED structure -- a pressure well with a
    tangential circulation around it, a reflectivity blob with a vertical
    profile -- which is the one thing a generic writer cannot express.
    ``field_anomaly`` writes a moving Gaussian into ANY published
    variable, named in the row, so exercising a new threat definition
    against this fixture is a row here exactly as adding the threat is a
    row in the metrics document.  A fixture that needed a new branch per
    threat would fail the same test the layer under it is held to.
    """

    kind: str
    latitude_deg: float
    longitude_deg: float
    bearing_deg: float
    speed_km_per_hour: float
    radius_km: float
    amplitude: float
    #: ``field_anomaly`` only: which published variable to write into.
    variable: str = ""
    #: ``field_anomaly`` only, and only for a variable with a level axis:
    #: an integer writes one level, ``-1`` writes every level.
    level_index: int = 0
    onset_hours: float = 0.0
    #: Linear trend in the amplitude, per hour of ABSOLUTE time.  This is
    #: what lets a scenario put two features near each other in rank and
    #: have them cross, which is the only way to exercise a hysteresis
    #: rule: a rule that never has to arbitrate anything is untested.
    amplitude_per_hour: float = 0.0
    #: A sinusoidal swing on top of the trend.  Two features in antiphase
    #: then trade rank every half period, which is the JITTER a hysteresis
    #: rule exists to absorb -- a single monotone crossing costs one
    #: eviction whether the rule is armed or not, and measuring on one
    #: proves nothing about the rule.
    amplitude_swing: float = 0.0
    amplitude_period_hours: float = 24.0
    amplitude_phase_hours: float = 0.0
    #: Scales the circulation independently of the depth, so a scenario can
    #: put a pressure minimum on the mesh with no wind around it -- the
    #: thermal-low case a confirmation row exists to reject.
    wind_factor: float = 1.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "StormRow":
        return cls(**raw)


def _positions(
    row: StormRow, hours: np.ndarray, offset_hours: float = 0.0
) -> list[tuple[float, float]]:
    from hexcore.swath.geometry import destination

    return [
        destination(
            row.latitude_deg, row.longitude_deg, row.bearing_deg,
            row.speed_km_per_hour * max(0.0, float(hour) + offset_hours),
        )
        for hour in hours
    ]


def _distance_km(
    latitude_deg: np.ndarray, longitude_deg: np.ndarray, centre: tuple[float, float]
) -> np.ndarray:
    phi = np.radians(latitude_deg)
    lam = np.radians(longitude_deg)
    phi0 = math.radians(centre[0])
    lam0 = math.radians(centre[1])
    h = (
        np.sin(0.5 * (phi - phi0)) ** 2
        + np.cos(phi) * math.cos(phi0) * np.sin(0.5 * (lam - lam0)) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(h)))


#: Every variable the shipped threat-metrics document's armed rows can
#: reach, with the axis shape the port's own history writer gives it.  The
#: fixture writes ALL of them, because the publication manifest is a
#: CONTRACT: a coarse run that publishes fewer disarms rows silently from
#: the reader's point of view and refuses loudly from the detector's, and
#: a fixture that could not exercise the shipped document would only ever
#: test the rows it happened to be written for.
SOIL_LEVELS = 4


def _base_state(
    latitude: np.ndarray,
    longitude: np.ndarray,
    t2: np.ndarray,
    pressure: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    levels: int,
) -> dict[str, np.ndarray]:
    """An atmosphere quiet enough that only the storm rows say anything.

    DELIBERATELY INERT, and each choice is a decision not to fire a
    shipped row by accident:

    * ``q2`` is set to 70 per cent relative humidity, so the fire row's
      20 per cent dryness margin is negative everywhere until a scenario
      dries somewhere out.
    * ``smois`` is 1.0, which is what a land surface writes over WATER, so
      the fire row's fuel margin is negative everywhere as well -- the
      base planet is an ocean and a fire row cannot fire on it.
    * ``qv`` falls off with height fast enough that the bulk vapour
      transport stays under the 250 kg m^-1 s^-1 atmospheric-river
      threshold at every latitude.
    * ``w`` and the three precipitation accumulators are zero, so the
      severe, heavy-rain and winter rows have nothing.
    * The winds carry a mid-latitude jet that GROWS with level, so deep
      layer shear is real and a shear ingredient is exercised rather than
      being identically zero.
    """

    frames, cells = t2.shape
    # 70 per cent relative humidity, through the same Bolton form the
    # metrics document's own humidity chain uses.
    celsius = t2 - 273.15
    saturation_hpa = 6.112 * np.exp(17.67 * celsius / (celsius + 243.5))
    vapour_hpa = 0.70 * saturation_hpa
    q2 = 0.62197 * vapour_hpa * 100.0 / np.maximum(pressure - vapour_hpa * 100.0, 1.0)

    jet = 25.0 * np.exp(-(((np.abs(latitude) - 45.0) / 15.0) ** 2))
    ramp = (np.arange(levels, dtype=np.float64) / max(1, levels - 1))[None, :, None]
    u_zonal = u10[:, None, :] + jet[None, None, :] * ramp
    v_meridional = np.repeat(v10[:, None, :], levels, axis=1)
    qv = q2[:, None, :] * np.exp(
        -np.arange(levels, dtype=np.float64)[None, :, None] / 6.0
    )
    return {
        "q2": q2,
        "u_zonal": u_zonal,
        "v_meridional": v_meridional,
        "qv": qv,
        "w": np.zeros((frames, levels + 1, cells), dtype=np.float64),
        "smois": np.ones((frames, SOIL_LEVELS, cells), dtype=np.float64),
        "rainc": np.zeros((frames, cells), dtype=np.float64),
        "rainnc": np.zeros((frames, cells), dtype=np.float64),
        "snownc": np.zeros((frames, cells), dtype=np.float64),
    }


def build_fields(
    mesh: FixtureMesh,
    storms: Sequence[StormRow],
    hours: np.ndarray,
    levels: int = 26,
    offset_hours: float = 0.0,
    terrain: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fields for every frame, from the storm rows and nothing else.

    ``terrain`` raises ground under the world.  The default is a sea-level
    planet, where the sea-level reduction is the identity and every
    calibrated number in this fixture is what it always was.  A scenario
    that names a plateau is how the reduction gets something to do -- and
    a raw-surface-pressure search gets something to fail on, which is what
    a real global forecast does to it.
    """

    latitude = np.degrees(mesh.arrays["latCell"])
    longitude = ((np.degrees(mesh.arrays["lonCell"]) + 180.0) % 360.0) - 180.0
    cells = latitude.size
    frames = hours.size

    pressure = np.full((frames, cells), 101325.0, dtype=np.float64)
    u10 = np.zeros((frames, cells), dtype=np.float64)
    v10 = np.zeros((frames, cells), dtype=np.float64)
    reflectivity = np.zeros((frames, levels, cells), dtype=np.float64)
    anomalies: list[tuple[StormRow, int, np.ndarray]] = []

    for source in storms:
        track = _positions(source, hours, offset_hours)
        for frame, hour in enumerate(hours.tolist()):
            absolute = hour + offset_hours
            if absolute < source.onset_hours:
                continue
            amplitude = source.amplitude + source.amplitude_per_hour * absolute
            if source.amplitude_swing:
                amplitude += source.amplitude_swing * math.sin(
                    2.0 * math.pi
                    * (absolute + source.amplitude_phase_hours)
                    / source.amplitude_period_hours
                )
            if source.kind == "field_anomaly":
                # A SIGNED amplitude, because most of what a compound
                # threat needs is a field going DOWN: a soil layer drying
                # out, a mixing ratio falling, a temperature dropping
                # below freezing.
                if amplitude == 0.0:
                    continue
            else:
                amplitude = max(0.0, amplitude)
                if amplitude <= 0.0:
                    continue
            row = replace(source, amplitude=amplitude)
            centre = track[frame]
            distance = _distance_km(latitude, longitude, centre)
            shape = np.exp(-((distance / row.radius_km) ** 2))
            if row.kind == "low":
                pressure[frame] -= row.amplitude * shape
                # Tangential wind peaking near the radius: a circulation, so
                # the confirming field is not a relabelled copy of the low.
                # Peak tangential wind of about amplitude/100, so a 42 hPa
                # low carries about 42 m/s.  It was amplitude/25 -- 168 m/s
                # for the same low -- until the shipped document grew rows
                # that read the wind for their own purposes: at 168 m/s the
                # atmospheric-river row's transport threshold was cleared by
                # the fixture's BASE STATE, so a row fired on every scenario
                # for a reason that had nothing to do with the scenario.
                tangential = (
                    row.wind_factor * row.amplitude / 100.0
                    * (distance / row.radius_km)
                    * np.exp(0.5 - 0.5 * (distance / row.radius_km) ** 2)
                )
                bearing = np.radians(
                    (
                        np.degrees(
                            np.arctan2(
                                np.sin(np.radians(longitude - centre[1]))
                                * np.cos(np.radians(latitude)),
                                np.cos(math.radians(centre[0]))
                                * np.sin(np.radians(latitude))
                                - math.sin(math.radians(centre[0]))
                                * np.cos(np.radians(latitude))
                                * np.cos(np.radians(longitude - centre[1])),
                            )
                        )
                        + 90.0
                    )
                )
                u10[frame] += tangential * np.sin(bearing)
                v10[frame] += tangential * np.cos(bearing)
            elif row.kind == "convection":
                profile = np.exp(-((np.arange(levels) - 0.4 * levels) / (0.3 * levels)) ** 2)
                reflectivity[frame] += (
                    row.amplitude * shape[None, :] * profile[:, None]
                )
            elif row.kind == "field_anomaly":
                # Deferred: the target array is part of the base state,
                # which is not built until the temperature and the
                # pressure are final.  The GEOMETRY is computed here, once.
                anomalies.append((row, frame, row.amplitude * shape))
            else:
                raise ValueError(
                    f"storm row kind {row.kind!r} is not one of 'low', 'convection' "
                    "or 'field_anomaly'"
                )

    # Terrain, and the surface pressure that stands ON it.  A real model
    # publishes pressure at the ground, not at sea level; the fixture only
    # tells the truth about that if it does the same.
    height = np.zeros(cells, dtype=np.float64)
    for plateau in terrain or ():
        centre = (float(plateau["latitude_deg"]), float(plateau["longitude_deg"]))
        distance = _distance_km(latitude, longitude, centre)
        height += float(plateau["height_m"]) * np.exp(
            -((distance / float(plateau["radius_km"])) ** 2)
        )
    # A plain latitude profile: warm equator, cold poles, nothing seasonal.
    # 302.15 K at the equator rather than 288.15, because a threat row is
    # allowed to ask whether a low sits in TROPICAL air and a planet whose
    # equator is 15 C has no tropics for it to find.
    t2 = 302.15 - 45.0 * np.sin(np.radians(latitude)) ** 2 - 0.0065 * height
    if np.any(height > 0.0):
        mean_temperature = t2 + 0.5 * 0.0065 * height
        pressure = pressure / np.exp(
            9.80665 * height[None, :] / (287.058 * mean_temperature[None, :])
        )

    t2_frames = np.repeat(t2[None, :], frames, axis=0)
    prepared: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {
        "surface_pressure": (pressure, ("Time", "nCells")),
        "u10": (u10, ("Time", "nCells")),
        "v10": (v10, ("Time", "nCells")),
        "refl10cm": (reflectivity, ("Time", "nVertLevels", "nCells")),
        "ter": (np.repeat(height[None, :], frames, axis=0), ("Time", "nCells")),
        "t2": (t2_frames, ("Time", "nCells")),
    }
    base = _base_state(latitude, longitude, t2_frames, pressure, u10, v10, levels)
    for name, values in base.items():
        if values.ndim == 3:
            axis = (
                "nSoilLevels" if name == "smois"
                else "nVertLevelsP1" if values.shape[1] == levels + 1
                else "nVertLevels"
            )
            prepared[name] = (values, ("Time", axis, "nCells"))
        else:
            prepared[name] = (values, ("Time", "nCells"))

    # The deferred anomalies, now that every target exists.  A row naming a
    # variable this fixture does not write is refused BY NAME: silently
    # dropping it would leave a scenario that looks like it exercises a
    # threat and does not.
    for row, frame, field in anomalies:
        if row.variable not in prepared:
            raise ValueError(
                f"storm row 'field_anomaly' names variable {row.variable!r}, which "
                f"this fixture does not write. Written variables: "
                f"{sorted(prepared)}"
            )
        array, dimensions = prepared[row.variable]
        if array.ndim == 2:
            array[frame] += field
        elif row.level_index < 0:
            array[frame, :, :] += field[None, :]
        else:
            if not 0 <= row.level_index < array.shape[1]:
                raise ValueError(
                    f"storm row 'field_anomaly' on {row.variable!r} asks for level "
                    f"{row.level_index}, but that variable has {array.shape[1]} levels"
                )
            array[frame, row.level_index, :] += field
    return prepared


DEFAULT_SCENARIO: list[dict[str, Any]] = [
    # Three lows of clearly different depth in three basins, so the ranking
    # has something to order and the separation rule has something to reject.
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0, "bearing_deg": 300.0,
     "speed_km_per_hour": 22.0, "radius_km": 420.0, "amplitude": 4200.0},
    {"kind": "low", "latitude_deg": 14.0, "longitude_deg": 132.0, "bearing_deg": 310.0,
     "speed_km_per_hour": 26.0, "radius_km": 380.0, "amplitude": 3400.0},
    {"kind": "low", "latitude_deg": -18.0, "longitude_deg": 62.0, "bearing_deg": 240.0,
     "speed_km_per_hour": 18.0, "radius_km": 400.0, "amplitude": 2200.0},
    # One that appears only after hour 12: the genesis case, which no track
    # archive and no operator could have seeded at hour zero.
    {"kind": "low", "latitude_deg": 12.0, "longitude_deg": -28.0, "bearing_deg": 290.0,
     "speed_km_per_hour": 20.0, "radius_km": 360.0, "amplitude": 2600.0,
     "onset_hours": 12.0},
    # A convective area that ignites at hour 9: the delayed-start case.
    {"kind": "convection", "latitude_deg": 38.0, "longitude_deg": -97.0,
     "bearing_deg": 75.0, "speed_km_per_hour": 45.0, "radius_km": 320.0,
     "amplitude": 52.0, "onset_hours": 9.0},
]


def build(
    path: str | Path,
    *,
    cells: int = 40962,
    hours: Sequence[float] | None = None,
    scenario: Sequence[dict[str, Any]] | None = None,
    grid: str | Path | None = None,
    #: Enough levels for the shipped document's mid-troposphere slice at
    #: index 23.  A fixture with fewer cannot exercise the shear ingredient
    #: of the severe row, and a document that reads level 23 out of an
    #: eight-level file refuses -- correctly, and uselessly, in a test.
    levels: int = 26,
    offset_hours: float = 0.0,
    mesh: "FixtureMesh | None" = None,
    terrain: Sequence[dict[str, Any]] | None = None,
) -> Path:
    if mesh is None:
        mesh = mesh_from_grid(Path(grid)) if grid is not None else fibonacci_mesh(cells)
    hour_values = np.asarray(
        list(hours) if hours is not None else [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
        dtype=np.float64,
    )
    rows = [StormRow.from_mapping(entry) for entry in (scenario or DEFAULT_SCENARIO)]
    prepared = build_fields(
        mesh, rows, hour_values, levels=levels, offset_hours=offset_hours,
        terrain=terrain,
    )
    fields = {name: value for name, (value, _) in prepared.items()}
    dimensions = {name: dims for name, (_, dims) in prepared.items()}
    from datetime import datetime, timedelta

    start = datetime(2026, 8, 26, 0, 0, 0) + timedelta(hours=float(offset_hours))
    times = [start + timedelta(hours=float(hour)) for hour in hour_values]
    return write_history(
        path,
        mesh,
        fields,
        times,
        initial_time=start,
        time_seconds=[float(hour) * 3600.0 for hour in hour_values],
        field_dimensions=dimensions,
        n_vert_levels=levels,
        global_attrs={
            "source": "gpuwm-hex swath placement fixture",
            "fixture_fields": "analytic; the mesh and the writer are real",
        },
        stream_options={"clobber_mode": "truncate"},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_swath_fixture_history",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=40962)
    parser.add_argument("--grid", type=Path, default=None,
                        help="a real MPAS grid file to take the cell graph from")
    parser.add_argument("--hours", type=float, nargs="*", default=None)
    parser.add_argument("--scenario", type=Path, default=None,
                        help="a JSON list of storm rows, replacing the default")
    arguments = parser.parse_args(argv)
    scenario = (
        json.loads(arguments.scenario.read_text(encoding="utf-8"))
        if arguments.scenario is not None
        else None
    )
    written = build(
        arguments.out,
        cells=arguments.cells,
        hours=arguments.hours,
        scenario=scenario,
        grid=arguments.grid,
    )
    print(json.dumps({"history": str(written), "bytes": written.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
