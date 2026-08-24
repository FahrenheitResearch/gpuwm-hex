"""Deterministic temporal/spatial matching for grid and station evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .bundle import GridBundle, StationBundle
from .errors import DataError, MeasurementUnavailable


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True, slots=True)
class AlignedValues:
    forecast: np.ndarray
    observation: np.ndarray
    latitude_deg: np.ndarray | None
    longitude_deg: np.ndarray | None
    audit: dict[str, Any]


def align_grid_field(
    model: GridBundle,
    observation: GridBundle,
    *,
    field: str,
    time_tolerance_seconds: int,
    space_tolerance_km: float,
) -> AlignedValues:
    if field not in model.fields:
        raise MeasurementUnavailable(
            f"model bundle {model.artifact_path.name} lacks field {field!r}"
        )
    if field not in observation.fields:
        raise MeasurementUnavailable(
            f"observation bundle {observation.artifact_path.name} lacks field {field!r}"
        )

    obs_indices, offsets = nearest_time_indices(
        model.time_unix_s,
        observation.time_unix_s,
        tolerance_seconds=time_tolerance_seconds,
    )
    valid_time = obs_indices >= 0
    if not np.any(valid_time):
        raise MeasurementUnavailable(
            f"no {field!r} times matched within {time_tolerance_seconds} seconds"
        )
    model_values = np.asarray(model.fields[field][valid_time], dtype=np.float64)
    selected_obs = np.asarray(
        observation.fields[field][obs_indices[valid_time]], dtype=np.float64
    )

    same_grid = (
        model.latitude_deg.shape == observation.latitude_deg.shape
        and np.array_equal(model.latitude_deg, observation.latitude_deg)
        and np.array_equal(model.longitude_deg, observation.longitude_deg)
    )
    if same_grid:
        obs_values = selected_obs
        nearest_distance = np.zeros(model.latitude_deg.shape, dtype=np.float64)
        spatial_valid = np.ones(model.latitude_deg.shape, dtype=bool)
        source_indices = np.arange(model.latitude_deg.size, dtype=np.int64).reshape(
            model.latitude_deg.shape
        )
    else:
        source_indices, nearest_distance = nearest_spatial_indices(
            source_latitude_deg=observation.latitude_deg,
            source_longitude_deg=observation.longitude_deg,
            target_latitude_deg=model.latitude_deg,
            target_longitude_deg=model.longitude_deg,
        )
        spatial_valid = nearest_distance <= space_tolerance_km
        flat_obs = selected_obs.reshape(selected_obs.shape[0], -1)
        obs_values = flat_obs[:, source_indices.ravel()].reshape(
            (selected_obs.shape[0], *model.latitude_deg.shape)
        )
        if not np.all(spatial_valid):
            obs_values = obs_values.copy()
            obs_values[:, ~spatial_valid] = np.nan

    model_values = np.asarray(model_values, dtype=np.float64)
    obs_values = np.asarray(obs_values, dtype=np.float64)
    model_values.setflags(write=False)
    obs_values.setflags(write=False)
    return AlignedValues(
        forecast=model_values,
        observation=obs_values,
        latitude_deg=model.latitude_deg,
        longitude_deg=model.longitude_deg,
        audit={
            "kind": "grid",
            "field": field,
            "model_times_total": int(model.time_unix_s.size),
            "matched_times": int(np.count_nonzero(valid_time)),
            "unmatched_times": int(np.count_nonzero(~valid_time)),
            "maximum_absolute_time_offset_seconds": (
                int(np.max(np.abs(offsets[valid_time]))) if np.any(valid_time) else None
            ),
            "spatial_points_total": int(model.latitude_deg.size),
            "spatial_points_within_tolerance": int(np.count_nonzero(spatial_valid)),
            "space_tolerance_km": float(space_tolerance_km),
            "maximum_nearest_distance_km": float(np.max(nearest_distance)),
            "same_grid": bool(same_grid),
        },
    )


def align_station_field(
    model: GridBundle,
    observation: StationBundle,
    *,
    field: str,
    time_tolerance_seconds: int,
    space_tolerance_km: float,
) -> AlignedValues:
    if field not in model.fields:
        raise MeasurementUnavailable(
            f"model bundle {model.artifact_path.name} lacks field {field!r}"
        )
    if field not in observation.fields:
        raise MeasurementUnavailable(
            f"station bundle {observation.artifact_path.name} lacks field {field!r}"
        )

    model_indices, offsets = nearest_time_indices(
        observation.time_unix_s,
        model.time_unix_s,
        tolerance_seconds=time_tolerance_seconds,
    )
    spatial_indices, distances = nearest_spatial_indices(
        source_latitude_deg=model.latitude_deg,
        source_longitude_deg=model.longitude_deg,
        target_latitude_deg=observation.latitude_deg,
        target_longitude_deg=observation.longitude_deg,
    )
    valid = (model_indices >= 0) & (distances <= space_tolerance_km)
    forecast = np.full(observation.time_unix_s.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        values = np.asarray(model.fields[field], dtype=np.float64).reshape(
            model.time_unix_s.size, -1
        )
        forecast[valid] = values[
            model_indices[valid],
            spatial_indices[valid],
        ]
    observed = np.asarray(observation.fields[field], dtype=np.float64).copy()
    observed[~valid] = np.nan
    forecast.setflags(write=False)
    observed.setflags(write=False)
    return AlignedValues(
        forecast=forecast,
        observation=observed,
        latitude_deg=observation.latitude_deg,
        longitude_deg=observation.longitude_deg,
        audit={
            "kind": "stations",
            "field": field,
            "records_total": int(observation.time_unix_s.size),
            "records_matched": int(np.count_nonzero(valid)),
            "records_unmatched": int(np.count_nonzero(~valid)),
            "time_unmatched": int(np.count_nonzero(model_indices < 0)),
            "space_unmatched": int(np.count_nonzero(distances > space_tolerance_km)),
            "maximum_absolute_time_offset_seconds": (
                int(np.max(np.abs(offsets[model_indices >= 0])))
                if np.any(model_indices >= 0)
                else None
            ),
            "space_tolerance_km": float(space_tolerance_km),
            "maximum_nearest_distance_km": float(np.max(distances)),
        },
    )


def nearest_time_indices(
    target_time_unix_s: np.ndarray,
    source_time_unix_s: np.ndarray,
    *,
    tolerance_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each target to nearest source; ties choose the earlier source."""

    target = np.asarray(target_time_unix_s, dtype=np.int64)
    source = np.asarray(source_time_unix_s, dtype=np.int64)
    if target.ndim != 1 or source.ndim != 1 or source.size == 0:
        raise DataError("nearest_time_indices requires non-empty one-dimensional times")
    if not np.all(np.diff(source) > 0):
        raise DataError("source times must be strictly increasing")
    positions = np.searchsorted(source, target, side="left")
    right = np.clip(positions, 0, source.size - 1)
    left = np.clip(positions - 1, 0, source.size - 1)
    left_delta = np.abs(target - source[left])
    right_delta = np.abs(source[right] - target)
    choose_left = left_delta <= right_delta
    indices = np.where(choose_left, left, right).astype(np.int64)
    offsets = source[indices] - target
    outside = np.abs(offsets) > int(tolerance_seconds)
    indices[outside] = -1
    offsets = offsets.astype(np.int64)
    offsets[outside] = np.iinfo(np.int64).min
    indices.setflags(write=False)
    offsets.setflags(write=False)
    return indices, offsets


def nearest_spatial_indices(
    *,
    source_latitude_deg: np.ndarray,
    source_longitude_deg: np.ndarray,
    target_latitude_deg: np.ndarray,
    target_longitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor match on the unit sphere.

    scipy's cKDTree provides O(n log n) behavior for real MRMS/model grids. A
    deterministic index epsilon resolves exact duplicate-coordinate ties toward
    the lower flattened source index.
    """

    source_lat = np.asarray(source_latitude_deg, dtype=np.float64).ravel()
    source_lon = np.asarray(source_longitude_deg, dtype=np.float64).ravel()
    target_lat = np.asarray(target_latitude_deg, dtype=np.float64).ravel()
    target_lon = np.asarray(target_longitude_deg, dtype=np.float64).ravel()
    if source_lat.size == 0:
        raise DataError("source spatial grid is empty")
    if source_lat.size != source_lon.size or target_lat.size != target_lon.size:
        raise DataError("latitude/longitude sizes differ")

    source_xyz = _unit_xyz(source_lat, source_lon)
    target_xyz = _unit_xyz(target_lat, target_lon)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        if source_lat.size * target_lat.size > 5_000_000:
            raise MeasurementUnavailable(
                "scipy is required to match large non-identical grids"
            ) from exc
        dot = target_xyz @ source_xyz.T
        indices = np.argmax(dot, axis=1).astype(np.int64)
    else:
        tree = cKDTree(source_xyz)
        _, indices = tree.query(target_xyz, k=1, workers=1)
        indices = np.asarray(indices, dtype=np.int64)

        # Duplicate source coordinates are rare but scientifically possible.
        # Resolve them explicitly to the lowest flat index.
        unique_map: dict[tuple[float, float, float], int] = {}
        duplicate_map: dict[int, int] = {}
        for idx, row in enumerate(source_xyz):
            key = (float(row[0]), float(row[1]), float(row[2]))
            first = unique_map.setdefault(key, idx)
            if first != idx:
                duplicate_map[idx] = first
        if duplicate_map:
            indices = np.asarray(
                [duplicate_map.get(int(idx), int(idx)) for idx in indices],
                dtype=np.int64,
            )

    selected_lat = source_lat[indices]
    selected_lon = source_lon[indices]
    distances = haversine_km(target_lat, target_lon, selected_lat, selected_lon)
    output_shape = np.asarray(target_latitude_deg).shape
    indices = indices.reshape(output_shape)
    distances = distances.reshape(output_shape)
    indices.setflags(write=False)
    distances.setflags(write=False)
    return indices, distances


def haversine_km(
    latitude_a_deg: np.ndarray | float,
    longitude_a_deg: np.ndarray | float,
    latitude_b_deg: np.ndarray | float,
    longitude_b_deg: np.ndarray | float,
) -> np.ndarray:
    lat_a = np.deg2rad(np.asarray(latitude_a_deg, dtype=np.float64))
    lon_a = np.deg2rad(np.asarray(longitude_a_deg, dtype=np.float64))
    lat_b = np.deg2rad(np.asarray(latitude_b_deg, dtype=np.float64))
    lon_b = np.deg2rad(np.asarray(longitude_b_deg, dtype=np.float64))
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _unit_xyz(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))
