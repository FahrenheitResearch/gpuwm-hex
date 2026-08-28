"""Deterministic scalar, categorical, FSS, and object verification metrics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .align import haversine_km
from .errors import MeasurementUnavailable, SchemaError


@dataclass(frozen=True, slots=True)
class MetricResult:
    status: str
    value: float | None
    n_valid: int
    components: dict[str, Any]
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "n_valid": self.n_valid,
            "components": self.components,
            "reason": self.reason,
        }


def calculate_metric(
    forecast: np.ndarray,
    observation: np.ndarray,
    *,
    family: str,
    statistic: str,
    threshold: float | None = None,
    neighborhood_radius_cells: int | None = None,
    minimum_object_cells: int = 1,
    maximum_object_match_km: float = 100.0,
    latitude_deg: np.ndarray | None = None,
    longitude_deg: np.ndarray | None = None,
    minimum_valid_samples: int = 1,
) -> MetricResult:
    fcst = np.asarray(forecast, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64)
    if fcst.shape != obs.shape:
        raise SchemaError(f"metric arrays differ: forecast {fcst.shape}, observation {obs.shape}")
    valid = np.isfinite(fcst) & np.isfinite(obs)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < minimum_valid_samples:
        return MetricResult(
            status="NOT_MEASURED",
            value=None,
            n_valid=n_valid,
            components={},
            reason=(
                f"{n_valid} finite paired samples is below minimum "
                f"{minimum_valid_samples}"
            ),
        )
    if family == "continuous":
        return _continuous(fcst[valid], obs[valid], statistic=statistic)
    if threshold is None:
        raise SchemaError(f"{family} metric requires threshold")
    if family == "categorical":
        return _categorical(
            fcst[valid], obs[valid], threshold=float(threshold), statistic=statistic
        )
    if family == "fss":
        if fcst.ndim != 3:
            return MetricResult(
                status="NOT_MEASURED",
                value=None,
                n_valid=n_valid,
                components={},
                reason="FSS requires arrays shaped (time, y, x)",
            )
        radius = int(neighborhood_radius_cells or 0)
        return _fss(
            fcst,
            obs,
            threshold=float(threshold),
            radius=radius,
            statistic=statistic,
            minimum_valid_samples=minimum_valid_samples,
        )
    if family == "objects":
        if fcst.ndim != 3 or latitude_deg is None or longitude_deg is None:
            return MetricResult(
                status="NOT_MEASURED",
                value=None,
                n_valid=n_valid,
                components={},
                reason="object metrics require (time, y, x) arrays and 2-D coordinates",
            )
        return _objects(
            fcst,
            obs,
            threshold=float(threshold),
            statistic=statistic,
            latitude_deg=np.asarray(latitude_deg, dtype=np.float64),
            longitude_deg=np.asarray(longitude_deg, dtype=np.float64),
            minimum_object_cells=int(minimum_object_cells),
            maximum_object_match_km=float(maximum_object_match_km),
        )
    raise SchemaError(f"unknown metric family {family!r}")


def _continuous(
    forecast: np.ndarray,
    observation: np.ndarray,
    *,
    statistic: str,
) -> MetricResult:
    error = forecast - observation
    count = int(error.size)
    components: dict[str, Any] = {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }
    if count >= 2 and np.std(forecast) > 0.0 and np.std(observation) > 0.0:
        components["correlation"] = float(np.corrcoef(forecast, observation)[0, 1])
    else:
        components["correlation"] = None
    if statistic not in components:
        raise SchemaError(
            f"continuous statistic must be one of {sorted(components)}, got {statistic!r}"
        )
    value = components[statistic]
    if value is None:
        return MetricResult(
            status="NOT_MEASURED",
            value=None,
            n_valid=count,
            components=components,
            reason=f"{statistic} is undefined for constant or singleton samples",
        )
    return MetricResult(
        status="MEASURED",
        value=float(value),
        n_valid=count,
        components=components,
    )


def _categorical(
    forecast: np.ndarray,
    observation: np.ndarray,
    *,
    threshold: float,
    statistic: str,
) -> MetricResult:
    forecast_event = forecast >= threshold
    observed_event = observation >= threshold
    hits = int(np.count_nonzero(forecast_event & observed_event))
    misses = int(np.count_nonzero(~forecast_event & observed_event))
    false_alarms = int(np.count_nonzero(forecast_event & ~observed_event))
    correct_negatives = int(np.count_nonzero(~forecast_event & ~observed_event))
    components: dict[str, Any] = {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "csi": _ratio(hits, hits + misses + false_alarms),
        "pod": _ratio(hits, hits + misses),
        "far": _ratio(false_alarms, hits + false_alarms),
        "frequency_bias": _ratio(hits + false_alarms, hits + misses),
        "accuracy": _ratio(hits + correct_negatives, forecast.size),
    }
    if statistic not in components or statistic in {
        "hits",
        "misses",
        "false_alarms",
        "correct_negatives",
    }:
        raise SchemaError(
            "categorical statistic must be csi, pod, far, frequency_bias, or accuracy"
        )
    value = components[statistic]
    if value is None:
        return MetricResult(
            status="NOT_MEASURED",
            value=None,
            n_valid=int(forecast.size),
            components=components,
            reason=f"{statistic} denominator is zero",
        )
    return MetricResult(
        status="MEASURED",
        value=float(value),
        n_valid=int(forecast.size),
        components=components,
    )


def _fss(
    forecast: np.ndarray,
    observation: np.ndarray,
    *,
    threshold: float,
    radius: int,
    statistic: str,
    minimum_valid_samples: int,
) -> MetricResult:
    if statistic != "fss":
        raise SchemaError("FSS family only supports statistic='fss'")
    valid = np.isfinite(forecast) & np.isfinite(observation)
    forecast_binary = ((forecast >= threshold) & valid).astype(np.float64)
    observation_binary = ((observation >= threshold) & valid).astype(np.float64)
    fractions_forecast: list[np.ndarray] = []
    fractions_observed: list[np.ndarray] = []
    valid_centers: list[np.ndarray] = []
    for time_index in range(forecast.shape[0]):
        window_count = _box_sum(valid[time_index].astype(np.float64), radius)
        ff = _box_sum(forecast_binary[time_index], radius)
        oo = _box_sum(observation_binary[time_index], radius)
        with np.errstate(divide="ignore", invalid="ignore"):
            ff = np.where(window_count > 0.0, ff / window_count, np.nan)
            oo = np.where(window_count > 0.0, oo / window_count, np.nan)
        fractions_forecast.append(ff)
        fractions_observed.append(oo)
        valid_centers.append(window_count > 0.0)
    ff_all = np.stack(fractions_forecast)
    oo_all = np.stack(fractions_observed)
    center_valid = np.stack(valid_centers) & np.isfinite(ff_all) & np.isfinite(oo_all)
    n = int(np.count_nonzero(center_valid))
    if n < minimum_valid_samples:
        return MetricResult(
            status="NOT_MEASURED",
            value=None,
            n_valid=n,
            components={"threshold": threshold, "radius_cells": radius},
            reason=f"only {n} valid neighborhood centers",
        )
    numerator = float(np.sum((ff_all[center_valid] - oo_all[center_valid]) ** 2))
    denominator = float(
        np.sum(ff_all[center_valid] ** 2 + oo_all[center_valid] ** 2)
    )
    fss = 1.0 if denominator == 0.0 else 1.0 - numerator / denominator
    return MetricResult(
        status="MEASURED",
        value=float(fss),
        n_valid=n,
        components={
            "fss": float(fss),
            "threshold": threshold,
            "radius_cells": radius,
            "numerator": numerator,
            "denominator": denominator,
        },
    )


def _objects(
    forecast: np.ndarray,
    observation: np.ndarray,
    *,
    threshold: float,
    statistic: str,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    minimum_object_cells: int,
    maximum_object_match_km: float,
) -> MetricResult:
    allowed = {
        "object_pod",
        "object_far",
        "mean_centroid_error_km",
        "median_centroid_error_km",
        "area_ratio",
    }
    if statistic not in allowed:
        raise SchemaError(f"object statistic must be one of {sorted(allowed)}")
    if latitude_deg.shape != forecast.shape[1:] or longitude_deg.shape != forecast.shape[1:]:
        raise SchemaError("object coordinate shapes must match each 2-D field")

    total_forecast = 0
    total_observed = 0
    total_matches = 0
    distances: list[float] = []
    forecast_area = 0
    observed_area = 0
    time_components: list[dict[str, Any]] = []
    for time_index in range(forecast.shape[0]):
        valid = np.isfinite(forecast[time_index]) & np.isfinite(observation[time_index])
        fcst_objects = _extract_objects(
            (forecast[time_index] >= threshold) & valid,
            forecast[time_index],
            latitude_deg,
            longitude_deg,
            minimum_cells=minimum_object_cells,
        )
        obs_objects = _extract_objects(
            (observation[time_index] >= threshold) & valid,
            observation[time_index],
            latitude_deg,
            longitude_deg,
            minimum_cells=minimum_object_cells,
        )
        matches = _match_objects(
            fcst_objects,
            obs_objects,
            maximum_distance_km=maximum_object_match_km,
        )
        total_forecast += len(fcst_objects)
        total_observed += len(obs_objects)
        total_matches += len(matches)
        forecast_area += sum(int(item["cell_count"]) for item in fcst_objects)
        observed_area += sum(int(item["cell_count"]) for item in obs_objects)
        distances.extend(float(item["distance_km"]) for item in matches)
        time_components.append(
            {
                "time_index": time_index,
                "forecast_objects": len(fcst_objects),
                "observed_objects": len(obs_objects),
                "matched_objects": len(matches),
            }
        )

    components: dict[str, Any] = {
        "forecast_objects": total_forecast,
        "observed_objects": total_observed,
        "matched_objects": total_matches,
        "object_pod": _ratio(total_matches, total_observed),
        "object_far": _ratio(total_forecast - total_matches, total_forecast),
        "mean_centroid_error_km": float(np.mean(distances)) if distances else None,
        "median_centroid_error_km": float(np.median(distances)) if distances else None,
        "area_ratio": _ratio(forecast_area, observed_area),
        "threshold": threshold,
        "minimum_object_cells": minimum_object_cells,
        "maximum_object_match_km": maximum_object_match_km,
        "by_time": time_components,
    }
    value = components[statistic]
    if value is None:
        return MetricResult(
            status="NOT_MEASURED",
            value=None,
            n_valid=int(np.count_nonzero(np.isfinite(forecast) & np.isfinite(observation))),
            components=components,
            reason=f"{statistic} is undefined because required objects are absent",
        )
    return MetricResult(
        status="MEASURED",
        value=float(value),
        n_valid=int(np.count_nonzero(np.isfinite(forecast) & np.isfinite(observation))),
        components=components,
    )


def _extract_objects(
    mask: np.ndarray,
    values: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    *,
    minimum_cells: int,
) -> list[dict[str, Any]]:
    if mask.ndim != 2:
        raise SchemaError("connected objects require a 2-D mask")
    visited = np.zeros(mask.shape, dtype=bool)
    objects: list[dict[str, Any]] = []
    ny, nx = mask.shape
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                cells.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        yy = cy + dy
                        xx = cx + dx
                        if (
                            0 <= yy < ny
                            and 0 <= xx < nx
                            and mask[yy, xx]
                            and not visited[yy, xx]
                        ):
                            visited[yy, xx] = True
                            queue.append((yy, xx))
            if len(cells) < minimum_cells:
                continue
            ys = np.asarray([cell[0] for cell in cells], dtype=np.int64)
            xs = np.asarray([cell[1] for cell in cells], dtype=np.int64)
            weights = np.maximum(values[ys, xs], 0.0)
            if not np.any(weights > 0.0):
                weights = np.ones(weights.shape, dtype=np.float64)
            lat = float(np.average(latitude_deg[ys, xs], weights=weights))
            lon = _weighted_longitude(longitude_deg[ys, xs], weights)
            objects.append(
                {
                    "object_index": len(objects),
                    "cell_count": len(cells),
                    "centroid_latitude_deg": lat,
                    "centroid_longitude_deg": lon,
                    "maximum": float(np.max(values[ys, xs])),
                    "sum": float(np.sum(values[ys, xs])),
                    "first_flat_index": int(np.min(ys * nx + xs)),
                }
            )
    objects.sort(key=lambda item: int(item["first_flat_index"]))
    for index, item in enumerate(objects):
        item["object_index"] = index
    return objects


def _match_objects(
    forecast: list[dict[str, Any]],
    observation: list[dict[str, Any]],
    *,
    maximum_distance_km: float,
) -> list[dict[str, Any]]:
    if not forecast or not observation:
        return []
    cost = np.empty((len(forecast), len(observation)), dtype=np.float64)
    distance = np.empty_like(cost)
    for i, fcst in enumerate(forecast):
        for j, obs in enumerate(observation):
            d = float(
                haversine_km(
                    fcst["centroid_latitude_deg"],
                    fcst["centroid_longitude_deg"],
                    obs["centroid_latitude_deg"],
                    obs["centroid_longitude_deg"],
                )
            )
            distance[i, j] = d
            area_penalty = abs(
                math.log((float(fcst["cell_count"]) + 0.5) / (float(obs["cell_count"]) + 0.5))
            )
            # Tiny stable epsilon makes tied assignments platform-independent.
            cost[i, j] = d + min(50.0, 10.0 * area_penalty) + (i * len(observation) + j) * 1e-12
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        candidates = sorted(
            (
                (float(cost[i, j]), i, j)
                for i in range(len(forecast))
                for j in range(len(observation))
            )
        )
        used_i: set[int] = set()
        used_j: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, i, j in candidates:
            if i not in used_i and j not in used_j:
                used_i.add(i)
                used_j.add(j)
                pairs.append((i, j))
    else:
        rows, columns = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), columns.tolist()))
    result = []
    for i, j in sorted(pairs):
        d = float(distance[i, j])
        if d <= maximum_distance_km:
            result.append(
                {
                    "forecast_object_index": i,
                    "observed_object_index": j,
                    "distance_km": d,
                }
            )
    return result


def _weighted_longitude(values_deg: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(values_deg)
    x = float(np.sum(weights * np.cos(radians)))
    y = float(np.sum(weights * np.sin(radians)))
    return float(np.rad2deg(math.atan2(y, x)))


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    if values.ndim != 2:
        raise SchemaError("_box_sum requires a 2-D array")
    if radius < 0:
        raise SchemaError("neighborhood radius cannot be negative")
    padded = np.pad(values, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    ny, nx = values.shape
    y = np.arange(ny)
    x = np.arange(nx)
    y0 = np.maximum(0, y - radius)
    y1 = np.minimum(ny, y + radius + 1)
    x0 = np.maximum(0, x - radius)
    x1 = np.minimum(nx, x + radius + 1)
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)
