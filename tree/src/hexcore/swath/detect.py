"""Finding what is worth resolving, in the model's own field, on its own mesh.

TWO DETECTOR KINDS, AND THEY RETURN THE SAME RECORD.  ``extremum_ball``
finds points -- a pressure minimum that is the deepest thing within a
search radius.  ``area_threshold_exceedance`` finds regions -- every
connected run of cells above a threshold whose total area clears a floor.
A cyclone is the first; a convective region is the second.  Both come back
as a :class:`Feature` with a centroid, an extremum value, an area and a
cell list, and NOTHING after this module can tell which detector produced
one.  That is what stops the pipeline growing a per-phenomenon branch.

DROPS ARE RECORDED, NOT DISCARDED.  A candidate that failed its
confirmation or its area floor leaves a row with the reason and the number
that decided it.  A gallery that publishes only what the machine placed
invites the reader to assume the misses were hidden; publishing them costs
nothing but nerve.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import SwathRefusal
from .history import HistoryReader, area_weighted_centroid, ball_indices
from .registry import ConfirmRow, MetricRegistry, MetricRow


@dataclass(frozen=True)
class Feature:
    """One detection, at one frame, from one metric row."""

    feature_id: str
    metric_id: str
    threat_class: str
    frame_index: int
    time_seconds: float
    latitude_deg: float
    longitude_deg: float
    extremum_value: float
    area_km2: float
    cell_count: int
    seed_cell: int
    confirmations: tuple[tuple[str, float, bool], ...] = ()

    def as_row(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "metric_id": self.metric_id,
            "threat_class": self.threat_class,
            "frame_index": self.frame_index,
            "time_seconds": self.time_seconds,
            "latitude_deg": round(self.latitude_deg, 6),
            "longitude_deg": round(self.longitude_deg, 6),
            "extremum_value": self.extremum_value,
            "area_km2": round(self.area_km2, 3),
            "cell_count": self.cell_count,
            "seed_cell": self.seed_cell,
            "confirmations": [
                {"field": name, "value": value, "passed": passed}
                for name, value, passed in self.confirmations
            ],
        }


@dataclass(frozen=True)
class Drop:
    """A candidate the machine looked at and did not keep, with the reason."""

    metric_id: str
    frame_index: int
    latitude_deg: float
    longitude_deg: float
    reason: str
    measured: float
    required: float

    def as_row(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "frame_index": self.frame_index,
            "latitude_deg": round(self.latitude_deg, 6),
            "longitude_deg": round(self.longitude_deg, 6),
            "reason": self.reason,
            "measured": self.measured,
            "required": self.required,
        }


@dataclass
class DetectionResult:
    features: list[Feature] = dataclass_field(default_factory=list)
    drops: list[Drop] = dataclass_field(default_factory=list)


class MeshSearch:
    """The mesh arrays every detector kind shares, read once."""

    def __init__(self, reader: HistoryReader) -> None:
        self.latitudes_deg = reader.latitudes_deg
        self.longitudes_deg = reader.longitudes_deg
        self.areas_km2 = reader.areas_km2
        self.neighbours = reader.neighbours()

    def ball(self, seed: int, radius_km: float) -> list[int]:
        return ball_indices(
            seed,
            radius_km,
            neighbours=self.neighbours,
            latitudes_deg=self.latitudes_deg,
            longitudes_deg=self.longitudes_deg,
        )

    def centroid(self, cells: Sequence[int]) -> tuple[float, float]:
        return area_weighted_centroid(
            cells,
            latitudes_deg=self.latitudes_deg,
            longitudes_deg=self.longitudes_deg,
            areas_km2=self.areas_km2,
        )

    def weighted_centroid(
        self, cells: Sequence[int], weights: np.ndarray
    ) -> tuple[float, float]:
        index = np.asarray(cells, dtype=np.int64)
        return area_weighted_centroid(
            index,
            latitudes_deg=self.latitudes_deg,
            longitudes_deg=self.longitudes_deg,
            areas_km2=self.areas_km2[index] * np.asarray(weights, dtype=np.float64),
            weights_are_absolute=True,
        )


# ---------------------------------------------------------------------------
# the two detector kinds
# ---------------------------------------------------------------------------
def _extremum_ball(
    metric: MetricRow, values: np.ndarray, search: MeshSearch, frame_index: int
) -> tuple[list[Feature], list[Drop]]:
    detector = metric.detector
    maximise = detector.extremum == "maximum"
    if maximise:
        candidates = np.flatnonzero(values >= detector.threshold)
    else:
        candidates = np.flatnonzero(values <= detector.threshold)
    radius = float(detector.search_radius_km or 0.0)

    kept: list[tuple[float, int, list[int]]] = []
    drops: list[Drop] = []
    for cell in candidates.tolist():
        ball = search.ball(int(cell), radius)
        window = values[np.asarray(ball, dtype=np.int64)]
        best = float(window.max() if maximise else window.min())
        here = float(values[cell])
        if here != best:
            continue
        # A plateau of equal values would emit one feature per cell; the
        # lowest index wins so the answer does not depend on iteration order.
        tied = [index for index in ball if float(values[index]) == here]
        if int(cell) != min(tied):
            continue
        kept.append((here, int(cell), ball))

    kept.sort(key=lambda entry: (-entry[0] if maximise else entry[0], entry[1]))
    separation = float(detector.minimum_separation_km or 0.0)
    accepted: list[tuple[float, int, list[int]]] = []
    from .geometry import great_circle_km

    for value, cell, ball in kept:
        lat = float(search.latitudes_deg[cell])
        lon = float(search.longitudes_deg[cell])
        clash = None
        for _, other, _ in accepted:
            distance = great_circle_km(
                lat, lon,
                float(search.latitudes_deg[other]), float(search.longitudes_deg[other]),
            )
            if distance < separation:
                clash = distance
                break
        if clash is not None:
            drops.append(
                Drop(
                    metric_id=metric.id,
                    frame_index=frame_index,
                    latitude_deg=lat,
                    longitude_deg=lon,
                    reason="within minimum_separation_km of a stronger extremum",
                    measured=round(clash, 3),
                    required=separation,
                )
            )
            continue
        accepted.append((value, cell, ball))

    features = []
    for value, cell, ball in accepted:
        index = np.asarray(ball, dtype=np.int64)
        # The centre is the ANOMALY-WEIGHTED centroid of the ball, not the
        # extremum cell.  A cell centre quantizes the position to the mesh
        # spacing, so on a 75 km parent a storm moving 22 km/h would sit
        # perfectly still for three hours and then jump 75 km: the track
        # speed, the projected path and the flare would all be built on a
        # staircase.  Weighting by how far each cell exceeds the threshold
        # -- the same construction the engine's own tracker uses -- puts the
        # centre between cells and lets a track move continuously.
        anomaly = (
            values[index] - detector.threshold if maximise
            else detector.threshold - values[index]
        )
        anomaly = np.maximum(anomaly, 0.0)
        peak = float(anomaly.max())
        if peak <= 0.0:
            weights = np.ones_like(anomaly)
        else:
            weights = anomaly
        latitude, longitude = search.weighted_centroid(ball, weights)
        # A size that means something for a point feature: the area over
        # which the anomaly is at least half its peak.  Constant ball area
        # would make the 'extent' rank term identical for every candidate.
        core = index[anomaly >= 0.5 * peak] if peak > 0.0 else index
        features.append(
            Feature(
                feature_id="",
                metric_id=metric.id,
                threat_class=metric.threat_class,
                frame_index=frame_index,
                time_seconds=0.0,
                latitude_deg=latitude,
                longitude_deg=longitude,
                extremum_value=value,
                area_km2=float(search.areas_km2[core].sum()),
                cell_count=int(core.size),
                seed_cell=cell,
            )
        )
    return features, drops


def _area_threshold_exceedance(
    metric: MetricRow, values: np.ndarray, search: MeshSearch, frame_index: int
) -> tuple[list[Feature], list[Drop]]:
    detector = metric.detector
    if detector.comparison == "at_least":
        mask = values >= detector.threshold
    else:
        mask = values <= detector.threshold
    neighbours = search.neighbours
    visited = np.zeros(values.shape[0], dtype=bool)
    features: list[Feature] = []
    drops: list[Drop] = []
    floor = float(detector.minimum_area_km2 or 0.0)

    for start in np.flatnonzero(mask).tolist():
        if visited[start]:
            continue
        component = [int(start)]
        visited[start] = True
        frontier = [int(start)]
        while frontier:
            nxt: list[int] = []
            for cell in frontier:
                for neighbour in neighbours[cell]:
                    index = int(neighbour)
                    if index < 0 or visited[index] or not mask[index]:
                        continue
                    visited[index] = True
                    component.append(index)
                    nxt.append(index)
            frontier = nxt
        index = np.asarray(component, dtype=np.int64)
        area = float(search.areas_km2[index].sum())
        window = values[index]
        peak = float(window.max() if detector.comparison == "at_least" else window.min())
        seed = int(index[int(np.argmax(window) if detector.comparison == "at_least" else np.argmin(window))])
        latitude, longitude = search.centroid(component)
        if area < floor:
            drops.append(
                Drop(
                    metric_id=metric.id,
                    frame_index=frame_index,
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    reason="below minimum_area_km2",
                    measured=round(area, 3),
                    required=floor,
                )
            )
            continue
        ceiling = float(detector.maximum_area_km2 or 0.0)
        if ceiling > 0.0 and area > ceiling:
            drops.append(
                Drop(
                    metric_id=metric.id,
                    frame_index=frame_index,
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    reason=(
                        "above maximum_area_km2: a connected region this large is a "
                        "climate belt, not a placeable feature -- its centroid is not "
                        "on anything and a swath could cover a fraction of it"
                    ),
                    measured=round(area, 3),
                    required=ceiling,
                )
            )
            continue
        features.append(
            Feature(
                feature_id="",
                metric_id=metric.id,
                threat_class=metric.threat_class,
                frame_index=frame_index,
                time_seconds=0.0,
                latitude_deg=latitude,
                longitude_deg=longitude,
                extremum_value=peak,
                area_km2=area,
                cell_count=len(component),
                seed_cell=seed,
            )
        )
    return features, drops


_DETECTORS = {
    "extremum_ball": _extremum_ball,
    "area_threshold_exceedance": _area_threshold_exceedance,
}


# ---------------------------------------------------------------------------
# where a definition applies
# ---------------------------------------------------------------------------
def _inside_region(
    metric: MetricRow, features: Sequence[Feature]
) -> tuple[list[Feature], list[Drop]]:
    """Keep the features whose centre falls inside the row's own region.

    A REGION IS PART OF A THREAT DEFINITION, not a filter bolted on.
    "Severe convection" as the term is operationally defined -- hail of a
    stated size, wind of a stated speed, a tornado -- is a North American
    construct with North American thresholds; a row carrying them fires
    continuously over the warm tropical oceans, where the same
    reflectivity and the same shear mean something else entirely and
    nobody is issuing a warning.  The alternative to saying so in the row
    is a global row whose thresholds are a compromise between two
    climates and correct in neither.

    The test is on the feature's CENTRE.  A region straddling the edge
    keeps or drops whole, which is the only answer that leaves the
    downstream track association with a feature it can follow.
    """

    if metric.region.kind == "global":
        return list(features), []
    kept: list[Feature] = []
    dropped: list[Drop] = []
    for feature in features:
        if metric.region.contains(feature.latitude_deg, feature.longitude_deg):
            kept.append(feature)
            continue
        dropped.append(
            Drop(
                metric_id=metric.id,
                frame_index=feature.frame_index,
                latitude_deg=feature.latitude_deg,
                longitude_deg=feature.longitude_deg,
                reason=(
                    f"outside this row's region "
                    f"{json.dumps(dict(metric.region.as_row()), sort_keys=True)}"
                ),
                measured=round(feature.extremum_value, 6),
                required=metric.detector.threshold,
            )
        )
    return kept, dropped


# ---------------------------------------------------------------------------
# confirmation
# ---------------------------------------------------------------------------
def _aggregate(kind: str, window: np.ndarray) -> float:
    if kind == "ball_maximum":
        return float(window.max())
    if kind == "ball_minimum":
        return float(window.min())
    return float(window.mean())


def _confirm(
    metric: MetricRow,
    feature: Feature,
    reader: HistoryReader,
    registry: MetricRegistry,
    search: MeshSearch,
    row: ConfirmRow,
) -> tuple[float, bool]:
    field_row = registry.field_rows[row.field]
    values = reader.derive(field_row, feature.frame_index, registry=registry)
    ball = search.ball(feature.seed_cell, row.radius_km)
    measured = _aggregate(row.aggregation_kind, values[np.asarray(ball, dtype=np.int64)])
    passed = measured >= row.value if row.comparison == "at_least" else measured <= row.value
    return measured, passed


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def detect(
    reader: HistoryReader, registry: MetricRegistry, *, search: MeshSearch | None = None
) -> DetectionResult:
    """Every armed metric row, over every frame of one history file."""

    mesh = search if search is not None else MeshSearch(reader)
    frames = reader.frames()
    if not frames:
        raise SwathRefusal(
            f"history file {reader.path.name} carries no time records, so there is "
            "no forecast to read a threat out of"
        )
    result = DetectionResult()
    for metric in registry.armed:
        field_row = registry.field_rows[metric.field]
        for frame in frames:
            values = reader.derive(field_row, frame.index, registry=registry)
            kind = metric.detector.kind
            handler = _DETECTORS.get(kind)
            if handler is None:  # pragma: no cover - registry closes this
                raise SwathRefusal(
                    f"detector kind {kind!r} loaded but has no implementation"
                )
            found, dropped = handler(metric, values, mesh, frame.index)
            result.drops.extend(dropped)
            found, outside = _inside_region(metric, found)
            result.drops.extend(outside)
            for ordinal, feature in enumerate(found):
                confirmations = []
                rejected = False
                for row in metric.confirm_with:
                    measured, passed = _confirm(
                        metric, feature, reader, registry, mesh, row
                    )
                    confirmations.append((row.field, round(measured, 6), passed))
                    if not passed:
                        rejected = True
                        result.drops.append(
                            Drop(
                                metric_id=metric.id,
                                frame_index=frame.index,
                                latitude_deg=feature.latitude_deg,
                                longitude_deg=feature.longitude_deg,
                                reason=f"confirm_with {row.field} {row.comparison} failed",
                                measured=round(measured, 6),
                                required=row.value,
                            )
                        )
                if rejected:
                    continue
                result.features.append(
                    Feature(
                        feature_id=f"{metric.id}:f{frame.index:03d}:{ordinal:03d}",
                        metric_id=feature.metric_id,
                        threat_class=feature.threat_class,
                        frame_index=frame.index,
                        time_seconds=frame.time_seconds,
                        latitude_deg=feature.latitude_deg,
                        longitude_deg=feature.longitude_deg,
                        extremum_value=feature.extremum_value,
                        area_km2=feature.area_km2,
                        cell_count=feature.cell_count,
                        seed_cell=feature.seed_cell,
                        confirmations=tuple(confirmations),
                    )
                )
    result.features.sort(key=lambda item: (item.frame_index, item.metric_id, item.feature_id))
    result.drops.sort(key=lambda item: (item.frame_index, item.metric_id, item.reason))
    return result


def detection_receipt(
    reader: HistoryReader, registry: MetricRegistry, result: DetectionResult
) -> Mapping[str, Any]:
    """The document a reader checks the placement against.

    Carries the history file's own sha256, so "you placed these by hand"
    is answered by showing that the feature list is a pure function of a
    named forecast and a named table.
    """

    return {
        "schema": "gpuwm-hex.threat-decision.v1",
        "history": dict(reader.provenance()),
        "metrics_document": {
            "schema": registry.schema,
            "sha256": registry.sha256,
            "path": str(registry.source_path) if registry.source_path else None,
            "armed": [row.id for row in registry.armed],
            "publication_manifest": list(registry.publication_manifest()),
        },
        "features": [feature.as_row() for feature in result.features],
        "drops": [drop.as_row() for drop in result.drops],
        "counts": {
            "features": len(result.features),
            "drops": len(result.drops),
        },
    }


__all__ = [
    "DetectionResult",
    "Drop",
    "Feature",
    "MeshSearch",
    "detect",
    "detection_receipt",
]
