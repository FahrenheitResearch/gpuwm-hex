"""Joining features into tracks, and reading the track forward.

PROJECTION HERE IS NOT EXTRAPOLATION, MOSTLY.  The detector runs over a
FORECAST, so where a feature will be during the fine level's lead window
is something the coarse model already said: the track's own later frames
ARE the anticipated path.  Extrapolation happens only past the last frame
the parent published, and when it does the plan says how many hours of the
path are extrapolated rather than presenting them as forecast.

Association is greedy over globally sorted (distance, track, feature)
triples under a per-metric speed gate.  Greedy rather than a global
assignment because the gate already rejects the pairings a global solver
would be needed to arbitrate, and because a deterministic, explainable
join is worth more here than an optimal one -- every drop and every join
appears in the decision receipt, and "the assignment solver preferred it"
is not an explanation an operator can act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .detect import Feature
from .errors import SwathRefusal
from .geometry import LatLon, destination, great_circle_km, initial_bearing_deg
from .registry import MetricRow


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    time_seconds: float
    latitude_deg: float
    longitude_deg: float
    extremum_value: float
    area_km2: float
    feature_id: str


@dataclass
class Track:
    track_id: str
    metric_id: str
    threat_class: str
    points: list[TrackPoint]

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def frames(self) -> int:
        return len(self.points)

    def displacement_km(self) -> float:
        return sum(
            great_circle_km(a.latitude_deg, a.longitude_deg, b.latitude_deg, b.longitude_deg)
            for a, b in zip(self.points, self.points[1:])
        )

    def velocity(self) -> tuple[float, float] | None:
        """(bearing degrees, speed km/h) from the last two points, or None.

        The LAST two rather than a fit over the whole track: a track that
        recurves has a mean heading nobody's storm ever took, and the
        swath must be built on where it is going, not on the average of
        where it has been.
        """

        if len(self.points) < 2:
            return None
        a, b = self.points[-2], self.points[-1]
        dt_hours = (b.time_seconds - a.time_seconds) / 3600.0
        if dt_hours <= 0.0:
            return None
        distance = great_circle_km(
            a.latitude_deg, a.longitude_deg, b.latitude_deg, b.longitude_deg
        )
        bearing = initial_bearing_deg(
            a.latitude_deg, a.longitude_deg, b.latitude_deg, b.longitude_deg
        )
        return (bearing, distance / dt_hours)

    def peak_anomaly(self, threshold: float, maximise: bool) -> float:
        values = [point.extremum_value for point in self.points]
        best = max(values) if maximise else min(values)
        return abs(best - threshold)

    def peak_area_km2(self) -> float:
        return max(point.area_km2 for point in self.points)

    def as_row(self) -> Mapping[str, Any]:
        return {
            "track_id": self.track_id,
            "metric_id": self.metric_id,
            "threat_class": self.threat_class,
            "frames": self.frames,
            "first_time_seconds": self.first.time_seconds,
            "last_time_seconds": self.last.time_seconds,
            "displacement_km": round(self.displacement_km(), 3),
            "points": [
                {
                    "frame_index": point.frame_index,
                    "time_seconds": point.time_seconds,
                    "latitude_deg": round(point.latitude_deg, 6),
                    "longitude_deg": round(point.longitude_deg, 6),
                    "extremum_value": point.extremum_value,
                    "feature_id": point.feature_id,
                }
                for point in self.points
            ],
        }


@dataclass(frozen=True)
class ProjectedPath:
    """The swath's axis over its own window, and how much of it is forecast."""

    points: tuple[LatLon, ...]
    hours: tuple[float, ...]
    extrapolated_hours: float
    start_seconds: float
    end_seconds: float


def associate(features: Sequence[Feature], metrics: Mapping[str, MetricRow]) -> list[Track]:
    """Features to tracks, one metric at a time."""

    by_metric: dict[str, list[Feature]] = {}
    for feature in features:
        by_metric.setdefault(feature.metric_id, []).append(feature)

    tracks: list[Track] = []
    for metric_id, group in sorted(by_metric.items()):
        metric = metrics.get(metric_id)
        if metric is None:  # pragma: no cover - detect only emits armed rows
            continue
        open_tracks: list[Track] = []
        frames = sorted({feature.frame_index for feature in group})
        for frame_index in frames:
            in_frame = sorted(
                (item for item in group if item.frame_index == frame_index),
                key=lambda item: item.feature_id,
            )
            pairs: list[tuple[float, int, int]] = []
            for track_ordinal, track in enumerate(open_tracks):
                last = track.last
                dt_hours = (in_frame[0].time_seconds - last.time_seconds) / 3600.0
                if dt_hours <= 0.0:
                    continue
                reach = metric.track.maximum_speed_km_per_hour * dt_hours
                for feature_ordinal, feature in enumerate(in_frame):
                    distance = great_circle_km(
                        last.latitude_deg, last.longitude_deg,
                        feature.latitude_deg, feature.longitude_deg,
                    )
                    if distance <= reach:
                        pairs.append((distance, track_ordinal, feature_ordinal))
            pairs.sort()
            taken_tracks: set[int] = set()
            taken_features: set[int] = set()
            for _, track_ordinal, feature_ordinal in pairs:
                if track_ordinal in taken_tracks or feature_ordinal in taken_features:
                    continue
                taken_tracks.add(track_ordinal)
                taken_features.add(feature_ordinal)
                open_tracks[track_ordinal].points.append(_point(in_frame[feature_ordinal]))
            for feature_ordinal, feature in enumerate(in_frame):
                if feature_ordinal in taken_features:
                    continue
                track = Track(
                    track_id=f"{metric_id}:t{len(open_tracks):03d}",
                    metric_id=metric_id,
                    threat_class=feature.threat_class,
                    points=[_point(feature)],
                )
                open_tracks.append(track)
        tracks.extend(
            track for track in open_tracks if track.frames >= metric.track.minimum_frames
        )
    tracks.sort(key=lambda track: track.track_id)
    return tracks


def _point(feature: Feature) -> TrackPoint:
    return TrackPoint(
        frame_index=feature.frame_index,
        time_seconds=feature.time_seconds,
        latitude_deg=feature.latitude_deg,
        longitude_deg=feature.longitude_deg,
        extremum_value=feature.extremum_value,
        area_km2=feature.area_km2,
        feature_id=feature.feature_id,
    )


def project(track: Track, *, start_seconds: float, lead_hours: float) -> ProjectedPath:
    """The track's own path over ``[start, start + lead]``, extrapolated only
    past the last frame the parent published."""

    end_seconds = start_seconds + lead_hours * 3600.0
    inside = [
        point for point in track.points
        if start_seconds - 1e-6 <= point.time_seconds <= end_seconds + 1e-6
    ]
    points: list[LatLon] = [(p.latitude_deg, p.longitude_deg) for p in inside]
    hours: list[float] = [
        (p.time_seconds - start_seconds) / 3600.0 for p in inside
    ]
    if not inside:
        # The window opens after the track's last published frame: the whole
        # path is extrapolated, from the last point at the last velocity.
        anchor = track.last
        points = [(anchor.latitude_deg, anchor.longitude_deg)]
        hours = [(anchor.time_seconds - start_seconds) / 3600.0]

    extrapolated = 0.0
    velocity = track.velocity()
    tail_seconds = (
        inside[-1].time_seconds if inside else track.last.time_seconds
    )
    if tail_seconds < end_seconds - 1e-6:
        missing_hours = (end_seconds - tail_seconds) / 3600.0
        if velocity is None:
            raise SwathRefusal(
                f"track {track.track_id!r} has one frame and no measurable velocity, "
                f"but its window runs {missing_hours:.2f} h past that frame. A path "
                "cannot be projected from a single position without inventing a "
                "heading, and a swath placed on an invented heading is an operator's "
                "guess with a receipt attached. Raise the metric row's "
                "track.minimum_frames, or shorten swath.lead_hours"
            )
        bearing, speed = velocity
        extrapolated = missing_hours
        points.append(
            destination(points[-1][0], points[-1][1], bearing, speed * missing_hours)
        )
        hours.append(hours[-1] + missing_hours)

    return ProjectedPath(
        points=tuple(points),
        hours=tuple(hours),
        extrapolated_hours=extrapolated,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def half_width_profile(
    path: ProjectedPath, *, base_km: float, flare_km_per_hour: float, maximum_km: float
) -> tuple[float, ...]:
    """The half-width at each path point.

    Widening with LEAD, not with distance: the flare exists to contain the
    coarse model's own track error, and that error grows with forecast
    hour whether the storm is moving fast or slowly.  A slow storm with a
    wide error cone gets a wide swath, which is the correct answer and the
    one a distance-based flare would get wrong.
    """

    return tuple(
        min(maximum_km, base_km + flare_km_per_hour * max(0.0, hour))
        for hour in path.hours
    )


__all__ = [
    "ProjectedPath",
    "Track",
    "TrackPoint",
    "associate",
    "half_width_profile",
    "project",
]
