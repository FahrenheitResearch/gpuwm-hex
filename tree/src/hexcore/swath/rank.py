"""Ranking, because compute is finite and four storms want three grids.

EVERY TERM IS A ROW.  A rank term is a kind from a closed vocabulary, a
weight and a scale; the score is the weighted sum of the scaled terms and
nothing else.  There is no term that reads a clock, a filesystem, a place
name or a phenomenon, which is what makes the same document usable by both
demo arms -- and what makes a threshold quietly fitted to one case visible,
because it would have to be the threshold the other case ran with.

The ORDER is total.  ``registry.PlacementPolicy`` refuses a tiebreak that
does not end in a per-candidate-unique key, so two candidates that compare
equal on score and class still have one deterministic order.  Without that,
the same history file could produce two different plans on two machines and
the "placed by the machine" claim would be untestable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .registry import MetricRow, PlacementPolicy, RankTerm
from .track import Track


@dataclass(frozen=True)
class RankBreakdown:
    """The score, and every term that made it, so the order is explainable."""

    score: float
    terms: tuple[tuple[str, str, float, float, float, float], ...]

    def as_row(self) -> Mapping[str, Any]:
        return {
            "score": round(self.score, 6),
            "terms": [
                {
                    "id": term_id,
                    "kind": kind,
                    "raw": round(raw, 6),
                    "reference": reference,
                    "scaled": round(scaled, 6),
                    "contribution": round(contribution, 6),
                }
                for term_id, kind, raw, reference, scaled, contribution in self.terms
            ],
        }


def _raw_term(term: RankTerm, track: Track, metric: MetricRow) -> float:
    if term.kind == "metric_extremum":
        return track.peak_anomaly(
            metric.detector.threshold,
            maximise=metric.detector.extremum == "maximum"
            or metric.detector.comparison == "at_least",
        )
    if term.kind == "track_frames":
        return float(track.frames)
    if term.kind == "track_displacement_km":
        return track.displacement_km()
    if term.kind == "feature_area_km2":
        return track.peak_area_km2()
    raise AssertionError(  # pragma: no cover - the registry closes this
        f"rank term kind {term.kind!r} loaded but has no implementation"
    )


#: A Voronoi cell of centre-to-centre spacing ``s`` covers about
#: ``sqrt(3)/2 * s^2``.  This is the same relation the placement layer's
#: own sizing integral uses; it is restated here rather than imported
#: because rank must not depend on the sizing module, which shells out.
_HEX_CELL_AREA_FACTOR = 0.8660254037844386


def resolvable_area_km2(metric: MetricRow, policy: PlacementPolicy) -> float:
    """The largest area this row's swath could ever actually resolve.

    Derived, not declared: the budget's cell ceiling times the area of one
    cell at the spacing this row asks for.  Nothing new has to be typed
    into a document for it to exist.
    """

    return (
        policy.budget.maximum_cells_per_swath
        * _HEX_CELL_AREA_FACTOR
        * metric.swath.spacing_km ** 2
    )


@dataclass(frozen=True)
class CycleExtent:
    """How much forecast there was to be persistent or to travel across.

    WHY THIS IS AN ARGUMENT AND NOT A CONSTANT.  ``track_frames`` and
    ``track_displacement_km`` are in shared units, but their CEILINGS are
    set by the parent forecast, not by the phenomenon: a track cannot last
    more frames than the forecast published, and it cannot travel further
    than its own speed gate times the forecast's length.  Against a fixed
    policy scale of 8 frames, a 25-frame forecast let one term reach 3.1
    on its own -- measured, 1.56 of a winning score of 4.97 with a weight
    of only 0.5 -- while an 8-frame forecast caps the same term at 1.0.
    The same document then ranks differently for a reason that is about
    the output interval and nothing to do with the weather.

    Both are therefore divided by what the cycle made possible, so both
    are fractions of their own ceiling and the policy weights are the only
    free numbers left in the ordering.
    """

    frames: int
    hours: float

    @classmethod
    def from_frame_times(cls, times: Sequence[float]) -> "CycleExtent":
        seconds = [float(value) for value in times]
        span = (max(seconds) - min(seconds)) / 3600.0 if len(seconds) > 1 else 0.0
        return cls(frames=max(1, len(seconds)), hours=max(span, 1e-6))


def _reference(
    term: RankTerm, metric: MetricRow, policy: PlacementPolicy, cycle: CycleExtent
) -> tuple[float, bool]:
    """This row's own divisor for one term, and whether the term saturates.

    TWO OF THE FOUR TERMS ARE NOT COMMENSURABLE ACROSS ROWS, and both were
    measured on real global forecasts before this function existed.

    ``metric_extremum`` is an anomaly in the DETECTED FIELD's units --
    pascals here, decibels there, a dimensionless margin somewhere else --
    and no policy-wide number can be right for all of them at once.  With
    one shared scale of 3,000 the top twenty-seven candidates of 205 were
    all one class.  Its divisor is the row's declared
    ``rank.intensity_reference``.

    ``feature_area_km2`` IS in shared units and that is exactly what made
    it look safe.  The distributions are not shared: measured on a 24 h
    global forecast, a cyclone core's half-max area ran to 517,000 km2 and
    a convective region to 211,000 km2, while a moisture corridor reached
    11,433,855 km2 -- one connected region covering two per cent of the
    planet.  Against a policy scale of 200,000 km2 that one term scored
    14.29 on its own, more than four times the whole of any other
    candidate's score, and the two slots it took were spent on a swath
    that could cover three per cent of the feature it was placed for.
    Rewarding a feature for being LARGER THAN ANY GRID CAN RESOLVE is
    backwards, so its divisor is the area this row's own swath could
    resolve and the term SATURATES there: extent means "how much of a
    whole fine grid would this fill", capped at one.

    ``track_frames`` and ``track_displacement_km`` are in shared units but
    their CEILINGS belong to the cycle: see :class:`CycleExtent`.  Each is
    divided by what this forecast made possible -- its frame count, and
    this row's own speed gate over its span -- so both are fractions.

    The result is that INTENSITY IS THE ONLY UNBOUNDED TERM, deliberately.
    A storm that is twice as deep as the reference should be able to run
    away with the ordering; a storm that lasted the whole forecast, or
    travelled as far as its gate allowed, has simply reached the ceiling
    and there is nothing further to reward.
    """

    if term.kind == "metric_extremum":
        return metric.rank.intensity_reference, False
    if term.kind == "feature_area_km2":
        return resolvable_area_km2(metric, policy), True
    if term.kind == "track_frames":
        return float(cycle.frames), True
    if term.kind == "track_displacement_km":
        return max(
            metric.track.maximum_speed_km_per_hour * cycle.hours, 1e-6
        ), True
    return 1.0, False


def score(
    track: Track,
    metric: MetricRow,
    policy: PlacementPolicy,
    *,
    cycle: CycleExtent,
) -> RankBreakdown:
    total = 0.0
    rows: list[tuple[str, str, float, float, float, float]] = []
    for term in policy.rank_terms:
        raw = _raw_term(term, track, metric)
        reference, saturates = _reference(term, metric, policy, cycle)
        # The raw value stays in the field's own units in the receipt, so
        # "944.6 hPa" is still readable in the explanation; the reference
        # that made it dimensionless is published beside it.
        ratio = raw / reference
        if saturates:
            ratio = min(1.0, ratio)
        scaled = ratio / term.scale
        contribution = term.weight * scaled
        total += contribution
        rows.append((term.id, term.kind, raw, reference, scaled, contribution))
    return RankBreakdown(score=total, terms=tuple(rows))


def order_key(
    *,
    breakdown: RankBreakdown,
    threat_class: str,
    metric_id: str,
    feature_id: str,
    policy: PlacementPolicy,
    effective_score: float | None = None,
) -> tuple[Any, ...]:
    """The tiebreak, read off the policy document in the order it declares.

    ``rank_score`` sorts DESCENDING (a bigger threat comes first); every
    other key sorts ascending, so the order is stable and readable.
    """

    values = {
        "rank_score": -(effective_score if effective_score is not None else breakdown.score),
        "threat_class": threat_class,
        "metric_id": metric_id,
        "feature_id": feature_id,
    }
    return tuple(values[key] for key in policy.tiebreak)


__all__ = [
    "CycleExtent",
    "RankBreakdown",
    "order_key",
    "resolvable_area_km2",
    "score",
]
