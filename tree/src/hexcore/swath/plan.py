"""One cycle's placement decision, end to end.

The whole layer in one function: read the coarse forecast, find features,
join them into tracks, project each track over its own lead window, build
its flared ring, score it, decide which survive against the previous
cycle's slots and the machine's budget, and emit the mesh-spec rows the
generator and the culler already understand.

THE OUTPUT IS DOCUMENTS, NOT CALLS.  ``gpuwm-hex.swath-plan.v1`` carries,
per admitted swath, a ``mesh_spec`` that ``rw_mpas_mesh --spec`` reads
unchanged and a ``cull_region`` that ``rw_mpas_mesh --cull-parent
--region`` reads unchanged.  Nothing downstream of this file had to grow a
swath concept, which is the test that the placement layer is a placement
layer and not a second mesh grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .detect import DetectionResult, detect
from .errors import SwathRefusal
from .geometry import (
    dilate_shape,
    LatLon,
    fit_swath_axis,
    great_circle_km,
    predicted_cells_in,
    resample_track,
    track_length_km,
)
from .history import HistoryReader
from .hysteresis import Continuity, SlotRecord, SwathState, effective_score, match, next_slot_id
from .rank import CycleExtent, RankBreakdown, order_key, score
from .registry import MetricRegistry, MetricRow, PlacementPolicy
from .track import ProjectedPath, Track, associate, half_width_profile, project

PLAN_SCHEMA = "gpuwm-hex.swath-plan.v1"

#: Below this multiple of its own base half-width, a projected path is not a
#: swath: the ring degenerates to a rounded blob and the two end caps carry
#: nearly all of it.  Such a candidate is emitted as a ``cap`` region, which
#: is the same grammar and the honest shape for something that is not going
#: anywhere.  The choice is made on the GEOMETRY -- a stalled cyclone gets a
#: cap and a fast-moving convective line gets a swath -- never on which
#: metric row produced it.
STATIONARY_PATH_FRACTION = 0.25


@dataclass
class Candidate:
    track: Track
    metric: MetricRow
    path: ProjectedPath
    axis: tuple[LatLon, ...]
    half_widths_km: tuple[float, ...]
    ring: tuple[LatLon, ...]
    shape: Mapping[str, Any]
    centroid: LatLon
    breakdown: RankBreakdown
    ignite_at_seconds: float
    smoothing_passes: int = 0
    smoothing_drift_km: float = 0.0
    continuity: Continuity | None = None
    effective: float = 0.0
    slot_id: str | None = None
    declined: str | None = None
    sizing: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    admitted: list[Candidate]
    declined: list[Candidate]
    tracks: list[Track]
    detection: DetectionResult
    state: SwathState
    churn: Mapping[str, Any]


# ---------------------------------------------------------------------------
# one candidate
# ---------------------------------------------------------------------------
def _ignition_seconds(
    metric: MetricRow, track: Track, frame_times: Sequence[float]
) -> float:
    """When this swath starts, derived from the coarse forecast itself.

    ``cycle_start`` is zero.  ``time_of_first_exceedance`` is the first
    frame at which this track's feature existed, less the row's declared
    margin, floored at zero and quantized DOWN to a parent history frame --
    always toward MORE lead, never less, because the quantization must not
    be able to make a swath start after the thing it was placed for.
    """

    if metric.start_policy.kind == "cycle_start":
        return 0.0
    raw = track.first.time_seconds - metric.start_policy.lead_margin_hours * 3600.0
    if raw <= 0.0:
        return 0.0
    earlier = [time for time in frame_times if time <= raw + 1e-6]
    return float(max(earlier)) if earlier else 0.0


def _shape_row(
    ring: Sequence[LatLon], path: ProjectedPath, half_widths: Sequence[float]
) -> tuple[Mapping[str, Any], tuple[LatLon, ...]]:
    return (
        {"kind": "polygon", "vertices_deg": [[lat, lon] for lat, lon in ring]},
        tuple(ring),
    )


def _cap_row(centre: LatLon, radius_km: float) -> tuple[Mapping[str, Any], tuple[LatLon, ...]]:
    from .geometry import destination

    ring = tuple(destination(centre[0], centre[1], bearing, radius_km) for bearing in range(0, 360, 10))
    return (
        {"kind": "cap", "center_deg": [centre[0], centre[1]], "radius_km": radius_km},
        ring,
    )


def build_candidate(
    track: Track, metric: MetricRow, frame_times: Sequence[float]
) -> Candidate:
    ignite = _ignition_seconds(metric, track, frame_times)
    path = project(track, start_seconds=ignite, lead_hours=metric.swath.lead_hours)
    dense_points = list(path.points)
    widths = list(half_width_profile(
        path,
        base_km=metric.swath.half_width_km,
        flare_km_per_hour=metric.swath.flare_km_per_hour,
        maximum_km=metric.swath.maximum_half_width_km,
    ))
    travelled = track_length_km(dense_points) if len(dense_points) > 1 else 0.0
    passes = 0
    drift = 0.0
    if travelled < STATIONARY_PATH_FRACTION * metric.swath.half_width_km:
        centre = dense_points[len(dense_points) // 2]
        radius = max(widths)
        shape, ring = _cap_row(centre, radius)
        centroid = centre
        axis = list(dense_points)
    else:
        dense, dense_widths = _densify(dense_points, widths, metric.swath.path_step_km)
        axis, ring_points, passes, drift = fit_swath_axis(
            dense, dense_widths, cap_points=metric.swath.cap_points
        )
        shape, ring = _shape_row(ring_points, path, dense_widths)
        centroid = axis[len(axis) // 2]
    return Candidate(
        track=track,
        metric=metric,
        path=path,
        axis=tuple(axis),
        half_widths_km=tuple(widths),
        ring=ring,
        shape=shape,
        centroid=centroid,
        breakdown=RankBreakdown(0.0, ()),
        ignite_at_seconds=ignite,
        smoothing_passes=passes,
        smoothing_drift_km=drift,
    )


def _densify(
    points: Sequence[LatLon], widths: Sequence[float], step_km: float
) -> tuple[list[LatLon], list[float]]:
    """Resample the axis, carrying the half-width along it by arc length."""

    dense = resample_track(points, step_km=step_km)
    if len(dense) == len(points):
        return list(dense), list(widths)
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + great_circle_km(a[0], a[1], b[0], b[1]))
    total = cumulative[-1]
    dense_cumulative = [0.0]
    for a, b in zip(dense, dense[1:]):
        dense_cumulative.append(dense_cumulative[-1] + great_circle_km(a[0], a[1], b[0], b[1]))
    out: list[float] = []
    for distance in dense_cumulative:
        if total <= 0.0:
            out.append(widths[0])
            continue
        for index in range(len(cumulative) - 1):
            if cumulative[index] <= distance <= cumulative[index + 1] + 1e-9:
                span = cumulative[index + 1] - cumulative[index]
                fraction = 0.0 if span <= 0.0 else (distance - cumulative[index]) / span
                out.append(widths[index] + fraction * (widths[index + 1] - widths[index]))
                break
        else:
            out.append(widths[-1])
    return list(dense), out


def cull_region_for(candidate: Candidate) -> Mapping[str, Any]:
    """The Shape row ``rw_mpas_mesh --cull-parent --region`` is handed.

    THE MESH SPEC IS NOT TOUCHED, and that separation is the point.  The
    refinement the row asks for is unchanged -- same fine spacing, same
    transition, same ring -- and only the LIMITED-AREA CUT moves outward, into
    the parent's own resolution ramp.  So a wider pad costs cells in the cull
    and nothing at all in the global mesh, and it cannot move a cell centre:
    the fine core the swath asked for is bit-identical whatever the pad is.

    ``cull_pad_scale`` of 1.0 returns the ring itself, which is what this
    function did before it existed.
    """

    return dilate_shape(candidate.shape, candidate.metric.swath.cull_pad_scale)


def mesh_spec_for(candidate: Candidate, policy: PlacementPolicy, slot_id: str) -> Mapping[str, Any]:
    """The resolution spec ``rw_mpas_mesh --spec`` reads, unchanged."""

    return {
        "background_km": policy.budget.background_km,
        "name": f"swath-{slot_id}-{candidate.metric.id}",
        "regions": [
            {
                "shape": dict(candidate.shape),
                "spacing_km": candidate.metric.swath.spacing_km,
                "transition_cells": candidate.metric.swath.transition_cells,
            }
        ],
    }


# ---------------------------------------------------------------------------
# the cycle
# ---------------------------------------------------------------------------
def plan_cycle(
    reader: HistoryReader,
    registry: MetricRegistry,
    policy: PlacementPolicy,
    *,
    state: SwathState | None = None,
    cycle_index: int | None = None,
) -> PlanResult:
    """Everything one cycle decides, from one history file and two documents."""

    previous = state if state is not None else SwathState.empty()
    index = cycle_index if cycle_index is not None else previous.cycle_index + 1
    detection = detect(reader, registry)
    tracks = associate(detection.features, registry.metric_rows)
    frame_times = [frame.time_seconds for frame in reader.frames()]
    # What this cycle made possible, so the two track terms are
    # fractions of their own ceiling rather than of a policy constant
    # that knows nothing about the output interval.
    cycle = CycleExtent.from_frame_times(frame_times)

    candidates: list[Candidate] = []
    declined: list[Candidate] = []
    for track in tracks:
        metric = registry.metric_rows[track.metric_id]
        try:
            candidate = build_candidate(track, metric, frame_times)
        except SwathRefusal as refusal:
            declined.append(
                Candidate(
                    track=track, metric=metric,
                    path=ProjectedPath((), (), 0.0, 0.0, 0.0), axis=(),
                    half_widths_km=(), ring=(), shape={},
                    centroid=(track.last.latitude_deg, track.last.longitude_deg),
                    breakdown=RankBreakdown(0.0, ()), ignite_at_seconds=0.0,
                    declined=f"DELAYED-START-LEAD: {refusal}",
                )
            )
            continue
        candidate.breakdown = score(track, metric, policy, cycle=cycle)
        candidates.append(candidate)

    claimed: set[str] = set()
    for candidate in candidates:
        candidate.continuity = match(
            state=previous,
            metric_id=candidate.metric.id,
            centroid=candidate.centroid,
            ring=candidate.ring,
            rule=policy.hysteresis,
            claimed=claimed,
        )
        if candidate.continuity.slot_id is not None:
            claimed.add(candidate.continuity.slot_id)
        candidate.effective = effective_score(
            candidate.breakdown.score, candidate.continuity, policy.hysteresis
        )

    candidates.sort(
        key=lambda item: (
            not (item.continuity is not None and item.continuity.protected),
            order_key(
                breakdown=item.breakdown,
                threat_class=item.track.threat_class,
                metric_id=item.metric.id,
                feature_id=item.track.track_id,
                policy=policy,
                effective_score=item.effective,
            ),
        )
    )

    admitted: list[Candidate] = []
    used_slots = [slot.slot_id for slot in previous.slots]
    held_by_class: dict[str, int] = {}
    for candidate in candidates:
        if len(admitted) >= policy.budget.maximum_swaths:
            candidate.declined = (
                f"SWATH-BUDGET: rank {candidates.index(candidate) + 1} of "
                f"{len(candidates)} against maximum_swaths "
                f"{policy.budget.maximum_swaths}. A cycle that admits more swaths "
                "than its wall clock allows places the NEXT cycle's swaths from a "
                "stale coarse forecast"
            )
            declined.append(candidate)
            continue
        crowded = _class_budget_gate(candidate, held_by_class, policy)
        if crowded is not None:
            candidate.declined = crowded
            declined.append(candidate)
            continue
        idle = _idle_lead_gate(candidate)
        if idle is not None:
            candidate.declined = idle
            declined.append(candidate)
            continue
        clash = _separation_clash(candidate, admitted, policy)
        if clash is not None:
            candidate.declined = clash
            declined.append(candidate)
            continue
        continuity = candidate.continuity
        if continuity is not None and continuity.slot_id is not None:
            candidate.slot_id = continuity.slot_id
        else:
            candidate.slot_id = next_slot_id(used_slots)
            used_slots.append(candidate.slot_id)
        cells = predicted_cells_in(
            list(candidate.ring), spacing_km=candidate.metric.swath.spacing_km
        )
        candidate.sizing = {
            "predicted_cells": round(cells, 1),
            "basis": "area_integral",
            "maximum_cells_per_swath": policy.budget.maximum_cells_per_swath,
        }
        if cells > policy.budget.maximum_cells_per_swath:
            candidate.declined = (
                f"SWATH-CAPACITY: {cells:,.0f} predicted cells against "
                f"maximum_cells_per_swath {policy.budget.maximum_cells_per_swath:,.0f}. "
                "A swath above the ceiling does not fail at generation, it fails "
                "during the forecast that consumes it, after the mesh, the statics "
                "and the boundaries have all been paid for"
            )
            declined.append(candidate)
            continue
        admitted.append(candidate)
        held_by_class[candidate.track.threat_class] = (
            held_by_class.get(candidate.track.threat_class, 0) + 1
        )

    churn = _churn(previous, admitted)
    next_state = SwathState(
        cycle_index=index,
        slots=[
            SlotRecord(
                slot_id=str(candidate.slot_id),
                metric_id=candidate.metric.id,
                threat_class=candidate.track.threat_class,
                latitude_deg=candidate.centroid[0],
                longitude_deg=candidate.centroid[1],
                rank_score=candidate.breakdown.score,
                admitted_cycle=(
                    candidate.continuity.previous.admitted_cycle
                    if candidate.continuity is not None and candidate.continuity.previous
                    else index
                ),
                cycles_held=(
                    candidate.continuity.cycles_held + 1
                    if candidate.continuity is not None and candidate.continuity.incumbent
                    else 1
                ),
                ring=candidate.ring,
            )
            for candidate in admitted
        ],
    )
    return PlanResult(
        admitted=admitted,
        declined=declined,
        tracks=tracks,
        detection=detection,
        state=next_state,
        churn=churn,
    )


#: A swath whose feature does not exist for more than this fraction of its
#: own lead window is refused.  The forecast still runs -- it just resolves
#: an atmosphere with nothing in it -- so nothing crashes and nothing looks
#: wrong on the frame; the cost is simply spent.
IDLE_LEAD_FRACTION = 0.5


def _idle_lead_gate(candidate: "Candidate") -> str | None:
    """Refuse a swath that pays for hours in which its feature does not exist.

    THE BREAKAGE THIS PREVENTS: a ``cycle_start`` swath placed for a
    cyclone the coarse forecast does not produce until hour 12 of a 12 h
    window integrates a fine mesh over an empty ocean and then stops at the
    moment the storm appears.  It costs a full slot -- 56.5 min of GPU plus
    9.0 min of init on the shipped ladder -- and produces frames that look
    like a quiet sea rather than like a failure, so nothing downstream
    catches it.  The remedy is named in the refusal because it is a ROW:
    ``start_policy.kind = time_of_first_exceedance`` ignites the swath at
    the hour the machine derived from its own forecast instead.
    """

    lead_seconds = candidate.metric.swath.lead_hours * 3600.0
    idle = candidate.track.first.time_seconds - candidate.ignite_at_seconds
    if idle <= IDLE_LEAD_FRACTION * lead_seconds:
        return None
    return (
        f"IGNITION-BEFORE-ONSET: the swath ignites at "
        f"{candidate.ignite_at_seconds / 3600.0:.2f} h but its track's first "
        f"frame is at {candidate.track.first.time_seconds / 3600.0:.2f} h, so "
        f"{idle / 3600.0:.2f} h of a {candidate.metric.swath.lead_hours:.2f} h "
        f"window would integrate a fine mesh over an atmosphere in which the "
        f"feature does not exist. Set this metric row's start_policy.kind to "
        f"'time_of_first_exceedance' with a lead_margin_hours that buys the "
        f"spin-up you want ahead of onset"
    )


def _class_budget_gate(
    candidate: Candidate, held: Mapping[str, int], policy: PlacementPolicy
) -> str | None:
    """Refuse a candidate whose threat class has already filled its share.

    THE BREAKAGE THIS PREVENTS, measured on a real 24 h global forecast:
    the detector formed 258 tracks -- 41 cyclone and 217 deep-convection --
    and the twenty-seven highest-ranked candidates of 205 were all
    cyclones, so a budget of four placed four cyclones and none of the 217
    convective regions the same forecast held.  Putting the classes in
    commensurable units (see ``registry.RankRow``) is necessary and it is
    not sufficient: it makes the scores mean the same thing, it does not
    make the atmosphere interleave them.  A cycle that can only ever
    resolve whatever its strongest class is doing is a tracker for that
    class with a threat table attached, and every slot it sweeps is a slot
    the other threat did not get -- 56.5 min of GPU plus 9.0 min of init
    on the shipped ladder, each.

    The cap applies to INCUMBENTS as well, deliberately.  A cap that
    exempted dwell-protected slots could never take effect on the only
    cycle it matters on: the one that already holds a full class.
    """

    limit = policy.budget.maximum_per_threat_class
    if limit >= policy.budget.maximum_swaths:
        return None
    threat_class = candidate.track.threat_class
    if held.get(threat_class, 0) < limit:
        return None
    return (
        f"SWATH-CLASS-BUDGET: threat class {threat_class!r} already holds "
        f"{held.get(threat_class, 0)} of this cycle's {limit} against "
        f"maximum_per_threat_class {limit} (maximum_swaths "
        f"{policy.budget.maximum_swaths}). One class sweeping the whole budget "
        "means the cycle resolves nothing else the forecast holds; the slot goes "
        "to the next highest-ranked candidate of another class"
    )


def _separation_clash(
    candidate: Candidate, admitted: Sequence[Candidate], policy: PlacementPolicy
) -> str | None:
    for other in admitted:
        distance = great_circle_km(
            candidate.centroid[0], candidate.centroid[1],
            other.centroid[0], other.centroid[1],
        )
        if distance < policy.budget.minimum_separation_km:
            return (
                f"SWATH-SEPARATION: {distance:.1f} km from admitted slot "
                f"{other.slot_id} against minimum_separation_km "
                f"{policy.budget.minimum_separation_km:.1f}. Two swaths this close "
                "would cull overlapping windows off separate parents and refine the "
                "same ground twice, paying two forecasts for one"
            )
    return None


def _churn(previous: SwathState, admitted: Sequence[Candidate]) -> Mapping[str, Any]:
    """What the hysteresis rule bought, counted.

    ``evictions`` are slots the previous cycle held that this cycle does
    not.  ``regenerations`` are admitted slots whose mesh must be built
    again.  Both are what a run with the rule disarmed is compared against.
    """

    held = {slot.slot_id for slot in previous.slots}
    kept = {
        candidate.slot_id for candidate in admitted
        if candidate.continuity is not None and candidate.continuity.incumbent
    }
    return {
        "previous_slots": len(previous.slots),
        "admitted": len(admitted),
        "continued": len(kept),
        "evictions": len(held - kept),
        "new_slots": len(admitted) - len(kept),
        "mesh_generate": sum(
            1 for candidate in admitted
            if candidate.continuity is None or candidate.continuity.mesh_action == "generate"
        ),
        "mesh_reuse": sum(
            1 for candidate in admitted
            if candidate.continuity is not None and candidate.continuity.mesh_action == "reuse"
        ),
    }


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------
def plan_document(
    reader: HistoryReader,
    registry: MetricRegistry,
    policy: PlacementPolicy,
    result: PlanResult,
) -> Mapping[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "cycle_index": result.state.cycle_index,
        "history": dict(reader.provenance()),
        "metrics_document": {
            "schema": registry.schema,
            "sha256": registry.sha256,
            "path": str(registry.source_path) if registry.source_path else None,
            "armed": [row.id for row in registry.armed],
            "publication_manifest": list(registry.publication_manifest()),
        },
        "policy_document": {
            "schema": policy.schema,
            "sha256": policy.sha256,
            "path": str(policy.source_path) if policy.source_path else None,
        },
        "admitted": [_admitted_row(candidate, policy) for candidate in result.admitted],
        "declined": [_declined_row(candidate) for candidate in result.declined],
        "tracks": [track.as_row() for track in result.tracks],
        "drops": [drop.as_row() for drop in result.detection.drops],
        "churn": dict(result.churn),
        "state": dict(result.state.as_document()),
    }


def _admitted_row(candidate: Candidate, policy: PlacementPolicy) -> Mapping[str, Any]:
    slot_id = str(candidate.slot_id)
    return {
        "slot_id": slot_id,
        "metric_id": candidate.metric.id,
        "threat_class": candidate.track.threat_class,
        "track_id": candidate.track.track_id,
        "ignite_at_seconds": candidate.ignite_at_seconds,
        "lead_hours": candidate.metric.swath.lead_hours,
        "extrapolated_hours": round(candidate.path.extrapolated_hours, 4),
        "track_first_time_seconds": candidate.track.first.time_seconds,
        "idle_lead_hours": round(
            max(
                0.0,
                (candidate.track.first.time_seconds - candidate.ignite_at_seconds) / 3600.0,
            ),
            4,
        ),
        "axis_smoothing_passes": candidate.smoothing_passes,
        "axis_smoothing_drift_km": round(candidate.smoothing_drift_km, 3),
        "axis_deg": [[round(lat, 6), round(lon, 6)] for lat, lon in candidate.axis],
        "centroid_deg": [round(candidate.centroid[0], 6), round(candidate.centroid[1], 6)],
        "path_deg": [[round(lat, 6), round(lon, 6)] for lat, lon in candidate.path.points],
        "half_widths_km": [round(width, 3) for width in candidate.half_widths_km],
        "cull_region": dict(cull_region_for(candidate)),
        "ring_deg": [[round(lat, 6), round(lon, 6)] for lat, lon in candidate.ring],
        "mesh_spec": dict(mesh_spec_for(candidate, policy, slot_id)),
        "sizing": dict(candidate.sizing),
        "rank": dict(candidate.breakdown.as_row()),
        "effective_score": round(candidate.effective, 6),
        "hysteresis": (
            dict(candidate.continuity.as_row()) if candidate.continuity else None
        ),
    }


def _declined_row(candidate: Candidate) -> Mapping[str, Any]:
    return {
        "slot_id": candidate.slot_id,
        "metric_id": candidate.metric.id,
        "threat_class": candidate.track.threat_class,
        "track_id": candidate.track.track_id,
        "centroid_deg": [round(candidate.centroid[0], 6), round(candidate.centroid[1], 6)],
        "rank": dict(candidate.breakdown.as_row()),
        "reason": candidate.declined,
    }


__all__ = [
    "PLAN_SCHEMA",
    "STATIONARY_PATH_FRACTION",
    "Candidate",
    "PlanResult",
    "build_candidate",
    "mesh_spec_for",
    "plan_cycle",
    "plan_document",
]
