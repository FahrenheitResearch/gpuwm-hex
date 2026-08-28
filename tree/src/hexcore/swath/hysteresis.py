"""Keeping a swath where it is until the world gives a reason to move it.

THE BREAKAGE, NAMED.  Ranking alone decides the admitted set every cycle
from scratch.  Two candidates whose scores differ by less than the
arithmetic that produced them then trade the last admitted slot every
cycle, and each trade throws away a fine domain that already ran -- its
mesh, its statics, its boundaries and its spin-up -- to build another on
ground the discarded one already covered.  On the shipped cascade ladder a
fine slot is 56.5 min of GPU plus 9.0 min of initialisation, so one
avoidable trade is over an hour out of a six-hour cycle.  It also breaks
the product: a slot's folder is a storm's whole life, and a slot that
changes storms mid-animation shows a cut nobody can explain from the
image.

THE RULE, in three parts, each with a knob in the policy document:

1. CONTINUATION.  A candidate whose window opens within
   ``continuation_radius_km`` of a prior slot's last centroid, under the
   same metric row, IS that slot.  Identity is geographic and continuous,
   never an index into a sorted list.

2. THE MARGIN.  An incumbent's score is multiplied by
   ``1 + promotion_margin`` before ordering.  A challenger must therefore
   beat it by more than the margin to take the slot -- the two-threshold
   construction that is what hysteresis means.  Expressed as a bonus
   rather than a special case so there is exactly one ordering in the
   whole layer.

3. DWELL.  A slot admitted fewer than ``minimum_dwell_cycles`` ago sorts
   ahead of everything, incumbent or not.  The margin alone cannot stop a
   one-cycle flicker when a genuinely stronger challenger arrives, runs
   one cycle and weakens; dwell is what makes the cost of admitting a
   slot buy at least that many cycles of use.

AND ONE MORE, WHICH IS NOT ABOUT ADMISSION.  An admitted slot that
continues does not automatically regenerate its mesh: it does so only when
its centroid has moved past ``regenerate_centroid_km`` or its ring's
overlap with last cycle's has fallen below ``regenerate_overlap_below``.
A slot that survives without moving is a free cache hit -- the mesh, the
statics and the cull are content-addressed by the same spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import SwathDocumentError
from .geometry import LatLon, great_circle_km, ring_overlap_fraction
from .registry import Hysteresis

STATE_SCHEMA = "gpuwm-hex.swath-state.v1"


@dataclass(frozen=True)
class SlotRecord:
    """One swath's identity, carried across cycles."""

    slot_id: str
    metric_id: str
    threat_class: str
    latitude_deg: float
    longitude_deg: float
    rank_score: float
    admitted_cycle: int
    cycles_held: int
    ring: tuple[LatLon, ...]

    def as_row(self) -> Mapping[str, Any]:
        return {
            "slot_id": self.slot_id,
            "metric_id": self.metric_id,
            "threat_class": self.threat_class,
            "latitude_deg": round(self.latitude_deg, 6),
            "longitude_deg": round(self.longitude_deg, 6),
            "rank_score": round(self.rank_score, 6),
            "admitted_cycle": self.admitted_cycle,
            "cycles_held": self.cycles_held,
            "ring": [[round(lat, 6), round(lon, 6)] for lat, lon in self.ring],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SlotRecord":
        known = {
            "slot_id", "metric_id", "threat_class", "latitude_deg", "longitude_deg",
            "rank_score", "admitted_cycle", "cycles_held", "ring",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise SwathDocumentError(
                f"swath-state slot row carries unknown key(s) {unknown}. A state "
                "document is honoured in full or refused: a slot silently loaded "
                "without one of its fields would lose the very continuity the "
                "document exists to carry"
            )
        return cls(
            slot_id=str(raw["slot_id"]),
            metric_id=str(raw["metric_id"]),
            threat_class=str(raw["threat_class"]),
            latitude_deg=float(raw["latitude_deg"]),
            longitude_deg=float(raw["longitude_deg"]),
            rank_score=float(raw["rank_score"]),
            admitted_cycle=int(raw["admitted_cycle"]),
            cycles_held=int(raw["cycles_held"]),
            ring=tuple((float(lat), float(lon)) for lat, lon in raw.get("ring", ())),
        )


@dataclass
class SwathState:
    cycle_index: int = -1
    slots: list[SlotRecord] = field(default_factory=list)

    def as_document(self) -> Mapping[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "cycle_index": self.cycle_index,
            "slots": [slot.as_row() for slot in self.slots],
        }

    @classmethod
    def empty(cls) -> "SwathState":
        return cls()

    @classmethod
    def from_document(cls, raw: Mapping[str, Any]) -> "SwathState":
        unknown = sorted(set(raw) - {"schema", "cycle_index", "slots"})
        if unknown:
            raise SwathDocumentError(
                f"swath-state carries unknown key(s) {unknown}"
            )
        schema = str(raw.get("schema", ""))
        if schema != STATE_SCHEMA:
            raise SwathDocumentError(
                f"swath-state declares schema {schema!r}; this build reads "
                f"{STATE_SCHEMA!r}. Continuing a cycle from a state document this "
                "build cannot read in full would silently restart every slot, and "
                "a restarted slot looks exactly like a continued one on the frame"
            )
        return cls(
            cycle_index=int(raw.get("cycle_index", -1)),
            slots=[SlotRecord.from_mapping(entry) for entry in raw.get("slots", ())],
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "SwathState":
        if path is None:
            return cls.empty()
        target = Path(path).expanduser()
        if not target.exists():
            return cls.empty()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SwathDocumentError(
                f"cannot read the swath-state document at {target}: {error}"
            ) from error
        return cls.from_document(raw)


@dataclass(frozen=True)
class Continuity:
    """What the previous cycle says about one candidate."""

    slot_id: str | None
    incumbent: bool
    protected: bool
    cycles_held: int
    previous: SlotRecord | None
    centroid_moved_km: float | None
    ring_overlap: float | None
    mesh_action: str
    reason: str

    def as_row(self) -> Mapping[str, Any]:
        return {
            "slot_id": self.slot_id,
            "incumbent": self.incumbent,
            "dwell_protected": self.protected,
            "cycles_held": self.cycles_held,
            "centroid_moved_km": (
                None if self.centroid_moved_km is None
                else round(self.centroid_moved_km, 3)
            ),
            "ring_overlap": (
                None if self.ring_overlap is None else round(self.ring_overlap, 4)
            ),
            "mesh_action": self.mesh_action,
            "reason": self.reason,
        }


def match(
    *,
    state: SwathState,
    metric_id: str,
    centroid: LatLon,
    ring: Sequence[LatLon],
    rule: Hysteresis,
    claimed: set[str],
) -> Continuity:
    """Which prior slot, if any, this candidate continues."""

    best: SlotRecord | None = None
    best_distance = float("inf")
    for slot in state.slots:
        if slot.metric_id != metric_id or slot.slot_id in claimed:
            continue
        distance = great_circle_km(
            centroid[0], centroid[1], slot.latitude_deg, slot.longitude_deg
        )
        if distance <= rule.continuation_radius_km and distance < best_distance:
            best, best_distance = slot, distance
    if best is None:
        return Continuity(
            slot_id=None,
            incumbent=False,
            protected=False,
            cycles_held=0,
            previous=None,
            centroid_moved_km=None,
            ring_overlap=None,
            mesh_action="generate",
            reason="new slot: no prior slot of this metric within continuation_radius_km",
        )
    overlap = ring_overlap_fraction(list(ring), list(best.ring)) if best.ring else 0.0
    if best_distance > rule.regenerate_centroid_km:
        action = "generate"
        reason = (
            f"centroid moved {best_distance:.1f} km, past regenerate_centroid_km "
            f"{rule.regenerate_centroid_km:.1f} km"
        )
    elif overlap < rule.regenerate_overlap_below:
        action = "generate"
        reason = (
            f"ring overlap {overlap:.3f} below regenerate_overlap_below "
            f"{rule.regenerate_overlap_below:.3f}"
        )
    else:
        action = "reuse"
        reason = (
            f"centroid moved {best_distance:.1f} km and ring overlap {overlap:.3f}: "
            "the swath still covers the same ground, so the mesh, statics and cull "
            "are a cache hit"
        )
    return Continuity(
        slot_id=best.slot_id,
        incumbent=True,
        protected=best.cycles_held < rule.minimum_dwell_cycles,
        cycles_held=best.cycles_held,
        previous=best,
        centroid_moved_km=best_distance,
        ring_overlap=overlap,
        mesh_action=action,
        reason=reason,
    )


def effective_score(raw_score: float, continuity: Continuity, rule: Hysteresis) -> float:
    """The score the ORDER is taken on: an incumbent carries the margin."""

    if not continuity.incumbent:
        return raw_score
    return raw_score * (1.0 + rule.promotion_margin)


def next_slot_id(used: Sequence[str]) -> str:
    taken = set(used)
    ordinal = 1
    while f"s{ordinal:02d}" in taken:
        ordinal += 1
    return f"s{ordinal:02d}"


__all__ = [
    "STATE_SCHEMA",
    "Continuity",
    "SlotRecord",
    "SwathState",
    "effective_score",
    "match",
    "next_slot_id",
]
