"""Per-architecture execution admission below the proven contract floor.

The port's numerical contract — the FTZ/subnormal contract, the NVRTC
contraction pin, and every frozen authority digest — was proven on sm_120
(compute capability 12.0).  ``require_cuda`` therefore refuses any other
architecture: an output produced on an unproven part could not be checked
against anything, so nothing it computed would be evidence.

Opening that pin for another architecture is not an edit to two integers.
An architecture is admitted below the floor only when it holds its own
anchor, minted on real hardware of that architecture:

* a **contract receipt** — the measured subnormal/FTZ behaviour of the
  kernels' actual arithmetic and the contraction pin under the port's own
  NVRTC flags, on a card of that architecture; and
* an **authority anchor** — the frozen-authority verification run on that
  architecture, recording per item either byte-identity with the sm_120
  masked digests or a stable, twice-reproduced per-architecture digest set.

The registry below is that record.  An entry is added only together with
the evidence it names; the evidence lives in this repository and the tests
hold each entry to it.  Architectures at or above the floor are untouched
by this module — the sm_120 path does not consult it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


#: The architecture the port's numerical contract was originally proven on.
#: Everything below this consults :data:`ADMITTED_BELOW_FLOOR`.
PROVEN_COMPUTE: tuple[int, int] = (12, 0)


@dataclass(frozen=True, slots=True)
class ArchAnchor:
    """The evidence that admits one architecture below the proven floor."""

    compute: tuple[int, int]
    card: str
    admitted_on: str
    contract_receipt: str
    authority_anchor: str
    basis: str

    @property
    def sm(self) -> str:
        return f"sm_{self.compute[0]}{self.compute[1]}"

    def as_dict(self) -> dict[str, object]:
        return {
            "compute_capability": f"{self.compute[0]}.{self.compute[1]}",
            "sm": self.sm,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "contract_receipt": self.contract_receipt,
            "authority_anchor": self.authority_anchor,
            "basis": self.basis,
        }


#: Architectures below :data:`PROVEN_COMPUTE` holding a verified anchor.
#: An entry lands only together with the receipt and authority evidence it
#: names, minted on real hardware of that architecture.
ADMITTED_BELOW_FLOOR: Mapping[tuple[int, int], ArchAnchor] = MappingProxyType(
    {
        (8, 6): ArchAnchor(
            compute=(8, 6),
            card=(
                "NVIDIA GeForce RTX 3080 (10,240 MiB, 68 SM, WDDM), "
                "driver 610.74"
            ),
            admitted_on="2026-08-25",
            contract_receipt="evidence/sm86-tier-20260825/RECEIPT.md",
            authority_anchor="evidence/sm86-tier-20260825/authority",
            basis=(
                "sm86-tier campaign: FTZ/contraction contract measured on "
                "this card by the port's own decks and the engine probe "
                "arms; determinism and authority anchoring per the campaign "
                "receipt"
            ),
        ),
    }
)


#: Per-architecture ceilings for the FTZ guarded-fallback timing control
#: (``cuda_ftz.run_normalized_fallback_performance_control``), beside the
#: admission registry because the two answer the same question -- what is
#: proven on THIS architecture -- and drift apart when kept in different
#: files.  Made per-architecture 2026-08-25 (stale-guard audit #347,
#: finding 8): the single global 1.25 was calibrated when sm_120 was the
#: only architecture and hard-refused the newly admitted sm_86 tier on a
#: timing-only deviation while bitwise identity held.  Timings are
#: SECONDARY evidence everywhere this ceiling is consulted; correctness is
#: bitwise enabled/disabled identity, which no ceiling relaxes.
PERFORMANCE_RATIO_CEILINGS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "sm_120": MappingProxyType(
            {
                "ceiling": 1.25,
                "basis": (
                    "the original sm_120 calibration: the five named "
                    "normalized-kernel microbenchmarks each measured below "
                    "1.25x median enabled/disabled on the proven-floor "
                    "card, and every archived sm_120 binding declares this "
                    "ceiling -- unchanged, so those receipts stay valid"
                ),
            }
        ),
        "sm_86": MappingProxyType(
            {
                "ceiling": 1.75,
                "basis": (
                    "set from the RECORDED sm_86 deviation, not a fresh "
                    "calibration: the desktop RTX 3080 measured 1.471975x "
                    "and 1.565028x at transport.transport_edge_values "
                    "(bitwise identity held; evidence/sm86-tier-20260825/"
                    "contract/perf-control-stability-sm86.json, STATE.md "
                    "section 8) while the four other benchmarks passed at "
                    "1.25.  1.75 covers the top recorded reading with the "
                    "recorded run-to-run spread (~0.093) as headroom; a "
                    "fresh multi-run calibration on the 3080 is the named "
                    "follow-up that replaces this basis"
                ),
            }
        ),
    }
)


def performance_ratio_ceiling(sm: str) -> float:
    """The FTZ guard-cost timing ceiling for one architecture, by ``sm_NN``.

    Refuses an unregistered architecture by name: a ceiling nobody measured
    or recorded for this silicon would turn the timing control into a
    number invented at the call site.
    """

    row = PERFORMANCE_RATIO_CEILINGS.get(sm)
    if row is None:
        roster = ", ".join(sorted(PERFORMANCE_RATIO_CEILINGS))
        raise LookupError(
            f"no FTZ performance-ratio ceiling is registered for {sm}: the "
            f"guard-cost timing control has no measured or recorded bound "
            f"on this architecture, so a verdict from it would be an "
            f"invented number; registered architectures: {roster}"
        )
    return float(row["ceiling"])  # type: ignore[arg-type]


def admitted_architecture(compute: tuple[int, int]) -> ArchAnchor | None:
    """The anchor admitting ``compute`` below the floor, or ``None``."""

    return ADMITTED_BELOW_FLOOR.get((int(compute[0]), int(compute[1])))


def admitted_summary() -> str:
    """Human-readable roster for refusal messages."""

    anchors = sorted(ADMITTED_BELOW_FLOOR.values(), key=lambda a: a.compute)
    if not anchors:
        return "none"
    return ", ".join(anchor.sm for anchor in anchors)


def below_floor_refusal(
    compute: tuple[int, int], floor: tuple[int, int]
) -> str:
    """The named refusal for an architecture holding no anchor.

    Names the architecture and the concrete breakage the refusal prevents:
    without a per-architecture contract receipt and authority anchor there
    is no digest set any output of this card could be verified against.
    """

    return (
        f"cuda.compute_capability={compute[0]}.{compute[1]} "
        f"(sm_{compute[0]}{compute[1]}) is below the proven contract floor "
        f"{floor[0]}.{floor[1]} and holds no per-architecture anchor: no "
        f"numerical-contract receipt or frozen-authority set exists for "
        f"this architecture, so nothing it computed could be verified; "
        f"architectures anchored below the floor: {admitted_summary()}"
    )


__all__ = [
    "ADMITTED_BELOW_FLOOR",
    "ArchAnchor",
    "PERFORMANCE_RATIO_CEILINGS",
    "PROVEN_COMPUTE",
    "admitted_architecture",
    "admitted_summary",
    "below_floor_refusal",
    "performance_ratio_ceiling",
]
