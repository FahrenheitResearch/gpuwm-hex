"""Every gate ``rw_mpas_mesh`` applies at GENERATION, asked of the SPEC.

THE BREAKAGE THIS PREVENTS, MEASURED 2026-08-28.  ``rw_mpas_mesh --dry-run``
sizes a resolution spec, prints a receipt and exits 0.  It does not apply the
gates the build applies, and two of those gates refuse specs that sizing has
just called fine:

* the **transition-band gate** (``rw-mpas`` ``mesh/hierarchy.rs``,
  ``generate_graded``) refuses a spec whose steepest requested spacing
  gradient packs a refinement level's transition band into fewer than twice
  the 3-cell surgery locality radius.  It is pure arithmetic on the SAME
  number the dry-run receipt already prints as
  ``steepest_requested_gradient_percent_per_cell``, and it refuses in about
  90 ms -- *after* a user has authored, sized and accepted the spec.
* the **short dual edge floor** (``rw-mpas`` ``mesh/validate.rs``,
  ``Limits::min_dv_edge_m``) reads the finished tessellation, so it cannot
  fire until the whole ladder has been relaxed.  Measured here on a
  0.75/3/15/75 km design that cleared the transition-band gate: **1,251 s**
  to derive 217,621 cells and then refuse edge 562175 -- ``dvEdge`` 36.8 m
  against ``dcEdge`` 886 m -- with no grid written.  The report that opened
  this lane measured the same shape at 711 s, 220,468 cells and edge 655199
  (39.2 m over 937 m) on another box.

  **THAT FLOOR MOVED WITH THE STORAGE ON 2026-08-29, and this module moved
  with it.**  The floor is not a length any more, it is 400 COORDINATE QUANTA
  (``mesh/validate.rs::DV_EDGE_FLOOR_QUANTA``): the breakage it prevents is
  the orthogonality defect storage rounding puts into the point set,
  ``1.935 * q / dvEdge`` at the worst edge, measured on both published
  statics' own bytes.  At the published binary32 representation the quantum
  is 0.5 m and 400 quanta is 200.0 m to the bit, so nothing moved for a mesh
  with a native MPAS-A counterpart.  A mesh ``rw_mpas_mesh`` GENERATES has no
  native counterpart and its static stores coordinates at binary64, where the
  quantum is 9.313e-10 m and the same budget reads **3.725e-7 m**.  Every
  refusal quoted above was a binary32 refusal of a generated mesh, and none of
  them would happen today.  Leaving 200 m transcribed here would have told a
  user their 1 km spec was heading for a refusal it can no longer reach --
  which is this lane's own fault, pointed the other way.

This module is the answer to the first: the cheap gate is applied where the
sizing happens, so a spec generation will refuse is refused by sizing first.
It is also the answer to the second in the only form the second admits -- the
floor is NOT decidable from a spec, and saying so with the numbers that
decide it beats saying nothing.  See :func:`short_dual_edge_exposure`.

WHAT THIS DOES NOT DO, AND WHY.  It does not move, soften or reinterpret
either floor.  The dual-edge floor has already survived a wrong-value episode
(the retired 7,500 m anchor refused the published ``x4.163842`` itself), and
the fix for "the user learns too late" is to change WHEN they learn, never
WHAT is allowed.  Every constant here is transcribed from the Rust that
enforces it, with the file named beside it, so a drift is a grep -- and the
2026-08-29 storage change is exactly the drift that discipline exists to
catch: the constant is now DERIVED here the same way it is derived there,
from the quanta and the representation, so the next storage change moves it
without an edit.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .errors import MpasPortError

# ---------------------------------------------------------------------------
# the constants, transcribed from the Rust that enforces them
# ---------------------------------------------------------------------------

#: ``rw-mpas`` ``mesh/hierarchy.rs::SURGERY_LOCALITY_CELLS``.  Surgery repairs
#: a dislocation by editing this many cells of neighbourhood.
SURGERY_LOCALITY_CELLS = 3.0

#: The gate: a level's transition band must be at least twice the locality
#: radius, or a repair at the band's centre reaches across the whole band.
MIN_TRANSITION_BAND_CELLS = 2.0 * SURGERY_LOCALITY_CELLS

#: The same gate expressed as the quantity a receipt prints.  The band is
#: ``ln 2 / ln(1 + g)`` cells wide, so the ceiling on ``g`` is exactly
#: ``2 ** (1 / 6) - 1`` -- 12.2462 % per cell.  Derived, never chosen.
MAX_GRADIENT_PER_CELL = 2.0 ** (1.0 / MIN_TRANSITION_BAND_CELLS) - 1.0

#: ``rw-mpas`` ``mesh/validate.rs::DV_EDGE_FLOOR_QUANTA``.  The dual-edge
#: floor in COORDINATE QUANTA of the representation the mesh will be stored
#: in, not in metres: the breakage it prevents is the orthogonality defect
#: storage rounding puts into the point set, ``1.935 * q / dvEdge`` at the
#: worst edge, so a floor stated in metres is stated in the wrong units and
#: has to be re-anchored by hand every time the storage changes.
DV_EDGE_FLOOR_QUANTA = 400.0

#: The coordinate quantum in metres at ``sphere_radius = 6 371 229``: the
#: spacing between representable values at that magnitude.  ``ulp`` is exact,
#: so these are exact.  ``rw-mpas``
#: ``staticfile::coordframe::CoordinateRepresentation::quantum_m``.
COORDINATE_QUANTUM_M = {
    "binary32_earth_centred": 0.5,
    "binary64_earth_centred": 9.313225746154785e-10,
}

#: What a mesh ``rw_mpas_mesh`` GENERATES is stored at.  It has no native
#: MPAS-A counterpart -- native MPAS-A cannot produce it -- so no dycore
#: byte-identity anchor binds its storage precision.  ``rw-mpas``
#: ``staticfile::coordframe::CoordinateRepresentation::for_generated_mesh``.
GENERATED_MESH_REPRESENTATION = "binary64_earth_centred"

#: The floor a mesh with a NATIVE COUNTERPART is judged by: 400 quanta of
#: 0.5 m, which is 200.0 m to the bit -- the 2026-08-25 ruling, unchanged.
PUBLISHED_MIN_DV_EDGE_M = (
    DV_EDGE_FLOOR_QUANTA * COORDINATE_QUANTUM_M["binary32_earth_centred"]
)

#: The floor a GENERATED mesh is judged by, in metres at earth radius, which
#: is what this module advises about: the same 400 quanta at the generated
#: representation's own quantum.  3.725e-7 m.
MIN_DV_EDGE_M = (
    DV_EDGE_FLOOR_QUANTA * COORDINATE_QUANTUM_M[GENERATED_MESH_REPRESENTATION]
)

#: ``Limits::default().min_dv_over_dc``, and equal to this package's own
#: :class:`hexcore.dual_edge_admission.DualEdgePolicy` floor.
MIN_DV_OVER_DC = 0.02

#: ``mesh/hierarchy.rs``: every ladder level must finish surgery at or above
#: this ratio.  It is the strongest per-edge promise the generator makes
#: while it is building, which is what makes it the right thing to compare a
#: length floor against.
LEVEL_SHIP_FLOOR_DV_OVER_DC = 0.03

#: ``mesh/surgery.rs`` ``SurgeryOptions::default().flag_floor``.  Surgery
#: FLAGS a quad for repair only below this ratio, so a drained level settles
#: JUST ABOVE it and this -- not the 0.03 ship floor -- is where a graded
#: mesh's worst dual-edge ratio actually lands.  MEASURED, one 0.75/3/15/75 km
#: build on 2026-08-28: its seven levels finished surgery at 0.0420, 0.0414,
#: 0.0407, 0.0401, 0.0400, 0.0404 and 0.0401, and the edge that refused it
#: read 0.0416.  Every graded mesh in :data:`MEASURED_GRADED_MESHES` reads
#: 0.0401 to 0.0427 as well.  That is what makes the length floor predictable
#: from a spec even though it is not provable from one.
SURGERY_FLAG_FLOOR_DV_OVER_DC = 0.04

#: The worst ``dvEdge/dcEdge`` in the published mesh family (``x4.163842``);
#: ``x1.40962`` reads 0.3945.  Quoted so the exposure figure has a reference
#: that is a measurement rather than a floor.
PUBLISHED_WORST_DV_OVER_DC = 0.033650

#: The lowest ``delivered / requested`` spacing 5th percentile across the ten
#: level runs tabulated in ``mesh/hierarchy.rs``.  Used ONE-SIDEDLY: a
#: delivered cell can be this much finer than its request, so a statement
#: that a floor "cannot bind" has to assume the fine tail, not the median.
DELIVERED_SPACING_P05 = 0.8484

#: Every graded mesh this tree has a shortest-dual-edge reading for:
#: ``(label, finest_km, background_km, min_dv_edge_m, min_dv_over_dc,
#: emitted)``.  ``emitted`` is False for a run the 200 m floor refused, and
#: those rows are the ones that make the table a measurement of the floor
#: rather than a survey of survivors.
#:
#: THE INSTRUMENT IS VALIDATED AGAINST THIS TABLE, IN BOTH DIRECTIONS, and it
#: had to be.  A first draft of this module predicted the shortest dual edge
#: as ``surgery flag floor x finest spacing`` -- 0.04 x 4 km = 160 m -- and
#: would have called every shipped 4 km swath a predicted refusal while rows
#: two and five of this table had BUILT, at 342.7 m and 338.3 m.  The
#: measured relation is not to the request alone: ``min dvEdge = min over
#: edges of (dv/dc x dcEdge)``, the ratio lands on the surgery flag floor
#: every time (0.0401 to 0.0427 over all six rows), and the ``dcEdge``
#: carrying it runs 1.2x to 2.7x the request.  So the ratio to the request is
#: a BAND, it narrows with nothing a spec knows, and it is reported as a band.
MEASURED_GRADED_MESHES: tuple[tuple[str, float, float, float, float, bool], ...] = (
    ("four-swaths-20260827 s01", 6.0, 75.0, 647.4565715154226, 0.040126487532503106, True),
    ("four-swaths-20260827 s02", 4.0, 75.0, 342.65141335832936, 0.04266799919812452, True),
    ("four-swaths-20260827 s03", 6.0, 75.0, 655.0911983085592, 0.042660408759148095, True),
    ("four-swaths-20260827 s04", 6.0, 75.0, 432.167852434209, 0.04092938767136717, True),
    ("swath-real-cascade-20260826 s01", 4.0, 75.0, 338.2686803998687, 0.04163822685616862, True),
    # This lane, 2026-08-28, rw_mpas_mesh 80178b69, 1,251 s to the refusal:
    # 217,621 cells derived, then edge 562175 refused at 36.8 m over an
    # 886 m dcEdge.  Quoted at the refusal's own printed precision because
    # a refused run writes no receipt to read a full float out of.
    ("meshgate-20260828 nmtx-ii", 0.75, 75.0, 36.8, 0.0416, False),
)

#: ``min dvEdge / finest requested spacing`` over that table, low and high.
#: A point estimate inside it would be a precision this does not have: the
#: low end is the 0.75 km seven-level run and the high end a 6 km four-level
#: one, and the spread is the ladder depth, which is why the band is wide
#: and why a spec between the two ends gets ``undecided`` rather than a
#: guess dressed as an answer.
DV_OVER_FINEST_BAND = (
    min(dv / (finest * 1000.0) for _, finest, _, dv, _, _ in MEASURED_GRADED_MESHES),
    max(dv / (finest * 1000.0) for _, finest, _, dv, _, _ in MEASURED_GRADED_MESHES),
)

#: The published variable-resolution mesh, for scale: 1.53 % per cell.
PUBLISHED_REFERENCE_GRADIENT_PERCENT = 1.53


class MeshSpecRefusal(MpasPortError):
    """A resolution spec that ``rw_mpas_mesh`` would refuse at GENERATION.

    Raised by sizing, from the spec and the dry-run receipt, so the refusal
    arrives before the build rather than after it.  The message carries the
    measured quantity, the limit, where the limit lives, and a spec change
    with numbers in it.
    """


# ---------------------------------------------------------------------------
# the transition-band gate: exact, cheap, and decided by the spec alone
# ---------------------------------------------------------------------------
def transition_band_cells(gradient_per_cell: float) -> float:
    """Cells across a doubling of spacing at a constant per-cell gradient.

    ``h`` grows by ``(1 + g)`` per cell, so a doubling takes
    ``ln 2 / ln(1 + g)`` cells.  The same expression ``generate_graded``
    evaluates before it seeds anything, spelled the same way: ``log(1 + g)``
    and not the more accurate ``log1p(g)``, because this function's job is to
    agree with the gate rather than to be right about the mathematics.

    WHERE THE TWO CAN STILL DISAGREE, STATED.  The gate reads its gradient as
    a fraction; a receipt prints that fraction times 100 and this reads it
    back divided by 100, which is a round trip that need not be exact.  The
    disagreement is at the last bit, so a spec within about 1e-15 of the
    limit can be judged differently by the two ARITHMETICS.

    AND WHY THAT SENTENCE USED TO BE WORTH NOTHING.  Agreement between the two
    gates was never evidence that either was right, because both read ONE
    number and neither measured it: this function is handed the gradient off
    the receipt, and until 2026-08-29 the generator sampled that gradient on a
    Fibonacci lattice uniform over the whole sphere.  At the 50,000 points it
    used, those sit 101 km apart, so any refinement transition narrower than
    that was stepped over and the receipt reported the flat background the
    lattice happened to land on.  MEASURED on the eight-rung 51.2 -> 0.2 km
    ladder: 12.1155 % per cell printed and agreed on to the last bit by both
    gates, for a field the corrected instrument reads at 37.0355 -- band 6.06
    against a floor of 6.0, so ADMITTED with one percent of margin, when the
    truth is a band of 2.2.  Two instruments sharing one blind spot read as
    corroboration, which is exactly what made it survive to two published
    releases.  The generator now probes where the spec says its regions are
    and stamps a coverage word beside the number; :func:`gates_from_receipt`
    refuses a receipt that does not carry one, because a stale engine would
    otherwise keep the old trust silently.
    """

    if not math.isfinite(gradient_per_cell) or gradient_per_cell <= 0.0:
        return math.inf
    return math.log(2.0) / math.log(1.0 + gradient_per_cell)


def finest_spacing_km(spec: Mapping[str, Any]) -> float:
    """``MeshSpec::finest_km``: the background, lowered by every region."""

    finest = float(spec["background_km"])
    for region in spec.get("regions") or ():
        finest = min(finest, float(region["spacing_km"]))
    return finest


def ladder_km(spec: Mapping[str, Any]) -> list[float]:
    """``mesh/hierarchy.rs::ladder``: the level spacings the build walks."""

    background = float(spec["background_km"])
    finest = finest_spacing_km(spec)
    if finest >= background:
        return [background]
    levels = math.ceil(math.log2(background / finest))
    return [max(background * 0.5 ** level, finest) for level in range(levels + 1)]


def scaled_transitions(spec: Mapping[str, Any], factor: float) -> dict[str, Any]:
    """The same spec with every region's ramp widened by ``factor``.

    Whichever spelling a region uses is the one that moves: ``transition_km``
    and ``transition_cells`` are the two the generator accepts, and a spec is
    written with one or the other per region.
    """

    widened = copy.deepcopy(dict(spec))
    for region in widened.get("regions") or ():
        for key in ("transition_km", "transition_cells"):
            if key in region:
                region[key] = float(region[key]) * factor
    return widened


def _transition_row(region: Mapping[str, Any], factor: float) -> str:
    for key in ("transition_km", "transition_cells"):
        if key in region:
            value = float(region[key])
            return f"{key} {value:g} -> {value * factor:g}"
    return "no transition field to widen"


def widening_that_clears(
    spec: Mapping[str, Any],
    gradient_per_cell: float,
    measure,
    *,
    attempts: int = 4,
) -> tuple[float, float] | None:
    """The smallest tried ramp widening whose gradient clears the gate.

    ``measure`` takes a spec and returns its steepest gradient as a fraction
    -- in production the generator's own ``--dry-run``, so the remedy is
    quoted from the instrument that will judge it, not from a model of it.

    WHY IT IS A SEARCH AND NOT A DIVISION.  This used to say the gradient is
    not monotone in the ramp width, and cited ``transition_cells``
    40/44/46/48 reading 21.45 / 23.93 / 21.51 / 24.82 % per cell on one
    0.75/6/75 km spec.  That non-monotonicity was SAMPLING NOISE, not a
    property of the field: the reading was a maximum over a Fibonacci lattice
    uniform over the whole sphere, and widening a ramp moved the field under a
    fixed point set rather than refining it.  RE-MEASURED 2026-08-29 on the
    corrected instrument, the same spec at ``transition_cells``
    40/42/44/46/48/50/60/80/108/160/240 reads 24.96 / 24.19 / 23.38 / 23.11 /
    21.94 / 21.67 / 19.56 / 16.15 / 13.52 / 10.20 / 7.60 -- monotone at every
    step, where the old instrument read 14.65 / 21.85 / 19.86 / 18.89 / 19.52
    / 17.35 / 10.64 / 14.17 / 9.55 / 7.48 / 7.31 and rose on five of them.

    IT IS STILL A SEARCH, and now for a reason that is about the FIELD.  One
    isolated ramp's peak gradient is exactly ``(h_bg - h_i) / (2 W)``, so
    widening it once would settle the remedy by division -- but a nested
    ladder is not one ramp.  Each rung's real neighbour is the next rung, and
    widening every ramp together changes which branch is steepest, so the
    composite falls SLOWER than ``1 / W``: over the sweep above, the value
    ``1/W`` would predict climbs from 24.96 to 45.59 % per cell.  A division
    would under-widen and quote a remedy that does not clear.  So ``g / g_max``
    stays the opening bid, each attempt widens it by half again, and the first
    that MEASURES clear is the one quoted.  ``None`` means no attempt cleared,
    which is reported as such rather than guessed past.

    WHAT THE OLD INSTRUMENT DID TO THIS FUNCTION, so the retirement is not
    mistaken for tidying: measured on the same spec, the search used to quote
    a widening whose own true gradient was still over the ceiling, and the
    build then ADMITTED it because the build under-read it identically. The
    remedy the door printed closed the loop on itself.
    """

    if gradient_per_cell <= 0.0:
        return None
    factor = gradient_per_cell / MAX_GRADIENT_PER_CELL
    for _ in range(max(1, attempts)):
        factor = math.ceil(factor * 10.0) / 10.0
        widened = measure(scaled_transitions(spec, factor))
        if widened <= MAX_GRADIENT_PER_CELL:
            return factor, widened
        factor *= 1.5
    return None


def transition_band_refusal(
    spec: Mapping[str, Any],
    gradient_per_cell: float,
    *,
    remedy: tuple[float, float] | None = None,
) -> str:
    """The refusal text: measured value, limit, where it lives, what to change."""

    band = transition_band_cells(gradient_per_cell)
    name = spec.get("name") or "this resolution spec"
    lines = [
        f"{name} is refused before anything is built: its steepest requested "
        f"spacing gradient packs each refinement level's transition band into "
        f"{band:.1f} cells. rw_mpas_mesh repairs a dislocation by editing a "
        f"{SURGERY_LOCALITY_CELLS:.0f}-cell neighbourhood, so a band under "
        f"{MIN_TRANSITION_BAND_CELLS:.0f} cells cannot contain its own repairs "
        "-- one repair at the band's centre reaches both of its edges -- and "
        "the build refuses for that reason after the spec has been authored, "
        "sized and accepted.",
        f"  measured   {gradient_per_cell * 100.0:.4f} % per cell "
        f"({band:.2f} cells across a doubling of spacing)",
        f"  limit      {MAX_GRADIENT_PER_CELL * 100.0:.4f} % per cell "
        f"({MIN_TRANSITION_BAND_CELLS:.0f} cells; the published "
        f"variable-resolution mesh runs {PUBLISHED_REFERENCE_GRADIENT_PERCENT} %)",
        "  gate       rw-mpas mesh/hierarchy.rs::generate_graded, pre-run "
        "arithmetic on the same number the dry-run receipt prints as "
        "steepest_requested_gradient_percent_per_cell",
    ]
    if remedy is not None:
        factor, widened = remedy
        lines.append(
            f"  change     widen every region's ramp by {factor:g}x and this "
            f"spec measures {widened * 100.0:.4f} % per cell "
            f"({transition_band_cells(widened):.2f} cells), which clears:"
        )
        for index, region in enumerate(spec.get("regions") or ()):
            lines.append(
                f"               region {index} "
                f"({float(region['spacing_km']):g} km): "
                f"{_transition_row(region, factor)}"
            )
        lines.append(
            "             Coarsening the finest spacing or lowering the "
            "background works as well: both shrink the ratio the ramp has to "
            "cross, and the gradient is that ratio divided by the ramp."
        )
    else:
        lines.append(
            "  change     widen every region's ramp. Four widenings, each "
            "measured through the generator's own dry-run, all still read "
            "over the limit, so the ramp is not the only thing to move: "
            "coarsen the finest spacing or lower the background as well."
        )
    return "\n".join(lines)


def check_transition_band(
    spec: Mapping[str, Any],
    gradient_per_cell: float,
    *,
    measure=None,
) -> None:
    """Refuse a spec the build's own pre-run arithmetic would refuse."""

    if transition_band_cells(gradient_per_cell) >= MIN_TRANSITION_BAND_CELLS:
        return
    remedy = (
        widening_that_clears(spec, gradient_per_cell, measure)
        if measure is not None
        else None
    )
    raise MeshSpecRefusal(
        transition_band_refusal(spec, gradient_per_cell, remedy=remedy)
    )


# ---------------------------------------------------------------------------
# the short dual edge floor: what sizing CANNOT decide, said with numbers
# ---------------------------------------------------------------------------
def short_dual_edge_exposure(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What the dual-edge floor demands of this spec, and what decides it.

    THE FLOOR THIS ANSWERS ABOUT IS THE GENERATED-MESH ONE, 3.725e-7 m --
    400 coordinate quanta of the binary64 representation ``rw_mpas_mesh``
    stores a generated mesh at (:data:`MIN_DV_EDGE_M`), not the 200.0 m a
    mesh with a native MPAS-A counterpart is judged by
    (:data:`PUBLISHED_MIN_DV_EDGE_M`).  Both are the same budget; they differ
    only by the quantum of the file the mesh lands in.  Every verdict below is
    therefore ``cannot_bind`` for any spec a person would write, and that is
    the answer, not a broken instrument: the arithmetic is kept because it is
    what SHOWS the floor cannot bind, and because it moves back on its own if
    a representation ever moves back.

    NOT A GATE, AND THAT IS THE POINT.  ``min_dv_edge_m`` reads the shortest
    dual edge of the FINISHED tessellation.  A spec fixes the spacing field;
    it does not fix where the pentagon-heptagon dislocations land or how far
    their two Voronoi vertices anneal apart, and those are what the floor
    measures.  Refusing a spec here on a prediction would refuse meshes the
    generator would have emitted -- the same failure this lane exists to
    close, pointed the other way.

    What IS decidable, and is computed here:

    * the floor is a LENGTH and the mesh's own guard is a RATIO, and the two
      meet at ``dcEdge``.  At a finest spacing ``s`` the 200 m floor demands
      ``dvEdge/dcEdge >= 200 / s`` of the finest edges.  Below
      ``s = 200 / 0.02 = 10 km`` that demand is STRICTER than the 0.02 ratio
      floor the generator and this port both admit on, by ``10 km / s``.
    * a PREDICTION BAND, from :data:`MEASURED_GRADED_MESHES`.  Surgery flags
      a quad for repair only below ``dv/dc`` 0.04 and stops there, so every
      graded mesh in that table lands on the flag floor (0.0401 to 0.0427)
      and its shortest dual edge is that ratio times the ``dcEdge`` carrying
      it, which is 1.2x to 2.7x the request.  The band of ``min dvEdge /
      finest requested`` is what gets quoted.  It never refuses: a wrong
      prediction in the strict direction would talk a user out of a spec
      that builds, which a first draft of this function did to every shipped
      4 km swath.
    * a one-sided immunity: if even the fine tail of the delivery
      distribution keeps ``0.02 * dcEdge >= 200 m``, then no edge can fail
      the length floor without already failing the ratio floor, so the
      length floor cannot be the binding refusal.
    * the earliest ladder level at which an offending edge can exist at all.
      Every level finishes surgery at or above 0.03, so a level whose
      spacing ``h`` satisfies ``0.03 * h >= 200 m`` is provably clear.  The
      first level below ``200 / 0.03 = 6.67 km`` is the first that can carry
      a refusal, and no amount of building before it can produce an answer.
    """

    finest_km = finest_spacing_km(spec)
    finest_m = finest_km * 1000.0
    required_ratio = MIN_DV_EDGE_M / finest_m if finest_m > 0.0 else math.inf
    immune_km = MIN_DV_EDGE_M / (MIN_DV_OVER_DC * DELIVERED_SPACING_P05) / 1000.0
    reachable_km = MIN_DV_EDGE_M / LEVEL_SHIP_FLOOR_DV_OVER_DC / 1000.0
    low_ratio, high_ratio = DV_OVER_FINEST_BAND
    predicted_low_m = low_ratio * finest_m
    predicted_high_m = high_ratio * finest_m
    clears_km = MIN_DV_EDGE_M / low_ratio / 1000.0
    refuses_below_km = MIN_DV_EDGE_M / high_ratio / 1000.0
    steps = ladder_km(spec)
    first_exposed = next(
        (index for index, h in enumerate(steps) if h < reachable_km), None
    )
    if finest_km >= immune_km:
        verdict = "cannot_bind"
    elif predicted_low_m >= MIN_DV_EDGE_M:
        verdict = "predicted_clear"
    elif predicted_high_m < MIN_DV_EDGE_M:
        verdict = "predicted_refusal"
    else:
        verdict = "undecided"
    if first_exposed is None:
        earliest = (
            "not reachable: every ladder level of this spec is coarser than "
            f"{reachable_km:.2f} km, so the generator's own "
            f"{LEVEL_SHIP_FLOOR_DV_OVER_DC} per-level ship floor already holds "
            f"every dual edge over {MIN_DV_EDGE_M:.0f} m by construction"
        )
    else:
        earliest = (
            f"ladder level {first_exposed} of {len(steps) - 1} "
            f"(h = {steps[first_exposed]:.2f} km). Levels 0 to "
            f"{first_exposed - 1} are provably clear -- the "
            f"{LEVEL_SHIP_FLOOR_DV_OVER_DC} per-level ship floor puts their "
            f"shortest dual edge over {MIN_DV_EDGE_M:.0f} m by construction -- "
            "so no build can answer this question earlier than that level, "
            "whatever it is asked"
        )
    return {
        "limit_m": MIN_DV_EDGE_M,
        "limit_quanta": DV_EDGE_FLOOR_QUANTA,
        "coordinate_representation": GENERATED_MESH_REPRESENTATION,
        "coordinate_quantum_m": COORDINATE_QUANTUM_M[GENERATED_MESH_REPRESENTATION],
        "limit_m_for_a_mesh_with_a_native_counterpart": PUBLISHED_MIN_DV_EDGE_M,
        "gate": "rw-mpas mesh/validate.rs, Limits::min_dv_edge_m",
        "decidable_before_the_build": False,
        "why_not": (
            "the floor reads the shortest dual edge of the finished "
            "tessellation. A spec fixes the spacing field; it does not fix "
            "where the pentagon-heptagon dislocations land or how far their "
            "two Voronoi vertices anneal apart, and that is what this floor "
            "measures. Nothing short of building the ladder decides it"
        ),
        "finest_requested_spacing_km": finest_km,
        "dv_over_dc_the_floor_demands_at_the_finest_edges": required_ratio,
        "ratio_floor_the_generator_and_the_port_both_admit_on": MIN_DV_OVER_DC,
        "how_much_stricter_the_length_floor_is": required_ratio / MIN_DV_OVER_DC,
        "level_ship_floor_the_generator_promises_while_building":
            LEVEL_SHIP_FLOOR_DV_OVER_DC,
        "surgery_flag_floor_dv_over_dc": SURGERY_FLAG_FLOOR_DV_OVER_DC,
        "predicted_shortest_dual_edge_m": [predicted_low_m, predicted_high_m],
        "prediction_basis": (
            "surgery flags a quad for repair only below dv/dc "
            f"{SURGERY_FLAG_FLOOR_DV_OVER_DC} (rw-mpas mesh/surgery.rs, "
            "SurgeryOptions::default) and stops there, so every graded mesh "
            "measured in this tree lands on that flag floor (0.0401 to "
            "0.0427) and its shortest dual edge is that ratio times the "
            "dcEdge carrying it, which runs 1.2x to 2.7x the request. The "
            "measured band of min dvEdge over finest request is therefore "
            f"{low_ratio:.4f} to {high_ratio:.4f} over "
            f"{len(MEASURED_GRADED_MESHES)} meshes at 0.75, 4 and 6 km "
            "finest -- the low end a seven-level ladder, the high end a "
            "four-level one, so the spread IS the ladder depth. A "
            "PREDICTION, never a refusal: the strict direction would talk a "
            "user out of a spec that builds"
        ),
        "prediction_samples": len(MEASURED_GRADED_MESHES),
        "measured_samples": [
            {
                "label": label,
                "finest_km": finest,
                "background_km": background,
                "min_dv_edge_m": dv,
                "min_dv_over_dc": ratio,
                "dv_over_finest": dv / (finest * 1000.0),
                "emitted": emitted,
            }
            for label, finest, background, dv, ratio, emitted
            in MEASURED_GRADED_MESHES
        ],
        "finest_spacing_km_the_prediction_clears": clears_km,
        "finest_spacing_km_below_which_no_sample_would_clear": refuses_below_km,
        "published_family_worst_dv_over_dc": PUBLISHED_WORST_DV_OVER_DC,
        "finest_spacing_km_at_which_this_floor_cannot_bind": immune_km,
        "verdict": verdict,
        "ladder_km": steps,
        "ladder_levels": len(steps) - 1,
        "first_ladder_level_that_can_carry_a_refusal": first_exposed,
        "earliest_this_can_be_known": earliest,
        "if_it_refuses_change": (
            "the finest spacing is the only spec knob this floor reads: it is "
            f"an ABSOLUTE length, so a finest spacing at or above "
            f"{clears_km:.2f} km clears it on every measured sample, one "
            f"below {refuses_below_km:.2f} km clears it on none, and one at "
            f"or above {immune_km:.2f} km puts the question out of reach "
            "entirely. Between those the remedies are the measured ones -- a "
            "different refinement layout, or fewer cells in the refined "
            "region. More relaxation re-rolls the dislocation rather than "
            "draining it"
        ),
    }


# ---------------------------------------------------------------------------
# the one call a sizing path makes
# ---------------------------------------------------------------------------
def gates_from_receipt(
    spec: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    measure=None,
) -> dict[str, Any]:
    """Apply what a spec decides; report what only a build can decide.

    Raises :class:`MeshSpecRefusal` for the transition-band gate, which is
    exact and cheap.  Returns the document a receipt carries: what was
    checked, and -- named, with its numbers -- what was not.
    """

    gradient_percent = receipt.get("steepest_requested_gradient_percent_per_cell")
    if gradient_percent is None:
        raise MeshSpecRefusal(
            "the dry-run receipt carries no "
            "steepest_requested_gradient_percent_per_cell, so the "
            "transition-band gate the build applies cannot be applied here. A "
            "sizing path that silently skips a gate is how a spec gets sized, "
            "accepted and then refused by the build, which is the defect this "
            "check exists for. Stage an rw_mpas_mesh that prints it"
        )
    coverage = receipt.get("gradient_probe_coverage")
    if coverage != "complete":
        raise MeshSpecRefusal(
            "the dry-run receipt reports its steepest-gradient probe coverage "
            f"as {coverage!r}, so the number beside it is not a measurement of "
            "this spec's field and this gate will not judge a spec on it. THE "
            "BREAKAGE THIS PREVENTS: an engine that samples the gradient on a "
            "lattice uniform over the whole sphere cannot see a refinement "
            "transition narrower than its own point spacing -- at the 50,000 "
            "points it used, 101 km -- and reports the flat background it "
            "landed on instead. Measured on an eight-rung 51.2 -> 0.2 km "
            "ladder, such an engine printed 12.1155 % per cell for a field "
            "that reads 37.0355, and this gate admitted it with one percent of "
            "margin. A missing coverage word means an engine from before "
            "2026-08-29; 'partial' means the probe budget could not cover the "
            "spec's own transition shells, and an unmeasured ramp is not a "
            "gentle one. Stage an rw_mpas_mesh that measures the gradient "
            "where the regions are"
        )
    gradient = float(gradient_percent) / 100.0
    check_transition_band(spec, gradient, measure=measure)
    return {
        "transition_band": {
            "gate": "rw-mpas mesh/hierarchy.rs::generate_graded",
            "decidable_before_the_build": True,
            "applied": True,
            "steepest_gradient_percent_per_cell": gradient * 100.0,
            "band_cells": transition_band_cells(gradient),
            "band_cells_floor": MIN_TRANSITION_BAND_CELLS,
            "gradient_percent_per_cell_ceiling": MAX_GRADIENT_PER_CELL * 100.0,
            # WHERE the number was measured, carried so a reader can tell a
            # measurement from a lattice that missed the ramp.  The two other
            # numbers on the same receipt -- predicted_cells and
            # region_attainment -- are still integrals over a global lattice
            # and do NOT follow the regions.
            "gradient_probe_coverage": coverage,
            "gradient_probe_points": receipt.get("gradient_probe_points"),
        },
        "short_dual_edge_floor": short_dual_edge_exposure(spec),
        "gates_this_sizing_cannot_apply": ["short_dual_edge_floor"],
    }


__all__ = [
    "DELIVERED_SPACING_P05",
    "DV_OVER_FINEST_BAND",
    "LEVEL_SHIP_FLOOR_DV_OVER_DC",
    "MAX_GRADIENT_PER_CELL",
    "MEASURED_GRADED_MESHES",
    "MIN_DV_EDGE_M",
    "MIN_DV_OVER_DC",
    "MIN_TRANSITION_BAND_CELLS",
    "MeshSpecRefusal",
    "PUBLISHED_REFERENCE_GRADIENT_PERCENT",
    "PUBLISHED_WORST_DV_OVER_DC",
    "SURGERY_FLAG_FLOOR_DV_OVER_DC",
    "SURGERY_LOCALITY_CELLS",
    "check_transition_band",
    "finest_spacing_km",
    "gates_from_receipt",
    "ladder_km",
    "scaled_transitions",
    "short_dual_edge_exposure",
    "transition_band_cells",
    "transition_band_refusal",
    "widening_that_clears",
]
