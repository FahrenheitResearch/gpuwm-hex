"""``gpuwm-hex mesh-plan --point``: a fine core at an arbitrary point.

WHAT THIS DOOR DOES.  Every registered sub-kilometre row this program
holds sits at 35.0 N 97.0 W, because a person typed its spec and its row.
This door takes a POINT, a fine spacing and a core radius, and does what
that person did, in order and with receipts:

1. writes the resolution spec -- the registered ``v0.9.120.110533`` recipe
   generalised: nested caps on one global background, each rung halving the
   spacing, each ramp a fixed multiple of its own spacing, each cap sitting
   a fixed number of ramps outside the next finer one -- centred on the
   point;
2. prices it through the generator's own ``--dry-run`` with the gates its
   build applies (:mod:`hexcore.swath.sizing`), predicts what the limited-
   area CULL will cost, and prices THAT against the named card on the one
   admission surface (:mod:`hexcore.device_admission`, the limited-area
   row), with the margin that surface names;
3. with ``--generate``: builds the pair (``rw_mpas_mesh``, then
   ``rw_mpas_static``), measures the pair's own admissions (dual edge, cell
   coordination, Courant at the declared timestep), registers it as a
   RUNTIME ROW (:mod:`hexcore.mesh_rows`, a file beside the mesh, never a
   checkout edit), cuts the limited-area cull at the shipped pad and
   registers that too; and, with ``--vertical-spec``, mints the parent's
   native-free vertical artifact and culls it so the child can be
   initialised from a regional meteorological source.

WHY THE LADDER IS DATA.  The spec is a pure function of six numbers
(:func:`ladder_spec`); nothing here branches on a place.  Moving the core
is moving the centre of every cap.  The transition and ring factors are
the registered row's own (18 and 3) and are arguments, not constants a
reader has to trust.

WHAT IS PRICED, AND WHAT IS NOT.  The generator prices the whole GRADED
GLOBAL mesh; only the cull reaches a card.  The plan therefore quotes both
and admits on the second, exactly as the swath layer does
(:mod:`hexcore.swath.sizing` states why).  The cull prediction is an area
integral at the generator's own attained spacing -- a bound, labelled
``basis: area_integral`` -- and the cull receipt replaces it with the count.

THE INITIAL CONDITION ROAD THIS DOOR PREPARES.  ``gpuwm-hex init`` refuses
a limited-area grid in its native-free mode: the closed-sphere vertical
authority does not invent exterior state (``hexcore.vertical``).  A global
init needs global meteorology, and a regional source (HRRR) covers only
its own domain.  The road that satisfies both is: mint the vertical on the
GLOBAL parent (grid + static + spec, no meteorology), cull the artifact
with the same region as the grid and static, and hand the culled artifact
to ``gpuwm-hex init --capsule/--reference`` with the regional intermediate.
The cull moves no cell centre, so the child's vertical IS the parent's,
cell for cell.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

#: The registered 937.5 m row's recipe, restated as the six numbers it is.
DEFAULT_BACKGROUND_KM = 120.0
#: Each rung's ramp is this many times its own spacing (the registered row:
#: "each ramp eighteen times its own target").
DEFAULT_TRANSITION_FACTOR = 18.0
#: Each cap sits this many ramps of the next finer rung outside it (the
#: registered row: 100 + 3 x 16.875 = 150.625 km, and so on up).
DEFAULT_RING_FACTOR = 3.0
#: The measured domain-size knee, and the shipped cull pad
#: (``evidence/nest-ratio-20260827/``: every field improves monotonically
#: with cut width and the knee is at 1.35x the fine core).
DEFAULT_CULL_PAD_SCALE = 1.35
#: Column count every measured row and every anchor in this tree runs at.
DEFAULT_LEVELS = 55
#: The generator's own hexagon area factor (rw-mpas density.rs).
HEXAGON_AREA_FACTOR = math.sqrt(3.0) / 2.0
EARTH_RADIUS_KM = 6371.229

#: ``--card`` spellings, mapped onto the two tables that price a mesh: the
#: admission surface's card profile (``hexcore.device_admission``) and the
#: generator's own measured part key (``rw_mpas_mesh --card``), plus the
#: nameplate memory a plan compares against when no card is in the room.
#: A card is a row here; a card with no generator row is priced by the
#: admission surface alone and the receipt says so.
CARD_ALIASES: Mapping[str, tuple[str, str | None, int]] = {
    "32gb": ("32gib-170sm", "rtx-5090", 32_607),
    "32gib": ("32gib-170sm", "rtx-5090", 32_607),
    "rtx-5090": ("32gib-170sm", "rtx-5090", 32_607),
    "32gib-170sm": ("32gib-170sm", "rtx-5090", 32_607),
    "16gb": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "16gib": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "rtx-5070-ti": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "16gib-70sm": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "10gb": ("10gib-68sm", None, 10_240),
    "10gib": ("10gib-68sm", None, 10_240),
    "rtx-3080": ("10gib-68sm", None, 10_240),
    "10gib-68sm": ("10gib-68sm", None, 10_240),
}


class PointPlanRefusal(MpasPortError):
    """A point plan is refused, and the message names what would break."""


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------
def parse_point(text: str) -> tuple[float, float]:
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) != 2:
        raise PointPlanRefusal(
            f"--point {text!r} is not LAT,LON; a fine core needs one centre "
            f"and nothing here guesses one"
        )
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError as error:
        raise PointPlanRefusal(f"--point {text!r} is not two numbers: {error}") from error
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 360.0:
        raise PointPlanRefusal(
            f"--point {text!r} is outside the sphere (latitude -90..90, "
            f"longitude -180..360)"
        )
    if lon > 180.0:
        lon -= 360.0
    return lat, lon


def ladder_rungs(background_km: float, fine_dx_m: float) -> list[float]:
    """The halving ladder from the background down to the fine spacing, km.

    Refuses a fine spacing that is not on the ladder, naming the rungs on
    either side: the generator snaps a request onto the nearest rung
    ALWAYS FINER (rw-mpas ``mesh::ladder_snap``), so a 900 m request would
    silently build 468.75 m and cost four times the cells the card was
    priced for.  Choosing the rung is the caller's decision.
    """

    if not background_km > 0.0:
        raise PointPlanRefusal(f"--background-km {background_km} is not positive")
    fine_km = float(fine_dx_m) / 1000.0
    if not fine_km > 0.0:
        raise PointPlanRefusal(f"--fine-dx-m {fine_dx_m} is not positive")
    if fine_km >= background_km:
        raise PointPlanRefusal(
            f"--fine-dx-m {fine_dx_m:g} m is not finer than the "
            f"{background_km:g} km background, so there is nothing to refine"
        )
    ratio = background_km / fine_km
    k = round(math.log2(ratio))
    if k < 1 or abs(background_km / (2.0 ** k) - fine_km) > 1e-9 * background_km:
        below = background_km / (2.0 ** math.floor(math.log2(ratio)))
        above = background_km / (2.0 ** math.ceil(math.log2(ratio)))
        raise PointPlanRefusal(
            f"--fine-dx-m {fine_dx_m:g} m is not a rung of the "
            f"{background_km:g} km halving ladder.  The graded generator "
            f"refines by midpoint insertion, so a refined core can only land "
            f"on background / 2^k, and rw_mpas_mesh snaps a request onto the "
            f"nearest such rung ALWAYS FINER -- {fine_dx_m:g} m would build "
            f"{above * 1000.0:g} m and cost about {(below / above) ** 2:.0f}x "
            f"the cells this plan would price.  The rungs on either side are "
            f"{below * 1000.0:g} m and {above * 1000.0:g} m; pass one of them, "
            f"or change --background-km"
        )
    return [background_km / (2.0 ** level) for level in range(1, k + 1)]


def ladder_spec(
    point: tuple[float, float],
    *,
    fine_dx_m: float,
    radius_km: float,
    background_km: float = DEFAULT_BACKGROUND_KM,
    transition_factor: float = DEFAULT_TRANSITION_FACTOR,
    ring_factor: float = DEFAULT_RING_FACTOR,
    name: str | None = None,
) -> dict[str, Any]:
    """The nested-cap resolution spec ``rw_mpas_mesh --spec`` reads unchanged.

    The registered ``v0.9.120.110533`` spec is ``ladder_spec((35.0, -97.0),
    fine_dx_m=937.5, radius_km=100.0)`` to the digit (tested).
    """

    if not radius_km > 0.0:
        raise PointPlanRefusal(f"--radius-km {radius_km} is not positive")
    if transition_factor <= 0.0 or ring_factor <= 0.0:
        raise PointPlanRefusal(
            "the transition and ring factors must be positive; a zero ramp "
            "is a step the surgery-locality gate refuses"
        )
    rungs = ladder_rungs(background_km, fine_dx_m)
    lat, lon = point
    regions: list[dict[str, Any]] = []
    radius = float(radius_km)
    for spacing in reversed(rungs):
        transition = transition_factor * spacing
        regions.append(
            {
                "shape": {"kind": "cap", "center_deg": [lat, lon], "radius_km": radius},
                "spacing_km": spacing,
                "transition_km": transition,
            }
        )
        radius = radius + ring_factor * transition
    regions.reverse()
    label = name or (
        f"point {lat:.4f}N {lon:.4f}E {fine_dx_m:g}m core {radius_km:g}km"
    )
    return {"name": label, "background_km": float(background_km), "regions": regions}


def cull_region(point: tuple[float, float], radius_km: float, pad_scale: float) -> dict[str, Any]:
    """The cap ``rw_mpas_mesh --cull-parent --region`` is handed."""

    if pad_scale < 1.0:
        raise PointPlanRefusal(
            f"--cull-pad-scale {pad_scale} is below 1.0: a cut inside the fine "
            f"core hands the boundary rings fine cells the parent's ramp "
            f"never reaches, which is the seam the pad exists to move outward"
        )
    lat, lon = point
    return {
        "kind": "cap",
        "center_deg": [lat, lon],
        "radius_km": float(radius_km) * float(pad_scale),
    }


def row_token(point: tuple[float, float]) -> str:
    lat, lon = point
    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"
    return f"{ns}{abs(lat):.2f}{ew}{abs(lon):.2f}"


def parent_row_name(point: tuple[float, float], fine_dx_m: float, background_km: float, cells: int) -> str:
    return f"p{fine_dx_m / 1000.0:g}.{background_km:g}.{cells}.{row_token(point)}"


def cull_row_name(point: tuple[float, float], fine_dx_m: float, background_km: float, cells: int) -> str:
    return f"q{fine_dx_m / 1000.0:g}.{background_km:g}.{cells}.{row_token(point)}"


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def cap_area_km2(radius_km: float) -> float:
    return 2.0 * math.pi * EARTH_RADIUS_KM ** 2 * (1.0 - math.cos(radius_km / EARTH_RADIUS_KM))


def spacing_profile(spec: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    """``(cap radius, spacing, transition)`` per cap region, innermost first.

    The generator's density puts a region's spacing INSIDE ``radius -
    transition`` and ramps it to the next coarser rung across the last
    ``transition`` of the cap (measured against the registered
    ``r0.9.120.40520``: "the whole 937.5 m core plus the first 25 km of the
    1.875 km ramp" is 40,520 cells, and this profile predicts it to 2 %).
    """

    rows: list[tuple[float, float, float]] = []
    for region in spec.get("regions", ()):
        shape = region.get("shape") or {}
        if shape.get("kind") != "cap":
            continue
        spacing = float(region["spacing_km"])
        transition = float(
            region.get("transition_km")
            if region.get("transition_km") is not None
            else float(region.get("transition_cells", 0.0)) * spacing
        )
        rows.append((float(shape["radius_km"]), spacing, transition))
    rows.sort()
    return rows


def spacing_at(profile: Sequence[tuple[float, float, float]], r_km: float,
               background_km: float) -> float:
    """The spacing the ladder asks for at ``r_km`` from the centre."""

    for index, (radius, spacing, transition) in enumerate(profile):
        if r_km > radius:
            continue
        flat_edge = radius - transition
        if r_km <= flat_edge:
            return spacing
        coarser = profile[index + 1][1] if index + 1 < len(profile) else background_km
        return spacing + (coarser - spacing) * (r_km - flat_edge) / max(transition, 1e-9)
    return background_km


def predicted_cull_cells(
    *,
    spec: Mapping[str, Any],
    radius_km: float,
    pad_scale: float,
    attained_fine_km: float | None = None,
    boundary_rings: int = 7,
    steps: int = 600,
) -> dict[str, Any]:
    """Cells the cull holds, integrated over the ladder's own spacing profile.

    ``basis: area_integral``.  The innermost rung's spacing is replaced by
    the generator's ATTAINED spacing when the plan measured one, so the
    count is a bound at what the mesh delivers rather than at the request.
    The seven boundary rings are integrated too: the disc is dilated by
    their width at the cut's own spacing.
    """

    profile = spacing_profile(spec)
    if not profile:
        raise PointPlanRefusal("the spec carries no cap region to price a cull of")
    background = float(spec.get("background_km", DEFAULT_BACKGROUND_KM))
    if attained_fine_km is not None:
        radius0, _, transition0 = profile[0]
        profile[0] = (radius0, float(attained_fine_km), transition0)
    cut = float(radius_km) * float(pad_scale)
    halo = boundary_rings * spacing_at(profile, cut, background)
    outer = cut + halo
    cells = 0.0
    core_cells = 0.0
    edges = [outer * i / steps for i in range(steps + 1)]
    for inner_r, outer_r in zip(edges, edges[1:]):
        mid = 0.5 * (inner_r + outer_r)
        area = cap_area_km2(outer_r) - cap_area_km2(inner_r)
        count = area / (HEXAGON_AREA_FACTOR * spacing_at(profile, mid, background) ** 2)
        cells += count
        if outer_r <= radius_km + 1e-9:
            core_cells += count
    return {
        "basis": "area_integral",
        "predicted_cells": round(cells, 1),
        "core_cells": round(core_cells, 1),
        "core_radius_km": float(radius_km),
        "fine_flat_radius_km": profile[0][0] - profile[0][2],
        "cut_radius_km": cut,
        "spacing_at_cut_km": spacing_at(profile, cut, background),
        "boundary_rings": boundary_rings,
        "halo_km": halo,
    }


def resolve_card(key: str | None) -> dict[str, Any] | None:
    """A ``--card`` spelling resolved onto the two pricing tables."""

    if key is None:
        return None
    normalized = str(key).strip().lower()
    if normalized not in CARD_ALIASES:
        raise PointPlanRefusal(
            f"--card {key!r} is not a card this plan can price.  Known "
            f"spellings: {', '.join(sorted(CARD_ALIASES))}.  A card is a row "
            f"in hexcore.mesh_point.CARD_ALIASES joined to a measured row of "
            f"hexcore.device_admission; a card nobody measured is refused "
            f"rather than given another card's number"
        )
    admission_key, mesh_key, nameplate_mib = CARD_ALIASES[normalized]
    return {
        "requested": key,
        "admission_card": admission_key,
        "generator_card": mesh_key,
        "nameplate_mib": nameplate_mib,
    }


def device_verdict(
    cells: float, card: Mapping[str, Any], *, budget_mib: float | None = None
) -> dict[str, Any]:
    """The limited-area admission row applied to the predicted cull."""

    from . import device_admission

    profile = device_admission.KNOWN_CARDS[str(card["admission_card"])]
    model = device_admission.model_for_card(profile, configuration="limited-area")
    required = model.required_bytes(int(math.ceil(cells)))
    predicted = model.predict_bytes(int(math.ceil(cells)))
    margin = model.margin_bytes()
    budget_bytes = int(
        (budget_mib if budget_mib is not None else float(card["nameplate_mib"])) * device_admission.MIB
    )
    fits = int(model.max_cells(budget_bytes))
    return {
        "row": device_admission.row_key("limited-area", profile),
        "row_measured": bool(model.measured),
        "card": profile.as_dict(),
        "cells_priced": int(math.ceil(cells)),
        "predicted_mib": round(predicted / device_admission.MIB, 1),
        "margin_mib": round(margin / device_admission.MIB, 1),
        "required_free_mib": round(required / device_admission.MIB, 1),
        "budget_mib": round(budget_bytes / device_admission.MIB, 1),
        "budget_basis": (
            "--vram-gib" if budget_mib is not None
            else "nameplate; the forecast door measures FREE memory at launch and "
                 "admits on that"
        ),
        "fits": required <= budget_bytes,
        "cells_that_fit": fits,
        "short_by_mib": round(max(0.0, (required - budget_bytes) / device_admission.MIB), 1),
    }


def max_radius_for_cells(
    cells: int, request: "PointRequest", *, attained_fine_km: float | None = None
) -> float:
    """The largest core radius whose predicted cull fits ``cells``."""

    low, high = 0.0, 4000.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        spec = ladder_spec(
            request.point, fine_dx_m=request.fine_dx_m, radius_km=mid,
            background_km=request.background_km,
            transition_factor=request.transition_factor, ring_factor=request.ring_factor,
        )
        predicted = predicted_cull_cells(
            spec=spec, radius_km=mid, pad_scale=request.cull_pad_scale,
            attained_fine_km=attained_fine_km,
        )["predicted_cells"]
        if predicted <= cells:
            low = mid
        else:
            high = mid
    return round(low, 1)


def choose_timestep(minimum_dc_edge_m: float, *, fine_dx_m: float) -> dict[str, Any]:
    """The largest ANCHORED timestep the mesh's own Courant limit admits.

    Sub-3-km meshes run convection off by the 2026-08-26 ruling
    (:mod:`hexcore.convection_admission`), so the anchors consulted are the
    convection-off rows; a coarser mesh consults the Grell-Freitas rows.
    The registry row declares what this returns and the bind re-admits it
    against the static's own ``dcEdge``.
    """

    from . import convection_admission, dt_admission
    from .timestep_admission import CourantPolicy

    policy = CourantPolicy()
    limit = policy.safety_factor * float(minimum_dc_edge_m) / policy.max_characteristic_speed_m_s
    sub_3km = float(fine_dx_m) < convection_admission.CONVECTION_OFF_BELOW_M
    # The anchor table spells Grell-Freitas "gf" (hexcore.dt_admission.dt_key).
    scheme = None if sub_3km else "gf"
    candidates = sorted(
        {
            anchor.dt_seconds
            for anchor in dt_admission.ADMITTED_TIMESTEPS.values()
            if anchor.cumulus_scheme == scheme
            and float(anchor.surface_pbl_seconds) == float(anchor.dt_seconds)
        },
        reverse=True,
    )
    for dt in candidates:
        if dt <= limit:
            return {
                "dt_seconds": float(dt),
                "courant_limit_seconds": limit,
                "margin": limit / float(dt),
                "cumulus_scheme": scheme,
                "anchors_consulted": candidates,
            }
    raise PointPlanRefusal(
        f"no anchored timestep fits this mesh: its Courant limit is "
        f"{limit:.3f} s (min dcEdge {minimum_dc_edge_m:.1f} m at "
        f"{policy.max_characteristic_speed_m_s:g} m/s x "
        f"{policy.safety_factor:g}) and the anchored "
        f"{'convection-off' if scheme is None else scheme.upper()} timesteps "
        f"are {candidates}.  A row declaring a smaller timestep would be "
        f"refused at bind for holding no anchor; earn one with "
        f"tools/mint_dt_anchor.py or coarsen --fine-dx-m"
    )


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PointRequest:
    point: tuple[float, float]
    fine_dx_m: float
    radius_km: float
    background_km: float
    transition_factor: float
    ring_factor: float
    cull_pad_scale: float
    card: Mapping[str, Any] | None
    vram_gib: float | None
    mesh_exe: Path | None
    name: str | None

    @property
    def fine_km(self) -> float:
        return self.fine_dx_m / 1000.0

    @property
    def transition_km(self) -> float:
        return self.transition_factor * self.fine_km


def request_from_arguments(arguments: argparse.Namespace) -> PointRequest:
    return PointRequest(
        point=parse_point(arguments.point),
        fine_dx_m=float(arguments.fine_dx_m),
        radius_km=float(arguments.radius_km),
        background_km=float(arguments.background_km),
        transition_factor=float(arguments.transition_factor),
        ring_factor=float(arguments.ring_factor),
        cull_pad_scale=float(arguments.cull_pad_scale),
        card=resolve_card(arguments.card),
        vram_gib=arguments.vram_gib,
        mesh_exe=arguments.mesh_exe,
        name=getattr(arguments, "name", None),
    )


def plan_point(request: PointRequest) -> dict[str, Any]:
    """Price the point: the spec, the parent, the cull, the card."""

    from .swath.sizing import dry_run

    spec = ladder_spec(
        request.point,
        fine_dx_m=request.fine_dx_m,
        radius_km=request.radius_km,
        background_km=request.background_km,
        transition_factor=request.transition_factor,
        ring_factor=request.ring_factor,
        name=request.name,
    )
    region = cull_region(request.point, request.radius_km, request.cull_pad_scale)
    generator_card = request.card["generator_card"] if request.card else None
    sizing = dry_run(
        spec,
        engine=request.mesh_exe,
        card=generator_card,
        vram_gib=request.vram_gib if generator_card else None,
    )
    attainment = sizing.get("region_attainment") or []
    attained_km = (
        float(attainment[-1]["attained_spacing_km"])
        if attainment and attainment[-1].get("attained_spacing_km")
        else request.fine_km
    )
    cull = predicted_cull_cells(
        spec=spec,
        radius_km=request.radius_km,
        pad_scale=request.cull_pad_scale,
        attained_fine_km=attained_km,
    )
    cull["attained_fine_spacing_km"] = attained_km
    cull["attainment_basis"] = (
        "generator region_attainment on the innermost cap (cap regions are "
        "the shape the generator's attainment reports correctly; see "
        "hexcore.swath.sizing)"
    )
    verdict = None
    if request.card is not None:
        verdict = device_verdict(
            cull["predicted_cells"], request.card, budget_mib=(
                None if request.vram_gib is None else float(request.vram_gib) * 1024.0
            ),
        )
        verdict["max_core_radius_km_on_this_card"] = max_radius_for_cells(
            int(verdict["cells_that_fit"]), request, attained_fine_km=attained_km,
        )
    return {
        "schema": "gpuwm-hex.point-plan/v1",
        "point_deg": list(request.point),
        "fine_dx_m": request.fine_dx_m,
        "core_radius_km": request.radius_km,
        "background_km": request.background_km,
        "transition_factor": request.transition_factor,
        "ring_factor": request.ring_factor,
        "ladder_km": ladder_rungs(request.background_km, request.fine_dx_m),
        "spec": spec,
        "cull_region": region,
        "cull_pad_scale": request.cull_pad_scale,
        "parent": {
            "predicted_cells": sizing.get("predicted_cells"),
            "basis": "generator_dry_run",
            "footprint_mib": sizing.get("footprint_mib"),
            "card": sizing.get("card"),
            "steepest_gradient_percent_per_cell": sizing.get(
                "steepest_requested_gradient_percent_per_cell"),
            "ladder_snap": sizing.get("ladder_snap"),
            "gates_applied_by_hexcore": sizing.get("gates_applied_by_hexcore"),
        },
        "cull": cull,
        "device": verdict,
        "card": request.card,
    }


def _report(plan: Mapping[str, Any]) -> list[str]:
    lat, lon = plan["point_deg"]
    parent = plan["parent"]
    cull = plan["cull"]
    gates = parent.get("gates_applied_by_hexcore") or {}
    band = gates.get("transition_band") or {}
    edge = gates.get("short_dual_edge_floor") or {}
    lines = [
        f"mesh-plan --point: {plan['spec']['name']}",
        f"  centre           {lat:.4f} N {lon:.4f} E",
        f"  ladder           {' > '.join(f'{r:g}' for r in plan['ladder_km'])} km "
        f"on a {plan['background_km']:g} km background "
        f"({len(plan['ladder_km'])} rungs, ramp {plan['transition_factor']:g}x, "
        f"ring {plan['ring_factor']:g}x)",
        f"  parent (global)  {parent['predicted_cells']:,.0f} cells (the generator's "
        f"own sizing integral)",
        f"  core             {plan['core_radius_km']:g} km at "
        f"{cull['attained_fine_spacing_km'] * 1000.0:.1f} m attained -> "
        f"{cull['core_cells']:,.0f} cells",
        f"  cull             cut at {cull['cut_radius_km']:g} km (pad "
        f"{plan['cull_pad_scale']:g}) -> {cull['predicted_cells']:,.0f} cells "
        f"predicted, {cull['boundary_rings']} boundary rings ({cull['basis']})",
    ]
    device = plan.get("device")
    if device is None:
        lines.append(
            "  footprint        not priced: no --card, and the fixed term is a "
            "property of the part"
        )
    else:
        lines.extend([
            f"  footprint        {device['predicted_mib']:,.1f} MiB predicted + "
            f"{device['margin_mib']:,.1f} MiB margin = {device['required_free_mib']:,.1f} "
            f"MiB free needed on row {device['row']} "
            f"({'measured' if device['row_measured'] else 'DERIVED'})",
            f"  budget           {device['budget_mib']:,.1f} MiB ({device['budget_basis']})",
            f"  verdict          {'FITS' if device['fits'] else 'DOES NOT FIT'}"
            + ("" if device["fits"] else f", short by {device['short_by_mib']:,.1f} MiB")
            + f"; this card holds {device['cells_that_fit']:,} cells, a core of up to "
            f"{device['max_core_radius_km_on_this_card']:g} km at this spacing and pad",
        ])
    lines.extend([
        "",
        "  GATES THE SPEC DECIDES, applied here:",
        f"    transition band   PASS  "
        f"{band.get('steepest_gradient_percent_per_cell', float('nan')):.4f} %/cell -> "
        f"{band.get('band_cells', float('nan')):.2f} cells (floor "
        f"{band.get('band_cells_floor', float('nan')):.0f}, ceiling "
        f"{band.get('gradient_percent_per_cell_ceiling', float('nan')):.4f} %/cell)",
        "  GATES ONLY A BUILD DECIDES:",
        f"    shortest dual edge  {str(edge.get('verdict', 'unknown')).upper()}  "
        f"limit {edge.get('limit_m', float('nan')):.0f} m ({edge.get('gate', 'unknown gate')})",
    ])
    return lines


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _run(argv: Sequence[str], *, what: str, log: Path | None = None) -> subprocess.CompletedProcess:
    started = time.perf_counter()
    completed = subprocess.run([str(item) for item in argv], capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"$ {' '.join(str(item) for item in argv)}\n[{elapsed:.1f} s, exit "
            f"{completed.returncode}]\n--- stdout ---\n{completed.stdout}\n--- stderr ---\n"
            f"{completed.stderr}\n",
            encoding="utf-8",
        )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise PointPlanRefusal(
            f"{what} exited {completed.returncode}"
            + (f"; its log is {log}" if log is not None else "")
            + (f".  Last line: {tail[-1]}" if tail else "")
        )
    completed.elapsed = elapsed  # type: ignore[attr-defined]
    return completed


def resolve_geog(explicit: Path | None) -> Path:
    """The WPS_GEOG root the static reads, or a refusal naming the ladder."""

    if explicit is not None:
        root = Path(explicit).expanduser()
        if not root.is_dir():
            raise PointPlanRefusal(
                f"--geog {root} is not a directory.  The static half of the "
                f"pair reads terrain, land use, soil, green-ness and albedo "
                f"from a WPS_GEOG archive, and the registry refuses a grid "
                f"with no matching static"
            )
        return root
    try:
        from gpuwm import rustwx_static
    except ImportError:
        rustwx_static = None  # type: ignore[assignment]
    if rustwx_static is not None:
        found = rustwx_static.default_geog_root()
        if found is not None:
            return Path(found)
        ladder = ", ".join(str(item) for item in rustwx_static.geog_root_candidates())
    else:
        ladder = "$GPUWM_WPS_GEOG (gpuwm is not importable here, so its ladder cannot be read)"
    raise PointPlanRefusal(
        f"no WPS_GEOG archive was found; pass --geog, or stage one with "
        f"`gpuwm fetch-geog --datasets mesh`.  Looked at: {ladder}"
    )


def admit_pair(grid: Path, static: Path, *, fine_dx_m: float, log=print) -> dict[str, Any]:
    """The door's own admission pass over a pair it just built."""

    import numpy as np

    from .cell_coordination_admission import (
        CellCoordinationAdmissionError, admit_cell_coordination,
    )
    from .dual_edge_admission import DualEdgeAdmissionError, admit_dual_edges
    from .mesh import Mesh
    from .timestep_admission import (
        CourantPolicy, TimestepAdmissionError, admit_timestep, edge_length_authority,
    )

    mesh = Mesh.from_netcdf(grid, static)
    try:
        dual = admit_dual_edges(
            mesh.dvEdge, mesh.dcEdge, cells_on_edge=mesh.cellsOnEdge,
            cells_on_edge_base=0, mesh_name=grid.name,
        )
    except DualEdgeAdmissionError as error:
        raise PointPlanRefusal(
            f"the generated pair is refused on its own dual edges and is NOT "
            f"registered: {error}"
        ) from error
    try:
        coordination = admit_cell_coordination(
            np.asarray(mesh.arrays["nEdgesOnCell"]), mesh_name=grid.name,
        )
    except CellCoordinationAdmissionError as error:
        raise PointPlanRefusal(
            f"the generated pair is refused on its cell coordination and is NOT "
            f"registered: {error}"
        ) from error
    from netCDF4 import Dataset

    with Dataset(str(static)) as dataset:
        dataset.set_auto_maskandscale(False)
        dc_edge = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
    authority = edge_length_authority(dc_edge)
    chosen = choose_timestep(authority.minimum_m, fine_dx_m=fine_dx_m)
    try:
        timestep = admit_timestep(chosen["dt_seconds"], authority, policy=CourantPolicy())
    except TimestepAdmissionError as error:
        raise PointPlanRefusal(str(error)) from error
    log(
        f"ADMIT {grid.name}: dual edges {dual.as_dict().get('minimum_ratio', '?')}, "
        f"coordination {coordination.as_dict().get('histogram', '?')}, "
        f"min dcEdge {authority.minimum_m:.3f} m, dt {chosen['dt_seconds']:g} s "
        f"(Courant limit {chosen['courant_limit_seconds']:.3f} s, "
        f"{chosen['margin']:.3f}x)"
    )
    return {
        "dual_edge_admission": dual.as_dict(),
        "cell_coordination_admission": coordination.as_dict(),
        "timestep_admission": timestep.as_dict(),
        "timestep_choice": chosen,
        "n_cells": int(mesh.dimensions["nCells"]),
        "n_edges": int(mesh.dimensions["nEdges"]),
        "regional": bool(mesh.is_regional),
    }


def generate_point(
    request: PointRequest,
    plan: Mapping[str, Any],
    *,
    out_dir: Path,
    geog: Path | None,
    vertical_spec: Path | None,
    clobber: bool,
    static_exe: Path | None = None,
    n_levels: int = DEFAULT_LEVELS,
    log=print,
) -> dict[str, Any]:
    """Build, admit, register and cull the pair the plan priced."""

    from . import mesh_rows
    from .cull_door import carry_lineage, cull_one
    from .engines import MESH, STATIC, EngineRefusal, resolve

    try:
        mesh_exe = resolve(MESH, request.mesh_exe)
        static_bin = resolve(STATIC, static_exe)
    except EngineRefusal as error:
        raise PointPlanRefusal(str(error)) from error
    geog_root = resolve_geog(geog)
    out_dir = Path(out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    stem = f"point-{row_token(request.point)}-{request.fine_dx_m:g}m-r{request.radius_km:g}km"
    if request.name:
        stem = request.name
    spec_path = out_dir / f"{stem}.spec.json"
    grid = out_dir / f"{stem}.grid.nc"
    static = out_dir / f"{stem}.static.nc"
    grid_receipt = out_dir / f"{stem}.grid.receipt.json"
    static_receipt = out_dir / f"{stem}.static.receipt.json"
    rows_path = out_dir / mesh_rows.MESH_ROWS_FILENAME
    for existing in (grid, static, rows_path):
        if existing.exists() and not clobber:
            raise PointPlanRefusal(
                f"{existing} exists; pass --clobber to replace it, or a fresh "
                f"--out-dir.  A generated pair that silently overwrote another "
                f"would leave a row file naming bytes that moved"
            )
    spec_path.write_text(
        json.dumps(plan["spec"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    legs: list[dict[str, Any]] = []

    argv: list[Any] = [mesh_exe, "--spec", spec_path, "--out", grid, "--receipt", grid_receipt]
    generator_card = request.card["generator_card"] if request.card else None
    if generator_card:
        argv += ["--card", generator_card]
        if request.vram_gib is not None:
            argv += ["--vram-gib", str(request.vram_gib)]
    if clobber:
        argv.append("--clobber")
    log(f"GENERATE {grid.name} ...")
    done = _run(argv, what="rw_mpas_mesh", log=logs / "mesh.log")
    legs.append({"leg": "mesh", "seconds": round(done.elapsed, 2)})
    log(f"GENERATE {grid.name} written in {done.elapsed:.1f} s")

    argv = [static_bin, "--grid", grid, "--out", static, "--geog", geog_root,
            "--nominal-dx-m", repr(float(request.fine_dx_m)), "--receipt", static_receipt]
    if clobber:
        argv.append("--clobber")
    log(f"STATIC {static.name} ...")
    done = _run(argv, what="rw_mpas_static", log=logs / "static.log")
    legs.append({"leg": "static", "seconds": round(done.elapsed, 2)})
    log(f"STATIC {static.name} written in {done.elapsed:.1f} s")

    admission = admit_pair(grid, static, fine_dx_m=request.fine_dx_m, log=log)
    if admission["regional"]:
        raise PointPlanRefusal(
            f"{grid} carries a boundary zone; the generator was asked for a "
            f"global mesh and this is not one"
        )
    parent = mesh_rows.describe_generated(
        name=parent_row_name(request.point, request.fine_dx_m, request.background_km,
                             admission["n_cells"]),
        grid=grid, static=static, spec_path=spec_path,
        generator_receipt=grid_receipt, static_receipt=static_receipt,
        point_deg=request.point, fine_dx_m=request.fine_dx_m,
        core_radius_km=request.radius_km, background_km=request.background_km,
        dt_seconds=admission["timestep_choice"]["dt_seconds"],
        n_levels=n_levels, admission=admission,
    )
    mesh_rows.write_rows(rows_path, [parent])
    log(f"ROW {parent.name} -> {rows_path}")

    # The cull: grid and static, the same region, the parent's own bytes.
    region_path = out_dir / f"{stem}.cull-region.json"
    region_path.write_text(
        json.dumps(plan["cull_region"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    cull_grid = out_dir / f"{stem}-cull.grid.nc"
    cull_static = out_dir / f"{stem}-cull.static.nc"
    started = time.perf_counter()
    cut: dict[str, Any] = {}
    cut["grid"] = cull_one(mesh_exe, grid, region_path, cull_grid,
                           graph=out_dir / f"{stem}-cull.graph.info", clobber=clobber)
    cut["grid"]["lineage"] = carry_lineage(grid, cull_grid, drives_boundaries=False)
    cut["static"] = cull_one(mesh_exe, static, region_path, cull_static, clobber=clobber)
    cut["static"]["lineage"] = carry_lineage(static, cull_static, drives_boundaries=False)
    legs.append({"leg": "cull", "seconds": round(time.perf_counter() - started, 2)})
    cull_admission = admit_pair(cull_grid, cull_static, fine_dx_m=request.fine_dx_m, log=log)
    if not cull_admission["regional"]:
        raise PointPlanRefusal(
            f"{cull_grid} carries no boundary zone after the cull; the "
            f"region {region_path} did not cut a limited-area mesh"
        )
    lbc_dir = out_dir / f"{stem}-cull.lbc"
    cull_row = mesh_rows.describe_cull(
        name=cull_row_name(request.point, request.fine_dx_m, request.background_km,
                           cull_admission["n_cells"]),
        parent=parent, grid=cull_grid, static=cull_static,
        cull_receipt=cull_grid.with_suffix(cull_grid.suffix + ".cull-receipt.json"),
        cull_region=plan["cull_region"], cull_pad_scale=request.cull_pad_scale,
        lbc_source=(
            f"{lbc_dir} (rw_mpas_lbc on the wps-intermediate route from the "
            f"same source that initialises the cull; the forecast door refuses "
            f"an absent or empty --lbc-dir)"
        ),
        admission=cull_admission,
        dt_seconds=parent.dt_seconds,
    )
    mesh_rows.append_row(rows_path, cull_row)
    log(f"ROW {cull_row.name} -> {rows_path} ({cull_row.n_cells:,} cells, "
        f"zone {cull_row.boundary_zone_width}, bdyMask {cull_row.bdy_mask_sha256[:16]}...)")

    vertical: dict[str, Any] | None = None
    if vertical_spec is not None:
        from .vertical_spec import materialize_vertical_artifact

        artifact = out_dir / f"{stem}.vertical.nc"
        started = time.perf_counter()
        log(f"VERTICAL {artifact.name} from {Path(vertical_spec).name} on the parent ...")
        payload = materialize_vertical_artifact(
            grid=grid, static=static, spec_path=vertical_spec, output=artifact,
            receipt_path=artifact.with_name(artifact.name + ".receipt.json"),
        )
        mint_seconds = time.perf_counter() - started
        cull_artifact = out_dir / f"{stem}-cull.vertical.nc"
        cut["vertical"] = cull_one(mesh_exe, artifact, region_path, cull_artifact, clobber=clobber)
        cut["vertical"]["lineage"] = carry_lineage(artifact, cull_artifact, drives_boundaries=False)
        legs.append({"leg": "vertical", "seconds": round(time.perf_counter() - started, 2)})
        vertical = {
            "spec": str(vertical_spec),
            "parent_artifact": str(artifact),
            "parent_receipt": payload.get("receipt") if isinstance(payload, dict) else None,
            "cull_artifact": str(cull_artifact),
            "mint_seconds": round(mint_seconds, 2),
            "n_vert_levels": int(payload["invariants"].get("n_vert_levels", n_levels))
            if isinstance(payload, dict) and isinstance(payload.get("invariants"), dict)
            else n_levels,
        }
        log(f"VERTICAL {cull_artifact.name} culled; init the cull with "
            f"--capsule/--reference {cull_artifact.name}")

    receipt = {
        "schema": "gpuwm-hex.point-generate/v1",
        "plan": dict(plan),
        "engines": {"rw_mpas_mesh": str(mesh_exe), "rw_mpas_static": str(static_bin)},
        "geog": str(geog_root),
        "spec": str(spec_path),
        "parent": {
            "row": parent.name, "grid": str(grid), "static": str(static),
            "n_cells": parent.n_cells, "n_edges": parent.n_edges,
            "dt_seconds": parent.dt_seconds,
            "generator_receipt": str(grid_receipt), "static_receipt": str(static_receipt),
            "admission": admission,
        },
        "cull": {
            "row": cull_row.name, "grid": str(cull_grid), "static": str(cull_static),
            "n_cells": cull_row.n_cells, "n_edges": cull_row.n_edges,
            "boundary_zone_width": cull_row.boundary_zone_width,
            "bdy_mask_sha256": cull_row.bdy_mask_sha256,
            "region": str(region_path), "files": cut, "admission": cull_admission,
            "predicted_cells": plan["cull"]["predicted_cells"],
            "lbc_dir": str(lbc_dir),
        },
        "vertical": vertical,
        "rows_file": str(rows_path),
        "legs": legs,
        "next": [
            "gpuwm-hex intermediate --source hrrr --from-plan "
            f"{out_dir / (stem + '.point-generate.json')} --grib-dir <fetched HRRR> "
            "--cycle <YYYY-MM-DDTHH> --hours 0-3 --out-dir <MET>",
            f"gpuwm-hex init --met <MET/FILE:YYYY-MM-DD_HH> --static {cull_static} "
            + (f"--capsule {vertical['cull_artifact']} --reference {vertical['cull_artifact']} "
               if vertical else "--capsule <culled vertical> --reference <culled vertical> ")
            + f"--out {out_dir / (stem + '-cull.init.nc')} --start-time ... "
            "--nfglevels 51 --nfgsoillevels 4 --extrap-airtemp constant --use-spechumd yes "
            "--theta-adv-order 3 --coef-3rd-order 0.25 --virtual-factor reproduce-fortran "
            "--deep-soil-moisture reproduce-fortran --landuse-table MODIFIED_IGBP_MODIS_NOAH "
            "--frac-seaice yes --tsk-seaice-threshold 100.0 --oned-underflow preserve",
            f"gpuwm-hex lbc --grid {out_dir / (stem + '-cull.init.nc')} --met-dir <MET> "
            f"--out-dir {lbc_dir} --start-time ... --stop-time ... --nfglevels 51 "
            "--extrap-airtemp constant --use-spechumd yes",
            f"GPUWM_HEX_MESH_ROWS={rows_path} gpuwm-hex forecast --mesh {cull_row.name} "
            f"--grid {cull_grid} --static {cull_static} --init {out_dir / (stem + '-cull.init.nc')} "
            f"--lbc-dir {lbc_dir} --hours 3 --start-time ... --out <RUN>",
        ],
    }
    receipt_path = out_dir / f"{stem}.point-generate.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    log(f"RECEIPT {receipt_path}")
    return receipt


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
def add_point_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("a fine core at a point (instead of --spec)")
    group.add_argument(
        "--point", default=None, metavar="LAT,LON",
        help="centre of the fine core; the spec is the registered 937.5 m "
             "ladder recipe moved to this point")
    group.add_argument(
        "--fine-dx-m", type=float, default=937.5, metavar="M",
        help="fine spacing in metres; must be a rung of the halving ladder "
             "under --background-km (default 937.5)")
    group.add_argument(
        "--radius-km", type=float, default=100.0, metavar="KM",
        help="radius of the fine core (default 100)")
    group.add_argument(
        "--background-km", type=float, default=DEFAULT_BACKGROUND_KM, metavar="KM",
        help=f"global background spacing (default {DEFAULT_BACKGROUND_KM:g})")
    group.add_argument(
        "--transition-factor", type=float, default=DEFAULT_TRANSITION_FACTOR, metavar="X",
        help=f"each rung's ramp as a multiple of its own spacing (default {DEFAULT_TRANSITION_FACTOR:g})")
    group.add_argument(
        "--ring-factor", type=float, default=DEFAULT_RING_FACTOR, metavar="X",
        help=f"how many ramps of the next finer rung a cap sits outside it (default {DEFAULT_RING_FACTOR:g})")
    group.add_argument(
        "--cull-pad-scale", type=float, default=DEFAULT_CULL_PAD_SCALE, metavar="X",
        help=f"the limited-area cut, as a multiple of the core radius (default {DEFAULT_CULL_PAD_SCALE:g}, the measured knee)")
    group.add_argument(
        "--name", default=None, metavar="TEXT",
        help="file-name stem and spec name (default: derived from the point)")
    group.add_argument(
        "--generate", action="store_true",
        help="build the pair, admit it, register it as a runtime row beside it, and cull it")
    group.add_argument(
        "--out-dir", type=Path, default=None, metavar="DIR",
        help="where --generate writes (required with --generate)")
    group.add_argument(
        "--geog", type=Path, default=None, metavar="DIR",
        help="WPS_GEOG root for the static (default: gpuwm's fetch-geog ladder)")
    group.add_argument(
        "--vertical-spec", type=Path, default=None, metavar="JSON",
        help="with --generate: mint the parent's native-free vertical artifact "
             "from this gpuwm-hex.vertical-spec/v1 declaration and cull it, so "
             "the cull can be initialised from a regional source")
    group.add_argument("--static-exe", type=Path, default=None, metavar="FILE")
    group.add_argument("--clobber", action="store_true")


def run_point(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    plan = plan_point(request)
    if arguments.json:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(_report(plan)))
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
    if not arguments.generate:
        return 0
    if arguments.out_dir is None:
        raise PointPlanRefusal(
            "--generate needs --out-dir: the pair, the row file and the cull "
            "land beside each other and nothing here guesses where"
        )
    device = plan.get("device")
    if device is not None and not device["fits"]:
        raise PointPlanRefusal(
            f"refused before anything is built: the predicted cull "
            f"({plan['cull']['predicted_cells']:,.0f} cells) needs "
            f"{device['required_free_mib']:,.1f} MiB free on "
            f"{request.card['admission_card']} and the budget is "
            f"{device['budget_mib']:,.1f} MiB, short by {device['short_by_mib']:,.1f} "
            f"MiB.  This card holds a core of up to "
            f"{device['max_core_radius_km_on_this_card']:g} km at this spacing "
            f"and pad; reduce --radius-km, coarsen --fine-dx-m, or name a bigger card"
        )
    generate_point(
        request, plan,
        out_dir=arguments.out_dir, geog=arguments.geog,
        vertical_spec=arguments.vertical_spec, clobber=bool(arguments.clobber),
        static_exe=arguments.static_exe,
    )
    return 0


__all__ = [
    "CARD_ALIASES",
    "DEFAULT_BACKGROUND_KM",
    "DEFAULT_CULL_PAD_SCALE",
    "DEFAULT_RING_FACTOR",
    "DEFAULT_TRANSITION_FACTOR",
    "PointPlanRefusal",
    "PointRequest",
    "add_point_arguments",
    "admit_pair",
    "choose_timestep",
    "cull_region",
    "cull_row_name",
    "device_verdict",
    "generate_point",
    "ladder_rungs",
    "ladder_spec",
    "max_radius_for_cells",
    "parent_row_name",
    "parse_point",
    "plan_point",
    "predicted_cull_cells",
    "spacing_at",
    "spacing_profile",
    "resolve_card",
    "resolve_geog",
    "run_point",
]
