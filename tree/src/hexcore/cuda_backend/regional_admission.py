"""Earned-anchor admission for regional (limited-area) CUDA execution.

Ruled 2026-08-25 as part of the LAM campaign: *regional execution
is refused by name until a registered regional anchor exists*, mirroring the
per-architecture pattern in :mod:`hexcore.cuda_backend.arch_admission`.

The reasoning is the same one that admits an architecture below the proven
floor.  The port's CUDA numbers are trusted because they are checked against
something: the global lane against the frozen v8.4.1 authority set, an
unproven architecture against its own minted anchor.  A regional
configuration has neither until somebody mints one, so a regional forecast
run without an anchor produces numbers no receipt could be compared to.  The
concrete breakage this gate prevents is therefore not a crash: it is a
regional forecast that looks finished, carries a receipt, and was never
checked against anything at all.

AN ANCHOR HAS TWO HALVES AND THEY DO NOT KEY THE SAME WAY.  This is the
2026-08-27 re-keying, and it retires a measured tax rather than relaxing a
gate.

* the **contract deck** measures every regional kernel's device bits against
  the CPU authority's host-derived expected bits ON ONE CULL'S OWN ZONE
  GEOMETRY, with a mutation control per deck proving the deck can fail.  What
  it certifies is a geometry.  It stays keyed to the geometry -- one row per
  ``bdyMask`` digest -- and it is cheap: 39 s at 4,440 cells, 142 s at 15,755
  (``evidence/nest-ratio-20260827/RECEIPT.md`` section 3b).

* the **forecast mint** runs the same configuration twice in two independent
  processes and compares the published history under masked digests.  What it
  certifies is that this KERNEL SET, at this TIMESTEP, on a domain of this
  SHAPE CLASS, reproduces itself.  It is two 1,080-step integrations -- 5.5
  to 8.7 minutes of card measured on four new meshes, against 4-6 minutes for
  the forecast it is admitting.

Keying the MINT to a cull's own bytes charged that 5.5-8.7 minutes again for
every re-placed cull.  A storm-following swath re-places itself every cycle
and a re-placed cull is a new digest, so the cascade paid roughly three
forecasts of card time per swath per cycle for permission to run one.

**Under the gate law that per-digest mint cannot name the breakage it
prevents.**  The measurement is in the nest-ratio receipt: five concentric
culls of ONE parent, 4,440 to 15,755 cells, all carrying the parent's own
finest edge of 4,457.233 m, all seven rings wide, all at dt 20 s -- five
independent mints, five identical verdicts, and all five earned the same 20 s
anchor at the same 1.605x Courant margin.  Nothing the mint measures
distinguished them.  the project rule is that a guard must name the concrete
breakage it prevents, and "this cull's bytes are new" is not one when every
input the mint reads is the same.

So the mint keys to the CONFIGURATION CLASS, and the class key is exactly the
set of inputs the mint's own instrument reads and its verdict depends on:

``boundary_zone_width``
    How many rings the culler grew.  Every regional stage -- the specified
    zone, the relaxation zone's Rayleigh and Laplacian terms, the mask-4/5
    scalar downgrade -- is indexed by ring, so a zone of a different width is
    a different set of stages and the mint says nothing about it.

``n_vert_levels``
    Every regional kernel launches over ``(levels, elements)``; a different
    column height is a different launch geometry for all 22 of them.

``finest_edge_m``
    ``min(dcEdge)``, measured off the mesh in hand.  It is what the Courant
    admission keys on, what ``config_len_disp`` follows, and what
    ``convection_admission`` takes its decision on.  It is also the one
    length a cull cannot change: a cull moves no cell centre, so every cull
    containing the parent's fine core carries the parent's own finest edge.

``dt_seconds``
    The relaxation-zone coefficients are hardwired at 50 and 10 times dt, so
    the boundary stages are dt-dependent by construction.  A mint at one
    timestep says nothing about another.

``kernel_set_sha256``
    A digest over the source of every translation unit the regional step
    launches through.  The mint is a statement about bytes; if the bytes
    move, the statement lapses.  Nothing checked this before -- the mint
    RECORDED its source digests and no admission ever re-read them.

**What class keying does NOT relax.**  A cull of an admitted class is still
refused until its own contract deck has been run and its receipt presented:
8 of 8 decks bitwise against the v8.4.1 CPU authority on THIS cull's zone
geometry, 22 of 22 declared kernels covered, every mutation control with
teeth.  The residual per-geometry admission cost is that deck and nothing
else.

**And what the pair of halves still does not admit** is unchanged and stated
on every row: the mint pair is the DRY regional dycore, so an anchor
certifies the 22 boundary kernels and says nothing on its own about the ArWen
physics seam a full-physics run also drives; and endpoint byte-identity
against the compiled reference is NOT measured here, and is not measured for
the global lane either.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .. import shipped_sources

#: Every translation unit the regional step launches through, as bytes.
#:
#: This is the mint instrument's own list (``tools/run_cuda_regional_forecast``)
#: minus the instrument itself: the instrument decides what is WRITTEN, never
#: what is COMPUTED, and hashing it would make an edit to a print statement
#: lapse every class.  ``regional_admission`` is deliberately absent for the
#: same reason it is absent from the mint receipt -- it decides ADMISSION, not
#: arithmetic, and the rows' own prose lives in it, so hashing it would make
#: every edit to a basis sentence invalidate the evidence that sentence names.
REGIONAL_KERNEL_SOURCES: tuple[str, ...] = (
    "src/hexcore/cuda_acoustic_v841.py",
    "src/hexcore/cuda_backend/containers.py",
    "src/hexcore/cuda_backend/runtime.py",
    "src/hexcore/cuda_driver.py",
    "src/hexcore/cuda_dynamics_v841.py",
    "src/hexcore/cuda_fp32.py",
    "src/hexcore/cuda_horizontal.py",
    "src/hexcore/cuda_horizontal_v841.py",
    "src/hexcore/cuda_regional_forecast_v841.py",
    "src/hexcore/cuda_regional_v841.py",
    "src/hexcore/cuda_transport.py",
    "src/hexcore/cuda_transport_v841.py",
    "src/hexcore/cuda_v841.py",
    "src/hexcore/regional_v841.py",
)

class RegionalAdmissionRefusal(RuntimeError):
    """A regional configuration is refused, and the message says why."""


def kernel_set_sha256(root: Path | None = None) -> str:
    """One digest over every source the regional step launches through.

    Framed name-then-payload with a NUL separator, in the declared order, so
    the digest cannot be moved by renaming a file into another's bytes.  The
    same framing convention ``mesh.regional_boundary_mask_digest`` uses, and
    for the same reason: this program has already lost a day to two
    classifiers hashing the same arrays two different ways.

    EACH NAME RESOLVES TO THE FILE THIS INTERPRETER WILL LAUNCH, through
    :mod:`hexcore.shipped_sources`, and that is ledger #379's fix.  The
    constant this replaced walked three directories up from here and assumed
    a checkout underneath; from an installed wheel that lands on
    ``<venv>/lib/python3.13``, where no ``src/`` exists, so this function
    refused and took the whole limited-area lane -- 0.2.0's headline -- with
    it on every machine, with no flag that could open it.  The names are
    unchanged, the bytes under them in a checkout are unchanged, and the
    digest is therefore unchanged: ``MINTED_KERNEL_SET_SHA256`` did not move
    and no admitted class lapsed.  Measured both ways in
    ``evidence/wheel-reach-20260827``.  That paragraph is about ledger #379
    and nothing else.  0.2.0's package rename moved this digest twice over --
    every declared NAME went from ``src/mpas_port/`` to ``src/hexcore/``, and
    eleven of the fourteen PAYLOADS moved with them, because the same token is
    an import line and an NVRTC ``module_key`` inside those files.  The digest
    therefore lapsed, which is the cost a name-framed digest exists to charge;
    the alternative, keeping a retired directory name in the declaration so the
    arithmetic would not notice, is a gate reporting on a path nothing
    executes.  What the move was NOT is measured and stated at
    :data:`MINTED_KERNEL_SET_SHA256`.

    The gate did not lose its teeth in the move.  It gained them where it had
    none: from a wheel it now digests the modules in ``site-packages`` that
    the run actually imports, so altering one of those bytes lapses every
    class, which is exactly what a statement about bytes has to do.
    """

    digest = hashlib.sha256()
    for name in REGIONAL_KERNEL_SOURCES:
        path = shipped_sources.resolve(name, root)
        try:
            payload = path.read_bytes()
        except OSError as error:  # pragma: no cover - a broken install
            raise RegionalAdmissionRefusal(
                f"the regional kernel set cannot be digested: {name} is not "
                f"readable at {path} under "
                f"{shipped_sources.describe_root(root)} ({error}).  A "
                f"regional anchor is a statement about these bytes, so an "
                f"install that cannot produce them cannot be admitted "
                f"against one"
            ) from error
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


#: A class whose finest edge was never measured in a checkout holding its
#: mesh.  Exactly one class carries it -- the 120 km v8.4.1 reference cull,
#: whose grid publishes UNIT-SPHERE ``dcEdge`` and whose Earth-scaled metrics
#: live in a static file this repository does not contain (§9 of STATE.md: the
#: regional reference bundle is a mirror on the nodes, with no fetch path).
#: Guessing the number would be a fabricated key, so the class declares it
#: unmeasured and PAYS FOR IT: a class with an unmeasured edge admits only the
#: cull geometries already listed in :data:`SHIPPED_CONTRACTS`, never a
#: presented receipt, so it cannot grow to cover a cull nobody looked at.
UNMEASURED_FINEST_EDGE_MM = -1


@dataclass(frozen=True, slots=True)
class RegionalClassKey:
    """The identity a forecast MINT actually measures, geometry removed.

    Integers, deliberately.  ``finest_edge_mm`` and ``dt_ms`` are quantised
    so a key can be compared for equality without a tolerance and without a
    float repr appearing in a refusal message.  A cull moves no cell centre,
    so two culls of one parent carry ``min(dcEdge)`` in identical float64
    bits; the quantisation is protection against a value that arrived through
    a printed number, not against real drift.
    """

    boundary_zone_width: int
    n_vert_levels: int
    finest_edge_mm: int
    dt_ms: int
    kernel_set_sha256: str

    @classmethod
    def build(
        cls,
        *,
        boundary_zone_width: int,
        n_vert_levels: int,
        finest_edge_m: float,
        dt_seconds: float,
        kernel_set: str,
    ) -> "RegionalClassKey":
        return cls(
            boundary_zone_width=int(boundary_zone_width),
            n_vert_levels=int(n_vert_levels),
            finest_edge_mm=int(round(float(finest_edge_m) * 1000.0)),
            dt_ms=int(round(float(dt_seconds) * 1000.0)),
            kernel_set_sha256=str(kernel_set),
        )

    @property
    def finest_edge_measured(self) -> bool:
        return self.finest_edge_mm != UNMEASURED_FINEST_EDGE_MM

    def matches(self, other: "RegionalClassKey") -> bool:
        """Is ``other`` -- a key measured off a real run -- this class?

        Every field compares exactly, except that a class declaring an
        unmeasured finest edge does not compare one.  Such a class buys that
        looseness back by refusing every geometry that is not already a
        shipped contract row (see :func:`require_regional_anchor`).
        """

        if (
            self.boundary_zone_width != other.boundary_zone_width
            or self.n_vert_levels != other.n_vert_levels
            or self.dt_ms != other.dt_ms
            or self.kernel_set_sha256 != other.kernel_set_sha256
        ):
            return False
        if not self.finest_edge_measured:
            return True
        return self.finest_edge_mm == other.finest_edge_mm

    def differing_fields(self, other: "RegionalClassKey") -> list[str]:
        names = ["boundary_zone_width", "n_vert_levels", "dt_ms", "kernel_set_sha256"]
        if self.finest_edge_measured:
            names.append("finest_edge_mm")
        return [name for name in names if getattr(self, name) != getattr(other, name)]

    def as_dict(self) -> dict[str, object]:
        return {
            "boundary_zone_width": self.boundary_zone_width,
            "n_vert_levels": self.n_vert_levels,
            "finest_edge_m": (
                None
                if not self.finest_edge_measured
                else self.finest_edge_mm / 1000.0
            ),
            "dt_seconds": self.dt_ms / 1000.0,
            "kernel_set_sha256": self.kernel_set_sha256,
        }

    def describe(self) -> str:
        edge = (
            "finest edge NOT MEASURED"
            if not self.finest_edge_measured
            else f"finest edge {self.finest_edge_mm / 1000.0:.3f} m"
        )
        return (
            f"zone width {self.boundary_zone_width}, "
            f"{self.n_vert_levels} levels, "
            f"{edge}, "
            f"dt {self.dt_ms / 1000.0:g} s, "
            f"kernel set {self.kernel_set_sha256[:16]}..."
        )


@dataclass(frozen=True, slots=True)
class RegionalClass:
    """One minted configuration class: the FORECAST half of an anchor."""

    class_id: str
    key: RegionalClassKey
    parent: str
    card: str
    admitted_on: str
    mint_receipts: tuple[str, ...]
    minted_on_geometries: tuple[str, ...]
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "key": self.key.as_dict(),
            "parent": self.parent,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "mint_receipts": list(self.mint_receipts),
            "minted_on_geometries": list(self.minted_on_geometries),
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class RegionalContract:
    """One cull's own contract deck: the GEOMETRY half of an anchor."""

    bdy_mask_sha256: str
    n_cells: int
    boundary_zone_width: int
    class_id: str
    mesh_row: str | None
    card: str
    admitted_on: str
    contract_receipt: str
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "bdy_mask_sha256": self.bdy_mask_sha256,
            "n_cells": self.n_cells,
            "boundary_zone_width": self.boundary_zone_width,
            "class_id": self.class_id,
            "mesh_row": self.mesh_row,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "contract_receipt": self.contract_receipt,
            "basis": self.basis,
        }


# ---------------------------------------------------------------------------
# the two registries
# ---------------------------------------------------------------------------

#: The kernel-set digest the shipped classes were minted at.  It is a
#: MEASURED constant of this tree, not a declaration: ``tests/
#: test_regional_forecast_anchor.py`` re-computes it from the sources and
#: refuses a mismatch by name, so a kernel edit that silently kept an anchor
#: alive is impossible.
#: RE-MINTED 2026-08-27 for the provenance scrub (ledger #377).  Three of the
#: fourteen hashed sources moved -- ``cuda_driver.py`` and ``regional_v841.py``
#: carried a person's name, a node label and a machine-shaped rig description
#: in comments and docstrings, and those had to leave the PRIVATE tree rather
#: than the public copy, because this digest is evaluated at ADMISSION time
#: from the shipped tree: a copy scrubbed after the fact refuses every regional
#: CUDA run at the door of the published package while its wheel builds clean
#: and its audit passes.  The move is measured inert
#: (``evidence/scrub-pins-20260827/``: identical ASTs with docstrings stripped
#: and string constants normalised), but a mint is a statement about BYTES, so
#: it lapsed and was RE-TAKEN rather than re-pointed -- both rows below carry
#: fresh two-process mint pairs on this digest.
#:
#: RE-DERIVED 2026-08-28 for the 0.2.0 package rename, and this is a different
#: KIND of move from the one above, which is why it is written out rather than
#: folded into that paragraph.  #377 moved BYTES: three sources were edited and
#: the two shipped classes were RE-MINTED on the card, two fresh 1,080-step
#: pairs, 17 min 29 s of a 5090.  The rename moved NAMES and the spelling of
#: one identifier: all fourteen declared names changed prefix, and eleven of
#: the fourteen files changed bytes by the substitution of the token
#: ``mpas_port`` with ``hexcore`` AND NOTHING ELSE -- measured, not argued.
#: Each of those eleven is byte-identical to its pre-rename self once that one
#: identifier is substituted, which is a stronger statement than #377's
#: AST comparison and needs no docstring-stripping to make.  The 32 lines
#: carrying the token are 19 comment, docstring and cross-reference lines, 12
#: NVRTC ``module_key`` / module-name labels, and one import.  Not one is
#: inside a CUDA source string, so no kernel's compiled text differs, and the
#: only thing a ``module_key`` selects is a cache slot and an observer label.
#:
#: WHAT IS THEREFORE OWED, said plainly rather than dodged: the two rows below
#: name mint receipts taken at kernel set ``237630342e...``, and this constant
#: is no longer that value.  Those receipts are NOT re-pointed and their prose
#: is not edited to pretend otherwise.  A re-mint at this digest -- the four
#: 1,080-step integrations #377 paid for, on a card -- is an OPEN follow-up of
#: the 0.2.0 cut and was not taken by the lane that re-derived this constant.
#: The constant moved anyway because the alternative is worse by a wide margin:
#: a stale value here is not a failing test, it is ``require_regional_anchor``
#: refusing every limited-area run at the door of the shipped package, which is
#: ledger #379's breakage restaged.  Only one half of the anchor is in
#: question: every cull's contract deck still measures device bits against the
#: v8.4.1 CPU authority on its own zone geometry, and a spelling cannot touch
#: that.
MINTED_KERNEL_SET_SHA256 = (
    "eeec7042024d89ebbdbd18d8a401b90a9843f2783f5eb57a695a515f3ef657db"
)

_DRY_MINT_CAVEAT = (
    "WHAT THIS CLASS DOES NOT ADMIT: the mint pair is the DRY regional "
    "dycore, as every regional mint this program has run is.  It certifies "
    "the 22 boundary kernels and says nothing on its own about the ArWen "
    "physics seam a full-physics run also drives; the full-physics arms are "
    "admitted on the reasoning the regional-physics lane recorded, that the "
    "boundary machinery is the same 22 kernels either way, and that "
    "reasoning is inherited rather than re-earned.  Endpoint byte-identity "
    "against the compiled reference is NOT measured here, and is not "
    "measured for the global lane either."
)

#: Configuration classes holding a verified forecast mint.  Adding a row is
#: table work -- one entry naming evidence that exists in this repository.
ADMITTED_CLASSES: Mapping[str, RegionalClass] = MappingProxyType(
    {
        "graded-4457m-dt20-z7": RegionalClass(
            class_id="graded-4457m-dt20-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=4_457_233,
                dt_ms=20_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="v4.75.121182",
            card="NVIDIA GeForce RTX 5090 (sm_120)",
            admitted_on="2026-08-27",
            mint_receipts=(
                "evidence/scrub-pins-20260827/graded-class-mint/"
                "mint-run1/forecast.json",
                "evidence/scrub-pins-20260827/graded-class-mint/"
                "mint-run2/forecast.json",
            ),
            minted_on_geometries=("r4.75.14050",),
            basis=(
                "The class every storm-following cull of a 4.6 km graded "
                "parent belongs to: seven boundary rings, 55 levels, the "
                "parent's own finest edge of 4,457.233 m, dt 20 s, and this "
                "tree's regional kernel set.  Two independent processes ran "
                "1,080 steps -- six forecast hours from 2026-08-12_06:00:00, "
                "config_apply_lbcs=true, config_len_disp 4,000 m -- and all "
                "THIRTEEN published history frames agree under masked digests "
                "while every whole-file digest differs on its own file_id.  "
                "THE CLASS IS EARNED ONCE AND THE EVIDENCE THAT IT IS A CLASS "
                "IS SEPARATE FROM THE MINT: the nest-ratio lane minted FIVE "
                "concentric culls of this same parent independently -- 4,440, "
                "7,975, 11,020, 14,050 and 15,755 cells, domains from 155 to "
                "565 km in radius, interface ratios from 13.5:1 to 3.6:1 -- "
                "and every one of the five returned the same verdict at the "
                "same 1.605x Courant margin under a 32.092 s limit.  Not one "
                "input this instrument reads distinguished them, which is why "
                "the geometry is not in the key.  RE-MINTED 2026-08-27 at "
                "the kernel set the provenance scrub produced (#377), on the "
                "same 14,050-cell cull and the same namelist: 1,080 steps in "
                "two independent processes, 192 s and 184 s, all THIRTEEN "
                "published frames identical under masked digests and every "
                "whole-file digest differing on its own file_id.  The "
                "superseded pair was taken at kernel set bdcfe014... and its "
                "receipts remain in the tree as the record of that run.  "
                + _DRY_MINT_CAVEAT
            ),
        ),
        "conus-x1-120km-dt120-z7": RegionalClass(
            class_id="conus-x1-120km-dt120-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=UNMEASURED_FINEST_EDGE_MM,
                dt_ms=120_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="x1.2562 (the v8.4.1 regional reference)",
            card="NVIDIA GeForce RTX 5090 (sm_120)",
            admitted_on="2026-08-27",
            mint_receipts=(
                "evidence/scrub-pins-20260827/conus-class-mint/"
                "mint-run1/forecast.json",
                "evidence/scrub-pins-20260827/conus-class-mint/"
                "mint-run2/forecast.json",
            ),
            minted_on_geometries=("conus-x1.2971",),
            basis=(
                "The reference-cull class, at the coarse end: 120 km cells, "
                "dt 120 s, seven rings, 55 levels.  FOUR independent "
                "processes ran the CANDIDATE-REGIONAL-DRY namelist "
                "(config_apply_lbcs=true) for 90 steps -- three forecast "
                "hours from 2026-08-12_06:00:00.  Two were assembled with the "
                "gate bypassed, which is what earned it; two went through the "
                "gate afterwards and reproduced it.  All SEVEN published "
                "history frames agree under masked digests across all four, "
                "while every whole-file digest differs on its own file_id.  "
                "RE-MINTED at the merged tip after the NVRTC "
                "reciprocal-rewrite fix (#355), which moved the third-order "
                "stencil denominator off a source literal in both cuda_driver "
                "and cuda_transport: the pre-fix pair is superseded.  ONE "
                "MEMBER ONLY, and that is a statement about the evidence "
                "rather than about the class -- no second cull of this parent "
                "has ever been made, so the five-member independence "
                "measurement the 4.6 km class carries has no counterpart "
                "here.  THE FINEST EDGE OF THIS CLASS IS NOT MEASURED and "
                "the row says so rather than guessing: this cull's grid "
                "publishes UNIT-SPHERE dcEdge and the Earth-scaled metrics "
                "sit in a static file no copy of this repository contains, "
                "which is why even the Courant admission reads the static for "
                "it.  The class therefore admits ONLY the geometries listed "
                "in SHIPPED_CONTRACTS and refuses a presented receipt, so it "
                "cannot grow to cover a cull nobody looked at.  Measuring the "
                "edge on a machine holding the reference mirror retires that "
                "restriction.  RE-MINTED 2026-08-27 at the kernel set the "
                "provenance scrub produced (#377), on the same cull and the "
                "same CANDIDATE-REGIONAL-DRY namelist: 90 steps in two "
                "independent processes, 13 s each on an RTX 5090, all SEVEN "
                "published frames identical under masked digests and every "
                "whole-file digest differing on its own file_id.  The "
                "superseded 2026-08-26 pair was taken at kernel set "
                "bdcfe014... on an RTX 5070 Ti and its receipts remain in the "
                "tree as the record of that run.  " + _DRY_MINT_CAVEAT
            ),
        ),
    }
)


#: Cull geometries whose own contract deck has been run and passed.  Keyed by
#: ``bdyMask`` digest, because that IS the zone geometry the deck measured.
SHIPPED_CONTRACTS: Mapping[str, RegionalContract] = MappingProxyType(
    {
        anchor.bdy_mask_sha256: anchor
        for anchor in (
            RegionalContract(
                bdy_mask_sha256=(
                    "acc95da7ecc58253e0085332eb5acc827d42287ecae4fe9ea88bd64060f4d67e"
                ),
                n_cells=2_971,
                boundary_zone_width=7,
                class_id="conus-x1-120km-dt120-z7",
                mesh_row="conus-x1.2971",
                card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
                admitted_on="2026-08-26",
                contract_receipt=(
                    "evidence/regional-cuda-l5-20260826/contract/contract-run1.json"
                ),
                basis=(
                    "8 of 8 decks bitwise against the v8.4.1 CPU authority "
                    "over 10,443,332 float32 values compared as raw bit "
                    "patterns, 22 of 22 declared kernels covered, dual-run "
                    "stable across two independent processes (84 of 84 "
                    "payload digests identical), and every deck re-run with a "
                    "deliberately wrong zone geometry FAILS."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "2baf091d718efcbbcb3f9385d55b0f224c1ea54fc3d75bfa8f108fc8a1fca158"
                ),
                n_cells=11_020,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.11020",
                card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/swath-as-lam-20260827/contract/contract-run1.json"
                ),
                basis=(
                    "The first regional geometry earned on a mesh this "
                    "program placed for itself: 121,182 cells become 11,020, "
                    "rings 9,220/240/245/251/257/263/269/275, Euler "
                    "characteristic 1.  8 of 8 decks bitwise against the "
                    "v8.4.1 CPU authority on THIS cull's zone geometry (544 "
                    "spec cells, 1,016 relax cells, 1,644 spec edges, 3,072 "
                    "relax edges), 22 of 22 declared kernels covered, "
                    "dual-run stable, every mutation control with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "e2d18b34c866e2768896467f21ec11d908c00425ed36409e100d25827ed6b68e"
                ),
                n_cells=4_440,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.4440",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d045/contract.json"
                ),
                basis=(
                    "The 0.45x cull of the same parent: 4,440 cells, rings "
                    "2,937/207/209/212/215/218/217/225.  8 of 8 decks bitwise "
                    "on THIS cull's zone geometry (442 spec cells, 854 relax "
                    "cells, 1,332 spec edges, 2,570 relax edges), 22 of 22 "
                    "kernels covered, every deck dual-run identical in "
                    "process, all eight mutation controls with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "401c8c160948dadffef45cf75632970d73e8491b1799ea1639a3da794e76e29f"
                ),
                n_cells=7_975,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.7975",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d070/contract.json"
                ),
                basis=(
                    "The 0.70x cull: 7,975 cells, rings "
                    "6,269/254/251/249/242/241/237/232.  Those ring "
                    "populations SHRINK outward, and that is the mesh being "
                    "variable resolution rather than torn.  8 of 8 decks "
                    "bitwise on THIS cull's zone geometry (469 spec cells, "
                    "983 relax cells, 1,399 spec edges, 2,935 relax edges), "
                    "22 of 22 kernels covered, all eight mutation controls "
                    "with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "a8e66046452db881bb4a9da08952610207ee5aa2e0a58d48b1d2348b48f84088"
                ),
                n_cells=14_050,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.14050",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d135/contract.json"
                ),
                basis=(
                    "The 1.35x cull -- the measured knee, and the cut every "
                    "shipped swath row takes from 2026-08-27: 14,050 cells, "
                    "rings 12,094/297/298/293/285/270/264/249, driven-ring "
                    "cells 10.86 to 14.30 km wide against a 71.0 km coarse "
                    "parent, about 5.7:1 at the interface.  8 of 8 decks "
                    "bitwise on THIS cull's zone geometry (513 spec cells, "
                    "1,146 relax cells, 1,508 spec edges, 3,405 relax "
                    "edges), 22 of 22 kernels covered, all eight mutation "
                    "controls with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "1726f1a23b6140d21a2d96897939d7b9eda990b78d31579722cb6293f66533db"
                ),
                n_cells=15_755,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.15755",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d170/contract.json"
                ),
                basis=(
                    "The gentlest boundary interface this program has run: "
                    "15,755 cells, rings 14,208/240/239/229/225/217/205/192, "
                    "driven-ring cells 15.89 to 25.27 km wide, about 3.4:1 at "
                    "the interface.  8 of 8 decks bitwise on THIS cull's zone "
                    "geometry (397 spec cells, 910 relax cells, 1,169 spec "
                    "edges, 2,696 relax edges), 22 of 22 kernels covered, all "
                    "eight mutation controls with teeth.  The zone is SMALLER "
                    "in element count than the 1.35x cull's while the domain "
                    "is larger, because the rings out here are made of much "
                    "wider cells."
                ),
            ),
        )
    }
)


# ---------------------------------------------------------------------------
# the presented-receipt route: a cull the cascade made this cycle
# ---------------------------------------------------------------------------

#: Where a freshly-run contract deck's receipt is looked for.  A cycling
#: cascade re-places its swath every cycle, so the cull it must admit did not
#: exist when this file was written and cannot be a row in it.
#:
#: THE ROUTE IS EVIDENCE, NOT A DECLARATION, and the difference is the whole
#: of it.  A presented receipt admits a geometry only if it says, in its own
#: recorded numbers, everything a shipped row's basis sentence says: it was
#: written by ``tools/run_cuda_regional_contract.py``, it names THIS mesh's
#: ``bdyMask`` digest and cell count, it was produced at the same kernel-set
#: digest as the class admitting it, all 8 decks are bitwise against the CPU
#: authority, all 22 declared kernels are covered, and every mutation control
#: had teeth.  Anything short of that is refused by name.  A receipt cannot
#: admit a configuration a shipped row could not.
CONTRACT_LEDGER_ENVIRONMENT = "GPUWM_HEX_REGIONAL_CONTRACT_DIR"

#: What a presented receipt must assert about itself before it admits
#: anything.  Each entry is (json key path, required value, the sentence the
#: refusal uses when it is not there).
_REQUIRED_CONTRACT_CLAIMS: tuple[tuple[str, Any, str], ...] = (
    ("instrument", "run_cuda_regional_contract",
     "it was not written by tools/run_cuda_regional_contract.py"),
    ("all_decks_bitwise", True,
     "it does not report all 8 decks bitwise against the v8.4.1 CPU authority"),
    ("all_kernels_covered", True,
     "it does not report all 22 declared regional kernels covered by a deck"),
    ("all_controls_have_teeth", True,
     "it does not report every mutation control failing on a deliberately "
     "wrong zone geometry, so its decks are not proven able to fail"),
    ("dual_run_identical", True,
     "it does not report its decks reproducing across two runs"),
    ("decks_selected", False,
     "it was run with --decks, so it is a subset of the deck set and not the "
     "contract"),
)


def _ledger_directories(extra: Sequence[Path | str] = ()) -> list[Path]:
    directories: list[Path] = [Path(item) for item in extra]
    raw = os.environ.get(CONTRACT_LEDGER_ENVIRONMENT, "")
    for piece in raw.split(os.pathsep):
        if piece.strip():
            directories.append(Path(piece.strip()))
    return directories


def _read_contract_receipt(path: Path) -> Mapping[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def contract_receipt_defects(
    document: Mapping[str, Any],
    *,
    bdy_mask_sha256: str,
    n_cells: int | None,
    kernel_set: str,
) -> list[str]:
    """Everything wrong with a presented contract receipt, by name."""

    defects: list[str] = []
    for key, required, sentence in _REQUIRED_CONTRACT_CLAIMS:
        if document.get(key) != required:
            defects.append(sentence)
    measured = str(document.get("bdy_mask_sha256") or "")
    if measured != bdy_mask_sha256:
        defects.append(
            f"it was run on bdyMask digest {measured[:16] or '(absent)'}... "
            f"and this mesh's is {bdy_mask_sha256[:16]}..., so it measured a "
            f"different zone geometry"
        )
    if n_cells is not None and document.get("n_cells") is not None:
        if int(document["n_cells"]) != int(n_cells):
            defects.append(
                f"it was run on {int(document['n_cells'])} cells and this "
                f"mesh carries {int(n_cells)}"
            )
    recorded = str(document.get("kernel_set_sha256") or "")
    if recorded != kernel_set:
        defects.append(
            f"it was produced at kernel set {recorded[:16] or '(absent)'}... "
            f"and this tree's regional kernel set digests to "
            f"{kernel_set[:16]}..., so its decks measured different bytes "
            f"from the ones this run would launch"
        )
    return defects


def presented_contract(
    bdy_mask_sha256: str,
    *,
    n_cells: int | None,
    kernel_set: str,
    directories: Sequence[Path | str] = (),
) -> tuple[RegionalContract | None, list[str]]:
    """A contract deck receipt presented for this exact geometry.

    Returns the admitted contract and the reasons every candidate receipt was
    rejected, so a refusal can say what it looked at rather than only that it
    found nothing.
    """

    rejected: list[str] = []
    for directory in _ledger_directories(directories):
        if not directory.is_dir():
            rejected.append(f"{directory} is not a directory")
            continue
        for path in sorted(directory.glob("*.json")):
            document = _read_contract_receipt(path)
            if document is None:
                rejected.append(f"{path.name} is not a readable JSON object")
                continue
            defects = contract_receipt_defects(
                document,
                bdy_mask_sha256=bdy_mask_sha256,
                n_cells=n_cells,
                kernel_set=kernel_set,
            )
            if defects:
                rejected.append(f"{path.name}: {defects[0]}")
                continue
            class_id = str(document.get("class_id") or "")
            return (
                RegionalContract(
                    bdy_mask_sha256=bdy_mask_sha256,
                    n_cells=int(document.get("n_cells") or (n_cells or 0)),
                    boundary_zone_width=int(
                        document.get("boundary_zone_width") or 0
                    ),
                    class_id=class_id,
                    mesh_row=document.get("mesh_row"),
                    card=str(document.get("card") or "unrecorded"),
                    admitted_on=str(document.get("date_utc") or "unrecorded"),
                    contract_receipt=str(path),
                    basis=(
                        "PRESENTED contract receipt, checked by content: 8 of "
                        "8 decks bitwise against the v8.4.1 CPU authority on "
                        "this cull's own zone geometry, 22 of 22 declared "
                        "kernels covered, every mutation control with teeth, "
                        "dual-run identical, at this tree's own regional "
                        "kernel-set digest."
                    ),
                ),
                rejected,
            )
    return None, rejected


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------
def admitted_summary() -> str:
    """Human-readable roster for refusal messages."""

    if not ADMITTED_CLASSES:
        return "none"
    return ", ".join(sorted(ADMITTED_CLASSES))


def contract_summary() -> str:
    if not SHIPPED_CONTRACTS:
        return "none"
    return ", ".join(
        sorted(
            row.mesh_row or row.bdy_mask_sha256[:16]
            for row in SHIPPED_CONTRACTS.values()
        )
    )


def admitted_class(class_id: str | None) -> RegionalClass | None:
    if not class_id:
        return None
    return ADMITTED_CLASSES.get(str(class_id))


def admitted_class_for_key(key: RegionalClassKey) -> RegionalClass | None:
    for row in ADMITTED_CLASSES.values():
        if row.key.matches(key):
            return row
    return None


def shipped_contract(bdy_mask_sha256: str | None) -> RegionalContract | None:
    if not bdy_mask_sha256:
        return None
    return SHIPPED_CONTRACTS.get(str(bdy_mask_sha256))


def contract_for_row(mesh_row: str | None) -> RegionalContract | None:
    if not mesh_row:
        return None
    for row in SHIPPED_CONTRACTS.values():
        if row.mesh_row == str(mesh_row):
            return row
    return None


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def unanchored_class_refusal(
    key: RegionalClassKey, mesh_row: str | None
) -> str:
    """The named refusal for a configuration class holding no mint."""

    named = "an unregistered regional mesh" if not mesh_row else repr(mesh_row)
    return (
        f"regional CUDA execution is refused for {named}: its configuration "
        f"class ({key.describe()}) holds no forecast mint, so no pair of runs "
        f"has ever shown that THIS kernel set reproduces itself at this "
        f"timestep on a domain of this shape -- the run would produce a "
        f"receipt nobody could verify, which is the one thing this gate "
        f"exists to prevent.  A class is minted ONCE, on any one cull of it, "
        f"by tools/run_cuda_regional_forecast.py --mint-anchor run twice, "
        f"and is then one row in "
        f"cuda_backend/regional_admission.ADMITTED_CLASSES.  Minted classes: "
        f"{admitted_summary()}.  NOTE WHAT IS AND IS NOT IN THE KEY: the "
        f"cull's own bytes are NOT, because five concentric culls of one "
        f"parent were minted independently and no input the mint reads "
        f"distinguished them; the kernel set IS, so an edit to a regional "
        f"translation unit lapses every class until it is re-minted"
    )


def uncontracted_geometry_refusal(
    bdy_mask_sha256: str | None,
    n_cells: int | None,
    class_id: str,
    rejected: Sequence[str] = (),
) -> str:
    """The named refusal for a cull whose own zone geometry is unchecked."""

    digest = (
        "no bdyMask digest was supplied"
        if not bdy_mask_sha256
        else f"bdyMask digest {bdy_mask_sha256[:16]}..."
    )
    cells = "" if n_cells is None else f", {int(n_cells)} cells"
    looked = ""
    if rejected:
        looked = "  Receipts looked at and why each was rejected: " + "; ".join(
            rejected[:6]
        )
    return (
        f"regional CUDA execution is refused for this cull ({digest}{cells}): "
        f"its configuration class {class_id!r} IS minted, but no contract "
        f"deck has been run on THIS cull's own zone geometry.  The concrete "
        f"breakage that prevents: the 22 regional kernels are indexed by "
        f"ring, and a deck run on another cull's rings measured another "
        f"cull's specified and relaxation zones -- a mask-4/5 downgrade, a "
        f"spec-zone pgrad mask or a one-cell ring-7 edge that is wrong HERE "
        f"would be reproduced identically by two runs and caught by neither.  "
        f"Run tools/run_cuda_regional_contract.py against this grid and "
        f"present its receipt through the {CONTRACT_LEDGER_ENVIRONMENT} "
        f"directory (or add a row to SHIPPED_CONTRACTS).  Contracted "
        f"geometries: {contract_summary()}.{looked}"
    )


def class_mismatch_refusal(
    mesh_row: str, expected: RegionalClassKey, measured: RegionalClassKey
) -> str:
    """The named refusal when a contracted geometry is run off its class."""

    differing = expected.differing_fields(measured)
    return (
        f"regional CUDA execution is refused for {mesh_row!r}: its contract "
        f"deck belongs to a class minted at [{expected.describe()}] and this "
        f"run would execute [{measured.describe()}].  The fields that differ "
        f"are {differing}.  The mint measured a different configuration from "
        f"the one this run would execute, so it says nothing about it"
    )


def digest_mismatch_refusal(mesh_row: str, expected: str, measured: str) -> str:
    """The named refusal when a registered row's bdyMask bytes moved."""

    return (
        f"regional CUDA execution is refused for {mesh_row!r}: the contract "
        f"deck registered bdyMask digest {expected[:16]}... but this mesh's "
        f"boundary masks digest {measured[:16]}... -- the zone geometry the "
        f"deck was run on is not the zone geometry this run would execute, so "
        f"its per-kernel receipt says nothing about it"
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RegionalAnchor:
    """What admitted one regional run: a class mint plus a geometry deck.

    Kept under the name the two call sites and every archived receipt already
    use.  ``mesh_row``, ``n_cells``, ``boundary_zone_width``,
    ``bdy_mask_sha256``, ``card``, ``admitted_on``, ``contract_receipt``,
    ``forecast_anchor`` and ``basis`` all still read, so a caller that
    recorded an anchor before the re-keying records the same fields after it.
    """

    mesh_row: str | None
    n_cells: int
    boundary_zone_width: int
    bdy_mask_sha256: str
    card: str
    admitted_on: str
    contract_receipt: str
    forecast_anchor: str
    basis: str
    class_id: str = ""
    class_key: RegionalClassKey | None = None
    contract_route: str = "shipped"

    def as_dict(self) -> dict[str, object]:
        return {
            "mesh_row": self.mesh_row,
            "n_cells": self.n_cells,
            "boundary_zone_width": self.boundary_zone_width,
            "bdy_mask_sha256": self.bdy_mask_sha256,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "contract_receipt": self.contract_receipt,
            "forecast_anchor": self.forecast_anchor,
            "basis": self.basis,
            "class_id": self.class_id,
            "class_key": None if self.class_key is None else self.class_key.as_dict(),
            "contract_route": self.contract_route,
        }


def _compose(
    contract: RegionalContract,
    klass: RegionalClass,
    *,
    n_cells: int | None,
    route: str,
) -> RegionalAnchor:
    return RegionalAnchor(
        mesh_row=contract.mesh_row,
        n_cells=int(contract.n_cells or (n_cells or 0)),
        boundary_zone_width=int(
            contract.boundary_zone_width or klass.key.boundary_zone_width
        ),
        bdy_mask_sha256=contract.bdy_mask_sha256,
        card=contract.card,
        admitted_on=contract.admitted_on,
        contract_receipt=contract.contract_receipt,
        forecast_anchor=klass.mint_receipts[0] if klass.mint_receipts else "",
        basis=(
            f"CLASS {klass.class_id}: {klass.basis}  GEOMETRY "
            f"{contract.bdy_mask_sha256[:16]}...: {contract.basis}"
        ),
        class_id=klass.class_id,
        class_key=klass.key,
        contract_route=route,
    )


def require_regional_anchor(
    mesh_row: str | None,
    *,
    bdy_mask_sha256: str | None,
    n_cells: int | None = None,
    boundary_zone_width: int | None = None,
    n_vert_levels: int | None = None,
    finest_edge_m: float | None = None,
    dt_seconds: float | None = None,
    contract_directories: Sequence[Path | str] = (),
    kernel_set: str | None = None,
) -> RegionalAnchor:
    """Admit one regional configuration for CUDA execution, or refuse by name.

    Two halves, checked in the order that makes the message useful.  First the
    GEOMETRY: has a contract deck ever been run on these rings?  Then the
    CLASS: is the configuration this run would execute one a forecast mint has
    ever been taken of?

    Callers that know the configuration -- the regional forecast opener, which
    holds the mesh and the namelist -- pass ``boundary_zone_width``,
    ``n_vert_levels``, ``finest_edge_m`` and ``dt_seconds``, and the class key
    is MEASURED from this run rather than trusted from a table.  The
    validation site inside the pinned global driver holds only a mesh, so it
    resolves the class through the geometry's own contract row and the class
    key it names; that site is reachable only by handing a culled mesh to the
    global door, which is a configuration nobody runs deliberately.

    Raises :class:`RuntimeError` (``RegionalAdmissionRefusal``) carrying a
    message that names the concrete breakage, per the gate law.  Callers
    convert it to their own refusal type when a typed refusal is the local
    convention.
    """

    digest = None if not bdy_mask_sha256 else str(bdy_mask_sha256)
    resolved_kernel_set = kernel_set or kernel_set_sha256()

    # A named row whose bytes moved is refused before anything else, because
    # the name and the bytes disagreeing is a different defect from either
    # half being missing.
    named = contract_for_row(mesh_row)
    if named is not None and digest and named.bdy_mask_sha256 != digest:
        raise RegionalAdmissionRefusal(
            digest_mismatch_refusal(str(mesh_row), named.bdy_mask_sha256, digest)
        )
    # NAMING A ROW IS A CLAIM, AND THE BYTES ARE THE PROOF: both must hold.
    # A caller that names a row nobody has a deck for is refused even when
    # its bytes happen to match a geometry that IS contracted.  Falling
    # through to the digest would admit a run whose own receipt names an
    # identity no registry holds -- two classifiers disagreeing about which
    # mesh they are looking at, which is a defect shape this program has
    # already lost a day to.
    if mesh_row and named is None and shipped_contract(digest) is not None:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for {mesh_row!r}: no "
            f"contract deck is registered under that name, and the bdyMask "
            f"digest supplied belongs to "
            f"{shipped_contract(digest).mesh_row!r}.  The name and the bytes "
            f"disagree about which cull this is, so any receipt this run "
            f"produced would name an identity no registry holds.  Contracted "
            f"geometries: {contract_summary()}"
        )

    contract = shipped_contract(digest) or named
    route = "shipped"
    rejected: list[str] = []
    if contract is None and digest:
        contract, rejected = presented_contract(
            digest,
            n_cells=n_cells,
            kernel_set=resolved_kernel_set,
            directories=contract_directories,
        )
        route = "presented"

    # The class key.  Measured from this run when the caller knows it, taken
    # from the geometry's contract row when it does not.
    measured_key: RegionalClassKey | None = None
    if None not in (boundary_zone_width, n_vert_levels, finest_edge_m, dt_seconds):
        measured_key = RegionalClassKey.build(
            boundary_zone_width=int(boundary_zone_width),  # type: ignore[arg-type]
            n_vert_levels=int(n_vert_levels),  # type: ignore[arg-type]
            finest_edge_m=float(finest_edge_m),  # type: ignore[arg-type]
            dt_seconds=float(dt_seconds),  # type: ignore[arg-type]
            kernel_set=resolved_kernel_set,
        )

    if contract is None:
        class_id = (
            "(unresolved)"
            if measured_key is None
            else (
                admitted_class_for_key(measured_key).class_id
                if admitted_class_for_key(measured_key) is not None
                else "(unminted)"
            )
        )
        if measured_key is not None and class_id == "(unminted)":
            raise RegionalAdmissionRefusal(
                unanchored_class_refusal(measured_key, mesh_row)
            )
        raise RegionalAdmissionRefusal(
            uncontracted_geometry_refusal(digest, n_cells, class_id, rejected)
        )

    klass = admitted_class(contract.class_id)
    if klass is None:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for this cull (bdyMask "
            f"digest {contract.bdy_mask_sha256[:16]}...): its contract deck "
            f"names configuration class {contract.class_id!r}, which holds no "
            f"forecast mint.  Minted classes: {admitted_summary()}"
        )

    if measured_key is not None and not klass.key.matches(measured_key):
        raise RegionalAdmissionRefusal(
            class_mismatch_refusal(
                str(mesh_row or contract.mesh_row or "an unregistered cull"),
                klass.key,
                measured_key,
            )
        )
    if route == "presented" and not klass.key.finest_edge_measured:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for this cull (bdyMask "
            f"digest {contract.bdy_mask_sha256[:16]}...): its contract deck "
            f"was presented at run time and it names class "
            f"{klass.class_id!r}, whose own finest edge was never measured -- "
            f"so the class key cannot tell this cull's resolution from any "
            f"other at the same zone width, column count and timestep, and "
            f"the mint would be inherited by a domain nobody compared it to.  "
            f"A class with an unmeasured finest edge admits only the "
            f"geometries written into SHIPPED_CONTRACTS"
        )
    if measured_key is None and klass.key.kernel_set_sha256 != resolved_kernel_set:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for "
            f"{mesh_row or contract.mesh_row!r}: class {klass.class_id!r} was "
            f"minted at kernel set {klass.key.kernel_set_sha256[:16]}... and "
            f"this tree's regional kernel set digests to "
            f"{resolved_kernel_set[:16]}...  The mint is a statement about "
            f"bytes and those bytes have moved, so it no longer describes "
            f"what this run would launch; re-mint the class"
        )

    if n_cells is not None and contract.n_cells and int(n_cells) != int(contract.n_cells):
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for "
            f"{mesh_row or contract.mesh_row!r}: the contract deck was run on "
            f"{contract.n_cells} cells and this mesh carries {int(n_cells)}.  "
            f"The bdyMask digest matched, so this is a mesh whose rings are "
            f"those of a different domain -- exactly the disagreement the "
            f"digest cross-check exists to catch.  For the GEOMETRY half of "
            f"an anchor a different cull is a different configuration, and "
            f"that did not change when the forecast mint moved to a class: "
            f"the 22 regional kernels are indexed by ring and a deck run on "
            f"other rings measured another domain's zones"
        )

    return _compose(contract, klass, n_cells=n_cells, route=route)


# ---------------------------------------------------------------------------
# retired surface, kept resolvable
# ---------------------------------------------------------------------------
#: RETIRED 2026-08-27 by the class re-keying, and kept only so a reader who
#: meets the name in an archived receipt can resolve it.  It was a mapping
#: from mesh-row name to a single row carrying BOTH halves of an anchor, so a
#: re-placed cull of an already-minted class could not be admitted without
#: paying for a second forecast mint.  The two halves now live in
#: ``ADMITTED_CLASSES`` and ``SHIPPED_CONTRACTS``; this view composes them
#: back so an old caller reads the same fields.
ADMITTED_REGIONS: Mapping[str, RegionalAnchor] = MappingProxyType(
    {
        contract.mesh_row: _compose(
            contract,
            ADMITTED_CLASSES[contract.class_id],
            n_cells=contract.n_cells,
            route="shipped",
        )
        for contract in SHIPPED_CONTRACTS.values()
        if contract.mesh_row and contract.class_id in ADMITTED_CLASSES
    }
)


def admitted_region(mesh_row: str) -> RegionalAnchor | None:
    """The anchor admitting ``mesh_row``, or ``None``."""

    return ADMITTED_REGIONS.get(str(mesh_row))


def admitted_region_by_digest(bdy_mask_sha256: str | None) -> RegionalAnchor | None:
    """The anchor whose registered boundary masks are exactly these bytes."""

    contract = shipped_contract(bdy_mask_sha256)
    if contract is None or contract.class_id not in ADMITTED_CLASSES:
        return None
    return _compose(
        contract,
        ADMITTED_CLASSES[contract.class_id],
        n_cells=contract.n_cells,
        route="shipped",
    )


def unanchored_refusal(mesh_row: str | None, bdy_mask_sha256: str | None) -> str:
    """The named refusal for a regional configuration holding no anchor."""

    return uncontracted_geometry_refusal(
        bdy_mask_sha256, None, "(unresolved)", ()
    )


__all__ = [
    "ADMITTED_CLASSES",
    "ADMITTED_REGIONS",
    "CONTRACT_LEDGER_ENVIRONMENT",
    "MINTED_KERNEL_SET_SHA256",
    "REGIONAL_KERNEL_SOURCES",
    "RegionalAdmissionRefusal",
    "RegionalAnchor",
    "RegionalClass",
    "RegionalClassKey",
    "RegionalContract",
    "SHIPPED_CONTRACTS",
    "UNMEASURED_FINEST_EDGE_MM",
    "admitted_class",
    "admitted_class_for_key",
    "admitted_region",
    "admitted_region_by_digest",
    "admitted_summary",
    "class_mismatch_refusal",
    "contract_for_row",
    "contract_receipt_defects",
    "contract_summary",
    "digest_mismatch_refusal",
    "kernel_set_sha256",
    "presented_contract",
    "require_regional_anchor",
    "shipped_contract",
    "unanchored_class_refusal",
    "unanchored_refusal",
    "uncontracted_geometry_refusal",
]
