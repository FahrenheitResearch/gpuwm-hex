"""The ONE device-memory admission surface for the v8.4.1 CUDA lane.

Every "will this mesh fit this card?" decision in the project answers from
this module: the forecast door's ``--preflight`` verdict, the door's real-run
admission, and the driver's own floor (``MIN_FREE_DEVICE_BYTES``, rebound per
mesh by ``tools/mpas_mesh_binding.py``) all call :func:`required_free_bytes`
with the same model, so they agree to the byte -- and therefore to the cell.
Before this module existed the door predicted from a measured affine row
while the driver held a separate linear floor scaled from an asserted 24 GiB
constant; a card between the two answers passed one gate and died on the
other, which is the concrete breakage a single surface prevents.

THE MODEL HAS A SHAPE NOW, AND THE SHAPE IS NOT A LINE (2026-08-27).
--------------------------------------------------------------------

Every row this module has ever shipped was ``fixed + slope * cells``.  That
form was refuted by measurement, not by argument:

* Across five full-physics forecast-door runs on placed variable-resolution
  meshes the affine row's error spanned **+3.89 % to -6.76 %**, and one mesh
  with 15,343 MORE cells peaked 318 MiB LOWER.  A function linear in cell
  count cannot do that.
* One of those runs (``v6.75.112676``) peaked **96 MiB past its own
  ``required_free``** -- the flat 512 MiB headroom did not cover it.
* On the limited-area path the affine row over-predicted every one of five
  concentric culls by 1.2-1.9 GiB: 4,440 -> 15,755 cells (3.5x) moved the
  measured peak only 5,552 -> 7,728 MiB (1.39x).

The per-allocation ledger (#264, ``tools/device_memory_ledger/``) says why,
and the arithmetic below reproduces its measured block sizes exactly.  The
process footprint is

    peak(cells, card) = core(card, configuration)
                      + SUM over tiled workspaces of ws(min(cells, tile(card)))
                      + bytes_per_cell * cells

THE TILED WORKSPACES ARE THE MISSING TERM.  Three physics kernels keep their
column arrays in a GLOBAL device workspace sized to the threads the CARD can
hold in flight, capped by the mesh -- ``min(cells, tile)``.  Below the tile
the term is proportional to cells; above it the term is CONSTANT.  That knee
is the shape a line cannot have, and the tiles are large enough to matter:

    Grell-Freitas  gf.py::gf_tile_columns    tile = SMs x 4 x 64
    YSU            ysu.py::ysu_tile_columns  tile = SMs x 16 x 32
    RRTMG SW/LW    rrtmg_lw.py::batch_column_chunk
                                             chunk = quantised(resident
                                             threads / g-points), ceiling

On the 170 SM part the knees are 43,520 cells (GF) and 87,040 cells (YSU).
Both published fit meshes straddle them: ``x1.40962`` sits BELOW both and
``x4.163842`` ABOVE both, so the affine fit charged the tile growth between
them as if it were per-cell.  That is where 2,167 B of the retired row's
98,748 B/cell slope came from, and it is why the retired row over-predicts
every small mesh and every limited-area cull.

MEASURED PROOF THE TILE ARITHMETIC IS THE RIGHT ARITHMETIC.  The #264 ledger
records each site's largest single allocation at both meshes.  Every site in
the model scales exactly 4.00x with the 4x cell count EXCEPT three, and the
formulas above reproduce all three to the byte:

    site                          x1 MiB     x4 MiB   ratio   this module
    gpuwm/core/gf.py (workspace)  1,262.0    1,338.8  1.06    641/680 blocks
    gpuwm/core/ysu.py (workspace)   157.6      334.7  2.12    1,281/2,720
    gpuwm/core/rrtmg_sw.py (wk_d) 1,745.6    1,745.6  1.00    2,048 columns

THE PEAK CHANGES SCHEME WITH MESH SIZE, which is the same finding from the
other side.  At 40,962 cells the largest thing live at the global peak is the
RRTMG shortwave ``spcvmc`` workspace -- one mesh-independent 1,830,420,480 B
block.  At 151,649 and 163,842 cells the peak instant is in the dycore and
that block is not live at it at all.  A single line through one radiation-set
point and one dycore-set point describes neither regime.

WHAT THE MARGIN ABSORBS, BY NAME (the gate law, 2026-08-16).
---------------------------------------------------------------

The retired 512 MiB headroom named nothing and failed by 96 MiB.  This model
carries a margin with two named, measured components and no padding:

1. **Arena placement of the largest mesh-independent block.**  There is no
   arena in this project: allocation goes to CuPy's default ``MemoryPool``,
   whose high-water is the set of blocks it has had to ``cudaMalloc``, not
   the peak live set.  Whether the 1,745.6 MiB shortwave workspace is
   servable from the pool's free list at the instant radiation runs depends
   on the arena layout, and the layout depends on the exact byte sizes of
   every per-cell array allocated before it.  This is MEASURED, twice, on an
   idle card: freeing 38.4 MiB EARLIER in the run moved ``x1.40962``'s pool
   high-water from 5,404.5 to 7,111.7 MiB -- **a 1,707.2 MiB rise from a
   38 MiB saving**, because that block stopped being servable from the free
   list (``STATE.md``, the construction-diagnostic hold).  The margin is that
   block, priced at the card's own chunk width: 1,745.6 MiB on a 170 SM
   part, 872.8 MiB on a 68 or 70 SM part.  The concrete breakage: a card
   admitted without it dies inside the CuPy allocation at the first
   radiation call, minutes into a run.
2. **Instrument convention.**  The fitted rows are per-process
   ``nvidia-smi``; the door faces whole-device free memory.  Measured side by
   side, the two conventions differ by 11.2 MiB (x4 arm) and 0.75 MiB
   (graded arm).  The margin carries the larger.

Nothing else is in the margin.  If a term is not measured it is not here.

PER-CARD ROWS, AND WHY #366 WAS A DEFECT.
-----------------------------------------

The retired module carried :data:`CARD_TIER_ROWS` and **nothing ever read
it**: ``forecast_door._resolve_model`` returned the 170 SM row unless the
user typed ``--device-fixed-mib`` and ``--device-bytes-per-cell`` by hand.
So the 10 GiB desktop was priced with a 32 GiB card's fixed term and refused
``x1.40962`` and ``v15.150.38857`` -- two meshes it had been MEASURED running
with 2,244 MiB to spare.  That is ledger #366.  The door now reads the card's
multiprocessor count at decision time and selects or derives its row, so a
card that was never measured still gets its own arithmetic.

A DERIVED row is not a measured one and says so.  It scales the reference
card's core by the two mechanisms that are known and measured to scale with
the card -- the CUDA local-memory backing store (per-context, sized to the
widest launched frame times resident threads) and the radiation chunk width
-- and it lands CONSERVATIVE on both cards that have their own measurement:
+182.7 MiB on the 70 SM part and +490.8 MiB on the 68 SM part, both
over-predictions.  Direction matters more than magnitude for a gate.

CONFIGURATIONS ARE DATA, NOT CODE PATHS (the arbitrary acceptance test).
------------------------------------------------------------------------

The global and limited-area paths build different device stacks -- the
regional one carries a padded atmosphere, two levels of lateral-boundary
state and 22 more kernels -- so they carry different cores.  They are two
rows of one table, not two functions.  Adding a card or a configuration is a
row; it is never a new branch.

STATED LIMITS, ON PAGE ONE OF THE MODULE.
-----------------------------------------

* The limited-area core is an ENVELOPE, not a fit: the per-allocation
  instrument has never been run on a regional mesh, so the core is the
  largest residue over five whole-device forecast-door samples.  One #264
  arm on a cull replaces it.
* The 16 GiB and 10 GiB cores were measured at hex ``5252421``, not at the
  merged tip, and carry :data:`PIN_RESTATEMENT_BYTES` for it.
* Every measurement in this module is 55 vertical levels, float32, WSM6 + GF
  + YSU + YSU-GWDO + revised-MO + NoahMP + cloud fraction + legacy RRTMG.
  The tiled-workspace terms are functions of ``nz``; nothing else here is.
* The four-swath spread that opened this lane is EXPLAINED IN MECHANISM but
  NOT SEPARATED IN MEASUREMENT: the four runs share card, pin, timestep,
  schedule and radiation cadence, so arena placement is the only candidate
  left standing, and no A/B on those four meshes has been run to prove it.

This module is stdlib-only at import, deliberately: the forecast door
imports it before any refusal, on boxes with no CUDA lane at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "CARD_TIER_ROWS",
    "CONVENTION_MARGIN_BYTES",
    "CardProfile",
    "DEFAULT_HEADROOM_BYTES",
    "FLOOR_DERIVATION",
    "FOOTPRINT_MODEL",
    "FootprintModel",
    "GRELL_FREITAS_WORKSPACE",
    "KNOWN_CARDS",
    "MIB",
    "MODEL_VERTICAL_LEVELS",
    "NATIVE_MESH_CELLS",
    "PIN_RESTATEMENT_BYTES",
    "REFERENCE_CARD",
    "RETIRED_AFFINE_ROW_20260826",
    "RETIRED_CARD_TIER_ROWS_20260826",
    "RETIRED_FLAT_HEADROOM_BYTES",
    "RETIRED_LINEAR_FLOOR_BYTES",
    "RETIRED_ROW_20260825",
    "SHAPED_ROWS",
    "ShapedFootprintModel",
    "TiledWorkspace",
    "YSU_WORKSPACE",
    "card_profile_from_attributes",
    "model_for_card",
    "native_device_floor_bytes",
    "radiation_chunk_columns",
    "required_free_bytes",
    "retired_affine_row_floor_bytes",
    "retired_converged_row_floor_bytes",
    "retired_linear_floor_bytes",
    "row_key",
    "shortwave_workspace_bytes",
    "tier_model",
]

MIB = 1024**2

#: The vertical level count every measurement in this module was taken at.
#: The tiled-workspace terms are functions of it; the per-cell slope and the
#: cores are NOT re-derivable at another ``nz`` from anything measured here.
MODEL_VERTICAL_LEVELS = 55

#: The native x4.163842 cell count the frozen proof runs at.  Restated here
#: (rather than imported from the registry) because this module must import
#: with no numpy on the box; the registry test asserts the two agree.
NATIVE_MESH_CELLS = 163_842


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CardProfile:
    """The card properties every card-scaled term in the model reads.

    Two numbers, both published by the CUDA driver and both read at decision
    time by the door: the multiprocessor count and the maximum resident
    threads per multiprocessor.  Nothing here is a nameplate memory size --
    the door measures free memory separately and never carries it between
    decisions.
    """

    name: str
    multiprocessors: int
    max_threads_per_sm: int = 1536

    @property
    def resident_threads(self) -> int:
        """Threads the card can hold resident: what CUDA prices a per-thread
        local frame at, and what the radiation chunk width is derived from."""

        return int(self.multiprocessors) * int(self.max_threads_per_sm)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "multiprocessors": int(self.multiprocessors),
            "max_threads_per_sm": int(self.max_threads_per_sm),
            "resident_threads": self.resident_threads,
        }


def card_profile_from_attributes(
    name: str, attributes: Mapping[str, Any]
) -> CardProfile:
    """A :class:`CardProfile` from a CuPy ``Device.attributes`` mapping.

    The door calls this with what the driver reports right now.  A card this
    project has never seen gets a real profile rather than a default, which
    is the whole point of #366's fix.
    """

    sms = attributes.get("MultiProcessorCount")
    threads = attributes.get("MaxThreadsPerMultiProcessor")
    if not sms or int(sms) <= 0:
        raise ValueError(
            "MultiProcessorCount missing or non-positive; the device-memory "
            "model prices the CUDA local-memory backing store and the "
            "radiation chunk width from it, so a card with no SM count "
            "cannot be given a footprint row"
        )
    if not threads or int(threads) <= 0:
        threads = 1536
    return CardProfile(
        name=str(name), multiprocessors=int(sms), max_threads_per_sm=int(threads)
    )


#: The cards this project has run on.  A row here is a convenience for
#: reports and tests; the door never looks a card up by name, it reads the
#: driver.
KNOWN_CARDS: Mapping[str, CardProfile] = MappingProxyType(
    {
        "32gib-170sm": CardProfile("RTX 5090 (32,607 MiB, sm_120)", 170, 1536),
        "16gib-70sm": CardProfile("RTX 5070 Ti (16,303 MiB, sm_120)", 70, 1536),
        "10gib-68sm": CardProfile("RTX 3080 (10,240 MiB, sm_86, WDDM)", 68, 1536),
    }
)

#: The card every measured row of record was taken on, and the card a
#: derived row is derived FROM.
REFERENCE_CARD = KNOWN_CARDS["32gib-170sm"]


# ---------------------------------------------------------------------------
# the tiled workspaces -- the term the affine row did not have
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TiledWorkspace:
    """A physics workspace sized to ``min(cells, tile)`` where the tile is a
    property of the CARD.

    Grell-Freitas and YSU both moved their column arrays out of the CUDA
    per-thread local frame and into a global workspace sized to the threads
    in flight (the #335 frame cut).  A local frame is priced by CUDA at the
    card's WHOLE resident-thread capacity; a workspace is priced at the tile.
    That is why these terms exist at all, and why they are card-scaled.

    ``slots`` and ``block`` are the engine's own constants, restated here
    because this module imports with no engine on the box.
    ``tests/test_device_admission.py`` asserts they still agree with
    ``gpuwm.core`` whenever the engine checkout is importable.
    """

    name: str
    slots: int
    block: int
    blocks_per_sm: int
    level_offset: int
    level_floor: int
    dtype_bytes: int
    site: str
    provenance: str

    def tile_columns(self, cells: int, card: CardProfile) -> int:
        """Columns kept in flight: the card's capacity, capped by the mesh."""

        tile = int(self.blocks_per_sm) * int(self.block) * int(card.multiprocessors)
        return max(int(self.block), min(int(cells), tile))

    def bytes_for(
        self, cells: int, card: CardProfile, levels: int = MODEL_VERTICAL_LEVELS
    ) -> int:
        """The workspace this card allocates for this mesh, in bytes.

        Rounded up to whole blocks, because both kernels interleave the
        workspace by lane within a block: the unit of allocation is one
        block's region, not one column's.
        """

        columns = self.tile_columns(cells, card)
        blocks = -(-columns // int(self.block))
        extent = max(int(self.level_floor), int(levels)) + int(self.level_offset)
        return blocks * int(self.slots) * extent * int(self.block) * int(self.dtype_bytes)

    def knee_cells(self, card: CardProfile) -> int:
        """The cell count above which this term stops growing."""

        return int(self.blocks_per_sm) * int(self.block) * int(card.multiprocessors)


#: Grell-Freitas.  ``GFWS_SLOT_COUNT_COL + GFWS_SLOT_COUNT_DRV = 90 + 36``,
#: ``GF_BLOCK = 64``, ``GF_TILE_BLOCKS_PER_SM = 4``, extent
#: ``gf_kernel_capacity(nz) + 9`` with ``gf_kernel_capacity`` floored at the
#: compiled ``_GF_KMAX_DEFAULT = 40``.  Reproduces the ledger's measured
#: 1,262.0 MiB (x1, 641 blocks) and 1,338.8 MiB (x4, 680 blocks) exactly.
GRELL_FREITAS_WORKSPACE = TiledWorkspace(
    name="grell_freitas_column_workspace",
    slots=90 + 36,
    block=64,
    blocks_per_sm=4,
    level_offset=9,
    level_floor=40,
    dtype_bytes=4,
    site="gpuwm/core/gf.py::gf_workspace_floats via gf_tile_columns",
    provenance=(
        "engine constants GFWS_SLOT_COUNT_COL=90, GFWS_SLOT_COUNT_DRV=36, "
        "GF_BLOCK=64, GF_TILE_BLOCKS_PER_SM=4, _GF_KMAX_DEFAULT=40.  "
        "Verified against the #264 merged-tip ledger: 1,262.0 MiB at 40,962 "
        "cells and 1,338.8 MiB at 163,842 on the 170 SM card, ratio 1.06, "
        "which is 680/641 blocks and not 4.00"
    ),
)

#: YSU.  ``YSUWS_SLOTS = 18``, ``YSU_BLOCK = 32``,
#: ``YSU_TILE_BLOCKS_PER_SM = 16``, extent ``nz + 1``.  Reproduces the
#: ledger's 157.6 MiB (x1, 1,281 blocks) and 334.7 MiB (x4, 2,720 blocks).
YSU_WORKSPACE = TiledWorkspace(
    name="ysu_column_workspace",
    slots=18,
    block=32,
    blocks_per_sm=16,
    level_offset=1,
    level_floor=0,
    dtype_bytes=4,
    site="gpuwm/core/physics_inventory.py::ysu_workspace_floats via ysu_tile_columns",
    provenance=(
        "engine constants YSUWS_SLOTS=18, YSU_BLOCK=32, "
        "YSU_TILE_BLOCKS_PER_SM=16.  Verified against the #264 merged-tip "
        "ledger: 157.6 MiB at 40,962 cells and 334.7 MiB at 163,842 on the "
        "170 SM card, ratio 2.12, which is 2,720/1,281 blocks and not 4.00"
    ),
)

#: The workspaces the model charges, in order.
TILED_WORKSPACES: tuple[TiledWorkspace, ...] = (
    GRELL_FREITAS_WORKSPACE,
    YSU_WORKSPACE,
)


# ---------------------------------------------------------------------------
# the radiation chunk -- the card-scaled step function
# ---------------------------------------------------------------------------
#: ``gpuwm.core.rrtmg_lw.BATCH_CHUNK_QUANTUM``.
RADIATION_CHUNK_QUANTUM = 256
#: ``gpuwm.core.rrtmg_sw.SW_BATCH_COLUMN_CHUNK_CEILING`` and ``NGPTSW``.
SHORTWAVE_CHUNK_CEILING = 2048
SHORTWAVE_GPOINTS = 112
#: ``SPCVMC_WK_ARRAYS`` -- the float32 arrays the spcvmc workspace holds per
#: thread, one thread per (column, g-point) pair.
SPCVMC_WK_ARRAYS = 35


def radiation_chunk_columns(
    card: CardProfile,
    threads_per_column: int = SHORTWAVE_GPOINTS,
    ceiling: int = SHORTWAVE_CHUNK_CEILING,
    quantum: int = RADIATION_CHUNK_QUANTUM,
) -> int:
    """Columns per radiation chunk on this card -- ``batch_column_chunk``.

    The engine's own narrowing (#310): the smallest multiple of ``quantum``
    whose launch covers the card's resident threads, clamped to the ceiling.
    Restated here rather than imported because this module must price a
    footprint on a box with no CUDA lane; the test asserts the two agree.

    Measured consequence, which is why it is in the model: on the 70 SM part
    the shortwave chunk halves from 2,048 to 1,024 columns and the step-6
    pool event shrinks from +1,745.7 MiB to +872.8 MiB.
    """

    resident = int(card.resident_threads)
    if resident <= 0:
        return int(ceiling)
    need = -(-resident // int(threads_per_column))
    need = -(-need // int(quantum)) * int(quantum)
    return max(int(quantum), min(int(ceiling), need))


def shortwave_workspace_bytes(
    card: CardProfile, levels: int = MODEL_VERTICAL_LEVELS
) -> int:
    """The RRTMG shortwave ``spcvmc`` workspace: ONE contiguous block.

    ``cp.zeros((nc * NGPTSW, SPCVMC_WK_ARRAYS * nl1), float32)`` at
    ``gpuwm/core/rrtmg_sw.py``, where ``nc`` is the chunk width and
    ``nl1 = nlayers + 1 = levels + 2``.  At 2,048 columns and 55 levels that
    is exactly 1,830,420,480 B = 1,745.6 MiB, which is the block the #264
    ledger records as the largest single allocation live at the 40,962-cell
    global peak -- **identical at 163,842 cells**.

    This is the margin's first named component: it is the largest thing in
    the footprint that does not scale with the mesh, so it is the thing whose
    arena placement can move the pool high-water without any input changing.
    """

    columns = radiation_chunk_columns(card)
    return columns * SHORTWAVE_GPOINTS * SPCVMC_WK_ARRAYS * (int(levels) + 2) * 4


#: The margin's second named component: the gap between the per-process
#: ``nvidia-smi`` convention the rows are fitted on and the whole-device
#: convention a door faces.  MEASURED both ways in one process, twice:
#: 11.2 MiB on the x4 arm and 0.75 MiB on the graded arm.  The larger is
#: carried.
CONVENTION_MARGIN_BYTES = int(round(11.2 * MIB))

#: What a row measured before the merged tip must be restated by.  The
#: 2026-08-26 re-fit measured the same mesh on the same card at the merged
#: tip and at hex ``7fe514b``: ``x1.40962`` rose 484.0 MiB.  A row taken at
#: an older tip is an under-statement by about that, and a gate must not
#: quote an under-statement without saying so.
PIN_RESTATEMENT_BYTES = int(round(484.0 * MIB))

#: The CUDA per-thread local frame the widest LAUNCHED kernel carries, from
#: the ledger's ``local_memory_model.max_local_size_bytes`` at the merged tip
#: (``wsm6_column``, 7,216 B).  CUDA reserves one per-context backing store
#: of this times the card's resident threads, at first launch, and never
#: returns it -- so it is the largest card-scaled term in the core and the
#: one a derived row rescales.  The reservation CUDA actually takes is for
#: achievable occupancy and is therefore <= this product, which makes a
#: derived row conservative.
WIDEST_LAUNCHED_LOCAL_FRAME_BYTES = 7_216


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapedFootprintModel:
    """One card's, one configuration's footprint row -- core plus shape.

    ``predict_bytes`` and ``max_cells`` keep the signatures the affine
    :class:`FootprintModel` had, so every caller of the admission sum is
    unchanged; what moved is what those methods compute.
    """

    core_bytes: float
    bytes_per_cell: float
    card: CardProfile
    configuration: str
    provenance: str
    levels: int = MODEL_VERTICAL_LEVELS
    measured: bool = True
    derived_from: str | None = None

    # -- the terms, each nameable on its own ------------------------------
    def tiled_bytes(self, cells: int) -> int:
        return sum(
            ws.bytes_for(int(cells), self.card, self.levels)
            for ws in TILED_WORKSPACES
        )

    def tiled_breakdown(self, cells: int) -> dict[str, int]:
        return {
            ws.name: ws.bytes_for(int(cells), self.card, self.levels)
            for ws in TILED_WORKSPACES
        }

    def predict_bytes(self, cells: int) -> float:
        if cells <= 0:
            raise ValueError("cells must be positive")
        return (
            float(self.core_bytes)
            + float(self.bytes_per_cell) * int(cells)
            + float(self.tiled_bytes(int(cells)))
        )

    # -- the margin, and what each part of it absorbs ----------------------
    def margin_terms(self) -> dict[str, dict[str, Any]]:
        """The margin, itemised.  Every entry names its breakage."""

        return {
            "arena_placement": {
                "bytes": shortwave_workspace_bytes(self.card, self.levels),
                "absorbs": (
                    "the RRTMG shortwave spcvmc workspace -- the largest "
                    "mesh-independent block in the footprint -- being mapped "
                    "fresh instead of served from the CuPy pool's free list "
                    "at the instant radiation runs"
                ),
                "breakage": (
                    "the run dies inside a CuPy allocation part-way through "
                    "the first radiation call, minutes in, on a card the gate "
                    "had already admitted"
                ),
                "measured": (
                    "freeing 38.4 MiB EARLIER in the run moved x1.40962's pool "
                    "high-water 5,404.5 -> 7,111.7 MiB, +1,707.2 MiB, "
                    "reproduced across two runs one of which was on an idle "
                    "card (STATE.md, the construction-diagnostic hold)"
                ),
                "scales_with": "the card's radiation chunk width",
            },
            "instrument_convention": {
                "bytes": CONVENTION_MARGIN_BYTES,
                "absorbs": (
                    "the gap between the per-process nvidia-smi rows these "
                    "cores are fitted on and the whole-device free memory a "
                    "door reads"
                ),
                "breakage": (
                    "a card with exactly the predicted peak free is short by "
                    "the convention gap and refuses or dies at the margin"
                ),
                "measured": (
                    "11.2 MiB on the x4 arm and 0.75 MiB on the graded arm, "
                    "both conventions sampled in one process"
                ),
                "scales_with": "nothing",
            },
        }

    def margin_bytes(self) -> int:
        return int(sum(term["bytes"] for term in self.margin_terms().values()))

    # -- the sums the gates use -------------------------------------------
    def required_bytes(self, cells: int, margin_bytes: int | None = None) -> int:
        margin = self.margin_bytes() if margin_bytes is None else int(margin_bytes)
        return int(round(self.predict_bytes(int(cells)))) + margin

    def max_cells(self, budget_bytes: int, margin_bytes: int | None = None) -> int:
        """Largest cell count this budget admits.

        Solved by bisection rather than by division because the model is
        piecewise linear: the tiled terms stop growing at their knees, so
        there is no single slope to divide by.
        """

        margin = self.margin_bytes() if margin_bytes is None else int(margin_bytes)
        budget = int(budget_bytes) - margin
        if budget <= 0 or self.bytes_per_cell <= 0.0:
            return 0
        if self.predict_bytes(1) > budget:
            return 0
        low, high = 1, 1
        while self.predict_bytes(high) <= budget:
            low = high
            high *= 2
            if high > 1 << 40:  # pragma: no cover - no mesh is this large
                return low
        while low + 1 < high:
            mid = (low + high) // 2
            if self.predict_bytes(mid) <= budget:
                low = mid
            else:
                high = mid
        return low

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": "core + sum(tiled workspaces at min(cells, tile)) + slope * cells",
            "core_bytes": float(self.core_bytes),
            "core_mib": float(self.core_bytes) / MIB,
            "bytes_per_cell": float(self.bytes_per_cell),
            "configuration": self.configuration,
            "levels": int(self.levels),
            "card": self.card.as_dict(),
            "measured": bool(self.measured),
            "derived_from": self.derived_from,
            "tiled_workspace_knee_cells": {
                ws.name: ws.knee_cells(self.card) for ws in TILED_WORKSPACES
            },
            "radiation_chunk_columns": radiation_chunk_columns(self.card),
            "margin_bytes": self.margin_bytes(),
            "margin_terms": self.margin_terms(),
            "provenance": self.provenance,
            # Kept so a receipt written against the affine row still finds a
            # field with this name; it is the core, and it is NOT the retired
            # row's fixed term.
            "fixed_bytes": float(self.core_bytes),
            "fixed_mib": float(self.core_bytes) / MIB,
        }


# ---------------------------------------------------------------------------
# the measured rows
# ---------------------------------------------------------------------------
def row_key(configuration: str, card: CardProfile) -> str:
    return f"{configuration}/{int(card.multiprocessors)}sm"


#: The measured rows.  Keyed by configuration and multiprocessor count,
#: because those are the two things that actually move the core: what device
#: stack is built, and how many threads the card holds resident.
_SHAPED_ROWS: dict[str, dict[str, Any]] = {
    "global/170sm": {
        "core_bytes": 3_860_338_719.0,
        "bytes_per_cell": 96_581.5833,
        "card": "32gib-170sm",
        "configuration": "global",
        "measured": True,
        "provenance": (
            "two-point fit on the #264 per-allocation ledger session measured "
            "2026-08-26 on the proving RTX 5090 (170 SM) at the merged tip, "
            "hex 2009db7 + engine 26daaab7e: x1.40962 peaked 8,874.0 MiB and "
            "x4.163842 20,446.0 MiB, both by this process's nvidia-smi row.  "
            "The tiled-workspace terms are subtracted from both points BEFORE "
            "the fit, which is the whole difference from the retired affine "
            "row: 2,167 B/cell of that row's slope was GF and YSU tile growth "
            "between a mesh below both knees and a mesh above both.  "
            "evidence/memory-row-refit-20260826/node2/"
        ),
    },
    "global/70sm": {
        "core_bytes": 1_084.9401 * MIB + PIN_RESTATEMENT_BYTES,
        "bytes_per_cell": 115_143.1,
        "card": "16gib-70sm",
        "configuration": "global",
        "measured": True,
        "provenance": (
            "two-point fit on the #264 ledger session measured 2026-08-25 on "
            "the proving RTX 5070 Ti (70 SM) at hex 5252421 + engine "
            "26daaab7e: x1.40962 peaked 6,272.0 MiB and u96.64002 8,802.0 MiB.  "
            "NOT MEASURED AT THE MERGED TIP -- the core carries "
            "PIN_RESTATEMENT_BYTES (+484.0 MiB, what the tip moved x1 by on "
            "the 170 SM card) so the row is not quoted as an under-statement.  "
            "CAVEAT CARRIED FROM THE 08-24 FIT ON THIS CARD: the two peaks "
            "land at different instants and the pair separates core from slope "
            "imperfectly, which is why this row's slope is 19 % above the 170 "
            "SM card's over a lever arm one fifth as long.  One #264 arm at "
            "the tip on a third mesh replaces it.  "
            "evidence/l6-capacity-20260825/node1/"
        ),
    },
    "global/68sm": {
        "core_bytes": 755.7227 * MIB + PIN_RESTATEMENT_BYTES,
        "bytes_per_cell": 96_581.5833,
        "card": "10gib-68sm",
        "configuration": "global",
        "measured": True,
        "provenance": (
            "single-point core from the #264 ledger row measured 2026-08-25 on "
            "the desktop RTX 3080 (68 SM, sm_86, WDDM) at hex 5252421 + engine "
            "26daaab7e: x1.40962 peaked 6,340.5 MiB device-view.  WDDM "
            "publishes no per-process row, so that figure INCLUDES the "
            "desktop's 1,142.5 MiB run-start baseline and the baseline is "
            "REMOVED here -- the door compares against free memory, which "
            "already has the desktop taken out of it, so a core carrying the "
            "baseline charges the desktop twice.  The retired tier row "
            "(2,483.0 MiB fixed) did exactly that.  Slope BORROWED from the "
            "170 SM row because the slope is a property of the build; core "
            "carries PIN_RESTATEMENT_BYTES.  "
            "evidence/l6-capacity-20260825/desktop3080/"
        ),
    },
    "limited-area/170sm": {
        "core_bytes": 6_324.7 * MIB,
        "bytes_per_cell": 96_581.5833,
        "card": "32gib-170sm",
        "configuration": "limited-area",
        "measured": True,
        "provenance": (
            "ENVELOPE, NOT A FIT.  The per-allocation instrument has never "
            "been run on a regional mesh, so this core is the LARGEST residue "
            "(peak minus tiled terms minus slope*cells) over the five "
            "concentric culls measured 2026-08-27 on the proving RTX 5090, "
            "all 1,080 steps of full physics at dt 20 s through "
            "`gpuwm-hex forecast --lbc-dir`, whole-device sampler on an idle "
            "card: r4.75.4440 5,552 MiB, r4.75.7975 7,336, r4.75.11020 7,670, "
            "r4.75.14050 7,712, r4.75.15755 7,728.  The five residues span "
            "4,988-6,325 MiB and the largest is taken, so this row "
            "over-predicts the other four by up to 1,337 MiB.  The regional "
            "device stack is a different configuration -- padded atmosphere, "
            "two levels of lateral-boundary state, 22 more kernels -- and it "
            "is 85 % mesh-independent at these cell counts, which is why the "
            "retired affine row over-predicted every cull by 1.2-1.9 GiB.  "
            "Slope BORROWED from the global 170 SM row: four of the five "
            "points are flat to within 228 MiB across a 1.98x cell count, so "
            "these five cannot determine a slope.  ONE #264 ledger arm on a "
            "cull replaces this whole row.  evidence/nest-ratio-20260827/"
        ),
    },
}

SHAPED_ROWS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {key: MappingProxyType(dict(row)) for key, row in _SHAPED_ROWS.items()}
)


def _row_model(key: str) -> ShapedFootprintModel:
    row = _SHAPED_ROWS[key]
    return ShapedFootprintModel(
        core_bytes=float(row["core_bytes"]),
        bytes_per_cell=float(row["bytes_per_cell"]),
        card=KNOWN_CARDS[str(row["card"])],
        configuration=str(row["configuration"]),
        provenance=str(row["provenance"]),
        measured=bool(row["measured"]),
    )


#: The row of record: the 170 SM card, the global configuration.  It is the
#: DEFAULT and the derivation reference, not a universal answer -- the door
#: reads the card and re-selects.
FOOTPRINT_MODEL = _row_model("global/170sm")


def model_for_card(
    card: CardProfile | None = None,
    configuration: str = "global",
    levels: int = MODEL_VERTICAL_LEVELS,
) -> ShapedFootprintModel:
    """The footprint row for this card and configuration.

    Measured if this card has been measured; DERIVED, and labelled so,
    otherwise.  This is the function ledger #366 was open for: the retired
    module had per-card rows and no caller, so a 10 GiB desktop was priced
    with a 32 GiB card's fixed term and refused two meshes it had been
    measured running.
    """

    if card is None:
        card = REFERENCE_CARD
    key = row_key(configuration, card)
    if key in _SHAPED_ROWS:
        model = _row_model(key)
        if levels != model.levels:
            model = ShapedFootprintModel(
                core_bytes=model.core_bytes,
                bytes_per_cell=model.bytes_per_cell,
                card=model.card,
                configuration=model.configuration,
                provenance=model.provenance,
                levels=int(levels),
                measured=model.measured,
            )
        # A row is measured on ONE card; if the caller hands us the same SM
        # count under a different name, the arithmetic is identical and the
        # name is the caller's.
        return ShapedFootprintModel(
            core_bytes=model.core_bytes,
            bytes_per_cell=model.bytes_per_cell,
            card=card,
            configuration=model.configuration,
            provenance=model.provenance,
            levels=int(levels),
            measured=True,
        )

    reference_key = row_key(configuration, REFERENCE_CARD)
    if reference_key not in _SHAPED_ROWS:
        raise KeyError(
            f"no measured row for configuration {configuration!r} on any card, "
            "so nothing can be derived for "
            f"{card.name!r}; add a measured row before admitting on it"
        )
    reference = _row_model(reference_key)
    local_store_saving = WIDEST_LAUNCHED_LOCAL_FRAME_BYTES * (
        REFERENCE_CARD.resident_threads - card.resident_threads
    )
    radiation_saving = shortwave_workspace_bytes(
        REFERENCE_CARD, levels
    ) - shortwave_workspace_bytes(card, levels)
    core = float(reference.core_bytes) - float(local_store_saving) - float(
        radiation_saving
    )
    if core < 0.0:
        core = float(reference.core_bytes)
    return ShapedFootprintModel(
        core_bytes=core,
        bytes_per_cell=reference.bytes_per_cell,
        card=card,
        configuration=configuration,
        levels=int(levels),
        measured=False,
        derived_from=reference_key,
        provenance=(
            f"DERIVED, NOT MEASURED: no #264 ledger arm exists for "
            f"{card.multiprocessors} SM on the {configuration} configuration, "
            f"so this row rescales the {reference_key} core by the two "
            "mechanisms that are known and measured to scale with the card -- "
            "the CUDA per-context local-memory backing store (widest launched "
            f"frame {WIDEST_LAUNCHED_LOCAL_FRAME_BYTES} B x resident threads, "
            "the upper bound CUDA prices, so the saving is understated and the "
            "row is conservative) and the RRTMG shortwave chunk width.  "
            "Checked against the two cards that DO have their own "
            "measurement, this derivation over-predicts by 182.7 MiB on the "
            "70 SM part and 490.8 MiB on the 68 SM part -- conservative on "
            "both, which is the direction a gate needs.  Slope is the "
            "reference's: the slope is a property of the build.  One #264 arm "
            "on this card replaces this row"
        ),
    )


# ---------------------------------------------------------------------------
# the ONE admission sum
# ---------------------------------------------------------------------------
def required_free_bytes(
    cells: int,
    model: ShapedFootprintModel | None = None,
    margin_bytes: int | None = None,
) -> int:
    """Free device bytes this mesh needs on this model: the ONE admission sum.

    ``predicted peak + margin``, rounded once.  The door's verdict, the
    driver argv the door builds, and the per-mesh floor the mesh binding
    installs all call this function; a test asserts they carry the same
    number to the byte.

    The margin is the model's own -- two named, measured components, not a
    flat constant.  Passing ``margin_bytes`` overrides it, which is what
    ``--headroom-mib`` does and why that flag now reports the model's figure
    as its default rather than 512 MiB.
    """

    model = FOOTPRINT_MODEL if model is None else model
    return model.required_bytes(int(cells), margin_bytes)


def native_device_floor_bytes() -> int:
    """The re-derived ``NATIVE_DEVICE_FLOOR``: the x4.163842 requirement."""

    return required_free_bytes(NATIVE_MESH_CELLS)


# ---------------------------------------------------------------------------
# retired arms -- computable, dated, never a gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FootprintModel:
    """A RETIRED affine row: fixed term plus a per-cell slope.

    This is the shape ledger #365/#366 retired on 2026-08-27.  It survives
    only so the before arm of a comparison has one home -- see
    :func:`retired_affine_row_floor_bytes`.  It is not a gate and must never
    become one.
    """

    fixed_bytes: float
    bytes_per_cell: float
    provenance: str

    def predict_bytes(self, cells: int) -> float:
        if cells <= 0:
            raise ValueError("cells must be positive")
        return self.fixed_bytes + self.bytes_per_cell * cells

    def max_cells(self, budget_bytes: int, headroom_bytes: int) -> int:
        usable = float(budget_bytes) - float(headroom_bytes) - self.fixed_bytes
        if usable <= 0.0 or self.bytes_per_cell <= 0.0:
            return 0
        return int(usable // self.bytes_per_cell)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixed_bytes": self.fixed_bytes,
            "fixed_mib": self.fixed_bytes / MIB,
            "bytes_per_cell": self.bytes_per_cell,
            "provenance": self.provenance,
        }


#: The flat headroom the shaped margin retired.  It named no breakage, and it
#: failed by 96 MiB on v6.75.112676 and by 28.2 MiB on v20.80.151649.  Kept
#: computable so a before/after table has one home.  NEVER a gate.
RETIRED_FLAT_HEADROOM_BYTES = 512 * MIB

#: Retired alias, kept so an existing import does not break at a tree that
#: still names it.  It is the RETIRED flat constant, not the model's margin;
#: call ``model.margin_bytes()`` for the number a gate uses.
DEFAULT_HEADROOM_BYTES = RETIRED_FLAT_HEADROOM_BYTES

#: The asserted constant the 2026-08-25 re-derivation retired.
RETIRED_LINEAR_FLOOR_BYTES = 24 * 1024**3


def retired_linear_floor_bytes(cells: int) -> int:
    """What the RETIRED 24 GiB floor demanded at ``cells`` -- before arm only.

    ``24 GiB * cells / 163,842``: 157,285 B per cell, no fixed term.
    NEVER call this to admit or refuse a run.
    """

    if cells <= 0:
        raise ValueError("cells must be positive")
    return RETIRED_LINEAR_FLOOR_BYTES * int(cells) // NATIVE_MESH_CELLS


#: The row the 2026-08-26 re-fit retired.
RETIRED_ROW_20260825 = FootprintModel(
    fixed_bytes=4339.1 * MIB,
    bytes_per_cell=103_696.0,
    provenance=(
        "RETIRED 2026-08-26.  Measured 2026-08-25 on the proving RTX 5090 "
        "(170 SM) at the converged seam pin, hex 7fe514b + engine 26daaab7e, "
        "both published meshes in one #264 ledger session; "
        "evidence/pin-move-335-20260825/node2/.  Superseded by the merged-tip "
        "re-fit at hex 2009db7 -- the same two meshes on the same card "
        "measured 8,874.0 / 20,446.0 MiB there against this row's "
        "8,390.0 / 20,542.0"
    ),
)


def retired_converged_row_floor_bytes(cells: int) -> int:
    """What the RETIRED 2026-08-25 row demanded -- before arm only."""

    if cells <= 0:
        raise ValueError("cells must be positive")
    return (
        int(round(RETIRED_ROW_20260825.predict_bytes(int(cells))))
        + RETIRED_FLAT_HEADROOM_BYTES
    )


#: The row THIS lane retired: the affine row of record from 2026-08-26.  It
#: was measured impeccably and it is the third generation of retired arm in
#: this module, and the three are different KINDS of wrong.  The 24 GiB proxy
#: was never measured.  The 08-25 row measured a different tree.  This one
#: measured the right tree on the right card with the right instrument and
#: had **the wrong shape**: a line in cell count, fitted across a mesh below
#: both tiled-workspace knees and a mesh above both, so it charged card-sized
#: workspace growth as if it were per-cell.
RETIRED_AFFINE_ROW_20260826 = FootprintModel(
    fixed_bytes=5016.5 * MIB,
    bytes_per_cell=98_748.0,
    provenance=(
        "RETIRED 2026-08-27.  Measured 2026-08-26 on the proving RTX 5090 "
        "(170 SM) at the merged tip, hex 2009db7 + engine 26daaab7e, both "
        "published meshes in one #264 ledger session -- the same session this "
        "module's global/170sm core is fitted from.  Retired for SHAPE, not "
        "for provenance: five placed-mesh forecast-door runs spanned "
        "+3.89 % to -6.76 % against it, one mesh with 15,343 more cells "
        "peaked 318 MiB lower, one run overran its own required_free by "
        "96 MiB, and five limited-area culls were over-predicted by "
        "1.2-1.9 GiB each.  evidence/memory-shape-20260827/"
    ),
)


def retired_affine_row_floor_bytes(cells: int) -> int:
    """What the RETIRED 2026-08-26 affine row demanded -- before arm only.

    Every receipt and chart that quotes the affine requirement computes it
    here rather than hand-typing it, because hand-typed before-arm numbers
    are exactly what outlive the constant they came from.

    NEVER call this to admit or refuse a run.
    """

    if cells <= 0:
        raise ValueError("cells must be positive")
    return (
        int(round(RETIRED_AFFINE_ROW_20260826.predict_bytes(int(cells))))
        + RETIRED_FLAT_HEADROOM_BYTES
    )


#: The per-card tier table the shaped rows retired.  It was correct data with
#: no caller: ``forecast_door._resolve_model`` never read it, which is ledger
#: #366.  Kept computable so the before arm of the #366 table has one home;
#: :func:`tier_model` still builds the affine model each row described.
RETIRED_CARD_TIER_ROWS_20260826: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "32gib-170sm": MappingProxyType(
            {
                "tier": "32 GiB",
                "card": "RTX 5090 (32,607 MiB, 170 SM, sm_120)",
                "measured": True,
                "fixed_mib": 5016.5,
                "bytes_per_cell": 98_748.0,
                "basis": "two-point #264 fit at the merged tip, 2026-08-26",
            }
        ),
        "16gib-70sm": MappingProxyType(
            {
                "tier": "16 GiB",
                "card": "RTX 5070 Ti (16,303 MiB, 70 SM, sm_120)",
                "measured": True,
                "fixed_mib": 1774.0,
                "bytes_per_cell": 115_143.0,
                "basis": "two-point #264 fit at hex 5252421, 2026-08-25",
            }
        ),
        "10gib-68sm": MappingProxyType(
            {
                "tier": "10 GiB",
                "card": "RTX 3080 (10,240 MiB, 68 SM, sm_86, WDDM)",
                "measured": True,
                "fixed_mib": 2483.0,
                "bytes_per_cell": 98_748.0,
                "basis": (
                    "single-point #264 row at hex 5252421, 2026-08-25, slope "
                    "borrowed.  RETIRED ALSO FOR A SECOND DEFECT: the "
                    "6,340.5 MiB device-view peak it was derived from includes "
                    "the desktop's 1,142.5 MiB baseline, and the door compares "
                    "against free memory which already excludes it, so this "
                    "row charged the desktop twice"
                ),
            }
        ),
        "12gib": MappingProxyType(
            {
                "tier": "12 GiB",
                "card": "no 12 GiB card exists in the fleet",
                "measured": False,
                "fixed_mib": 5016.5,
                "bytes_per_cell": 98_748.0,
                "basis": (
                    "DERIVED, NOT MEASURED: the of-record 170 SM model quoted "
                    "at a 12 GiB budget.  Superseded by "
                    "device_admission.model_for_card, which derives from the "
                    "card's SM count instead of from its memory size"
                ),
            }
        ),
    }
)

#: Retired alias for the tier table, kept so an existing import resolves.
CARD_TIER_ROWS = RETIRED_CARD_TIER_ROWS_20260826


def tier_model(key: str) -> FootprintModel:
    """The RETIRED affine model one tier row described -- before arm only."""

    row = RETIRED_CARD_TIER_ROWS_20260826[key]
    label = "measured row" if row["measured"] else "DERIVED (not measured)"
    return FootprintModel(
        fixed_bytes=float(row["fixed_mib"]) * MIB,
        bytes_per_cell=float(row["bytes_per_cell"]),
        provenance=f"RETIRED 2026-08-27.  {row['card']} -- {label}: {row['basis']}",
    )


# ---------------------------------------------------------------------------
# the derivation, machine-readable
# ---------------------------------------------------------------------------
FLOOR_DERIVATION: Mapping[str, Any] = MappingProxyType(
    {
        "derived": "2026-08-27",
        "schema": "gpuwm-hex.device-admission.shaped/v1",
        "ruling": (
            "the four-swaths receipt's named follow-up, 2026-08-27: 'A linear "
            "model in cell count is the wrong shape for a placed mesh: "
            "something other than the cell count is setting the peak.  Until "
            "that is understood the 512 MiB headroom is not sufficient on "
            "this class of mesh.'  This is the shape, and it is arithmetic on "
            "the per-allocation ledger rather than another fit"
        ),
        "model": (
            "peak(cells, card) = core(card, configuration) "
            "+ sum(tiled workspace at min(cells, tile(card))) "
            "+ bytes_per_cell * cells; "
            "required = peak + arena_placement_margin(card) + convention_margin"
        ),
        "what_moved": (
            "the tiled physics workspaces came OUT of the fixed term and OUT "
            "of the slope and became their own term.  Grell-Freitas and YSU "
            "size a global device workspace to min(cells, SMs x blocks/SM x "
            "block) after the #335 local-frame cut, and RRTMG sizes its column "
            "chunk from the card's resident threads after #310.  Both fit "
            "meshes straddle the GF knee (43,520 cells) and the YSU knee "
            "(87,040) on the 170 SM card, so the affine fit charged 2,167 "
            "B/cell of tile growth as per-cell and put the saturated remainder "
            "in a fixed term that a smaller card does not have"
        ),
        "reproduces_the_ledger": {
            "grell_freitas_x1_mib": 1261.97,
            "grell_freitas_x4_mib": 1338.75,
            "ysu_x1_mib": 157.62,
            "ysu_x4_mib": 334.69,
            "shortwave_spcvmc_bytes_170sm": 1_830_420_480,
            "note": (
                "the three sites the #264 ledger records as NOT scaling 4.00x "
                "with a 4x mesh, and the only three; the formulas in this "
                "module reproduce all three from card constants alone"
            ),
        },
        "margin_replaces": (
            "512 MiB flat, shared by every card and every mesh, naming no "
            "breakage.  It failed by 96 MiB on v6.75.112676 (measured 16,236 "
            "MiB against a 16,139.6 MiB required_free) and by 28.2 MiB on "
            "v20.80.151649.  Still computable at "
            "device_admission.RETIRED_FLAT_HEADROOM_BYTES"
        ),
        "retired_rows_computable_at": (
            "retired_affine_row_floor_bytes (2026-08-26 affine row of record), "
            "retired_converged_row_floor_bytes (2026-08-25 row), "
            "retired_linear_floor_bytes (the 24 GiB proxy), "
            "tier_model (the per-card affine tier table)"
        ),
        "open_and_stated": {
            "limited_area_core_is_an_envelope": (
                "the per-allocation instrument has never run on a regional "
                "mesh; the core is the largest of five whole-device residues "
                "and over-predicts four of them by up to 1,337 MiB.  ONE #264 "
                "arm on a cull replaces it"
            ),
            "four_swath_spread_not_separated": (
                "the +3.89 %/-6.76 % spread across four placed meshes has a "
                "named mechanism -- arena placement of the 1,745.6 MiB "
                "shortwave block, measured at 1,707.2 MiB in the "
                "construction-diagnostic A/B -- but NO A/B on those four "
                "meshes has been run, so the attribution is inference.  The "
                "settling arm is one #264 ledger run per swath mesh with the "
                "pool high-water and the peak-instant site census recorded"
            ),
            "70sm_slope": (
                "115,143 B/cell against the 170 SM card's 96,582 over a lever "
                "arm one fifth as long, same engine pin.  Probably a "
                "short-baseline artefact and treated as the card's own row "
                "because a gate must not under-predict; one #264 arm on a "
                "third mesh settles it"
            ),
            "nz_dependence": (
                "the tiled terms are functions of nz; the cores and the slope "
                "were measured only at nz=55 and are NOT re-derivable at "
                "another level count from anything in this module"
            ),
        },
    }
)
