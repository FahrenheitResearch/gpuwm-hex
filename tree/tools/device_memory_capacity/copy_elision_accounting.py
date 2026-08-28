#!/usr/bin/env python3
"""Static allocation accounting for the default-on capacity copy elisions.

The output is geometry arithmetic, not a process-memory measurement.  It says
how many bytes particular removed allocation events would have requested for a
specified mesh.  It deliberately does not add those events into a claimed peak:
CUDA allocation lifetimes, pool reuse, and the process-wide local backing store
must be measured by the hardware protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Sequence

GIB = 1024**3
MIB = 1024**2
SCHEMA = "gpuwm-hex.copy-elision-accounting.v1"

#: Emitted when the footprint model is not named on the command line.
#:
#: THE BREAKAGE THIS PREVENTS, measured: these two arguments used to default
#: to 9797.8 MiB and 86630 B/cell, the 2026-08-20 ledger. The Grell-Freitas
#: local-memory frame cut superseded that model on 2026-08-24 and moved both
#: terms in OPPOSITE directions, so the two errors partly cancelled and a bare
#: run kept returning a plausible budget verdict -- wrong, at exit status 0.
#: A refusal is the only outcome that cannot be mistaken for an answer.
MODEL_REFUSAL = """copy-elision accounting refused: the footprint model must be named.

--prior-fixed-mib and --prior-bytes-per-cell are REQUIRED and have no
defaults. Missing: {missing}

WHY THERE IS NO DEFAULT. These once defaulted to the 2026-08-20 model. The
Grell-Freitas local-memory frame cut superseded it on 2026-08-24, and both
terms moved in OPPOSITE directions: the fixed term FELL by 3501.3 MiB and the
slope ROSE by 6844 B/cell. The two errors partly cancel, so a bare run did not
fail loudly -- it produced a plausible fit verdict that was wrong, and exited
0. Naming the model is what makes the answer readable. (And the same drift
then bit this very text: until 2026-08-25 the "of record" arm below still
named the 08-24 row after the converged row superseded it -- stale-guard
audit #347, finding 7. A test now pins the of-record arm to
device_admission.FLOOR_DERIVATION so the two cannot disagree again.)

THERE IS NO AFFINE ROW OF RECORD ANY MORE, AND THIS TOOL CANNOT ASK THE
CURRENT QUESTION. As of 2026-08-27 the admission surface is not
`fixed + slope * cells`: it is a card core, plus physics workspaces sized to
`min(cells, tile(card))` -- which stop growing once the mesh is bigger than
the card -- plus a per-cell term. Every arm below is HISTORICAL. For any NEW
question about what fits a card today, ask the surface instead:

    from hexcore.device_admission import KNOWN_CARDS, model_for_card
    model_for_card(KNOWN_CARDS["10gib-68sm"]).required_bytes(40_962)

Passing an affine pair to this tool now answers a question about a retired
row, and the answer is only meaningful against the arm it was computed
against. evidence/memory-shape-20260827/ is why.

THE HISTORICAL MODELS. Pick by the arm you are reproducing:

  retired     --prior-fixed-mib 5016.5  --prior-bytes-per-cell 98748
  2026-08-27
              measured 2026-08-26, merged tip (hex 2009db7 + engine
              26daaab7e), 170 SM card, both published meshes in one #264
              session; evidence/memory-row-refit-20260826/node2/.
              RETIRED FOR SHAPE, not for provenance: it did not cover six of
              seventeen recorded runs and was 1,104 MiB short on a
              limited-area cull.  Computable at
              device_admission.retired_affine_row_floor_bytes.

  superseded  --prior-fixed-mib 4339.1  --prior-bytes-per-cell 103696
  (historical)
              measured 2026-08-25, converged seam pin (hex 7fe514b +
              engine 26daaab7e), 170 SM card
              evidence/pin-move-335-20260825/node2/
              PASS THESE only to reproduce an 08-25-era projection against
              the arm it was actually computed against.  This one is the
              cautionary case: it was a good measurement of a tree that had
              already moved on, and the tip moved BOTH terms in opposite
              directions, so the errors partly cancel between about 100k and
              200k cells and a bare comparison looks plausible.

  superseded  --prior-fixed-mib 6296.5  --prior-bytes-per-cell 93474
  (historical)
              measured 2026-08-24, Arwen seam pin 0d04db712, POST frame cut
              evidence/gf-pin-move-measured-20260824/
              PASS THESE only to reproduce an 08-24-era projection against
              the arm it was actually computed against.

  prior arm   --prior-fixed-mib 9797.8  --prior-bytes-per-cell 86630
  (historical)
              measured 2026-08-20, Arwen seam pin 629ddb6f0, PRE frame cut
              docs/device-memory-ledger.md (superseded as a current account)
              PASS THESE to reproduce the #308 copy-elision accounting as it
              was landed, against the arm it was actually computed against.

Whichever pair you pass is written into the report's prior_gap block, so the
artefact records the model it was projected against instead of leaving it to
be inferred from a file date."""


@dataclass(frozen=True, slots=True)
class Geometry:
    cells: int
    edges: int
    vertical_levels: int
    scalars: int = 6
    dtype_bytes: int = 4

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class AllocationEvent:
    name: str
    bytes: int
    reason: str
    lifetime_note: str

    def as_report(self, *, cells: int) -> dict[str, object]:
        return {
            **asdict(self),
            "mib": self.bytes / MIB,
            "bytes_per_cell": self.bytes / cells,
            "claim_status": "STATIC ALLOCATION EVENT; PROCESS PEAK NOT MEASURED",
        }


def allocation_events(geometry: Geometry) -> tuple[AllocationEvent, ...]:
    geometry.validate()
    ncell = geometry.cells
    nedge = geometry.edges
    nlev = geometry.vertical_levels
    nscalar = geometry.scalars
    item = geometry.dtype_bytes

    cell_field = nlev * ncell * item
    edge_field = nlev * nedge * item
    interface_field = (nlev + 1) * ncell * item
    scalar_block = nscalar * cell_field

    # _copy_state(saved_state) plus _copy_saved(saved_diag), removed by binding
    # current_* to the read-only substep-start images for RK stage 1.
    rk1_duplicate = (
        2 * cell_field
        + edge_field
        + interface_field
        + scalar_block
        + 5 * cell_field
        + edge_field
        + interface_field
    )

    return (
        AllocationEvent(
            "saved_state_scalar_copy",
            scalar_block,
            "_copy_state(..., share_scalars=True) binds the dynamics-read scalar block",
            "one dynamics-subcycle start image",
        ),
        AllocationEvent(
            "rk1_current_state_and_diagnostics_copy",
            rk1_duplicate,
            "RK1 reads the same substep-start state and diagnostics and rebinds before RK2",
            "from subcycle entry until the first recovered candidate replaces current_*",
        ),
        AllocationEvent(
            "candidate_scalar_copy",
            scalar_block,
            "_recover_candidate binds scalars that are read until transport returns a new block",
            "one allocation event per RK stage; old current scalars remain readable during recovery",
        ),
        AllocationEvent(
            "discarded_recovery_pressure_fields",
            6 * cell_field,
            "the caller already owns theta/exner/rho_p/rtheta_p/pressure_p; full pressure is unconsumed",
            "one allocation event per RK stage, previously live through recovery return",
        ),
        AllocationEvent(
            "cached_tangential_velocity_copy",
            edge_field,
            "RK1/RK2 consumers take the cached field as read-only and write separate outputs",
            "one allocation event at each cached-diagnostics solve",
        ),
    )


def prior_gap(
    *,
    cells: int,
    fixed_mib: float,
    bytes_per_cell: float,
    budget_gib: float,
    headroom_mib: float,
) -> dict[str, float | str | None]:
    if cells <= 0 or fixed_mib < 0 or bytes_per_cell <= 0:
        raise ValueError("prior model coefficients are invalid")
    if budget_gib <= 0 or headroom_mib < 0:
        raise ValueError("budget/headroom are invalid")
    fixed = fixed_mib * MIB
    budget = budget_gib * GIB
    headroom = headroom_mib * MIB
    projected = fixed + bytes_per_cell * cells
    available_for_slope = budget - headroom - fixed
    allowed_bpc = available_for_slope / cells
    return {
        "prior_fixed_mib": fixed_mib,
        "prior_bytes_per_cell": bytes_per_cell,
        "target_cells": cells,
        "projected_peak_gib": projected / GIB,
        "budget_gib": budget_gib,
        "headroom_mib": headroom_mib,
        "allowed_bytes_per_cell_if_fixed_unchanged": (
            None if available_for_slope < 0 else allowed_bpc
        ),
        "bytes_per_cell_reduction_needed_if_fixed_unchanged": (
            bytes_per_cell if available_for_slope < 0
            else max(0.0, bytes_per_cell - allowed_bpc)
        ),
        "claim_status": (
            "PRIOR LEDGER PROJECTION; MUST BE RE-FIT ON THE MODIFIED BRANCH/CARD"
        ),
    }


def build_report(
    geometry: Geometry,
    *,
    prior_fixed_mib: float,
    prior_bytes_per_cell: float,
    budget_gib: float,
    headroom_mib: float,
) -> dict[str, object]:
    events = allocation_events(geometry)
    event_reports = [event.as_report(cells=geometry.cells) for event in events]
    return {
        "schema": SCHEMA,
        "geometry": asdict(geometry),
        "events": event_reports,
        "sum_of_removed_allocation_events_bytes": sum(event.bytes for event in events),
        "sum_of_removed_allocation_events_mib": sum(event.bytes for event in events) / MIB,
        "sum_warning": (
            "Do not subtract this sum from nvidia-smi peak. Events repeat and overlap "
            "differently; the CuPy pool may reuse or retain their blocks."
        ),
        "prior_gap": prior_gap(
            cells=geometry.cells,
            fixed_mib=prior_fixed_mib,
            bytes_per_cell=prior_bytes_per_cell,
            budget_gib=budget_gib,
            headroom_mib=headroom_mib,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--edges", type=int, required=True)
    parser.add_argument("--vertical-levels", type=int, required=True)
    parser.add_argument("--scalars", type=int, default=6)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    # REQUIRED, with no default.  A default here is a footprint model chosen
    # by whoever last edited this file rather than by the person asking the
    # question, and it silently decides the answer.  See MODEL_REFUSAL.
    parser.add_argument(
        "--prior-fixed-mib",
        type=float,
        default=None,
        help="REQUIRED, no default. Fixed term of the footprint model to project against",
    )
    parser.add_argument(
        "--prior-bytes-per-cell",
        type=float,
        default=None,
        help="REQUIRED, no default. Per-cell slope of the footprint model to project against",
    )
    parser.add_argument("--budget-gib", type=float, default=12.0)
    parser.add_argument("--headroom-mib", type=float, default=512.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    missing = [
        flag
        for flag, value in (
            ("--prior-fixed-mib", args.prior_fixed_mib),
            ("--prior-bytes-per-cell", args.prior_bytes_per_cell),
        )
        if value is None
    ]
    if missing:
        # Not argparse's own required=True: its message names the flag and
        # stops there, which tells a reader what to type without telling them
        # which number to type, and picking the wrong model is the whole
        # hazard.  Exit 2 is argparse's usage-error code and is kept.
        print(MODEL_REFUSAL.format(missing=", ".join(missing)), file=sys.stderr)
        return 2
    geometry = Geometry(
        cells=args.cells,
        edges=args.edges,
        vertical_levels=args.vertical_levels,
        scalars=args.scalars,
        dtype_bytes=args.dtype_bytes,
    )
    try:
        report = build_report(
            geometry,
            prior_fixed_mib=args.prior_fixed_mib,
            prior_bytes_per_cell=args.prior_bytes_per_cell,
            budget_gib=args.budget_gib,
            headroom_mib=args.headroom_mib,
        )
    except ValueError as exc:
        raise SystemExit(f"copy-elision accounting refused: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
