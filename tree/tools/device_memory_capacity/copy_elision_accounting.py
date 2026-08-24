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
from typing import Sequence

GIB = 1024**3
MIB = 1024**2
SCHEMA = "gpuwm-hex.copy-elision-accounting.v1"


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
    parser.add_argument("--prior-fixed-mib", type=float, default=9797.8)
    parser.add_argument("--prior-bytes-per-cell", type=float, default=86630.0)
    parser.add_argument("--budget-gib", type=float, default=12.0)
    parser.add_argument("--headroom-mib", type=float, default=512.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
