"""Read the |w| band out of a forecast receipt's per-step health rows.

WHY THIS EXISTS.  Every measurement of the short-timestep vertical-velocity
growth this project holds was taken on ``x1.40962`` -- a 120 km mesh whose
own Courant limit is 698.95 s -- at timesteps 35x to 140x beneath it.  The
(20 s, gf) anchor says so in its own words and names what would settle it:
"the same trend on a mesh whose Courant limit is near this timestep".

This tool computes the SAME statistic those anchors quote, so the two can
be read side by side: the mean of the per-step ``vertical_velocity_abs_max``
over each half-hour window, and the maximum over the whole arm.  It reads
``driver_receipt.forecast.step_health`` -- the model's own per-step record,
not a re-derivation from history frames.

It is an analysis instrument over scalars.  It plots nothing and it renders
no weather field.

Usage::

    python tools/measure_w_band.py RECEIPT.json [--window-minutes 30]
                                   [--out BAND.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The 120 s reference band on x1.40962 from the 2026-08-12 06Z native init,
#: quoted by dt_admission's 120 s anchor as "REFERENCE." -- the band every
#: other row is read against.  Half-hour window means, then |w| max.
CONTROL_120S = {
    "mesh": "x1.40962",
    "dt_seconds": 120.0,
    "window_means": (1.15, 1.17, 1.20, 1.48),
    "abs_max": 1.680,
    "courant_limit_seconds": 698.95,
}

#: The (20 s, gf) anchor's own band, on the same mesh and init.  Its
#: physics_health reads "DIVERGES", and its basis says the cause is NOT
#: MEASURED because 20 s is 35x below that mesh's limit.
ANCHOR_20S_GF = {
    "mesh": "x1.40962",
    "dt_seconds": 20.0,
    "window_means": (1.99, 3.48, 4.11, 5.53),
    "abs_max": 7.511,
    "courant_limit_seconds": 698.95,
}


def _dig(document: Mapping[str, Any], *path: str) -> Any:
    node: Any = document
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def step_health(receipt: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = _dig(receipt, "driver_receipt", "forecast", "step_health")
    if not rows:
        raise SystemExit(
            "this receipt carries no driver_receipt.forecast.step_health, so "
            "there is no per-step |w| to read.  A receipt from a preflight or "
            "from a run that never reached the integration has none, and a "
            "band derived from history frames instead would be a different "
            "statistic from the one the anchors quote"
        )
    return list(rows)


def windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dt_seconds: float,
    window_minutes: float,
) -> list[dict[str, Any]]:
    """Split the per-step rows into equal wall-clock windows."""

    per_window = int(round(window_minutes * 60.0 / dt_seconds))
    if per_window < 1:
        raise SystemExit(
            f"a {window_minutes:g} min window holds {per_window} steps at "
            f"dt={dt_seconds:g} s; widen the window or the mean is taken over "
            f"nothing"
        )
    out: list[dict[str, Any]] = []
    for start in range(0, len(rows), per_window):
        chunk = rows[start : start + per_window]
        values = [float(row["vertical_velocity_abs_max"]) for row in chunk]
        out.append(
            {
                "window": len(out),
                "steps": len(chunk),
                "complete": len(chunk) == per_window,
                "from_hours": start * dt_seconds / 3600.0,
                "to_hours": (start + len(chunk)) * dt_seconds / 3600.0,
                "w_abs_max_mean": sum(values) / len(values),
                "w_abs_max_max": max(values),
                "theta_m_max": max(float(r["theta_m_max"]) for r in chunk),
                "theta_m_min": min(float(r["theta_m_min"]) for r in chunk),
                "all_finite": all(bool(r.get("finite", True)) for r in chunk),
            }
        )
    return out


def band(receipt_path: Path, *, window_minutes: float) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = step_health(receipt)
    dt_seconds = float(
        _dig(receipt, "driver_receipt", "forecast", "schedule", "dt_seconds")
        or _dig(receipt, "timestep_admission", "resolved_dt_seconds")
        or 0.0
    )
    # The door writes the Courant verdict under the BIND block, not at the
    # top level; a preflight receipt writes it at the top level.  Read both
    # rather than reporting None, because the whole reading of this band
    # turns on how far the timestep sits beneath the mesh's own limit.
    courant_limit = (
        _dig(receipt, "mesh_binding", "timestep_admission",
             "maximum_admitted_dt_seconds")
        or _dig(receipt, "timestep_admission", "maximum_admitted_dt_seconds")
    )
    minimum_dc_edge_m = (
        _dig(receipt, "mesh_binding", "timestep_admission",
             "edge_length_authority", "minimum_m")
        or _dig(receipt, "timestep_admission", "edge_length_authority",
                "minimum_m")
    )
    if dt_seconds <= 0.0:
        raise SystemExit(
            "the receipt does not state the timestep it integrated at, so the "
            "per-step rows cannot be placed on a clock and a half-hour window "
            "cannot be cut"
        )
    every = windows(rows, dt_seconds=dt_seconds, window_minutes=window_minutes)
    values = [float(row["vertical_velocity_abs_max"]) for row in rows]
    monotone = all(
        b["w_abs_max_mean"] >= a["w_abs_max_mean"]
        for a, b in zip(every, every[1:])
    )
    return {
        "schema": "gpuwm-hex.w-band/v1",
        "receipt": str(receipt_path),
        "mesh": _dig(receipt, "admission", "mesh"),
        "cells": _dig(receipt, "admission", "cells"),
        "dt_seconds": dt_seconds,
        "courant_limit_seconds": courant_limit,
        "minimum_dc_edge_m": minimum_dc_edge_m,
        # The number that makes this run different from every earlier |w|
        # measurement: how many times beneath its own mesh's limit the
        # timestep sits.  The 20 s and 5 s anchors read 35x and 140x.
        "courant_limit_over_dt": (
            None if not courant_limit else float(courant_limit) / dt_seconds
        ),
        "cumulus_scheme": _dig(
            receipt, "driver_receipt", "convection", "constructor_scheme"
        ),
        "convection_source": _dig(
            receipt, "driver_receipt", "convection", "source"
        ),
        "steps": len(rows),
        "window_minutes": window_minutes,
        "windows": every,
        "w_abs_max": max(values),
        "w_abs_max_mean_all": sum(values) / len(values),
        "w_first_step": values[0],
        "w_last_step": values[-1],
        "monotone_rising_windows": monotone,
        "all_steps_finite": all(bool(r.get("finite", True)) for r in rows),
        "reference_120s_x1": CONTROL_120S,
        "anchor_20s_gf_x1": ANCHOR_20S_GF,
        "reading_caveat": (
            "The two quoted bands were measured on x1.40962, a different mesh "
            "with a different init state resolved at 120 km.  They are an "
            "ORIENTATION and not a like-for-like control: nothing here holds "
            "the mesh fixed while the timestep moves.  What this row settles "
            "is narrower and is the thing the anchor asked for -- whether the "
            "rise appears AT ALL at a sane dt/dx."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--window-minutes", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)
    document = band(arguments.receipt, window_minutes=arguments.window_minutes)
    text = json.dumps(document, indent=2, sort_keys=True)
    if arguments.out is not None:
        arguments.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
