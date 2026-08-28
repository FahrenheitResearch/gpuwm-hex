"""Does the generator's own polygon attainment tell the truth? Measured.

the ruling is that a receipt quotes ATTAINED spacing and never
requested.  The swath layer emits ``polygon`` regions, so the number it
would naturally quote is the ``attained_spacing_km`` in the polygon's row
of ``rw_mpas_mesh --dry-run``'s ``region_attainment`` block.  This probe
asks whether that number can be trusted, by putting the SAME request
through the SAME binary as a polygon and as a cap covering comparable
ground, and comparing.

The verifier has teeth in both directions: the cap arm is the control, and
if it also reported its request exactly met at every size the probe would
prove nothing about polygons.

Committed so the claim is re-checkable rather than remembered.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.swath.geometry import destination  # noqa: E402
from hexcore.swath.sizing import dry_run, resolve_engine  # noqa: E402

BACKGROUND_KM = 75.0
SPACING_KM = 4.0
TRANSITION_KM = 100.0

#: Half-widths to sweep, in km.  The small end is where a region cannot
#: reach its request and an honest attainment must say so.
HALF_WIDTHS = (22.0, 60.0, 150.0, 300.0, 600.0)


def _spec(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        "background_km": BACKGROUND_KM,
        "regions": [
            {"shape": shape, "spacing_km": SPACING_KM, "transition_km": TRANSITION_KM}
        ],
    }


def _square(centre: tuple[float, float], half_width_km: float) -> dict[str, Any]:
    """A square polygon whose inscribed circle has the given half-width."""

    corners = [
        destination(centre[0], centre[1], bearing, half_width_km * math.sqrt(2.0))
        for bearing in (45.0, 135.0, 225.0, 315.0)
    ]
    return {"kind": "polygon", "vertices_deg": [[lat, lon] for lat, lon in corners]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_polygon_attainment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mesh-exe", type=Path, default=None)
    arguments = parser.parse_args(argv)

    engine = Path(arguments.mesh_exe) if arguments.mesh_exe else resolve_engine(None)
    centre = (0.2, 0.2)
    rows: list[dict[str, Any]] = []
    for half_width in HALF_WIDTHS:
        cap = dry_run(
            _spec({"kind": "cap", "center_deg": list(centre), "radius_km": half_width}),
            engine=engine,
        )
        polygon = dry_run(_spec(_square(centre, half_width)), engine=engine)
        cap_row = cap["region_attainment"][0]
        polygon_row = polygon["region_attainment"][0]
        rows.append({
            "half_width_km": half_width,
            "requested_spacing_km": SPACING_KM,
            "cap": {
                "attained_spacing_km": cap_row["attained_spacing_km"],
                "interior_depth_km": cap_row["interior_depth_km"],
                "predicted_cells": cap["predicted_cells"],
            },
            "polygon": {
                "attained_spacing_km": polygon_row["attained_spacing_km"],
                "interior_depth_km": polygon_row["interior_depth_km"],
                "predicted_cells": polygon["predicted_cells"],
            },
        })

    cap_varies = len({round(row["cap"]["attained_spacing_km"], 4) for row in rows}) > 1
    polygon_always_met = all(
        abs(row["polygon"]["attained_spacing_km"] - SPACING_KM) < 1e-6 for row in rows
    )
    depths_absurd = all(row["polygon"]["interior_depth_km"] > 10000.0 for row in rows)
    # The density FIELD is a separate question from the attainment REPORT:
    # if the field were also wrong, a polygon region would refine its own
    # antipode and the cell count would explode.  It does not.
    cells_comparable = all(
        0.5 < row["polygon"]["predicted_cells"] / row["cap"]["predicted_cells"] < 2.0
        for row in rows
    )

    verdict = {
        "schema": "gpuwm-hex.polygon-attainment-probe.v1",
        "engine": str(engine),
        "engine_bytes": Path(engine).stat().st_size,
        "background_km": BACKGROUND_KM,
        "requested_spacing_km": SPACING_KM,
        "transition_km": TRANSITION_KM,
        "rows": rows,
        "control_cap_attainment_varies_with_size": cap_varies,
        "polygon_attainment_always_reports_the_request_met": polygon_always_met,
        "polygon_interior_depth_is_a_hemisphere": depths_absurd,
        "polygon_density_field_is_unaffected": cells_comparable,
        "finding": (
            "the generator's polygon region_attainment is not usable as an "
            "attained-spacing figure; the cap path is. The emitted meshes are "
            "unaffected -- a polygon's predicted cell count tracks the equivalent "
            "cap's, so the density field integrates correctly and only the "
            "reporting is wrong. Cause: rw-mpas/src/mesh/density.rs::"
            "polygon_contains accepts on abs(winding) > pi, and a closed ring "
            "divides a SPHERE into two discs whose windings are +2*pi and -2*pi, "
            "so the complement is called interior and the attainment lattice "
            "finds its deepest point near the antipode"
        )
        if (polygon_always_met and depths_absurd and cap_varies)
        else "no defect reproduced at this engine build",
    }
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
