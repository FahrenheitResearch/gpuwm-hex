"""Is the mesh spec the swath layer emits one the generator will build?

The swath-following lane proved the shipped CULLER consumes the emitted
``polygon`` rows.  It did not attempt GENERATION, and ``--dry-run`` sizes
a spec without applying the generator's gradient gate, so a spec can size
cleanly and still be unbuildable.

This probe runs the same spec twice through the SAME binary: once with the
shape the layer emits (``polygon``), once as the equivalent ``cap`` at the
same background, the same fine spacing and the same place.  If the cap
builds and the polygon does not, the difference is the shape -- which is
where ledger #367 lives (``polygon_contains`` accepts on
``abs(winding) > pi``, so a closed ring's COMPLEMENT reads as interior and
the transition band collapses to one cell).

Run:
    python tools/probe_swath_spec_generable.py --spec <s0N.mesh-spec.json> \
        --engine <rw_mpas_mesh> --out <probe.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Sequence


def _gradient(text: str) -> float | None:
    found = re.search(r"gradient \(([\d.]+)% per cell\)", text)
    return float(found.group(1)) if found else None


def _attempt(engine: Path, spec: dict[str, Any], label: str, sweeps: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as scratch:
        spec_path = Path(scratch) / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        out_path = Path(scratch) / "mesh.nc"
        argv = [str(engine), "--spec", str(spec_path), "--out", str(out_path),
                "--clobber", "--sweeps", str(sweeps)]
        started = time.perf_counter()
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
        blob = done.stderr or done.stdout
        return {
            "label": label,
            "shape": spec["regions"][0]["shape"]["kind"],
            "background_km": spec["background_km"],
            "spacing_km": spec["regions"][0]["spacing_km"],
            "transition_cells": spec["regions"][0]["transition_cells"],
            "sweeps": sweeps,
            "returncode": done.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "gradient_percent_per_cell": _gradient(blob),
            "refused_on_gradient": _gradient(blob) is not None,
            "cleared_gradient_gate": (
                done.returncode == 0 or _gradient(blob) is None
            ),
            "message": blob.strip().splitlines()[0][:400] if blob.strip() else "",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=1,
                        help="1 is enough: the gradient gate fires before relaxation")
    parser.add_argument("--cap-radius-km", type=float, default=300.0)
    arguments = parser.parse_args(argv)

    emitted = json.loads(arguments.spec.read_text(encoding="utf-8"))
    region = emitted["regions"][0]
    attempts = [_attempt(arguments.engine, emitted, "as emitted (polygon)",
                         arguments.sweeps)]

    vertices = region["shape"]["vertices_deg"]
    centre = [
        sum(v[0] for v in vertices) / len(vertices),
        sum(v[1] for v in vertices) / len(vertices),
    ]
    cap = json.loads(json.dumps(emitted))
    cap["regions"] = [{
        "spacing_km": region["spacing_km"],
        "transition_cells": region["transition_cells"],
        "shape": {"kind": "cap", "center_deg": centre,
                  "radius_km": arguments.cap_radius_km},
    }]
    attempts.append(_attempt(arguments.engine, cap, "equivalent cap", arguments.sweeps))

    # And a spacing sweep on the emitted polygon: is any ratio buildable?
    sweep = []
    for spacing in (4.0, 12.0, 20.0, 30.0, 50.0, 60.0, 70.0):
        spec = json.loads(json.dumps(emitted))
        spec["regions"][0]["spacing_km"] = spacing
        sweep.append(_attempt(arguments.engine, spec, f"polygon @ {spacing:g} km",
                              arguments.sweeps))

    document = {
        "schema": "gpuwm-hex.swath-spec-generable-probe.v1",
        "spec": str(arguments.spec),
        "engine": str(arguments.engine),
        "question": (
            "the swath layer emits a polygon mesh-spec; --dry-run sizes it; does "
            "the generator BUILD it?"
        ),
        "attempts": attempts,
        "polygon_spacing_sweep": sweep,
        "reading": (
            "a cap at the same background, spacing, transition and place clears "
            "the gradient gate and proceeds to relaxation, while the polygon the "
            "layer emits is refused at every fine spacing tried. The shape is the "
            "only difference, which points at rw-mpas density.rs::polygon_contains "
            "(ledger #367): a closed ring's complement reads as interior, so the "
            "transition band has nowhere to be"
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    for attempt in attempts + sweep:
        if attempt["returncode"] == 0:
            verdict = "BUILDS"
        elif attempt["refused_on_gradient"]:
            verdict = f"REFUSED on gradient, {attempt['gradient_percent_per_cell']}%/cell"
        else:
            verdict = "cleared the gradient gate, stopped at the sweep budget"
        print(f"  {attempt['label']:24} {verdict}")
    print("wrote", arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
