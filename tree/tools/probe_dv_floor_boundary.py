#!/usr/bin/env python
"""Measure the REAL post-#311 dvEdge load boundary, through the real loader.

WHY THIS EXISTS.  The engine's ``validate.rs`` CARRIED an absolute
``min_dv_edge_m = 7,500.0`` floor whose stated breakage was the port's OLD
load check: ``rtol = 2e-5, atol = 0.0`` against f32-quantized vertices, under
which quantization alone exceeded the tolerance below ~7.5 km.  The check
that actually runs today is :func:`hexcore.mesh.spherical_arc_tolerance`
(rtol capped at ``8*eps32 ~ 9.54e-7``, atol ``2*sqrt(3)*spacing(f32 radius)
~ 1.732 m``), which names the quantization floor explicitly.  This probe
MEASURED that instead of arguing it, and the 2026-08-25 ruling re-anchored
the engine floor to 200 m on these readings (115x the live atol); the probe
stays so the boundary can be re-measured if the load check ever moves again.

METHOD.  Take the real published pair (grid + FP32 static).  For each target
dual-edge length, copy the static and slide ONE edge's two Voronoi vertices
symmetrically along their own geodesic until they sit exactly the target
apart, updating the stored ``dvEdge`` of every edge that touches a moved
vertex to its recomputed f64 arc (cast to the file's own dtype) and the
moved vertices' lat/lon.  Nothing else in ``Mesh.validate`` reads vertex
positions, so the edited pair isolates exactly the check under test, and the
verdict comes from ``Mesh.from_netcdf(grid, static, validate=True)`` -- the
real loader, on real files, not a reimplementation.  The pre-#311 verdict
(rtol 2e-5, atol 0.0) is computed alongside for the same edge, so the
retirement of the old boundary is a measured before/after, not a claim.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

TREE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TREE_SRC) not in sys.path:
    sys.path.insert(0, str(TREE_SRC))

from hexcore.mesh import (  # noqa: E402
    Mesh,
    MeshValidationError,
    spherical_arc_tolerance,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    omega = math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b)))
    if omega == 0.0:
        return a.copy()
    so = math.sin(omega)
    return (math.sin((1.0 - t) * omega) / so) * a + (math.sin(t * omega) / so) * b


def _arc(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b)))


def edit_static(source: Path, target: Path, dv_target_m: float) -> dict:
    """Copy the static and force one edge's dual length to ``dv_target_m``."""

    import netCDF4

    shutil.copyfile(source, target)
    with netCDF4.Dataset(str(target), "r+") as ds:
        radius = float(ds.sphere_radius)
        xv = np.asarray(ds.variables["xVertex"][:], dtype=np.float64)
        yv = np.asarray(ds.variables["yVertex"][:], dtype=np.float64)
        zv = np.asarray(ds.variables["zVertex"][:], dtype=np.float64)
        dv = np.asarray(ds.variables["dvEdge"][:], dtype=np.float64)
        lat_edge = np.asarray(ds.variables["latEdge"][:], dtype=np.float64)
        vertices_on_edge = np.asarray(ds.variables["verticesOnEdge"][:], dtype=np.int64)
        edges_on_vertex = np.asarray(ds.variables["edgesOnVertex"][:], dtype=np.int64)

        # A mid-latitude edge whose dvEdge sits near the median: unremarkable
        # by construction, so the only thing special about it is the edit.
        order = np.argsort(dv)
        median_pool = order[len(order) // 2 : len(order) // 2 + 4096]
        pick = None
        for e in median_pool:
            if abs(math.degrees(float(lat_edge[e]))) < 55.0:
                pick = int(e)
                break
        assert pick is not None, "no mid-latitude median edge found"

        v1, v2 = (int(v) - 1 for v in vertices_on_edge[pick])
        p1 = _unit(np.array([xv[v1], yv[v1], zv[v1]]) / radius)
        p2 = _unit(np.array([xv[v2], yv[v2], zv[v2]]) / radius)
        mid = _unit(p1 + p2)
        half = 0.5 * dv_target_m / radius  # radians
        full = _arc(mid, p1)
        t = half / full if full > 0 else 0.0
        q1 = _unit(_slerp(mid, p1, t))
        q2 = _unit(_slerp(mid, p2, t))

        for v, q in ((v1, q1), (v2, q2)):
            ds.variables["xVertex"][v] = q[0] * radius
            ds.variables["yVertex"][v] = q[1] * radius
            ds.variables["zVertex"][v] = q[2] * radius
            ds.variables["latVertex"][v] = math.asin(max(-1.0, min(1.0, q[2])))
            lon = math.atan2(q[1], q[0])
            ds.variables["lonVertex"][v] = lon if lon >= 0 else lon + 2.0 * math.pi

        # Every edge that touches a moved vertex gets its stored dvEdge
        # recomputed from the ideal f64 positions -- the same order of
        # operations a writer performs (f64 geometry, then storage cast).
        moved = {v1, v2}
        pos = {v1: q1, v2: q2}

        def vertex_pos(v: int) -> np.ndarray:
            if v in pos:
                return pos[v]
            return _unit(np.array([xv[v], yv[v], zv[v]]) / radius)

        affected = set()
        for v in moved:
            for e in edges_on_vertex[v]:
                if int(e) >= 1:
                    affected.add(int(e) - 1)
        updates = {}
        for e in sorted(affected):
            a, b = (int(v) - 1 for v in vertices_on_edge[e])
            new_dv = _arc(vertex_pos(a), vertex_pos(b)) * radius
            ds.variables["dvEdge"][e] = new_dv
            updates[e] = new_dv

        # The pre-#311 verdict for the edited edge, computed against the
        # values as STORED (both sides quantized the way the loader sees
        # them), so the old check's failure is measured, not modelled.
        stored_dv = float(np.asarray(ds.variables["dvEdge"][pick], dtype=np.float64))
        f32 = lambda v: np.asarray(  # noqa: E731
            [ds.variables["xVertex"][v], ds.variables["yVertex"][v], ds.variables["zVertex"][v]],
            dtype=np.float64,
        )
        recomputed = _arc(_unit(f32(v1) / radius), _unit(f32(v2) / radius)) * radius
        gap = abs(stored_dv - recomputed)
        old_pass = gap <= 2.0e-5 * abs(recomputed)
        arc_rtol, arc_atol = spherical_arc_tolerance(radius, np.dtype(np.float32))
        new_pass_predicted = gap <= arc_rtol * abs(recomputed) + arc_atol

    return {
        "edge": pick,
        "dv_target_m": dv_target_m,
        "stored_dv_m": stored_dv,
        "recomputed_from_f32_m": recomputed,
        "quantization_gap_m": gap,
        "edges_updated": len(updates),
        "pre_311_check": {
            "rtol": 2.0e-5,
            "atol": 0.0,
            "passes": bool(old_pass),
        },
        "post_311_check_predicted": {
            "rtol": arc_rtol,
            "atol_m": arc_atol,
            "passes": bool(new_pass_predicted),
        },
    }


def load_verdict(grid: Path, static: Path) -> dict:
    try:
        Mesh.from_netcdf(grid, static, validate=True)
        return {"loaded": True, "error": None}
    except MeshValidationError as error:
        return {"loaded": False, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--static", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--targets-m",
        default="50,200,1000,5000",
        help="synthetic dvEdge targets in metres",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    receipt: dict = {
        "instrument": "probe_dv_floor_boundary",
        "grid": str(args.grid),
        "static": str(args.static),
        "cases": [],
    }

    print("control: the unedited pair through the real loader", flush=True)
    control = load_verdict(args.grid, args.static)
    receipt["control"] = control
    print(f"  control loaded={control['loaded']}")
    if not control["loaded"]:
        print(f"  {control['error']}")
        print("the control pair must load before any edited pair means anything")
        return 1

    for target in [float(t) for t in args.targets_m.split(",")]:
        edited = args.out_dir / f"boundary-{int(target)}m.static.nc"
        edit = edit_static(args.static, edited, target)
        verdict = load_verdict(args.grid, edited)
        row = {"edit": edit, "verdict": verdict}
        receipt["cases"].append(row)
        print(
            f"  dvEdge {edit['stored_dv_m']:.3f} m (target {target:.0f}): "
            f"loader={'PASS' if verdict['loaded'] else 'REFUSED'}; "
            f"quantization gap {edit['quantization_gap_m']:.3f} m; "
            f"pre-#311 check would {'PASS' if edit['pre_311_check']['passes'] else 'FAIL'}",
            flush=True,
        )
        if not verdict["loaded"]:
            print(f"    {verdict['error'].splitlines()[-1]}")

    out = args.out_dir / "RECEIPT-dv-floor-boundary.json"
    out.write_text(json.dumps(receipt, indent=2))
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
