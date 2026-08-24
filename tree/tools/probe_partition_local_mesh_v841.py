#!/usr/bin/env python
"""Census gate: slice the REAL x4.163842 mesh onto a partition and verify.

CPU only.  Builds the partition-local ``Mesh`` for one partition, re-derives
the row-completeness verdicts from the sliced arrays, and prints the full
per-array census (name, entity dimension, remap class, global -> local shape).
Also slices the global advection coefficients so ``advCellsForEdge`` /
``adv_coefs`` participate in the edge verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpas_port.partition_assets_v841 import build_partition_layouts  # noqa: E402
from mpas_port.partition_local_mesh_v841 import (  # noqa: E402
    build_local_mesh,
    prepared_slice_report,
    slice_prepared_host,
    verify_row_completeness,
)


def _asset_root() -> Path:
    """The asset store root, or a refusal naming the flag and the setting.

    No path is baked in: a wrong default would be joined with a filename and
    the run would refuse as a missing netCDF, hiding that the asset store was
    simply never named.
    """
    declared = os.environ.get("MPAS_ASSET_ROOT")
    if not declared:
        raise SystemExit(
            "no asset store: pass --assets/--static/--grid explicitly or set "
            "MPAS_ASSET_ROOT to the asset store root"
        )
    return Path(declared).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=Path,
        default=None,
        help="official METIS distribution directory; defaults to "
        "$MPAS_ASSET_ROOT/meshes/official-vr-92to25",
    )
    parser.add_argument(
        "--static",
        type=Path,
        default=None,
        help="defaults to $MPAS_ASSET_ROOT/x4.163842.static.nc",
    )
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--halo-rings", type=int, default=60)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--grid",
        type=Path,
        default=None,
        help="the runner's authoritative grid (connectivity is identical to "
        "the official distribution beside the part files); defaults to "
        "$MPAS_ASSET_ROOT/x4.163842.grid.nc",
    )
    args = parser.parse_args(argv)

    assets = args.assets if args.assets is not None else (
        _asset_root() / "meshes" / "official-vr-92to25"
    )
    grid = args.grid if args.grid is not None else (
        _asset_root() / "x4.163842.grid.nc"
    )
    static = args.static if args.static is not None else (
        _asset_root() / "x4.163842.static.nc"
    )
    graph = assets / "x4.163842.graph.info"
    part = assets / f"x4.163842.graph.info.part.{args.parts}"

    from netCDF4 import Dataset

    with Dataset(grid) as data:
        coe = np.asarray(data.variables["cellsOnEdge"][:], dtype=np.int32) - 1
        cov = np.asarray(data.variables["cellsOnVertex"][:], dtype=np.int32) - 1

    started = time.time()
    layouts = build_partition_layouts(
        graph,
        part,
        halo_rings=args.halo_rings,
        cells_on_vertex=cov,
        cells_on_edge=coe,
        partitions=[args.partition],
    )
    layout = layouts[0]
    print(
        f"layout p{layout.partition}: owned={layout.n_owned_cells} "
        f"local={layout.n_local_cells} localE={layout.n_local_edges} "
        f"localV={layout.n_local_vertices} ({time.time() - started:.1f}s)"
    )

    from mpas_port.mesh import load_precision_preserving_mesh_pair

    loaded = time.time()
    mesh, output_mesh, _ = load_precision_preserving_mesh_pair(grid, static)
    del output_mesh
    print(f"global mesh loaded in {time.time() - loaded:.1f}s")

    local = build_local_mesh(mesh, layout, verify=False)
    print(f"local mesh built; {len(local.arrays)} arrays")

    # Slice the global advection coefficients so advCellsForEdge takes part in
    # the edge verdict, then fold its clamp mask into the verification.
    from mpas_port.partition_local_mesh_v841 import _remap_values

    adv_report: dict[str, object] = {"available": False}
    clamp: dict[str, np.ndarray] = {}
    for name, value in dict(local.arrays).items():
        pass
    # rebuild the clamp report by re-running the remap on the global arrays
    from mpas_port.partition_local_mesh_v841 import (
        CELL_VALUED,
        EDGE_VALUED,
        NOT_REMAPPED,
        VERTEX_VALUED,
    )

    g2l = {
        "nCells": layout.cell_g2l(),
        "nEdges": layout.edge_g2l(),
        "nVertices": layout.vertex_g2l(),
    }
    l2g = {
        "nCells": layout.cell_l2g,
        "nEdges": layout.edge_l2g,
        "nVertices": layout.vertex_l2g,
    }
    dims = dict(getattr(mesh, "variable_dimensions", {}) or {})
    for name in sorted(set(CELL_VALUED) | set(EDGE_VALUED) | set(VERTEX_VALUED)):
        if name in NOT_REMAPPED or name not in mesh.arrays:
            continue
        target = (
            "nCells"
            if name in CELL_VALUED
            else "nEdges"
            if name in EDGE_VALUED
            else "nVertices"
        )
        source = np.asarray(mesh.arrays[name])
        entity = next((d for d in dims.get(name, ()) if d in l2g), None)
        if entity is None:
            continue
        axis = dims[name].index(entity)
        gathered = np.take(source, l2g[entity], axis=axis)
        _, clamped = _remap_values(gathered, g2l[target], name=name)
        clamp[name] = clamped

    try:
        from mpas_port.transport import build_advection_coefficients

        coefficients = build_advection_coefficients(mesh)
        adv_cells = np.asarray(coefficients.adv_cells_for_edge)
        adv_local = adv_cells[layout.edge_l2g]
        _, adv_clamped = _remap_values(
            adv_local, g2l["nCells"], name="adv_cells_for_edge"
        )
        clamp["adv_cells_for_edge"] = adv_clamped
        local.arrays["nAdvCellsForEdge"] = np.asarray(
            coefficients.n_adv_cells_for_edge
        )[layout.edge_l2g]
        adv_report = {
            "available": True,
            "width": int(adv_cells.shape[1]),
            "n_adv_max": int(np.asarray(coefficients.n_adv_cells_for_edge).max()),
            "adv_coefs_shape": list(np.asarray(coefficients.adv_coefs).shape),
        }
        print(f"advection coefficients sliced: {adv_report}")
    except Exception as error:  # pragma: no cover - reported, never swallowed
        adv_report = {"available": False, "error": f"{type(error).__name__}: {error}"}
        print(f"advection coefficients NOT sliced: {adv_report}")

    verdict = verify_row_completeness(local, layout, clamp)
    print("row-completeness verification:")
    print(json.dumps(verdict, indent=2))

    census = sorted(
        local.partition_census, key=lambda row: (str(row.get("entity")), row["name"])
    )
    print(f"\n{'array':<34} {'entity':<10} {'remap':<10} shape")
    for row in census:
        print(
            f"{row['name']:<34} {str(row.get('entity')):<10} "
            f"{str(row.get('remapped_to')):<10} {row['shape']}"
        )

    receipt = {
        "partition": layout.partition,
        "parts": args.parts,
        "halo_rings": args.halo_rings,
        "layout": layout.receipt(),
        "row_completeness": verdict,
        "advection": adv_report,
        "clamp_counts": {k: int(np.count_nonzero(v)) for k, v in clamp.items()},
        "census": census,
    }
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"\nreceipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
