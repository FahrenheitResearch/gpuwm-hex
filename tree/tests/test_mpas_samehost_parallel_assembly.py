"""The owner assembly of a multi-rank frame is parallel and writes the serial assembly's words.

THE BREAKAGE THIS PREVENTS (MPAS-ROUTE-SPEED-2026-10-24): one host thread copied and scattered every word of a
frame while every rank waited, so a 60 km MPAS-suite frame on 2 cards stalled its step 1.55 s (0.31 s on one
card).  Parallel, the same frame takes 0.62 s more than an ordinary step, and the 2-card history and diag files
are value-identical (0 of 377,818,714 history words different at +3 h).
"""
from types import SimpleNamespace

import numpy as np

from hexcore.mpasphys import samehost_v842
from hexcore.output import HistoryField


def _partitions():
    # Three ranks over 9 global cells; owned cells first in local order, then halo cells.
    ids = [np.array([0, 3, 6, 1]), np.array([1, 4, 7, 2, 0]), np.array([2, 5, 8, 4])]
    return [SimpleNamespace(cell_l2g=i, n_owned_cells=3, edge_l2g=np.arange(1), n_owned_edges=1,
                            vertex_l2g=np.arange(1), n_owned_vertices=1) for i in ids]


def test_parallel_assembly_writes_the_serial_words() -> None:
    partitions = _partitions()
    rng = np.random.default_rng(7)
    rows = {"theta": ({"pool": "diag", "name": "theta"}, None),
            "rainnc": ({"pool": "diag_physics", "name": "rainnc"}, None),
            "nlev": ({"pool": "mesh", "name": "nlev"}, None)}
    snapshots = []
    for part in partitions:
        n = len(part.cell_l2g)
        theta = rng.standard_normal((n, 4)).astype(np.float32)
        theta[0, 0] = np.float32("nan")
        snapshots.append({"theta": HistoryField(theta, ("nCells", "nVertLevels"), {}),
                          "rainnc": HistoryField(rng.standard_normal(n).astype(np.float32), ("nCells",), {}),
                          "nlev": HistoryField(np.arange(4, dtype=np.int32), ("nVertLevels",), {}),
                          "Time": "2025-05-19_12:00:00"})
    layout = SimpleNamespace(dimensions=dict(nCells=9, nEdges=1, nVertices=1))
    serial = samehost_v842.assemble_snapshot(snapshots, partitions, rows, layout, workers=1)
    parallel = samehost_v842.assemble_snapshot(snapshots, partitions, rows, layout, workers=8)
    assert list(serial) == list(parallel)
    for name in ("theta", "rainnc", "nlev"):
        assert serial[name].values.tobytes() == parallel[name].values.tobytes()
    # Owner words only: global cell 1 comes from rank 1's first owned cell, never rank 0's halo copy.
    assert parallel["rainnc"].values[1] == snapshots[1]["rainnc"].values[0]
    assert parallel["theta"].values.shape == (9, 4)
