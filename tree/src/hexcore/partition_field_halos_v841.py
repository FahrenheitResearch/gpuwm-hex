"""Field-specific owner truth within the MPAS-A v8.4.1 halo widths.

Authority: src/core_atmosphere/mpas_atm_halos.F at v8.4.1, lines 78-154.
The layout retains K3 stencil metadata. Packets select field-specific halo
tails without changing local indices. Port rounds aggregate native groups,
so each field uses the maximum width of its corresponding native groups.
"""
from __future__ import annotations

import numpy as np

from .partition_samehost_tables_v841 import build_all

FIELD_WIDTHS = {
    "rho": 2, "rho_theta": 2, "rho_u": 3, "rho_w": 2,
    "scalars": 2, "theta_m": 2, "exner": 2,
    "density_perturbation": 2, "rho_theta_perturbation": 2,
    "pressure_perturbation": 2, "normal_velocity": 3, "vertical_velocity": 2,
    # The combined acoustic group uses rho_pp[1,2] and rtheta_pp[2].
    # The later one-layer damping groups do not reduce that maximum.
    "rho_pp": 2, "rtheta_pp": 2, "ru_p": 2,
    "scale_in": 2, "scale_out": 2,
}
FIELD_KINDS = {name: "edge" if name in ("rho_u", "normal_velocity", "ru_p") else "cell"
               for name in FIELD_WIDTHS}


def build_field_indices(layouts, cells_on_edge, *, tables=None):
    # ``tables`` is build_all(layouts) when the caller already holds it (the
    # same-host driver does); one build of the 20 km tables is 2.4-3.9 s.
    if tables is None:
        tables = build_all(layouts)
    global_edges = np.asarray(cells_on_edge)
    if global_edges.ndim != 2 or global_edges.shape[1] != 2 or global_edges.dtype.kind not in "iu":
        raise ValueError("field halo planning needs integer edge endpoints; prevents dropping a required seam value")
    if (len(global_edges) != layouts[0]._global_edges or
            global_edges.size and (global_edges.min() < -1 or global_edges.max() >= layouts[0]._global_cells)):
        raise ValueError("edge endpoints do not match the global mesh; prevents selecting a halo from another grid")
    rings = []
    for layout in layouts:
        inverse = layout.cell_g2l()
        endpoints = global_edges[layout.edge_l2g]
        local = inverse[np.maximum(endpoints, 0)]
        distance = np.full(endpoints.shape, 32760, np.int32)
        present = (endpoints >= 0) & (local >= 0)
        distance[present] = layout.cell_ring[local[present]]
        # MPAS includes all edges bounding owned cells in its solve region,
        # even when the unique owner is another rank. Edge layer therefore
        # follows the minimum incident cell layer, as in the layout builder.
        # Adding one drops the outer native u[3] layer needed by diagnostics.
        edge_ring = np.min(distance, axis=1)
        edge_ring[:layout.n_owned_edges] = 0
        rings.append({"cell": np.asarray(layout.cell_ring), "edge": edge_ring})
    indexes = {}
    for rank, table in enumerate(tables):
        for peer in table.peers:
            reverse = next(row for row in tables[peer.peer].peers if row.peer == rank)
            for name, width in FIELD_WIDTHS.items():
                kind = FIELD_KINDS[name]
                received = getattr(peer, f"{kind}_recv")
                sent = getattr(reverse, f"{kind}_send")
                if not np.array_equal(getattr(layouts[rank], f"{kind}_l2g")[received],
                                      getattr(layouts[peer.peer], f"{kind}_l2g")[sent]):
                    raise ValueError("field halo send/receive identities disagree; prevents copying the wrong global entity")
                selected = rings[rank][kind][received] <= width
                for key, values in (((rank, peer.peer, name, "recv"), received[selected]),
                                    ((peer.peer, rank, name, "send"), sent[selected])):
                    values = np.ascontiguousarray(values, dtype=np.int64)
                    values.setflags(write=False)
                    indexes[key] = values
    return indexes
