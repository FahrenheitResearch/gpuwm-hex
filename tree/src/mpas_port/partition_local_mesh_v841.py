"""Partition-local sub-mesh construction (brief-1, D2).

``build_local_mesh(global_mesh, layout)`` returns a :class:`mpas_port.mesh.Mesh`
whose arrays are the sliced/renumbered locals for one partition.

Slicing law
-----------
Every mesh array is gathered along its entity axis (the axis whose declared
dimension is ``nCells``/``nEdges``/``nVertices``); slots inside a row are never
reordered, so "aligned" float rows (``weightsOnEdge`` with ``edgesOnEdge``,
``kiteAreasOnVertex`` with ``cellsOnVertex``, ``defc_a``/``defc_b`` with
``edgesOnCell``, ``adv_coefs``/``adv_coefs_3rd`` with ``advCellsForEdge``) stay
aligned by construction.  Arrays with no entity axis (the global 1D vertical
profiles ``fzm, fzp, etp, etm, ewp, ewm, u_init, v_init`` and the reference
profiles) are shared verbatim.

Remap law
---------
Integer arrays whose *values* are entity ids are additionally mapped through
the global->local dictionary of the referenced class.  A referenced id absent
from the local set is clamped to local index 0 (deterministic garbage; such
rows only ever feed discarded outputs).  Negative padding sentinels are not
references and are copied verbatim.

The row-completeness masks carried on the layout are *predicted* from the
adjacency graph.  This module re-derives them from the real arrays and refuses
if the prediction and the observation disagree, so a census gap cannot hide.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, is_dataclass, replace
from typing import Any, Mapping

import numpy as np

from .partition_assets_v841 import PartitionLayout

# value-space census: which entity class an integer array's VALUES refer to.
CELL_VALUED = frozenset(
    {
        "cellsOnCell",
        "cellsOnEdge",
        "cellsOnVertex",
        "advCellsForEdge",
        "adv_cells_for_edge",
    }
)
EDGE_VALUED = frozenset({"edgesOnCell", "edgesOnEdge", "edgesOnVertex"})
VERTEX_VALUED = frozenset({"verticesOnCell", "verticesOnEdge"})
# integer arrays that are counts or global identifiers, never local ids
NOT_REMAPPED = frozenset(
    {
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "nAdvCellsForEdge",
        "n_adv_cells_for_edge",
        "indexToCellID",
        "indexToEdgeID",
        "indexToVertexID",
        "landmask",
        "edgeSignOnCell",
        "edgesOnCell_sign",
        "obsoleteSlot",
    }
)

_CELL_DIMS = ("nCells",)
_EDGE_DIMS = ("nEdges",)
_VERTEX_DIMS = ("nVertices",)


class PartitionCensusError(RuntimeError):
    """A mesh array could not be classified, or a mask prediction failed."""


def _remap_values(
    values: np.ndarray, g2l: np.ndarray, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Map global ids to local ids; clamp absent references to 0.

    Returns ``(local_values, clamped_mask)`` where ``clamped_mask`` marks the
    non-negative slots whose referent is outside the local set.
    """

    source = np.asarray(values)
    if source.dtype.kind not in "iu":
        raise PartitionCensusError(f"{name}: expected an integer id array")
    as_int = source.astype(np.int64, copy=False)
    padding = as_int < 0
    safe = np.where(padding, 0, as_int)
    if safe.size and safe.max() >= g2l.size:
        raise PartitionCensusError(f"{name}: id {int(safe.max())} outside global extent")
    mapped = g2l[safe]
    clamped = (mapped < 0) & ~padding
    result = np.where(padding, as_int, np.where(mapped < 0, 0, mapped))
    return np.ascontiguousarray(result.astype(source.dtype, copy=False)), clamped


def _entity_axis(
    name: str,
    array: np.ndarray,
    dims: tuple[str, ...] | None,
    counts: Mapping[str, int],
) -> tuple[str | None, int | None]:
    if dims:
        for axis, dim in enumerate(dims):
            if dim in counts:
                return dim, axis
        return None, None
    matches = [
        dim
        for dim, size in counts.items()
        if array.ndim and array.shape[-1] == size
    ]
    if len(matches) == 1:
        return matches[0], array.ndim - 1
    if len(matches) > 1:
        raise PartitionCensusError(
            f"{name}: shape {array.shape} is ambiguous between {matches}; "
            "declare variable_dimensions"
        )
    return None, None


def build_local_mesh(
    global_mesh: Any, layout: PartitionLayout, *, verify: bool = True
) -> Any:
    """Construct the partition-local ``Mesh``."""

    from .mesh import Mesh

    counts = {
        "nCells": int(layout._global_cells),
        "nEdges": int(layout._global_edges),
        "nVertices": int(layout._global_vertices),
    }
    index_of = {
        "nCells": layout.cell_l2g,
        "nEdges": layout.edge_l2g,
        "nVertices": layout.vertex_l2g,
    }
    g2l = {
        "nCells": layout.cell_g2l(),
        "nEdges": layout.edge_g2l(),
        "nVertices": layout.vertex_g2l(),
    }
    value_class = {
        **{name: "nCells" for name in CELL_VALUED},
        **{name: "nEdges" for name in EDGE_VALUED},
        **{name: "nVertices" for name in VERTEX_VALUED},
    }

    variable_dimensions = dict(getattr(global_mesh, "variable_dimensions", {}) or {})
    arrays: dict[str, np.ndarray] = {}
    clamp_report: dict[str, np.ndarray] = {}
    census: list[dict[str, Any]] = []

    for name, value in dict(global_mesh.arrays).items():
        array = np.asarray(value)
        dim, axis = _entity_axis(
            name, array, variable_dimensions.get(name), counts
        )
        if dim is None:
            arrays[name] = array
            census.append({"name": name, "entity": None, "shape": list(array.shape)})
            continue
        if axis != 0 and axis != array.ndim - 1:
            gathered = np.take(array, index_of[dim], axis=axis)
        else:
            gathered = np.take(array, index_of[dim], axis=axis)
        gathered = np.ascontiguousarray(gathered)
        target = value_class.get(name)
        if target is not None and name not in NOT_REMAPPED:
            gathered, clamped = _remap_values(gathered, g2l[target], name=name)
            clamp_report[name] = clamped
        arrays[name] = gathered
        census.append(
            {
                "name": name,
                "entity": dim,
                "axis": int(axis),
                "remapped_to": value_class.get(name),
                "shape": list(gathered.shape),
                "dtype": str(gathered.dtype),
            }
        )

    dimensions = dict(getattr(global_mesh, "dimensions", {}) or {})
    dimensions.update(
        {
            "nCells": layout.n_local_cells,
            "nEdges": layout.n_local_edges,
            "nVertices": layout.n_local_vertices,
        }
    )
    local = Mesh(
        arrays=arrays,
        dimensions=dimensions,
        attrs=dict(getattr(global_mesh, "attrs", {}) or {}),
        grid_attrs=dict(getattr(global_mesh, "grid_attrs", {}) or {}),
        static_attrs=dict(getattr(global_mesh, "static_attrs", {}) or {}),
        variable_dimensions=variable_dimensions,
        variable_attrs=getattr(global_mesh, "variable_attrs", {}) or {},
        variable_sources=getattr(global_mesh, "variable_sources", {}) or {},
        converted_connectivity=tuple(
            getattr(global_mesh, "converted_connectivity", ()) or ()
        ),
    )
    local.partition_layout = layout
    local.partition_census = census
    local.partition_clamp_report = {
        name: int(np.count_nonzero(mask)) for name, mask in clamp_report.items()
    }
    if verify:
        verify_row_completeness(local, layout, clamp_report)
    return local


def verify_row_completeness(
    local_mesh: Any,
    layout: PartitionLayout,
    clamp_report: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Refuse if observed clamping disagrees with the predicted masks.

    Only *active* slots count: a row is incomplete when a slot inside the
    entity's declared count was clamped.  ``advCellsForEdge`` participates in
    the edge verdict, per brief-1 law 5.
    """

    def active_mask(count_name: str, width: int, n_rows: int) -> np.ndarray:
        counts = local_mesh.arrays.get(count_name)
        if counts is None:
            return np.ones((n_rows, width), dtype=bool)
        counts = np.asarray(counts).astype(np.int64, copy=False)
        return np.arange(width)[None, :] < counts[:, None]

    observed: dict[str, np.ndarray] = {}
    saw_all: dict[str, bool] = {}
    for entity, names, required in (
        ("cell", ("cellsOnCell", "edgesOnCell", "verticesOnCell"), ("cellsOnCell",)),
        (
            "edge",
            (
                "cellsOnEdge",
                "verticesOnEdge",
                "edgesOnEdge",
                "advCellsForEdge",
                "adv_cells_for_edge",
            ),
            ("cellsOnEdge", "edgesOnEdge"),
        ),
        ("vertex", ("cellsOnVertex", "edgesOnVertex"), ("cellsOnVertex",)),
    ):
        n_rows = {
            "cell": layout.n_local_cells,
            "edge": layout.n_local_edges,
            "vertex": layout.n_local_vertices,
        }[entity]
        bad = np.zeros(n_rows, dtype=bool)
        for name in names:
            mask = clamp_report.get(name)
            if mask is None:
                continue
            mask = np.asarray(mask)
            if mask.ndim == 1:
                bad |= mask
                continue
            width = mask.shape[1]
            if entity == "cell":
                mask = mask & active_mask("nEdgesOnCell", width, n_rows)
            elif name == "edgesOnEdge":
                mask = mask & active_mask("nEdgesOnEdge", width, n_rows)
            elif name in ("advCellsForEdge", "adv_cells_for_edge"):
                mask = mask & active_mask("nAdvCellsForEdge", width, n_rows)
            bad |= mask.any(axis=1)
        observed[entity] = ~bad
        # The stencil array advCellsForEdge lives outside Mesh (it is carried
        # by the advection-coefficient structure), so an edge verdict built
        # from the Mesh alone is an upper bound on completeness.
        adv_present = any(
            name in clamp_report for name in ("advCellsForEdge", "adv_cells_for_edge")
        )
        saw_all[entity] = all(name in clamp_report for name in required) and (
            entity != "edge" or adv_present
        )

    predicted = {
        "cell": np.asarray(layout.cell_row_complete),
        "edge": np.asarray(layout.edge_row_complete),
        "vertex": np.asarray(layout.vertex_row_complete),
    }
    report: dict[str, Any] = {}
    for entity, mask in observed.items():
        unsound = int(np.count_nonzero(predicted[entity] & ~mask))
        conservative = int(np.count_nonzero(mask & ~predicted[entity]))
        report[entity] = {
            "predicted_complete": int(np.count_nonzero(predicted[entity])),
            "observed_complete": int(np.count_nonzero(mask)),
            "rows_predicted_complete_but_clamped": unsound,
            "rows_observed_complete_but_predicted_incomplete": conservative,
            "all_contributing_arrays_present": bool(saw_all[entity]),
            "agree": unsound == 0 and (conservative == 0 or not saw_all[entity]),
        }
        if unsound:
            raise PartitionCensusError(
                f"{entity}_row_complete claims {unsound} rows complete that the "
                f"sliced arrays show clamped (partition {layout.partition})"
            )
        if conservative and saw_all[entity]:
            raise PartitionCensusError(
                f"{entity}_row_complete marks {conservative} rows incomplete that "
                f"the sliced arrays show clean (partition {layout.partition})"
            )
    return report


# ---------------------------------------------------------------------------
# prepared-host slicing
# ---------------------------------------------------------------------------


def slice_prepared_host(
    prepared: Any,
    layout: PartitionLayout,
    *,
    _seen: dict[int, Any] | None = None,
    _path: str = "",
    _report: list[dict[str, Any]] | None = None,
) -> Any:
    """Recursively gather every mesh-shaped array in a prepared host object.

    Handles dataclasses, mappings and plain attribute objects.  An array is
    treated as entity-indexed when its LAST axis equals a global entity count
    (the port's layout law puts the entity index last).  Integer arrays whose
    names appear in the value-space census are additionally renumbered.
    Everything else -- vertical profiles, reference profiles, scalars -- is
    shared verbatim.

    Returns the sliced structure; the per-field report is attached to the
    returned object as ``partition_slice_report`` when possible, and is always
    available via :func:`prepared_slice_report`.
    """

    report: list[dict[str, Any]] = [] if _report is None else _report
    seen: dict[int, Any] = {} if _seen is None else _seen
    result = _slice_any(prepared, layout, seen, _path, report)
    try:
        object.__setattr__(result, "partition_slice_report", report)
    except Exception:  # pragma: no cover - immutable containers
        pass
    _LAST_REPORT.clear()
    _LAST_REPORT.extend(report)
    return result


_LAST_REPORT: list[dict[str, Any]] = []


def prepared_slice_report() -> list[dict[str, Any]]:
    """Field list (name, entity dim, shape, remap class) of the last slice."""

    return list(_LAST_REPORT)


def _slice_any(
    value: Any,
    layout: PartitionLayout,
    seen: dict[int, Any],
    path: str,
    report: list[dict[str, Any]],
) -> Any:
    from .mesh import Mesh

    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return value
    key = id(value)
    if key in seen:
        return seen[key]
    if isinstance(value, Mesh):
        sliced = build_local_mesh(value, layout)
        seen[key] = sliced
        report.append({"path": path, "kind": "Mesh", "entity": "mesh"})
        return sliced
    if isinstance(value, np.ndarray):
        return _slice_array(value, layout, path.rsplit(".", 1)[-1], path, report)
    if isinstance(value, Mapping):
        out = {
            name: _slice_any(item, layout, seen, f"{path}.{name}" if path else str(name), report)
            for name, item in value.items()
        }
        seen[key] = out
        return out
    if isinstance(value, (list, tuple)):
        out = type(value)(
            _slice_any(item, layout, seen, f"{path}[{i}]", report)
            for i, item in enumerate(value)
        )
        return out
    if is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for field in dataclass_fields(value):
            current = getattr(value, field.name)
            new = _slice_any(
                current, layout, seen, f"{path}.{field.name}" if path else field.name, report
            )
            if new is not current:
                changes[field.name] = new
        try:
            out = replace(value, **changes) if changes else value
        except Exception:
            out = value
            for name, new in changes.items():
                object.__setattr__(out, name, new)
        seen[key] = out
        return out
    if hasattr(value, "__dict__"):
        import copy as _copy

        out = _copy.copy(value)
        seen[key] = out
        for name, current in list(vars(value).items()):
            if name.startswith("__"):
                continue
            new = _slice_any(
                current, layout, seen, f"{path}.{name}" if path else name, report
            )
            if new is not current:
                object.__setattr__(out, name, new)
        return out
    return value


def _slice_array(
    array: np.ndarray,
    layout: PartitionLayout,
    name: str,
    path: str,
    report: list[dict[str, Any]],
) -> np.ndarray:
    counts = {
        "nCells": int(layout._global_cells),
        "nEdges": int(layout._global_edges),
        "nVertices": int(layout._global_vertices),
    }
    index_of = {
        "nCells": layout.cell_l2g,
        "nEdges": layout.edge_l2g,
        "nVertices": layout.vertex_l2g,
    }
    if array.ndim == 0:
        return array
    # The port's layout law puts the entity index LAST for fields, but
    # coefficient/metric carriers put it elsewhere (advection rows
    # ``(nEdges, width)`` lead with it; terrain ``zb_cell`` carries it in the
    # middle, ``(nlev+1, nCells, maxEdges)``).  Match every axis against the
    # global entity extents: prefer a trailing match (the field law), else
    # accept exactly one non-trailing match; more than one is a refusal,
    # never a guess.  Non-entity extents (levels, stencil widths) can never
    # equal a global entity count.
    pairs = [
        (axis, dim)
        for axis in range(array.ndim)
        for dim, size in counts.items()
        if array.shape[axis] == size
    ]
    if not pairs:
        return array
    trailing = [(axis, dim) for axis, dim in pairs if axis == array.ndim - 1]
    if len(trailing) == 1:
        axis, dim = trailing[0]
    elif len(pairs) == 1:
        axis, dim = pairs[0]
    else:
        raise PartitionCensusError(
            f"{path}: shape {array.shape} matches entity extents ambiguously: {pairs}"
        )
    if axis == array.ndim - 1:
        gathered = np.ascontiguousarray(array[..., index_of[dim]])
    else:
        gathered = np.ascontiguousarray(np.take(array, index_of[dim], axis=axis))
    target = None
    if name in CELL_VALUED:
        target = "nCells"
    elif name in EDGE_VALUED:
        target = "nEdges"
    elif name in VERTEX_VALUED:
        target = "nVertices"
    if target is not None and name not in NOT_REMAPPED:
        g2l = {
            "nCells": layout.cell_g2l,
            "nEdges": layout.edge_g2l,
            "nVertices": layout.vertex_g2l,
        }[target]()
        gathered, _ = _remap_values(gathered, g2l, name=path)
    report.append(
        {
            "path": path,
            "entity": dim,
            "remapped_to": target,
            "global_shape": list(array.shape),
            "local_shape": list(gathered.shape),
            "dtype": str(gathered.dtype),
        }
    )
    return gathered
