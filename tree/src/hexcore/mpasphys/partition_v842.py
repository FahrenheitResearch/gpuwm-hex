"""Registry pool transport for process-isolated route-B ranks.

This is metadata and word transport, not a physics implementation. Native
connectivity is one based, unlike the CUDA partition mesh. A missing local
referent must name the local native dummy, never the first solved cell.
"""
from __future__ import annotations

from copy import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..partition_local_mesh_v841 import CELL_VALUED, EDGE_VALUED, VERTEX_VALUED
from .pool_v842 import NativePoolHydration, RegistryLayout

_ENTITIES = {
    'nCells': ('cell_l2g', 'n_owned_cells'),
    'nEdges': ('edge_l2g', 'n_owned_edges'),
    'nVertices': ('vertex_l2g', 'n_owned_vertices'),
}
_REFERENCES = {
    **{name: 'nCells' for name in CELL_VALUED | {'cellsOnCellsOnCell'}},
    **{name: 'nEdges' for name in EDGE_VALUED},
    **{name: 'nVertices' for name in VERTEX_VALUED},
}


def _maps(layout, partition):
    maps = {}
    for entity, (ids_name, owned_name) in _ENTITIES.items():
        ids = np.asarray(getattr(partition, ids_name))
        total = layout.dimensions[entity]
        owned = getattr(partition, owned_name)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
                not 0 <= owned <= len(ids) or np.any(ids < 0) or
                np.any(ids >= total) or len(np.unique(ids)) != len(ids)):
            raise ValueError(f'invalid {entity} partition map')
        inverse = np.full(total + 2, len(ids) + 1, dtype=np.int32)
        inverse[0] = 0
        inverse[ids + 1] = np.arange(1, len(ids) + 1, dtype=np.int32)
        maps[entity] = (ids, owned, inverse)
    return maps


def partition_layout(layout: RegistryLayout, partition) -> RegistryLayout:
    """Select local Registry extents, retaining species and pool identities.

    Halo ends are intentionally required from the caller for a provider that
    needs ring-specific exchanges. Cell rings alone cannot classify edge and
    vertex halo rings under every native decomposition convention.
    """
    return _partition_layout(layout, _maps(layout, partition))


def _partition_layout(layout, maps):
    local = copy(layout)
    local.dimensions = dict(layout.dimensions)
    local.configs = dict(layout.configs)
    local.indices = dict(layout.indices)
    local.fields = []
    if hasattr(local, 'halo_ends'):
        # Global empty-halo metadata is not a distributed decomposition.
        del local.halo_ends
    for entity, (ids, owned, _) in maps.items():
        local.dimensions.update({entity: len(ids), entity + 'All': len(ids),
                                 entity + 'Solve': owned})
    for row in layout.fields:
        field = dict(row)
        field['dimensions'] = list(row['dimensions'])
        field['components'] = list(row.get('components', ()))
        field['shape'] = [local.dimensions[d] + int(d in _ENTITIES)
                          for d in row['dimensions']]
        local.fields.append(field)
    return local


def project_buffer(buffer, row, layout, partition):
    """Slice every decomposed axis and remap only declared connectivity.

    Real words, including NaN payloads and signed zero, are copied unchanged.
    The final native dummy column is carried from the global dummy column.
    Zero/negative padding is preserved; positive absent/dummy references use
    the local dummy. Global identity arrays are sliced without renumbering.
    """
    return _project_buffer(buffer, row, layout, _maps(layout, partition))


def _project_buffer(buffer, row, layout, maps):
    values = np.asarray(buffer)
    expected = tuple(row['shape'])
    if values.shape != expected:
        raise ValueError(f"{row['pool']}.{row['name']} shape {values.shape} != {expected}")
    dtype = np.dtype('<f4' if row['kind'] == 'real' else '<i4')
    if row['kind'] not in ('real', 'integer') or values.dtype != dtype:
        raise ValueError(f"{row['pool']}.{row['name']} is not a native word buffer")
    for axis, dim in enumerate(row['dimensions']):
        for entity, (ids, owned, _) in maps.items():
            if dim == entity:
                indices = np.concatenate((ids, [layout.dimensions[entity]]))
            elif dim == entity + 'All':
                indices = ids
            elif dim == entity + 'Solve':
                indices = ids[:owned]
            else:
                continue
            values = np.take(values, indices, axis=axis)
            break
    referent = _REFERENCES.get(row['name']) if row['pool'] == 'mesh' else None
    if referent is not None:
        if row['kind'] != 'integer':
            raise ValueError(f"connectivity {row['name']} requires native integers")
        inverse = maps[referent][2]
        if np.any(values > layout.dimensions[referent] + 1):
            raise ValueError(f"connectivity {row['name']} exceeds native {referent} dummy")
        values = values.copy()
        positive = values > 0
        values[positive] = inverse[values[positive]]
    return np.array(values, dtype=dtype, order='F', copy=True)


def project_payload(layout: RegistryLayout, payload: Path, partition,
                    output: Path) -> RegistryLayout:
    """Write an independently validated rank payload to a fresh directory.

    The caller owns process creation and GPU selection. This function does
    not create a native session or claim that an executor supports ranks.
    """
    source = NativePoolHydration(None, layout, payload)
    maps = _maps(layout, partition)
    local = _partition_layout(layout, maps)
    # A fresh directory prevents a failed/reused rank from mixing manifests.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (row, field) in enumerate(zip(layout.fields, local.fields)):
        record = {name: field[name] for name in
                  ('pool', 'name', 'kind', 'dimensions', 'shape')}
        original = source.records[row['pool'], row['name']]
        if row['kind'] == 'character':
            record['text'] = original['text']
        else:
            words = _project_buffer(source.field_buffer(row['pool'], row['name']),
                                    row, layout, maps).tobytes(order='F')
            name = f'field-{index:06d}.bin'
            (output / name).write_bytes(words)
            record.update(payload=name, sha256=hashlib.sha256(words).hexdigest())
        records.append(record)
    (output / 'manifest.json').write_text(json.dumps({
        'schema': 'mpas-native-pool-payload-v1', 'fields': records}, indent=2) + '\n')
    NativePoolHydration(None, local, output)
    return local


def gather_owned_field(row, layout, partitions, buffers):
    """Assemble one carried/output field using owner words only.

    The result excludes native dummy storage. Connectivity is deliberately
    excluded: its rank-local values need inverse remapping before output.
    Fields with no decomposed axis must agree bit for bit on every rank.
    This supports arbitrary Registry species counts, including LES TKE.
    """
    if not partitions or len(partitions) != len(buffers):
        raise ValueError('one field buffer is required for each partition')
    if row['kind'] not in ('real', 'integer'):
        raise ValueError('owned field assembly requires a numeric Registry field')
    if row['pool'] == 'mesh':
        raise ValueError('mesh fields require global connectivity reconstruction')
    axes = [(axis, entity) for axis, dim in enumerate(row['dimensions'])
            for entity in _ENTITIES if dim in (entity, entity + 'All', entity + 'Solve')]
    if len(axes) > 1:
        raise ValueError('owned field assembly requires at most one decomposed axis')
    maps = [_maps(layout, part) for part in partitions]
    dtype = np.dtype('<f4' if row['kind'] == 'real' else '<i4')
    for mapping, buffer in zip(maps, buffers):
        dimensions = dict(layout.dimensions)
        for entity, (ids, owned, _) in mapping.items():
            dimensions.update({entity: len(ids), entity + 'All': len(ids),
                               entity + 'Solve': owned})
        expected = tuple(dimensions[d] + int(d in _ENTITIES) for d in row['dimensions'])
        if np.asarray(buffer).shape != expected or np.asarray(buffer).dtype != dtype:
            raise ValueError('rank field shape/dtype differs from local Registry')
    if not axes:
        first = np.asarray(buffers[0])
        if any(first.tobytes(order='F') != np.asarray(b).tobytes(order='F') for b in buffers[1:]):
            raise ValueError('replicated Registry words differ between ranks')
        return first.copy(order='F')
    axis, entity = axes[0]
    count = layout.dimensions[entity]
    seen = np.zeros(count, dtype=np.bool_)
    shape = list(row['shape'])
    shape[axis] = count
    result = np.empty(shape, dtype=dtype, order='F')
    for mapping, buffer in zip(maps, buffers):
        ids, owned, _ = mapping[entity]
        ids = ids[:owned]
        if np.any(seen[ids]):
            raise ValueError('duplicate Registry field owners')
        seen[ids] = True
        target = [slice(None)] * result.ndim
        source = [slice(None)] * result.ndim
        target[axis] = ids
        source[axis] = slice(0, owned)
        result[tuple(target)] = np.asarray(buffer)[tuple(source)]
    if not np.all(seen):
        raise ValueError('Registry field ownership does not cover the global domain')
    return result
