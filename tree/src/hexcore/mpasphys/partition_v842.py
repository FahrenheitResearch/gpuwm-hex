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


def partition_maps(layout, partitions):
    """Every partition's validated maps, for many gather_owned_field calls (each _maps sorts every id list)."""
    return [_maps(layout, part) for part in partitions]


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
    fresh = False
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
            # Native buffers are Fortran-ordered: gather on the transposed (C-ordered) view, so the result is
            # Fortran-ordered already and each entity's words are copied as one contiguous run.  THE COST THIS
            # REMOVES: np.take on the Fortran array returned a C-ordered gather that the final copy then
            # transposed word by word, 13.5 s for four LTS classes of the 98k-cell storm-following payload on one
            # thread (0.8 s now), mostly under the GIL (MPAS-ROUTE-SPEED-2026-10-24).  The words are the same.
            values = np.take(values.T, indices, axis=values.ndim - 1 - axis).T
            fresh = True
            break
    referent = _REFERENCES.get(row['name']) if row['pool'] == 'mesh' else None
    if referent is not None:
        if row['kind'] != 'integer':
            raise ValueError(f"connectivity {row['name']} requires native integers")
        inverse = maps[referent][2]
        if np.any(values > layout.dimensions[referent] + 1):
            raise ValueError(f"connectivity {row['name']} exceeds native {referent} dummy")
        values = values if fresh else values.copy(order='F')
        positive = values > 0
        values[positive] = inverse[values[positive]]
        fresh = True
    if fresh and values.dtype == dtype and values.flags.f_contiguous:
        return values                   # already a private Fortran-ordered copy of the words
    return np.array(values, dtype=dtype, order='F', copy=True)


def project_payload(layout: RegistryLayout, payload: Path, partition,
                    output: Path) -> RegistryLayout:
    """Write an independently validated rank payload to a fresh directory.

    The caller owns process creation and GPU selection. This function does
    not create a native session or claim that an executor supports ranks.
    """
    return project_payloads(layout, payload, [partition], [output])[0]


def project_payloads(layout: RegistryLayout, payload: Path, partitions, outputs, *, workers=None):
    """``project_payload`` for several partitions at once, every field on its own worker.

    Each source field is decoded once for every partition, and the fields are projected, written and hashed on
    all cores.  THE COST THIS REMOVES: the MPAS LTS class executor projected the whole payload once per class,
    field after field (numpy take, file write, sha256), about 30 s before the first step of a storm-following
    segment on box-aws2x and the largest part of a resumed segment's 90 s preparation
    (MPAS-ROUTE-SPEED-2026-10-24).  The bytes and manifests written are project_payload's.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    source = NativePoolHydration(None, layout, payload)
    maps = [_maps(layout, partition) for partition in partitions]
    locals_ = [_partition_layout(layout, item) for item in maps]
    # A fresh directory prevents a failed/reused rank from mixing manifests.
    for output in outputs:
        output.mkdir(parents=True, exist_ok=False)

    def field(index):
        row = layout.fields[index]
        original = source.records[row['pool'], row['name']]
        buffer = None if row['kind'] == 'character' else source.field_buffer(row['pool'], row['name'])
        records = []
        for mapping, local, output in zip(maps, locals_, outputs):
            projected = local.fields[index]
            record = {name: projected[name] for name in ('pool', 'name', 'kind', 'dimensions', 'shape')}
            if 'origins' in original:
                # Which input stream filled the words: SuiteIau admits an IAU run only
                # when the iau stream read tend_iau, and a rank payload without origins
                # refused every multi-rank IAU run before its first step.
                record['origins'] = original['origins']
            if row['kind'] == 'character':
                record['text'] = original['text']
            else:
                words = _project_buffer(buffer, row, layout, mapping).tobytes(order='F')
                name = f'field-{index:06d}.bin'
                (output / name).write_bytes(words)
                record.update(payload=name, sha256=hashlib.sha256(words).hexdigest())
            records.append(record)
        return records

    count = len(layout.fields)
    workers = max(1, min(count, workers or os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='hex-project-payload') as pool:
        per_field = list(pool.map(field, range(count)))
    for which, (local, output) in enumerate(zip(locals_, outputs)):
        (output / 'manifest.json').write_text(json.dumps({
            'schema': 'mpas-native-pool-payload-v1', 'fields': [records[which] for records in per_field]},
            indent=2) + '\n')
        NativePoolHydration(None, local, output)
    return locals_


def gather_owned_field(row, layout, partitions, buffers, *, maps=None):
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
    # ``maps``: the partitions' _maps, built once by a caller that assembles many fields (partition_maps).
    maps = [_maps(layout, part) for part in partitions] if maps is None else maps
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
