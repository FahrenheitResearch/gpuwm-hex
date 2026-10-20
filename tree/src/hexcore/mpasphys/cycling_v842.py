"""MPAS-native pools at the CUDA JEDI cycling boundary.

The existing JEDI contract owns CDF-5 I/O and coupled-state initialization.
No ArWen physics state or soil interpolation is used on this route.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import jedi_contract as contract


def background_fields(session, layout, names):
    """Copy the native physical fields, with selected suite soil/species axes."""
    rows = {}
    for row in layout.fields:
        if row['name'] in rows:
            raise ValueError(f"ambiguous Registry field {row['name']}")
        rows[row['name']] = row
    fields, provenance = {}, {}
    required = tuple(name for name in contract.BACKGROUND_FIELDS if name != 'scalars')
    for name in (*required, 'scalars'):
        if name not in rows:
            raise ValueError(f'native JEDI background requires active Registry field {name}')
        row = rows[name]
        value = np.asarray(session.native.get_field(row['pool'], row['pool_name']))
        # Remove native garbage storage, along the Registry entity axis.
        for axis, dim in enumerate(row['dimensions']):
            if dim in ('nCells', 'nEdges', 'nVertices'):
                value = np.take(value, np.arange(layout.dimensions[dim]), axis=axis)
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ValueError(f'native JEDI background {name} has invalid RKIND or non-finite words')
        if name == 'scalars':
            if tuple(row['components']) != tuple(names):
                raise ValueError('native JEDI scalar component order differs')
            for index, species in enumerate(names):
                fields[species] = np.ascontiguousarray(value[index])
                provenance[species] = f"native MPAS pool {row['pool']}.{row['pool_name']}[{species}]"
        else:
            fields[name] = np.ascontiguousarray(value)
            provenance[name] = f"native MPAS pool {row['pool']}.{row['pool_name']}"
    soil = layout.dimensions['nSoilLevels']
    label = contract.NATIVE_SOIL_LABEL.format(levels=soil)
    for name in ('smois', 'tslb'):
        provenance[name] += '; ' + label
    return fields, provenance


def write_background(arguments, session, layout, names, *, step, dt, out):
    """Write one requested boundary through the existing Rust CDF-5 writer."""
    fields, provenance = background_fields(session, layout, names)
    native_soil = layout.dimensions['nSoilLevels']
    selected_soil = getattr(arguments, 'jedi_soil_levels', None)
    if selected_soil is not None and int(selected_soil) != native_soil:
        raise ValueError(f'native JEDI uses the selected suite soil column ({native_soil} levels); '
                         f'--jedi-soil-levels {selected_soil} differs')
    valid = (datetime.strptime(layout.configs['config_start_time'], '%Y-%m-%d_%H:%M:%S')
             + timedelta(seconds=step * dt))
    seconds = int(step * dt)
    if seconds != step * dt:
        raise ValueError('JEDI boundary must have an integral-second timestamp')
    label = f'f{seconds // 3600:03d}' if not seconds % 3600 else f't{seconds:08d}'
    return contract.write_background(Path(out) / f'mpasout.{label}.nc',
        fields=fields, provenance=provenance, valid=valid,
        n_cells=layout.dimensions['nCells'], n_edges=layout.dimensions['nEdges'],
        n_levels=layout.dimensions['nVertLevels'], soil_levels=native_soil,
        scalar_names=names, n_vertices=layout.dimensions['nVertices'],
        identity_path=Path(arguments.init),
        global_attributes={'hex_physics_route': 'MPAS native physics',
                           'source': 'gpuwm-hex: MPAS CUDA dynamics and MPAS native physics'})


def recouple_analysis(arguments, layout, inputs, *, work_dir):
    """Replace analysed species while retaining every unanalysed LES carrier."""
    analysis = getattr(arguments, 'da_analysis', None)
    if analysis is None:
        return inputs, None
    if not layout.configs.get('config_do_DAcycling'):
        raise ValueError('native --da-analysis requires config_do_DAcycling=true')
    if not layout.configs.get('config_do_restart'):
        raise ValueError('native --da-analysis requires a native restart or moving carrier')
    mesh, grid, profiles, old_state, reference, old_saved, names = inputs
    analysed = tuple(name for name in names if name in contract.DEFAULT_SCALAR_NAMES)
    if not analysed or analysed[0] != 'qv':
        raise ValueError('native JEDI cycling requires the selected suite water-vapour carrier')
    state, saved, receipt = contract.recouple_da_cycling(Path(analysis),
        init_path=Path(arguments.init), mesh=mesh, vertical=grid.vertical_grid,
        terrain_metrics=grid.terrain_metrics, expected_reference=reference,
        expected_valid_time=layout.configs['config_start_time'], scalar_names=analysed,
        n_cells=layout.dimensions['nCells'], n_edges=layout.dimensions['nEdges'],
        n_levels=layout.dimensions['nVertLevels'],
        allow_regional_sentinels=bool(layout.configs.get('config_apply_lbcs', False)),
        time_seconds=old_state.time_seconds, work_dir=Path(work_dir))
    # JEDI analyses the water species, not TKE/number concentrations held by
    # the native suite. Copy those words from this member's carried state.
    scalars = np.array(old_state.scalars, copy=True, order='C')
    for index, name in enumerate(analysed):
        scalars[names.index(name)] = state.scalars[index]
    state.scalars = scalars
    receipt['native_species_retained'] = [name for name in names if name not in analysed]
    return (mesh, grid, profiles, state, reference, saved, names), receipt
