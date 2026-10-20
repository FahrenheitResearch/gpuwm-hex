"""Preserve published vertical metrics and complete native init landing sites."""
from __future__ import annotations

import json
from pathlib import Path
import shutil


def materialize(source, grid, output, declarations, *, static=None):
    from netCDF4 import Dataset
    import numpy as np
    from .init_door import _COMPATIBILITY_VARIABLES
    from .vertical_spec import (_declare_met_state_schema, encode_config_attribute,
                                _complete_derived_geometry)
    from .preset_prepare import describe

    source, grid, output = Path(source), Path(grid), Path(output)
    static = None if static is None else Path(static)
    derived_names = ('edgeNormalVectors', 'localVerticalUnitVectors',
                     'cellTangentPlane', 'coeffs_reconstruct')
    required = (*_COMPATIBILITY_VARIABLES, 'edgeNormalVectors',
                'localVerticalUnitVectors', 'cellTangentPlane', 'coeffs_reconstruct')
    with Dataset(source) as invariant, Dataset(grid) as mesh:
        missing = [name for name in required if name not in invariant.variables]
        if missing:
            raise ValueError(f"public invariant lacks the native vertical contract: {missing}")
        # Matching dimensions alone can attach correct geometry to wrong cells.
        for name in ('latCell', 'lonCell', 'cellsOnEdge', 'edgesOnCell', 'nEdgesOnCell'):
            left, right = invariant.variables[name], mesh.variables[name]
            if left.shape != right.shape:
                raise ValueError(f"public invariant and grid disagree on {name} shape")
            for start in range(0, left.shape[0], 65536):
                a, b = left[start:start+65536], right[start:start+65536]
                # Public invariant coordinates may be stored as float32 while
                # their mesh uses float64. Compare exact rounded coordinates,
                # without accepting an arbitrary spatial tolerance.
                if name in ('latCell', 'lonCell') and a.dtype.kind == b.dtype.kind == 'f':
                    stored_dtype = a.dtype if a.dtype.itemsize <= b.dtype.itemsize else b.dtype
                    a, b = a.astype(stored_dtype), b.astype(stored_dtype)
                if not np.array_equal(a, b):
                    raise ValueError(f"public invariant and grid disagree on {name} cell order")
        if 'nSoilLevels' in invariant.dimensions:
            raise ValueError("public vertical source already carries meteorological soil state; "
                             "select an invariant, not a supplied init")
        placeholders = [name for name in derived_names
                        if not np.any(invariant.variables[name][:])]
    # Defect: public invariant fields exist but are all zero. Native init
    # copies those placeholders, then forecast refuses non-unit edge normals.
    geometry = None
    if placeholders:
        if static is None:
            raise ValueError('public invariant has zero derived geometry; '
                             'the grid/static pair is required for native completion')
        from .rust_geometry import geometry_engine
        engine, digest = geometry_engine()
        geometry = {'engine': str(engine), 'sha256': digest}
    static_receipt = None if static is None else describe(static)
    receipt = output.with_name(output.name + '.receipt.json')
    if output.exists():
        held = json.loads(receipt.read_text()) if receipt.exists() else None
        if (held and held.get('schema') == 'gpuwm-hex.published-vertical/v2'
                and held.get('static') == static_receipt
                and held.get('geometry_engine') == geometry
                and held['declarations'] == declarations and held['source'] == describe(source)
                and held['grid'] == describe(grid) and held['output'] == describe(output)):
            return held
        raise ValueError(f"{output} is not a complete matching vertical artifact")
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise ValueError(f"{partial} is incomplete; select a fresh preparation directory")
    shutil.copyfile(source, partial)
    with Dataset(partial, 'r+') as nc:
        completed = []
        if placeholders:
            from .mesh import Mesh
            completed, _ = _complete_derived_geometry(nc, Mesh.from_netcdf(grid, static))
        declared = _declare_met_state_schema(nc, start_time=declarations['config_start_time'])
        nc.variables['Time'].units = 'seconds since ' + declarations['config_start_time'].replace('_', ' ')
        for name, value in declarations.items():
            nc.setncattr(name, encode_config_attribute(value))
        nc.gpuwm_hex_vertical_source = 'published-invariant-vertical-metrics-unchanged'
    partial.rename(output)
    result = {'schema': 'gpuwm-hex.published-vertical/v2',
              'source': describe(source), 'grid': describe(grid), 'output': describe(output),
              'static': static_receipt, 'geometry_engine': geometry,
              'derived_geometry_completed': completed,
              'declarations': declarations, 'declared_met_state': declared,
              'claim': 'Published vertical metrics are unchanged; zero derived geometry '
                       'and meteorology are completed by the existing native bridges'}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result
