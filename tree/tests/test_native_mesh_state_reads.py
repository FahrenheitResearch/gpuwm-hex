"""Real NetCDF read suppression; no native or weather identity claim."""
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset
import pytest

from hexcore import mesh as module
from hexcore.mpasphys.runtime_v842 import native_mesh_state_exclusions


def layout():
    return SimpleNamespace(fields=[
        dict(pool='mesh', name='xCell', pool_name='xCell', components=[]),
        dict(pool='state', name='u', pool_name='u_native', components=[]),
        dict(pool='state', name='scalars', pool_name='scalars', components=['qv', 'qc']),
        # An alias collision must never discard a mesh field.
        dict(pool='state', name='xCell', pool_name='xCell', components=[]),
    ])


def write(path, *, overlay=False, cells=2):
    with Dataset(path, 'w') as ds:
        ds.createDimension('nCells', cells)
        ds.createDimension('nVertLevels', 8)
        ds.createDimension('maxEdges', 1)
        ds.file_id = 'init-id' if overlay else 'grid-id'
        if overlay:
            ds.parent_id = 'grid-id'
        x = ds.createVariable('xCell', 'f4', ('nCells',))
        x.units = 'm'
        x[:] = -0.0 if overlay else 7.0
        edge = ds.createVariable('cellsOnCell', 'i4', ('nCells', 'maxEdges'))
        edge[:] = 1
        count = ds.createVariable('nEdgesOnCell', 'i4', ('nCells',))
        count[:] = 1
        for name in ('u', 'qv'):
            state = ds.createVariable(name, 'f4', ('nCells', 'nVertLevels'))
            state[:] = 42.0


def test_state_exclusions_include_components_and_protect_mesh_aliases():
    assert native_mesh_state_exclusions(layout()) == {'u', 'u_native', 'scalars', 'qv', 'qc'}


def test_excluded_state_is_never_read_and_mesh_words_and_overlay_survive(tmp_path, monkeypatch):
    grid, init = tmp_path / 'grid.nc', tmp_path / 'init.nc'
    write(grid)
    write(init, overlay=True)
    reads = []
    excluded = native_mesh_state_exclusions(layout())
    class Variable:
        def __init__(self, name, value):
            self.name, self.value = name, value
        def __getattr__(self, name):
            return getattr(self.value, name)
        def __getitem__(self, selection):
            assert self.name not in excluded, 'state payload was read as mesh'
            reads.append(self.name)
            return self.value[selection]
    class Reader:
        def __init__(self, *args, **kwargs):
            self.dataset = Dataset(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.dataset, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.dataset.close()
        @property
        def variables(self):
            return {name: Variable(name, value) for name, value in self.dataset.variables.items()}
    monkeypatch.setattr(module, 'Dataset', Reader)
    validated = []
    monkeypatch.setattr(module.Mesh, 'validate', lambda self: validated.append(self))
    result = module.Mesh.from_netcdf(grid, init, exclude_variables=excluded)
    assert validated == [result]
    assert reads == ['xCell', 'cellsOnCell', 'nEdgesOnCell'] * 2
    assert set(result.arrays) == {'xCell', 'cellsOnCell', 'nEdgesOnCell'}
    assert result.arrays['xCell'].view(np.uint32).tolist() == [0x80000000] * 2
    assert result.arrays['cellsOnCell'].tolist() == [[0], [0]]
    assert result.variable_attrs['xCell']['units'] == 'm'
    assert result.variable_dimensions['xCell'] == ('nCells',)
    assert result.variable_sources['xCell'] == 'static'
    assert result.attrs['file_id'] == 'init-id'
    assert result.dimensions['nVertLevels'] == 8


def test_default_still_reads_state_and_exclusion_keeps_dimension_refusal(tmp_path):
    grid, init = tmp_path / 'grid.nc', tmp_path / 'init.nc'
    write(grid)
    result = module.Mesh.from_netcdf(grid, validate=False)
    assert result.arrays['u'].shape == result.arrays['qv'].shape == (2, 8)
    write(init, overlay=True, cells=3)
    with pytest.raises(module.MeshValidationError, match='dimensions disagree'):
        module.Mesh.from_netcdf(grid, init, validate=False,
                               exclude_variables=native_mesh_state_exclusions(layout()))
