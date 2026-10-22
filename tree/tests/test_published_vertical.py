from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import pytest

from hexcore.published_vertical import materialize
from hexcore.init_door import _COMPATIBILITY_VARIABLES


def source_files(tmp_path, levels=2):
    grid, source = tmp_path / 'grid.nc', tmp_path / 'invariant.nc'
    for path in (grid, source):
        with Dataset(path, 'w') as nc:
            for name, length in (('Time', None), ('nCells', 2), ('nEdges', 3),
                                 ('nVertLevels', levels), ('nVertLevelsP1', levels + 1),
                                 ('maxEdges', 2), ('TWO', 2), ('R3', 3)):
                nc.createDimension(name, length)
            for name in ('latCell', 'lonCell', 'edgesOnCell', 'nEdgesOnCell'):
                nc.createVariable(name, 'f8', ('nCells',))[:] = [1, 2]
            nc.createVariable('cellsOnEdge', 'i4', ('nEdges',))[:] = [1, 2, 1]
            if path == source:
                for i, name in enumerate((*_COMPATIBILITY_VARIABLES, 'edgeNormalVectors',
                                         'localVerticalUnitVectors', 'cellTangentPlane',
                                         'coeffs_reconstruct')):
                    nc.createVariable(name, 'f4', ('nCells',))[:] = [i + .25, i + .5]
    return source, grid


def test_published_low_top_cannot_be_relabelled_as_high_top(tmp_path):
    from hexcore.global_door import DATA
    source, grid = source_files(tmp_path, levels=55)
    output = tmp_path / 'artifact.nc'
    with pytest.raises(ValueError, match='level count'):
        materialize(source, grid, output, {'config_nvertlevels': 80})
    with pytest.raises(ValueError, match='level count'):
        materialize(source, grid, output, {}, vertical_spec=DATA / 'countries/s2s80-v1.json')
    assert not output.exists()


def test_published_binding_checks_the_top_and_receipts_the_spec(tmp_path):
    from hexcore.global_door import DATA
    from hexcore.vertical_spec import VerticalSpec
    spec_path = DATA / 'countries/s2s80-v1.json'
    source, grid = source_files(tmp_path, levels=80)
    output = tmp_path / 'artifact.nc'
    with Dataset(source, 'r+') as nc:
        nc.config_ztop = 30000.
    with pytest.raises(ValueError, match='vertical top'):
        materialize(source, grid, output, {}, vertical_spec=spec_path)
    assert not output.exists()
    with Dataset(source, 'r+') as nc:
        nc.config_ztop = 50000.
    declarations = {'config_start_time': '2018-02-05_00:00:00'}
    receipt = materialize(source, grid, output, declarations, vertical_spec=spec_path)
    assert receipt['vertical_binding']['sha256'] == VerticalSpec.from_file(spec_path).sha256()
    assert materialize(source, grid, output, declarations, vertical_spec=spec_path) == receipt


def test_public_geometry_is_byte_identical_and_source_is_unchanged(tmp_path):
    source, grid = source_files(tmp_path)
    before = source.read_bytes()
    output = tmp_path / 'artifact.nc'
    declaration = {'config_start_time': '2026-10-03_12:00:00', 'config_nsoillevels': 4}
    record = materialize(source, grid, output, declaration)
    assert source.read_bytes() == before
    with Dataset(source) as left, Dataset(output) as right:
        for name, variable in left.variables.items():
            assert variable[:].tobytes() == right.variables[name][:].tobytes()
        assert right.variables['theta'][0].tobytes() == np.zeros((2, 2), 'f4').tobytes()
        assert len(right.dimensions['nSoilLevels']) == 4
        assert right.config_start_time == declaration['config_start_time']
        assert right.variables['Time'].units == 'seconds since 2026-10-03 12:00:00'
    assert materialize(source, grid, output, declaration) == record


def test_cell_order_mismatch_cannot_publish_an_artifact(tmp_path):
    source, grid = source_files(tmp_path)
    with Dataset(grid, 'r+') as nc:
        nc.variables['latCell'][:] = [2, 1]
    output = tmp_path / 'artifact.nc'
    with pytest.raises(ValueError, match='cell order'):
        materialize(source, grid, output, {})
    assert not output.exists()


def test_a_supplied_soil_initialization_is_not_an_invariant(tmp_path):
    source, grid = source_files(tmp_path)
    with Dataset(source, 'r+') as nc:
        nc.createDimension('nSoilLevels', 4)
    with pytest.raises(ValueError, match='supplied init'):
        materialize(source, grid, tmp_path / 'artifact.nc', {})


def test_exact_stored_coordinate_precision_and_cycle_metadata(tmp_path):
    source, grid = source_files(tmp_path)
    coordinates = np.array([.5235458246837469, .5235987857481674])
    with Dataset(source, 'r+') as nc:
        nc.renameVariable('latCell', 'original_latCell')
        nc.createVariable('latCell', 'f4', ('nCells',))[:] = coordinates
        time = nc.createVariable('Time', 'f8', ('Time',))
        time[:] = [0]
        time.units = 'seconds since 2026-03-17 15:00:00'
    with Dataset(grid, 'r+') as nc:
        nc.variables['latCell'][:] = coordinates
    result = tmp_path / 'artifact.nc'
    materialize(source, grid, result, {'config_start_time': '2026-10-03_12:00:00'})
    with Dataset(result) as nc:
        assert nc.variables['Time'].units == 'seconds since 2026-10-03 12:00:00'
        assert np.array_equal(nc.variables['latCell'][:], coordinates.astype('f4'))
    with Dataset(grid, 'r+') as nc:
        nc.variables['latCell'][0] += .001
    with pytest.raises(ValueError, match='cell order'):
        materialize(source, grid, tmp_path / 'wrong.nc', {})


def test_zero_geometry_is_completed_without_changing_vertical_metrics(tmp_path, monkeypatch):
    from hexcore import mesh, rust_geometry, vertical_spec
    source, grid = source_files(tmp_path)
    static = tmp_path / 'static.nc'
    static.write_bytes(b'fixture static authority')
    with Dataset(source, 'r+') as nc:
        nc.variables['edgeNormalVectors'][:] = 0
    output = tmp_path / 'artifact.nc'
    with pytest.raises(ValueError, match='zero derived geometry'):
        materialize(source, grid, output, {})
    assert not output.exists()
    monkeypatch.setattr(mesh.Mesh, 'from_netcdf', lambda *args: object())
    monkeypatch.setattr(rust_geometry, 'geometry_engine', lambda: (Path('/native/geometry'), 'bridge-sha'))
    # Existing completion keeps populated fields. Exercise that policy here.
    monkeypatch.setattr(vertical_spec, '_derived_geometry_arrays', lambda *args:
        ({name: np.array([1, 1], 'f4') for name in
          ('edgeNormalVectors', 'localVerticalUnitVectors', 'cellTangentPlane', 'coeffs_reconstruct')},
         'computed'))
    record = materialize(source, grid, output,
                         {'config_start_time': '2026-10-03_12:00:00'}, static=static)
    assert record['derived_geometry_completed'] == ['edgeNormalVectors']
    with Dataset(source) as before, Dataset(output) as after:
        for name in _COMPATIBILITY_VARIABLES:
            assert before[name][:].tobytes() == after[name][:].tobytes()
        assert after['edgeNormalVectors'][:].tobytes() == np.ones(2, 'f4').tobytes()
        assert before['coeffs_reconstruct'][:].tobytes() == after['coeffs_reconstruct'][:].tobytes()
