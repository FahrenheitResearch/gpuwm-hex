"""Native fork lineage admission retains grid identity and dimension guards."""
import pytest
import hexcore.mesh as module


@pytest.mark.parametrize('grid_id,selected,accepted', [
    ('ancestor', True, True), ('immediate', True, True),
    ('ancestor', False, False), ('other', True, False),
    ('anc', True, False),
])
def test_native_parent_chain_requires_an_exact_ancestor(tmp_path, monkeypatch,
                                                       grid_id, selected, accepted):
    grid = tmp_path / 'grid.nc'
    static = tmp_path / 'init.nc'
    grid.touch()
    static.touch()
    def read(path):
        attrs = ({'file_id': grid_id} if path == grid else
                 {'parent_id': 'immediate\nancestor'})
        return {}, attrs, {}, {}, {}
    monkeypatch.setattr(module, '_read_netcdf', read)
    if accepted:
        mesh = module.Mesh.from_netcdf(grid, static, validate=False,
                                      native_parent_chain=selected)
        assert mesh.provenance['static_attrs']['parent_id'] == 'immediate\nancestor'
    else:
        with pytest.raises(module.MeshValidationError, match='parent_id'):
            module.Mesh.from_netcdf(grid, static, validate=False,
                                   native_parent_chain=selected)


def test_native_parent_chain_keeps_dimension_guard(tmp_path, monkeypatch):
    grid = tmp_path / 'grid.nc'
    static = tmp_path / 'init.nc'
    grid.touch()
    static.touch()
    def read(path):
        if path == grid:
            return {'nCells': 2}, {'file_id': 'ancestor'}, {}, {}, {}
        return {'nCells': 3}, {'parent_id': 'immediate\nancestor'}, {}, {}, {}
    monkeypatch.setattr(module, '_read_netcdf', read)
    with pytest.raises(module.MeshValidationError, match='dimensions disagree'):
        module.Mesh.from_netcdf(grid, static, validate=False, native_parent_chain=True)
