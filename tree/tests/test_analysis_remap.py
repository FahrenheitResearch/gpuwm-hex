"""The transfer's file seam must preserve native staggering and all water species."""
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore import analysis_remap as module


def fields():
    rho = np.full((3, 2), 1.1, np.float32)
    scalars = np.arange(1, 7, dtype=np.float32)[:, None, None] * np.full((6, 3, 2), .001, np.float32)
    theta = np.full_like(rho, 299)
    moist = theta * (np.float32(1) + np.float32(461.6) / np.float32(287) * scalars[0])
    state = SimpleNamespace(rho=rho, rho_theta=rho * moist,
                            rho_u=np.arange(9, dtype=np.float32).reshape(3, 3),
                            rho_w=np.arange(8, dtype=np.float32).reshape(4, 2), scalars=scalars)
    saved = SimpleNamespace(theta_m=moist, normal_velocity=state.rho_u / np.float32(1.1),
                            vertical_velocity=state.rho_w / np.float32(1.1))
    vertical = SimpleNamespace(zz=np.full_like(rho, .9))
    return state, saved, vertical


def test_native_carriers_and_all_six_water_species_are_written(tmp_path):
    state, saved, vertical = fields()
    values = module.native_fields(state, saved, vertical, module.SPECIES)
    np.testing.assert_allclose(values['theta'], 299, rtol=1e-7)
    np.testing.assert_array_equal(values['ru'], state.rho_u)
    np.testing.assert_array_equal(values['rw'], state.rho_w)
    path = tmp_path / 'native.nc'
    with Dataset(path, 'w') as d:
        for name, size in [('Time', 1), ('nCells', 2), ('nEdges', 3),
                           ('nVertLevels', 3), ('nVertLevelsP1', 4)]:
            d.createDimension(name, size)
        # Deliberately mix layouts, including a non-leading time dimension.
        d.createVariable('theta', 'f4', ('nVertLevels', 'Time', 'nCells'))
        d.createVariable('qv', 'f4', ('nCells', 'nVertLevels'))
        module.write_fields(d, values)
    with Dataset(path) as d:
        np.testing.assert_array_equal(d['theta'][:, 0, :], values['theta'])
        np.testing.assert_array_equal(d['qv'][:].T, state.scalars[0])
        for k, name in enumerate(module.SPECIES[1:], 1):
            np.testing.assert_array_equal(d[name][0].T, state.scalars[k])
        np.testing.assert_array_equal(d['rw'][0].T, state.rho_w)


def test_bad_stagger_and_nonfinite_fields_are_refused(tmp_path):
    state, saved, vertical = fields()
    state.rho_w[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite transferred field: rw'):
        module.native_fields(state, saved, vertical, module.SPECIES)
    with Dataset(tmp_path / 'bad.nc', 'w') as d:
        d.createDimension('nCells', 2)
        d.createVariable('theta', 'f4', ('nCells',))
        with pytest.raises(ValueError, match='unsupported dimensions'):
            module.write_fields(d, {'theta': np.ones((3, 2), np.float32)})


def test_species_order_is_explicit():
    state, saved, vertical = fields()
    names = tuple(reversed(module.SPECIES))
    state.scalars = state.scalars[::-1]
    values = module.native_fields(state, saved, vertical, names)
    np.testing.assert_allclose(values['theta'], 299, rtol=1e-7)
    with pytest.raises(ValueError, match='unique species'):
        module.native_fields(state, saved, vertical, ('qv',) * 6)


def test_all_clocks_follow_record_time_not_old_start_attribute(tmp_path):
    with Dataset(tmp_path / 'source.nc', 'w') as source, Dataset(tmp_path / 'target.nc', 'w') as target:
        for d in (source, target):
            d.createDimension('Time', 1)
            d.createDimension('StrLen', 64)
            for name in ('xtime', 'initial_time'):
                v = d.createVariable(name, 'S1', ('Time', 'StrLen'))
                v[0] = np.array(list('2022-06-13_21:00:00'.ljust(64)), dtype='S1')
            d.setncattr('config_start_time', '2022-06-13_21:00:00')
        source['xtime'][0] = np.array(list('2022-06-14_00:00:00'.ljust(64)), dtype='S1')
        target.createVariable('Time', 'f8', ('Time',))[:] = 10800
        assert module.stamp_analysis_time(target, source) == '2022-06-14_00:00:00'
        for name in ('xtime', 'initial_time'):
            assert target[name][0].tobytes().decode().strip() == '2022-06-14_00:00:00'
        assert target.config_start_time == '2022-06-14_00:00:00'
        assert target['Time'][0] == 0


def test_identity_requires_equal_terrain_and_native_edge_geometry():
    import copy
    vertical = SimpleNamespace(**{n: np.ones(2) for n in ('zgrid', 'zz', 'dzw', 'zxu', 'fzm', 'fzp')})
    arrays = {n: np.ones(2) for n in ('areaCell', 'cellsOnEdge', 'edgesOnCell', 'coeffs_reconstruct', 'edgeNormalVectors')}
    source = SimpleNamespace(vertical=vertical, mesh=SimpleNamespace(arrays=arrays),
                             terrain_metrics=SimpleNamespace(zb_cell=np.ones(2), zb3_cell=np.ones(2)))
    target = copy.deepcopy(source)
    cells = SimpleNamespace(fingerprint=lambda: 'same spherical mesh')
    assert module.same_geometry(source, target, cells, cells)
    target.terrain_metrics.zb_cell[0] = 2
    assert not module.same_geometry(source, target, cells, cells)
    target = copy.deepcopy(source)
    target.mesh.arrays['edgeNormalVectors'][0] = 2
    assert not module.same_geometry(source, target, cells, cells)


def test_vertical_artifact_clock_has_nonrecord_initial_time(tmp_path):
    with Dataset(tmp_path / 'clock.nc', 'w') as d:
        d.createDimension('Time', None); d.createDimension('StrLen', 64)
        d.createVariable('xtime', 'S1', ('Time', 'StrLen'))[0] = np.array(list('2022-06-14_00:00:00'.ljust(64)), dtype='S1')
        d.createVariable('initial_time', 'S1', ('StrLen',))
        d.createVariable('Time', 'f4', ('Time',))
        module.stamp_analysis_time(d, d)
        assert d['initial_time'][:].tobytes().decode().strip() == '2022-06-14_00:00:00'
        assert d['Time'].units == 'seconds since 2022-06-14 00:00:00'


def test_real_data_reference_is_positive_and_has_250k_physical_temperature():
    z = np.broadcast_to(np.linspace(0, 30000, 56, dtype=np.float32)[:, None], (56, 3)).copy()
    vertical = SimpleNamespace(zgrid=z, zz=np.full((55, 3), .92, np.float32))
    reference, theta = module.real_data_reference(vertical)
    reference.validate((55, 3))
    np.testing.assert_allclose(theta * reference.exner_base, 250., rtol=4e-7)
    np.testing.assert_allclose(reference.pressure_base / (reference.rho_base * vertical.zz * np.float32(287)), 250., rtol=4e-7)


def test_surface_pressure_matches_native_linear_density_extrapolation():
    rho = np.array([[1.2], [1.]], np.float32)
    state = SimpleNamespace(rho=rho, rho_theta=rho * np.float32(300),
                            scalars=np.full((6, 2, 1), .01, np.float32))
    vertical = SimpleNamespace(zz=np.ones((2, 1), np.float32),
                               zgrid=np.array([[500], [600], [900]], np.float32),
                               dzw=np.array([100, 300], np.float32))
    pressure, vapor = module.atmospheric_surface_diagnostics(state, vertical)
    # Independent double-precision calculation of MPAS_to_physics.F formula.
    p0 = (287 * 1.2 * 300 / 100000) ** (1004.5 / (1004.5 - 287)) * 100000
    r1, r2 = 1.2 * 1.01, 1.0 * 1.01
    expected = p0 + .5 * 9.80616 * 100 * (r1 - .5 * (r2-r1)*100/400)
    np.testing.assert_allclose(pressure, expected, rtol=3e-7)
    np.testing.assert_allclose(vapor, 4.2, rtol=1e-7)


def test_surface_transfer_keeps_soil_snow_column_and_sea_ice_identity(tmp_path, monkeypatch):
    from scipy.sparse import csr_matrix
    from hexcore.moving_remap import ConservativeMap
    from hexcore.vertical_spec import MET_STATE_VARIABLES, ZERO_STATE_VARIABLES, REFERENCE_PROFILE_VARIABLES
    weights = ConservativeMap(csr_matrix([[.75, .25], [.25, .75]]), np.ones(2), np.ones(2), {})
    monkeypatch.setattr(module, 'horizontal_map', lambda *args: weights)
    vertical = SimpleNamespace(zgrid=np.broadcast_to(np.array([0, 100, 400], np.float32)[:, None], (3, 2)),
                               dzw=np.array([100, 300], np.float32), zz=np.ones((2, 2), np.float32))
    state = SimpleNamespace(rho=np.ones((2, 2), np.float32),
                            rho_theta=np.full((2, 2), 300, np.float32),
                            scalars=np.full((6, 2, 2), .01, np.float32))
    mesh = SimpleNamespace(arrays=dict(latCell=np.zeros(2), lonCell=np.zeros(2)),
                           attrs=dict(sphere_radius=6371229.))
    source = SimpleNamespace(vertical=vertical, mesh=mesh, state=state)
    target = SimpleNamespace(vertical=vertical, mesh=mesh)
    geometry = SimpleNamespace(centers=np.array([[1, 0, 0], [1, 0, 0]]))
    with Dataset(tmp_path/'parent.nc', 'w') as parent, Dataset(tmp_path/'child.nc', 'w') as child:
        for dataset in (parent, child):
            for name, size in [('Time', 1), ('nCells', 2), ('nEdges', 2), ('nVertLevels', 2),
                               ('nVertLevelsP1', 3), ('nSoilLevels', 4)]:
                dataset.createDimension(name, size)
            for name, dims, _, _ in MET_STATE_VARIABLES + ZERO_STATE_VARIABLES + REFERENCE_PROFILE_VARIABLES:
                dataset.createVariable(name, 'f4', dims)[:] = 1.
            for name in ('landmask', 'ivgtyp', 'isltyp'):
                dataset.createVariable(name, 'f4', ('nCells',))[:] = [1, 0] if name=='landmask' else [1, 2]
            dataset.createVariable('isice_lu', 'i4').assignValue(24)
        # The larger overlap is the wrong land type for both target columns.
        child['landmask'][:] = [0, 1]
        parent['seaice'][0] = [0, 1]
        parent['tslb'][0] = [[271, 272, 273, 274], [281, 282, 283, 284]]
        parent['snow'][0] = [10, 20]
        result = module.carry_surface(parent, child, source, target, geometry, geometry, tmp_path/'operator.npz', state)
        np.testing.assert_array_equal(child['tslb'][0], parent['tslb'][0][::-1])
        np.testing.assert_array_equal(child['snow'][0], [20, 10])
        np.testing.assert_array_equal(child['xland'][0], [1, 1])
        assert result['nearest_compatible_columns'] == 0
        assert np.all(child['surface_pressure'][0] > 80000)
