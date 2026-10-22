"""Independent CPU oracles for wind, empirical windows, and regional products."""
from datetime import datetime
import numpy as np
import pytest
from hexcore.s2s_door import S2SError
from hexcore.s2s_products import (FIELDS, WIND_LEVELS, ensemble_statistics,
    integrate_window, interp_log_pressure, region_indices)
from hexcore.s2s_climatology import historical_start, reduce_window, window_distribution


@pytest.mark.parametrize('first_hour', [0, 24])
def test_archive_origin_comes_from_stored_cf_words(first_hour):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timedelta
    from hexcore.s2s_climatology import Era5Store
    store = object.__new__(Era5Store)
    store.receipts = []
    store.metadata = {
        'time/.zattrs': {'_ARRAY_DIMENSIONS':['time'], 'calendar':'proleptic_gregorian',
                         'units':'hours since 1959-01-01'},
        'time/.zarray': {'shape':[40], 'chunks':[16]},
        'temperature/.zattrs': {'_ARRAY_DIMENSIONS':['time','longitude','latitude']},
        'temperature/.zarray': {'shape':[40,1,1], 'chunks':[8,1,1]}}
    def read(name, indices):
        if name == 'time':
            return first_hour + (indices[0]*16 + np.arange(16))*6, {'time_chunk':indices[0]}
        return (indices[0]*8 + np.arange(8)).reshape(8,1,1), {'field_chunk':indices[0]}
    store.read = read
    origin = datetime(1959,1,1)+timedelta(hours=first_hour)
    assert store.time_origin(np) == origin
    assert len(store.receipts) == 3
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(store.source_windows(np,'temperature',[1959],origin,[('week-1',0,7)],pool))
    np.testing.assert_equal(rows[0][1]['week-1'], [[14.]])
    # A gap after the first chunk must still be rejected, rather than accepted
    # from a plausible first offset alone.
    del store._time_origin
    def broken(name, indices):
        values, receipt = read(name, indices)
        if name == 'time' and indices[0] == 1:
            values = values.copy(); values[3] += 6
        return values, receipt
    store.read = broken
    with pytest.raises(S2SError, match='complete six-hour clock'):
        store.time_origin(np)


def test_wind_window_uses_components_and_exact_endpoints():
    times = range(29)
    samples = [{field: np.array([3. + index * .25]) for field in FIELDS} for index in times]
    for sample in samples:
        sample['precipitation_rate'] = np.array([12.])
    result = integrate_window(np, samples, 6, 7)
    for level in WIND_LEVELS:
        for component in ('u', 'v'):
            np.testing.assert_allclose(result[f'{component}{level}'], 6.5)
    with pytest.raises(S2SError, match='exact endpoints'):
        integrate_window(np, samples, 5, 7)


def test_pressure_winds_include_exact_levels_and_mask_below_ground():
    pressure = np.array([[92500., 85000., 20000.], [90000., 60000., 20000.]])
    wind = np.log(pressure) * 7. - 30.
    for level in WIND_LEVELS:
        result = interp_log_pressure(np, pressure, wind, level * 100)
        np.testing.assert_allclose(result[0], np.log(level * 100) * 7 - 30)
    assert np.isnan(interp_log_pressure(np, pressure, wind, 92500)[1])


def test_terciles_partition_ties_and_preserve_complete_roster():
    members = np.array([[0., 1., np.nan, 1.], [1., 1., 2., 2.], [2., 1., 3., 3.]])
    baseline = np.array([1., 1., 2., 2.])
    lower, upper = np.array([.5, 1., 1., 3.]), np.array([1.5, 1., 2., 2.])
    stats = ensemble_statistics(np, members, baseline, (lower, upper))
    np.testing.assert_allclose(stats['tercile_probability'][:, 0], [1/3, 1/3, 1/3])
    np.testing.assert_allclose(stats['tercile_probability'][:, 1], [0, 1, 0])
    assert np.isnan(stats['tercile_probability'][:, 2:]).all()
    np.testing.assert_allclose(stats['tercile_probability'][:, :2].sum(axis=0), 1)


def test_historical_quantiles_are_over_means_not_daily_values():
    # Historical windows with means 1,2,100. Linear quantiles are 5/3,104/3.
    years = [reduce_window(np, [np.array([v])] * 29, precipitation=False, days=7)
             for v in (1., 2., 100.)]
    mean, thresholds = window_distribution(np, years)
    np.testing.assert_allclose(mean, 103/3)
    np.testing.assert_allclose(thresholds[:, 0], [5/3, 104/3])
    years[1][0] = np.nan
    mean, thresholds = window_distribution(np, years)
    assert np.isnan(mean).all() and np.isnan(thresholds).all()


def test_calendar_matches_month_day_across_leap_years():
    assert historical_start(datetime(2020, 3, 1, 6), 2019) == datetime(2019, 3, 1, 6)
    assert historical_start(datetime(2020, 2, 29, 18), 2019) == datetime(2019, 2, 28, 18)


def test_regional_view_orders_date_line_columns_without_changing_fields():
    lat, lon = np.meshgrid(np.arange(-90, 91), np.arange(-180, 180), indexing='ij')
    name, ys, xs, unwrapped = region_indices(np, lat, lon, 'crossing:-10,10,170,-170')
    assert name == 'crossing' and len(ys) == 21 and len(xs) == 21
    np.testing.assert_array_equal(unwrapped, np.arange(170, 191))
    np.testing.assert_array_equal(lon[ys[:, None], xs[None, :]][0], np.r_[np.arange(170, 180), np.arange(-180, -169)])
    for text in ('bad', '../escape:0,10,0,10', 'flat:0,0,0,10', 'flat:0,10,1,1'):
        with pytest.raises(S2SError):
            region_indices(np, lat, lon, text)


def test_rust_decoder_rejects_corrupt_archive_chunk():
    import os
    from pathlib import Path
    from hexcore.s2s_climatology import decode_chunk
    library = os.environ.get('GPUWM_HEX_S2S_ZARR_LIBRARY')
    if not library or not Path(library).is_file():
        pytest.skip('source-built Rust decoder not selected')
    info = {'zarr_format': 2, 'order': 'C', 'filters': None,
            'compressor': {'id': 'blosc'}, 'dtype': '<f8', 'chunks': [13]}
    with pytest.raises(S2SError, match='Rust ERA5 Blosc decode failed'):
        decode_chunk(b'not-a-valid-blosc-frame', info)
