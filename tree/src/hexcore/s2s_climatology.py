"""Public ERA5 normals: threaded Zarr chunk acquisition and GPU window means.

Provider and calendar convention:
https://weatherbench2.readthedocs.io/en/latest/data-guide.html
https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py
Only the four required fields and their intersecting day-of-year chunks are read.
Raw compressed/decoded chunks stay in memory and are released after reduction.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from .s2s_door import cores, write_json

STORE = ('https://storage.googleapis.com/weatherbench2/datasets/era5-hourly-climatology/'
         '1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr')
OROGRAPHY_STORE = ('https://storage.googleapis.com/weatherbench2/datasets/era5/'
    '1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
FIELDS = {'temperature2': ('2m_temperature', 'K', 1.),
          'precipitation_rate': ('total_precipitation_6hr', 'm', 1000.),
          'z500': ('geopotential', 'm**2 s**-2', 1 / 9.80665),
          'mslp': ('mean_sea_level_pressure', 'Pa', .01)}


def calendar_samples(start, low, high):
    """Use the archive's UTC hour and ordinary Gregorian day-of-year convention."""
    return [(stamp.hour, stamp.timetuple().tm_yday) for stamp in
            (start + timedelta(hours=6 * n) for n in range(low * 4, high * 4 + 1))]


def reduce_window(xp, samples, *, precipitation, days):
    values = xp.stack(samples)
    if precipitation:
        # Six-hour precipitation is an accumulation ending at each timestamp.
        return values[1:].sum(axis=0) / days
    return (values[0] * .5 + values[-1] * .5 + values[1:-1].sum(axis=0)) / (days * 4)


def _fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise


def fetch_normals(value, args, latitude, longitude):
    import cupy as cp
    import numpy as np
    from numcodecs import Blosc, blosc
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    from .s2s_products import TIME_FORMAT, windows, FIELDS as UNITS
    started = time.monotonic()
    root = Path(value['paths']['work'])
    target = root / 'era5-window-climatology.nc'
    receipt_path = root / 'era5-climatology-receipt.json'
    identity = {'store': STORE, 'start': value['start_time'], 'days': args.days,
                'shape': list(latitude.shape), 'schema': 's2s-era5-normal/v2',
                'orography_store': OROGRAPHY_STORE}
    if target.is_file() and receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if prior.get('identity') == identity and prior.get('sha256') == hashlib.sha256(target.read_bytes()).hexdigest():
            return target, prior['baseline']
    metadata_bytes = _fetch(STORE + '/.zmetadata')
    metadata = json.loads(metadata_bytes)['metadata']
    source_receipts = []
    blosc.set_nthreads(1)  # Independent chunks run across the host cores.

    def read_chunk(name, indices, *, store=STORE, metadata=metadata):
        info = metadata[name + '/.zarray']
        if (info['zarr_format'] != 2 or info['order'] != 'C' or info.get('filters')
                or info['compressor']['id'] != 'blosc'):
            raise ValueError('ERA5 climatology requires C-order, unfiltered Blosc Zarr v2 chunks')
        key = name + '/' + '.'.join(map(str, indices))
        payload = _fetch(store + '/' + key)
        decoded = Blosc(**{key: item for key, item in info['compressor'].items() if key != 'id'}).decode(payload)
        array = np.frombuffer(decoded, dtype=info['dtype']).reshape(info['chunks'])
        return array, {'url': store + '/' + key, 'bytes': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest(), 'decoded_bytes': len(decoded)}

    workers = min(cores(), 16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        names = ('hour', 'dayofyear', 'level', 'latitude', 'longitude')
        coordinate_results = list(pool.map(lambda name: read_chunk(name, [0]), names))
        coordinates = {name: result[0] for name, result in zip(names, coordinate_results)}
        source_receipts.extend(result[1] for result in coordinate_results)
        if (not np.array_equal(coordinates['hour'], [0, 6, 12, 18])
                or not np.array_equal(coordinates['dayofyear'], np.arange(1, 367))):
            raise ValueError('ERA5 climatology UTC hour/day coordinates changed')
        levels = np.flatnonzero(coordinates['level'] == 500)
        if len(levels) != 1:
            raise ValueError('ERA5 climatology needs exactly one 500 hPa level')
        level = int(levels[0])
        start = datetime.strptime(value['start_time'], TIME_FORMAT)
        product_windows = windows(args.days)
        samples = {name: calendar_samples(start, low, high) for name, low, high in product_windows}
        day_indices = sorted({(day - 1) // 30 for rows in samples.values() for _, day in rows})
        requests = []
        for field, (source, units, _) in FIELDS.items():
            attrs = metadata[source + '/.zattrs']
            expected_dims = ['hour', 'dayofyear'] + (['level'] if field == 'z500' else []) + ['longitude', 'latitude']
            info = metadata[source + '/.zarray']
            if (attrs['units'] != units or attrs['_ARRAY_DIMENSIONS'] != expected_dims
                    or info['chunks'] != ([4, 30, 3, 240, 121] if field == 'z500' else [4, 30, 240, 121])):
                raise ValueError('ERA5 climatology field units, dimensions or chunk layout changed: ' + source)
            for day_index in day_indices:
                indices = [0, day_index] + ([level // 3] if field == 'z500' else []) + [0, 0]
                requests.append((field, day_index, source, indices))
        chunks = {}
        for request, result in zip(requests, pool.map(lambda row: read_chunk(row[2], row[3]), requests)):
            field, day_index, _, _ = request
            array, evidence = result
            chunks[field, day_index] = array[:, :, level % 3] if field == 'z500' else array
            source_receipts.append(evidence)

    # The climatology has no surface geopotential. Fetch the matching ERA5
    # constant on exactly its conservative 240x121 grid, never another terrain.
    from .s2s_terrain import GRAVITY_M_S2
    terrain_metadata_bytes = _fetch(OROGRAPHY_STORE + '/.zmetadata')
    terrain_metadata = json.loads(terrain_metadata_bytes)['metadata']
    attrs = terrain_metadata['geopotential_at_surface/.zattrs']
    info = terrain_metadata['geopotential_at_surface/.zarray']
    if (attrs['units'] != 'm**2 s**-2' or attrs['_ARRAY_DIMENSIONS'] != ['longitude', 'latitude']
            or info['shape'] != [240, 121] or info['chunks'] != [240, 121]):
        raise ValueError('ERA5 source orography units, dimensions or grid changed')
    with ThreadPoolExecutor(max_workers=3) as pool:
        terrain_names = ('latitude', 'longitude', 'geopotential_at_surface')
        terrain_results = list(pool.map(lambda name: read_chunk(name,
            [0, 0] if name == 'geopotential_at_surface' else [0],
            store=OROGRAPHY_STORE, metadata=terrain_metadata), terrain_names))
    for name, result in zip(terrain_names[:2], terrain_results[:2]):
        if not np.array_equal(result[0], coordinates[name]):
            raise ValueError('ERA5 source orography coordinates differ from the climatology')
    source_receipts.extend(result[1] for result in terrain_results)
    terrain = cp.asarray(terrain_results[2][0]) / GRAVITY_M_S2
    if not bool(cp.isfinite(terrain).all()):
        raise ValueError('nonfinite ERA5 source orography')
    source_lat = cp.asarray(coordinates['latitude'])
    source_lon = cp.asarray(coordinates['longitude'])
    latitude_index = cp.abs(latitude[..., None] - source_lat).argmin(axis=-1)
    delta = cp.abs((longitude[..., None] - source_lon + 180) % 360 - 180)
    longitude_index = delta.argmin(axis=-1)
    baseline = 'ERA5 1990-2019 WeatherBench2 smoothed six-hour climatology; 1.5 degree source; GPU window means and nearest-grid remap'
    schema = ClassicSchema()
    y = schema.def_dim('south_north', latitude.shape[0])
    x = schema.def_dim('west_east', latitude.shape[1])
    payloads = []

    def variable(name, units, array):
        number = schema.def_var(name, 'f4', (y, x))
        schema.put_var_attr(number, 'units', units)
        payloads.append((number, cp.asnumpy(array).astype('f4')))

    schema.put_global_attr('s2s_climatology_source', 'ERA5')
    schema.put_global_attr('s2s_climatology_start_mmdd', start.strftime('%m-%d'))
    schema.put_global_attr('s2s_climatology_baseline', baseline)
    variable('XLAT', 'degrees_north', latitude)
    variable('XLONG', 'degrees_east', longitude)
    variable('orography', 'm', terrain[longitude_index, latitude_index])
    for field, (_, _, scale) in FIELDS.items():
        device_chunks = {index: cp.asarray(chunks[field, index]) for index in day_indices}
        for name, low, high in product_windows:
            selected = [device_chunks[(day - 1) // 30][hour // 6, (day - 1) % 30] * scale
                        for hour, day in samples[name]]
            reduced = reduce_window(cp, selected, precipitation=field == 'precipitation_rate', days=high-low)
            remapped = reduced[longitude_index, latitude_index]
            if not bool(cp.isfinite(remapped).all()):
                raise ValueError('nonfinite ERA5 climatology: ' + field + ' ' + name)
            variable(name.replace('-', '_') + '_' + field, UNITS[field], remapped)
        del device_chunks
    temporary = target.with_suffix('.part')
    with schema.create(temporary) as writer:
        for number, payload in payloads:
            writer.write_var(number, payload)
    temporary.replace(target)
    write_json(receipt_path, {'identity': identity, 'baseline': baseline,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
        'wall_seconds': time.monotonic() - started, 'source_chunks': source_receipts,
        'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'orography_metadata_sha256': hashlib.sha256(terrain_metadata_bytes).hexdigest(),
        'orography_conversion': 'geopotential_at_surface / 9.80665; metres',
        'calendar': 'Gregorian dt.dayofyear and UTC hour, matching WeatherBench2 ACC selection',
        'parallelism': {'fetch_decode_threads': workers, 'blosc_threads_per_chunk': 1, 'reductions_and_remap': 'GPU'},
        'raw_disposition': 'compressed and decoded chunks held only in memory, released on function return'})
    return target, baseline
