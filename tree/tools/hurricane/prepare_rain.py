"""Collocate exact 24-hour model windows and Stage IV for tc-verify.

Model endpoints are the retained, Rust product-remapped CF surface files from
render_wind.py --keep-cf. Both endpoints are independently remapped before
subtraction. This does not establish the error of the preceding native-mesh
product sampling; its receipts must accompany the resulting scores.

The installed gpuwm Rust rain gate owns polygon overlap. Parallel destination
stripes keep this offline reduction off the forecast's critical path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from netCDF4 import Dataset, num2date
from pyproj import Transformer
from gpuwm.verify import rain_gate as gate


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc(nc):
    v = nc['time']
    value = num2date(np.asarray(v[:]).item(), v.units,
                    only_use_cftime_datetimes=False)
    return datetime(value.year, value.month, value.day, value.hour,
                    value.minute, value.second, tzinfo=timezone.utc)


def stripe(job):
    field, valid, x, y, edges_x, edges_y, projection = job
    source = gate.QuadGrid(x, y, projection)
    target = gate.RectGrid(edges_x, edges_y, projection)
    return gate.conservative_remap(field, valid, source, target)


def remap(field, valid, x, y, target, workers):
    jobs = []
    for rows in np.array_split(np.arange(target.shape[0]), workers):
        if len(rows):
            jobs.append((field, valid, x, y, target.x_edges_m,
                         target.y_edges_m[rows[0]:rows[-1]+2], target.projection_id))
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pieces = list(pool.map(stripe, jobs))
    values = np.concatenate([p[0] for p in pieces])
    good = np.concatenate([p[1] for p in pieces])
    return values, good, dict(stripes=[p[2] for p in pieces],
                             wall_seconds=time.monotonic()-began)


def model_endpoint(path, target, center, workers):
    path = Path(path)
    with Dataset(path) as nc:
        valid_time = utc(nc)
        if any(nc[name].units != 'mm' for name in ('rainc', 'rainnc')):
            raise ValueError('both model rain counters must declare mm')
        total = np.ma.asarray(nc['rainc'][0], dtype=float) + nc['rainnc'][0]
        field = total.filled(np.nan)
        valid = ~np.ma.getmaskarray(total) & np.isfinite(field) & (field >= 0)
        lat, lon = np.asarray(nc['lat'][:]), np.asarray(nc['lon'][:])
    x, y = gate.project_laea(lat, lon, *center)
    x, y = gate.corners_from_centers(x, y)
    value, good, receipt = remap(field, valid, x, y, target, workers)
    receipt.update(path=str(path), sha256=digest(path), time=valid_time.isoformat(),
        source_geometry='Rust corner recovery from product-lattice centers in common LAEA',
        upstream_geometry='Rust native-mesh product sampling; see endpoint render receipt')
    return valid_time, value, good, receipt


def center_edges(values):
    values = np.asarray(values, dtype=float)
    delta = np.diff(values)
    if len(values) < 2 or not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError('observation coordinates must be monotonic')
    return np.r_[values[0]-delta[0]/2, (values[:-1]+values[1:])/2,
                 values[-1]+delta[-1]/2]


def observation(path, target, center, workers):
    with Dataset(path) as nc:
        # These daily files encode the nominal date at 00Z in `time`, but
        # `data_time` gives the 12Z accumulation endpoint. NOAA documents
        # daily Stage IV as 12Z-to-12Z. Preserve both and refuse other offsets.
        nominal = utc(nc)
        end = datetime.strptime(nc.data_time, '%Y%m%dT%H:%M:%S').replace(tzinfo=timezone.utc)
        if end.hour != 12 or end.minute or end.second:
            raise ValueError('Stage IV daily endpoint must be exactly 12Z')
        if (end-nominal).total_seconds() not in (0, 43200):
            raise ValueError('unrecognized nominal-date versus accumulation-end metadata')
        v = nc['observation']
        if v.units.lower() != 'inches':
            raise ValueError(f'unexpected Stage IV units: {v.units}')
        data = np.ma.asarray(v[:], dtype=float)
        field = data.filled(np.nan) * 25.4
        valid = ~np.ma.getmaskarray(data) & np.isfinite(field) & (field >= 0)
        x, y = np.meshgrid(center_edges(nc['x'][:]), center_edges(nc['y'][:]))
        projection = nc['crs'].proj4
    transform = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True)
    lon, lat = transform.transform(x, y)
    x, y = gate.project_laea(lat, lon, *center)
    value, good, receipt = remap(field, valid, x, y, target, workers)
    receipt.update(path=str(path), sha256=digest(path), time=end.isoformat(),
        nominal_time=nominal.isoformat(), endpoint_authority='data_time attribute; NOAA daily 12Z-to-12Z convention',
        source_projection=projection, source_units='inches', factor_to_mm=25.4,
        source_geometry='native projected cell edges, inverse source CRS, common LAEA')
    return end, value, good, receipt


def nullable(values, valid):
    return np.where(valid, values, None).ravel().tolist()


def write_cf(path, target, center, end, observed, forecast, valid):
    """Keep the exact collocated fields available to gpuwm.rustwx."""
    x, y = np.meshgrid((target.x_edges_m[:-1]+target.x_edges_m[1:])/2,
                       (target.y_edges_m[:-1]+target.y_edges_m[1:])/2)
    projection = (f'+proj=laea +lon_0={center[0]} +lat_0={center[1]} '
                  f'+R={gate.EARTH_RADIUS_M} +units=m +no_defs')
    lon, lat = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True).transform(x,y)
    with Dataset(path, 'w', format='NETCDF3_64BIT_OFFSET') as nc:
        nc.createDimension('time',1)
        nc.createDimension('y',target.shape[0]); nc.createDimension('x',target.shape[1])
        nc.source = 'Collocated Stage IV and hex exact 24-hour window; common valid mask'
        nc.projection = target.projection_id
        t = nc.createVariable('time','f8',('time',))
        t.units = 'seconds since 1970-01-01 00:00:00 UTC'
        t.standard_name = 'time'; t[:] = [end.timestamp()]
        for name, values, units, standard in [('lat',lat,'degrees_north','latitude'),
                                             ('lon',lon,'degrees_east','longitude')]:
            v = nc.createVariable(name,'f8',('y','x'))
            v.units = units; v.standard_name = standard; v[:] = values
        for name, values in [('observed_rain',observed),('forecast_rain',forecast)]:
            v = nc.createVariable(name,'f4',('time','y','x'),fill_value=-9999.)
            v.units = 'mm'
            v[:] = np.ma.array(values,mask=~valid)[None]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--end', type=Path, required=True)
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--observation-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--center-lon', type=float, required=True)
    p.add_argument('--center-lat', type=float, required=True)
    p.add_argument('--half-width-km', type=float, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.workers <= 24 or not 3 <= a.half_width_km <= 1500:
        p.error('workers must be 1..24 and half-width 3..1500 km')
    began = time.monotonic()
    center = a.center_lon, a.center_lat
    identity = f'LAEA sphere R={gate.EARTH_RADIUS_M:g} lon={center[0]:g} lat={center[1]:g}'
    n = int(np.ceil(a.half_width_km / 3))
    edges = np.arange(-n, n+1, dtype=float) * 3000
    target = gate.RectGrid(edges, edges, identity)
    t0, first, first_good, r0 = model_endpoint(a.start, target, center, a.workers)
    t1, last, last_good, r1 = model_endpoint(a.end, target, center, a.workers)
    tobs, obs, obs_good, robs = observation(a.observation, target, center, a.workers)
    if t1 != tobs or (t1-t0).total_seconds() != 86400:
        raise ValueError('model endpoints must exactly bracket the Stage IV 24-hour window')
    amount = last-first
    supported = first_good & last_good
    decrease = supported & (amount < 0)
    # A carried cumulative counter can redistribute across moves. Do not hide
    # decreases by clipping them or call the resulting comparison complete.
    good = supported & (amount >= 0) & np.isfinite(amount)
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='incomplete' if np.any(decrease) else 'collocated',
        reason='negative endpoint differences excluded; transfer redistribution unresolved' if np.any(decrease) else None,
        endpoint_start=r0, endpoint_end=r1, observation=robs,
        model_negative_cells=int(np.count_nonzero(decrease)),
        minimum_difference_mm=float(amount[supported].min()) if np.any(supported) else None,
        common_valid_cells=int(np.count_nonzero(good & obs_good)),
        target_corner_geometry='3 km equal-area squares',
        upstream_sampling_error_bound=None, wall_seconds=time.monotonic()-began)
    result = dict(nx=target.shape[1], ny=target.shape[0], dx_km=3,
        cell_area_km2=9, grid_id=f'{identity}; edges={edges[0]:g}:{edges[-1]:g}:3000',
        start_utc=t0.isoformat(), end_utc=t1.isoformat(),
        observation_source=f'{a.observation_url}; sha256={robs["sha256"]}; exact native corners; Rust conservative LAEA overlap',
        observed=nullable(obs, obs_good), forecasts={a.model:nullable(amount, good)},
        thresholds_mm=[50,100,150], neighborhoods_km=[25,50,100])
    for name, value in [('receipt.json',receipt), ('rain.json',result)]:
        (a.out/name).write_text(json.dumps(value, allow_nan=False, separators=(',',':'))+'\n')
    if not receipt['common_valid_cells']:
        raise ValueError('no common observation/model support')
    write_cf(a.out/'rain.cf.nc',target,center,t1,obs,amount,good & obs_good)
    return 1 if np.any(decrease) else 0


if __name__ == '__main__':
    raise SystemExit(main())
