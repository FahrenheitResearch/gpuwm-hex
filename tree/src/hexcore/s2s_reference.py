"""Observed S2S reference preparation: native I/O and CUDA calculations.

This driver fits only independent observed years. It never fits a forecast.
Compressed source chunks stay in bounded memory; disposable array transports
are removed immediately after transfer to the device.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from .s2s_door import S2SError, cores, write_json
from .s2s_climatology import Era5Store
from .s2s_mjo import backend, receipt_attempt

ARCO = ('https://storage.googleapis.com/gcp-public-data-arco-era5/ar/'
        '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr')
OLR = 'https://psl.noaa.gov/thredds/fileServer/Datasets/interp_OLR/olr.day.mean.nc'
TIME = '%Y-%m-%d_%H:%M:%S'


def transport(cp, path, names, output):
    """Native CF unpacking, unchanged host buffer, device transfer, disposal."""
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    request = output / 'extract-plan.json'
    write_json(request, {'input': str(path), 'output': str(output), 'variables': names})
    subprocess.run([str(backend()), 'extract', str(request)], check=True)
    metadata = json.loads((output / 'arrays.json').read_text())['arrays']
    result = {}
    for name, row in metadata.items():
        raw = Path(row['path'])
        result[name] = cp.asarray(np.fromfile(raw, dtype='<f8').reshape(row['shape']))
        cp.cuda.Stream.null.synchronize()
        raw.unlink()
    return result, metadata


def bin_weights(cp, lat, lon, domain, nx, ny):
    """Area weights for regular source centres and declared fixed feature bins."""
    yy, xx = cp.meshgrid(lat, lon, indexing='ij')
    xx %= 360
    valid = ((yy >= domain[0]) & (yy <= domain[1]) &
             (xx >= domain[2]) & (xx < domain[3]))
    iy = cp.minimum(((yy-domain[0])/(domain[1]-domain[0])*ny).astype(cp.int32), ny-1)
    ix = cp.minimum(((xx-domain[2])/(domain[3]-domain[2])*nx).astype(cp.int32), nx-1)
    labels = iy * nx + ix
    weights = cp.where(valid[..., None] & (labels[..., None] == cp.arange(nx*ny)),
                       cp.cos(cp.deg2rad(yy))[..., None], 0.).reshape(-1, nx*ny)
    sums = weights.sum(axis=0)
    if bool((sums <= 0).any()):
        raise S2SError('observed source does not support every requested spatial bin')
    return weights / sums


def read_olr(cp, root, dates, definitions, receipts):
    target = root / 'noaa-interpolated-olr.nc'
    with urllib.request.urlopen(urllib.request.Request(OLR,method='HEAD'),timeout=60) as head:
        expected_bytes = int(head.headers['Content-Length'])
    if not target.is_file() or target.stat().st_size != expected_bytes:
        started = time.monotonic()
        ranges = [(first,min(first+8*1024*1024,expected_bytes))
                  for first in range(0,expected_bytes,8*1024*1024)]
        def fetch_range(bounds):
            first,last = bounds
            for attempt in range(4):
                try:
                    request = urllib.request.Request(OLR,headers={'Range':f'bytes={first}-{last-1}'})
                    with urllib.request.urlopen(request,timeout=120) as source:
                        if source.status != 206 or source.headers.get('Content-Range') != f'bytes {first}-{last-1}/{expected_bytes}':
                            raise S2SError('NOAA file server did not honor the exact byte range')
                        payload=source.read()
                    if len(payload) != last-first:
                        raise OSError('NOAA file server truncated an OLR byte range')
                    return payload
                except OSError:
                    if attempt == 3: raise
        with target.with_suffix('.part').open('wb') as out, ThreadPoolExecutor(max_workers=min(cores(),8)) as pool:
            for payload in pool.map(fetch_range,ranges): out.write(payload)
        if target.with_suffix('.part').stat().st_size != expected_bytes:
            raise S2SError('NOAA OLR content length differs from its source header')
        target.with_suffix('.part').replace(target)
        receipts['olr_download_seconds'] = time.monotonic() - started
    receipts['olr_source'] = {'url': OLR, 'bytes': target.stat().st_size,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'license': 'public-domain',
        'terms': 'https://psl.noaa.gov/disclaimer/',
        'attribution': 'Data provided by NOAA/OAR/PSL, Boulder, Colorado, USA; interpolated OLR'}
    coordinates, metadata = transport(cp, target, [{'name': n} for n in ('lat', 'lon', 'time')], root/'transport')
    units = metadata['time']['units']
    if not units.startswith('hours since 1800-01-01'):
        raise S2SError('NOAA OLR time units changed: ' + units)
    epoch = datetime(1800, 1, 1)
    times = coordinates['time']
    weights = [bin_weights(cp, coordinates['lat'], coordinates['lon'],
                           d['domain'], d['nx'], d['ny']) for d in definitions]
    selected = {}
    # One calendar year per native read bounds both the file transport and GPU memory.
    for year in sorted({date.year for date in dates}):
        year_dates = [date for date in dates if date.year == year]
        low, high = min(year_dates), max(year_dates)
        first_matches = cp.where(times == (low-epoch).total_seconds()/3600)[0]
        if first_matches.size != 1:
            raise S2SError('NOAA OLR lacks exact daily observation ' + str(low))
        first, count = int(first_matches[0]), (high-low).days+1
        expected = (low-epoch).total_seconds()/3600 + cp.arange(count)*24
        if not bool(cp.array_equal(times[first:first+count], expected)):
            raise S2SError('NOAA OLR daily clock is incomplete')
        arrays, meta = transport(cp, target, [{'name': 'olr', 'start': first, 'count': count}], root/'transport')
        if meta['olr']['units'] not in ('W/m^2', 'W m-2', 'W/m2'):
            raise S2SError('NOAA OLR units changed: ' + meta['olr']['units'])
        values = arrays['olr'].reshape(count, -1)
        if not bool(cp.isfinite(values).all()):
            raise S2SError('NOAA OLR contains unavailable observations')
        reduced = [values @ weight for weight in weights]
        for date in year_dates:
            selected[date] = [item[(date-low).days] for item in reduced]
        print('NOAA OLR reduced', year, count, flush=True)
    return selected


def observed_features(cp, root, dates, definitions, store, receipts):
    olr = read_olr(cp, root, dates, definitions, receipts)
    lat, lon, levels = (store.coordinate(cp, name) for name in ('latitude', 'longitude', 'level'))
    weights = [bin_weights(cp, lat, lon, d['domain'], d['nx'], d['ny']) for d in definitions]
    source = 'u_component_of_wind'
    info = store.metadata[source+'/.zarray']
    if (store.metadata[source+'/.zattrs']['_ARRAY_DIMENSIONS'] != ['time','level','longitude','latitude']
            or info['chunks'][1:] != info['shape'][1:]):
        raise S2SError('ARCO wind layout changed')
    level_indices = [int(cp.where(levels == level)[0][0]) for level in (850, 200)]
    epoch, chunk = store.time_origin(cp), info['chunks'][0]
    indices = sorted({int((date-epoch).total_seconds()//21600)+k for date in dates for k in range(5)})
    chunks = sorted({index//chunk for index in indices})
    required = set(indices)
    winds = {}
    with ThreadPoolExecutor(max_workers=min(cores(), 8)) as pool:
        for at in range(0, len(chunks), 4):
            batch = chunks[at:at+4]
            for index, (raw, receipt) in zip(batch, pool.map(lambda i: store.read(source, [i,0,0,0]), batch)):
                device = cp.asarray(raw)
                reduced = [(device[:, level_indices, :, :].transpose(0,1,3,2).reshape(chunk,2,-1) @ weight)
                           for weight in weights]
                for k in range(chunk):
                    if index*chunk+k in required:
                        winds[index*chunk+k] = [item[k].copy() for item in reduced]
                store.receipts.append(receipt)
            if at % 100 == 0:
                print('ARCO observed winds', min(at+4,len(chunks)), '/', len(chunks), flush=True)
    result = []
    for dindex, d in enumerate(definitions):
        rows = []
        for date in dates:
            first = int((date-epoch).total_seconds()//21600)
            wind = (winds[first][dindex]*.5 + winds[first+4][dindex]*.5 +
                    sum(winds[first+k][dindex] for k in (1,2,3)))/4
            fields = [olr[date][dindex], wind[0]] + ([wind[1]] if d['id']=='rmm' else [])
            rows.append(cp.concatenate(fields))
        result.append(cp.stack(rows).astype(cp.float64))
    return result


def seasonal_fit(cp, dates, features, training):
    # Fixed annual mean plus three harmonics, fitted only to independent years.
    phase = cp.asarray([(date-datetime(date.year,1,1)).days/365.2425*2*math.pi for date in dates])
    design = cp.stack([cp.ones(len(dates))]+[fn(k*phase) for k in range(1,4) for fn in (cp.cos,cp.sin)],axis=1)
    train = cp.asarray(training)
    coefficients = cp.linalg.lstsq(design[train], features[train], rcond=None)[0]
    return design @ coefficients


def prepare_mjo(args):
    import cupy as cp
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with receipt_attempt(root/'reference-preparation.json', 'gpuwm-hex.s2s-reference-preparation/v1'):
        start = datetime.strptime(args.start, TIME)
        if start.hour or start.minute or start.second:
            raise S2SError('NOAA daily OLR supports only 00 UTC interval starts')
        train_start, train_end = datetime(args.first_year,1,1), datetime(args.last_year+1,1,1)
        if args.last_year-args.first_year < 4 or not train_end <= start-timedelta(days=120):
            raise S2SError('EOFs need at least five complete independent pre-event years')
        early = train_start-timedelta(days=120)
        dates = sorted({early+timedelta(days=k) for k in range((train_end-early).days)} |
                       {start+timedelta(days=k) for k in range(-120,args.days)})
        definitions = [{'id':'rmm','fields':['olr','u850','u200'],'domain':[-15,15,0,360], 'nx':72,'ny':1},
                       {'id':'bsiso1','fields':['olr','u850'],'domain':[-10,40,40,160], 'nx':12,'ny':5}]
        store, receipts = Era5Store(args.era5_store), {}
        features = observed_features(cp, root, dates, definitions, store, receipts)
        training = [train_start <= date < train_end for date in dates]
        positions = {date:i for i,date in enumerate(dates)}
        reference = {'schema':'gpuwm-hex.s2s-mjo-reference/v1','source':'NOAA PSL interpolated OLR + ARCO ERA5 u850/u200',
                     'license':'CC-BY-4.0','training_period':f'{args.first_year}-01-01/{args.last_year}-12-31',
                     'daily_interval_start_hour':0,'method':'fixed-observed-eofs',
                     'raw_fields':{'olr':'olr','pressure':'pressure','u':'u_zonal'},'definitions':[]}
        observed_rmm = {}
        for d, raw in zip(definitions, features):
            seasonal = seasonal_fit(cp, dates, raw, training)
            anomalies = raw-seasonal
            valid = [(i,date) for i,date in enumerate(dates)
                     if date-timedelta(days=120) in positions
                     and i-positions[date-timedelta(days=120)] == 120]
            valid_dates = [date for _,date in valid]
            chosen = cp.asarray([i for i,_ in valid])
            prefix = cp.concatenate((cp.zeros((1,raw.shape[1])),cp.cumsum(anomalies,axis=0)),axis=0)
            filtered = anomalies[chosen] - (prefix[chosen]-prefix[chosen-120])/120
            train = cp.asarray([train_start <= date < train_end for date in valid_dates])
            spatial = d['nx']*d['ny']
            # A fixed observed-orography mask excludes pressure surfaces inside mountains.
            terrain, receipt = store.read('geopotential_at_surface', [0,0])
            store.receipts.append(receipt)
            latitude, longitude = (store.coordinate(cp,n) for n in ('latitude','longitude'))
            weight = bin_weights(cp,latitude,longitude,d['domain'],d['nx'],d['ny'])
            support = ((cp.asarray(terrain).T.reshape(-1) / 9.80665 < 1100).astype(cp.float64) @ weight)
            mask = cp.concatenate([cp.ones(spatial,dtype=bool),support>=.95] +
                                  ([cp.ones(spatial,dtype=bool)] if d['id']=='rmm' else []))
            filtered[:,~mask] = 0
            anomalies[:,~mask] = 0
            seasonal[:,~mask] = 0
            scale = cp.stack([cp.sqrt(cp.mean(filtered[train, k*spatial:(k+1)*spatial]**2)) for k in range(len(d['fields']))])
            normalized = filtered / cp.repeat(scale,spatial)
            covariance = normalized[train].T @ normalized[train] / int(train.sum())
            eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
            eofs = eigenvectors[:,-4:][:,::-1].T.copy()
            # Deterministic signs only; no claim of operational phase equivalence.
            for k in range(4):
                peak = int(cp.argmax(cp.abs(eofs[k])))
                if float(eofs[k,peak]) < 0: eofs[k] *= -1
            pcs = normalized @ eofs.T
            pc_scale = cp.sqrt(cp.mean(pcs[train]**2,axis=0))
            modes = [(d['id'],0,2)] if d['id']=='rmm' else [('bsiso1',0,2),('bsiso2',2,4)]
            for name, low, high in modes:
                item = {**d,'id':name,'standard_deviation':cp.asnumpy(scale).tolist(),
                        'eofs':cp.asnumpy(eofs[low:high]).tolist(), 'pc_standard_deviation':cp.asnumpy(pc_scale[low:high]).tolist(),
                        'eof_modes':list(range(low+1,high+1)), 'phase_order':[5,6,7,8,1,2,3,4],
                        'minimum_bin_area_fraction':.9,'feature_mask':cp.asnumpy(mask).tolist(),
                        'seasonal':{date.strftime('%Y-%m-%d'):cp.asnumpy(seasonal[positions[date]]).tolist()
                                    for date in dates if start <= date < start+timedelta(days=args.days)},
                        'prior_anomalies':{date.strftime('%Y-%m-%d'):cp.asnumpy(anomalies[positions[date]]).tolist()
                                           for date in dates if start-timedelta(days=120) <= date < start}}
                reference['definitions'].append(item)
            if d['id']=='rmm':
                observed_rmm = {date.strftime('%Y-%m-%d'):cp.asnumpy(pcs[i,:2]/pc_scale[:2]).tolist()
                                for i,date in enumerate(valid_dates) if start <= date < start+timedelta(days=args.days)}
            receipts[d['id']] = {'training_days':int(train.sum()),'retained_feature_count':int(mask.sum()),
                'explained_variance_fraction':cp.asnumpy(eigenvalues[-4:][::-1]/eigenvalues.sum()).tolist()}
        identity = hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest()
        reference['basis_id'] = 'observed-'+identity
        write_json(root/'mjo-reference.json', reference)
        write_json(root/'observed-rmm.json', {'basis_id':reference['basis_id'],'rmm':observed_rmm})
        cp.cuda.Stream.null.synchronize()
        write_json(root/'reference-preparation.json', {'status':'complete','schema':'gpuwm-hex.s2s-reference-preparation/v1',
            'seconds':time.monotonic()-started,'source_store':store.url,'source_metadata_sha256':store.metadata_hash,
            'first_stored_era5_timestamp':store.time_origin(cp).strftime(TIME),
            'source_chunks':store.receipts,'details':receipts,'basis_id':reference['basis_id'],
            'preprocessing':'three seasonal harmonics; causal prior-120-day anomaly mean; field RMS scaling; fixed terrain support mask',
            'qualification':'independently fitted RMM-style / BSISO modes, not operational index phase numbers',
            'parallelism':{'fetch_decode_threads':min(cores(),8),'linear_algebra':'CUDA cuSOLVER / cuBLAS'},
            'raw_disposition':'chunks in bounded memory; native array transports removed after transfer'})


def prepare_weekly(args):
    import cupy as cp
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    from .s2s_climatology import FIELDS
    root = args.output.absolute()
    root.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    start = datetime.strptime(args.start,TIME)
    weeks = [(f'week-{week}',(week-1)*7,week*7) for week in range(1,args.days//7+1)]
    with receipt_attempt(root/'weekly-reference-preparation.json','gpuwm-hex.s2s-weekly-reference-preparation/v1'):
        store = Era5Store(args.era5_store)
        names = ['XLAT','XLONG','model_orography']
        grid, _ = transport(cp,args.product_dir/'week-1.nc',[{'name':n} for n in names],root/'transport')
        climatology_names = ['orography']+[name.replace('-','_')+'_'+field
                            for name,_,_ in weeks for field in list(FIELDS)[:4]]
        climate, _ = transport(cp,args.climatology,[{'name':n} for n in climatology_names],root/'transport')
        lat,lon,levels = (store.coordinate(cp,name) for name in ('latitude','longitude','level'))
        iy = cp.abs(grid['XLAT'][...,None]-lat).argmin(axis=-1)
        ix = cp.abs((grid['XLONG'][...,None]-lon+180)%360-180).argmin(axis=-1)
        terrain,receipt=store.read('geopotential_at_surface',[0,0])
        store.receipts.append(receipt)
        observation_terrain=cp.asarray(terrain)[ix,iy]/9.80665
        observed = {name:{} for name,_,_ in weeks}
        with ThreadPoolExecutor(max_workers=min(cores(),8)) as pool:
            for field,(source,units,scale,level) in list(FIELDS.items())[:4]:
                if store.metadata[source+'/.zattrs']['units'] != units:
                    raise S2SError('ERA5 observation units changed: '+source)
                _, windows = next(store.source_windows(cp,source,[start.year],start,weeks,pool))
                index=None if level is None else int(cp.where(levels==level)[0][0])
                for name,_,_ in weeks:
                    plane=windows[name] if index is None else windows[name][index]
                    data=plane[ix,iy]*scale
                    if field=='temperature2':
                        data=data-.0065*(grid['model_orography']-observation_terrain)
                    observed[name][field]=data
        rows=[]
        for week,(name,low,high) in enumerate(weeks,1):
            begin=(start+timedelta(days=low)).strftime(TIME)
            end=(start+timedelta(days=high)).strftime(TIME)
            paths={}
            for kind in ('observations','climatology'):
                target=root/(name+'-'+kind+'.nc')
                schema=ClassicSchema('cdf5')
                y=schema.def_dim('south_north',grid['XLAT'].shape[0])
                x=schema.def_dim('west_east',grid['XLAT'].shape[1])
                for key,value in {'s2s_reference_source':'ERA5','s2s_reference_kind':kind,
                    's2s_window_start':begin,'s2s_window_end':end,'s2s_data_license':'CC-BY-4.0',
                    'attribution':'Contains modified Copernicus Climate Change Service information; neither the European Commission nor ECMWF is responsible for use',
                    'temperature_terrain_adjustment':'source T2 - 0.0065 K/m * (model terrain - source terrain)'}.items():
                    schema.put_global_attr(key,value)
                arrays={key:(grid[key],units) for key,units in
                        [('XLAT','degrees_north'),('XLONG','degrees_east'),('model_orography','m')]}
                for field,units in [('temperature2','K'),('precipitation_rate','mm day-1'),('z500','m'),('mslp','hPa')]:
                    data=observed[name][field] if kind=='observations' else climate[name.replace('-','_')+'_'+field]
                    if kind=='climatology' and field=='temperature2':
                        data=data-.0065*(grid['model_orography']-climate['orography'])
                    if not bool(cp.isfinite(data).all()):
                        raise S2SError('nonfinite weekly '+kind+' '+field)
                    arrays[field]=(data,units)
                payloads=[]
                for key,(data,units) in arrays.items():
                    number=schema.def_var(key,'f4',(y,x))
                    schema.put_var_attr(number,'units',units)
                    payloads.append((number,cp.asnumpy(data.astype(cp.float32))))
                with schema.create(target.with_suffix('.part')) as writer:
                    for number,data in payloads: writer.write_var(number,data)
                target.with_suffix('.part').replace(target)
                paths[kind]=str(target)
            rows.append({'week':week,'start':begin,'end':end,**paths})
        rmm=json.loads((root/'observed-rmm.json').read_text())
        manifest={'schema':'gpuwm-hex.s2s-observations/v1','source':'ERA5','license':'CC-BY-4.0',
            'provenance':json.dumps({'store':store.url,'metadata_sha256':store.metadata_hash,
                'weekly_method':'six-hour endpoint trapezoids; precipitation sum of following six-hour accumulations divided by days',
                'climatology':str(args.climatology),'climatology_sha256':hashlib.sha256(args.climatology.read_bytes()).hexdigest(),
                'rmm_training_period':f'{args.first_year}-01-01/{args.last_year}-12-31',
                'weekly_baseline':'declared in the independently supplied window climatology',
                'rmm':'NOAA daily OLR + ARCO six-hour winds, same independent fixed observed basis'},sort_keys=True),
            'minimum_area_fraction':.95,'weeks':rows,'rmm':rmm['rmm'],'rmm_basis_id':rmm['basis_id'],
            'rmm_sampling':'daily-interval-start','rmm_interval_start':args.start}
        write_json(root/'observations.json',manifest)
        write_json(root/'weekly-reference-preparation.json',{'status':'complete',
            'schema':'gpuwm-hex.s2s-weekly-reference-preparation/v1','seconds':time.monotonic()-started,
            'source_chunks':store.receipts,'parallelism':{'fetch_decode_threads':min(cores(),8),'reductions_remap':'CUDA'},
            'weeks':len(rows),'observations':str(root/'observations.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('mjo','weekly'),default='mjo')
    parser.add_argument('--start',required=True)
    parser.add_argument('--days',type=int,required=True)
    parser.add_argument('--first-year',type=int,required=True)
    parser.add_argument('--last-year',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--era5-store',default=ARCO)
    parser.add_argument('--product-dir',type=Path)
    parser.add_argument('--climatology',type=Path)
    args=parser.parse_args()
    if args.mode=='mjo': prepare_mjo(args)
    else: prepare_weekly(args)


if __name__ == '__main__':
    main()
