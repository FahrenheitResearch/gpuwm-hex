"""Box-only scalar scoring on a fixed MRMS lattice. This creates no maps.

The nearest spherical-cell comparison follows the historical moving-patch
verification. Every history supplies its own coordinates; cell indices do
not identify locations across members or segments.
"""
from pathlib import Path
import datetime as dt, hashlib, json
import numpy as np
from scipy.ndimage import label, uniform_filter
from scipy.spatial import cKDTree

def xyz(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)

def distance(a, b):
    return float(np.arccos(np.clip(xyz(*a) @ xyz(*b), -1, 1))*6371.229)

def scores(model, observed, latitude, longitude, *, neighborhoods=(20, 50), threshold=35):
    valid = np.isfinite(model) & np.isfinite(observed)
    weights = np.cos(np.deg2rad(latitude))*valid
    m, o = (model >= threshold) & valid, (observed >= threshold) & valid
    hit, miss, false = (float(weights[a].sum()) for a in (m&o, ~m&o, m&~o))
    fss = {}
    km_y = 111.2*abs(float(latitude[1,0]-latitude[0,0]))
    km_x = 111.2*np.cos(np.deg2rad(latitude.mean()))*abs(float(longitude[0,1]-longitude[0,0]))
    for km in neighborhoods:
        size = tuple(max(1, round(km/spacing)) for spacing in (km_y, km_x))
        coverage = uniform_filter(valid.astype(float), size=size, mode='constant')
        fractions = [np.divide(uniform_filter(event.astype(float), size=size, mode='constant'),
                     coverage, out=np.zeros_like(model, dtype=float), where=coverage>0) for event in (m,o)]
        fm, fo = fractions
        den = float((weights*(fm*fm+fo*fo)).sum())
        fss[str(km)] = None if den == 0 else 1-float((weights*(fm-fo)**2).sum())/den
    def objects(field):
        ids, count = label((field >= threshold)&valid)
        result = []
        for k in range(1, count+1):
            mask = ids == k
            if mask.sum() < 5: continue
            vote = weights[mask]*(field[mask]-threshold+1)
            result.append(dict(pixels=int(mask.sum()), center=[float(np.average(c[mask], weights=vote))
                           for c in (latitude, longitude)], max_dbz=float(field[mask].max())))
        return result
    oc, mc = objects(observed), objects(model)
    target = max(oc, key=lambda c:c['pixels']) if oc else None
    matched = min(mc, key=lambda c:distance(c['center'], target['center'])) if mc and target else None
    if matched and distance(matched['center'], target['center']) > 300: matched = None
    model_max = float(model[valid].max()) if valid.any() else None
    obs_max = float(observed[valid].max()) if valid.any() else None
    return dict(valid_pixels=int(valid.sum()), weighted_hit=hit, weighted_miss=miss,
        weighted_false_alarm=false, csi_35=None if hit+miss+false == 0 else hit/(hit+miss+false),
        fss=fss, model_max_dbz=model_max, observed_max_dbz=obs_max,
        max_reflectivity_bias_dbz=None if model_max is None else model_max-obs_max,
        position_metric='largest observed 35 dBZ object, nearest model object within 300 km; minimum five pixels',
        observed_object=target, model_object=matched,
        position_error_km=distance(target['center'],matched['center']) if matched else None)

def match_history(history, observation_root, frame):
    from netCDF4 import Dataset
    meta = json.loads((observation_root/'nowcast.json').read_text())
    lattice = meta['lattice']
    stride = 3
    olat = lattice['lat0'] + np.arange(0,lattice['ny'],stride)*lattice['dlat']
    olon = lattice['lon0'] + np.arange(0,lattice['nx'],stride)*lattice['dlon']
    longitude, latitude = np.meshgrid(olon, olat)
    path = observation_root/frame['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
        raise ValueError('MRMS frame changed after fetching')
    observed = np.fromfile(path, dtype='<f4').reshape(frame['shape'])[0,::stride,::stride]
    with Dataset(history) as f:
        lat, lon = np.asarray(f['latCell'][:]), np.asarray(f['lonCell'][:])
        if abs(lat).max() > np.pi/2+1e-6: raise ValueError('unexpected model coordinate units')
        refl = np.asarray(f['refl10cm'][0]).max(axis=-1)
        if not np.isfinite(refl).all(): raise ValueError('nonfinite model reflectivity')
        tree = cKDTree(xyz(np.rad2deg(lat),np.rad2deg(lon)))
        chord, indices = tree.query(xyz(latitude,longitude),workers=2)
        model = refl[indices]
    mapping = dict(method='nearest spherical model cell on fixed MRMS lattice', stride=stride,
        maximum_distance_km=float((2*np.arcsin(np.minimum(chord/2,1))*6371.229).max()),
        observation_sha256=frame['sha256'], observation_offset_seconds=frame['offset_seconds'])
    return model, observed, latitude, longitude, mapping

def ensemble_scores(models, observed, latitude, longitude):
    weights = np.cos(np.deg2rad(latitude))*np.isfinite(observed)
    event = observed >= 35
    probability = (models >= 35).mean(axis=0)
    total = weights.sum()
    spread = models.std(axis=0,ddof=1) if len(models)>1 else np.zeros_like(observed)
    reliability = []
    for k in range(len(models)+1):
        mask = probability == k/len(models)
        weight = weights[mask].sum()
        reliability.append(dict(probability=k/len(models), pixels=int((mask&(weights>0)).sum()),
            observed_frequency=None if weight==0 else float((weights[mask]*event[mask]).sum()/weight)))
    return dict(members=len(models), brier_score_35=float((weights*(probability-event)**2).sum()/total),
        weighted_mean_spread_dbz=float((weights*spread).sum()/total),
        rms_spread_dbz=float(np.sqrt((weights*spread**2).sum()/total)), reliability=reliability,
        mean=scores(models.mean(axis=0), observed, latitude, longitude),
        maximum=scores(models.max(axis=0), observed, latitude, longitude))
