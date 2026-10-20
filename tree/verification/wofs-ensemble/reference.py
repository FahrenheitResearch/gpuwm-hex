"""CPU observation coverage and Brier references, never weather maps."""
import os
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.ndimage import label
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))

def summarize(observed,lat,lon):
    valid=np.isfinite(observed)
    weight=np.cos(np.deg2rad(lat))*valid
    event=(observed>=35)&valid
    frequency=float((weight*event).sum()/weight.sum())
    ids,count=label(event)
    sizes=np.bincount(ids.ravel()); sizes[0]=0
    target=int(sizes.argmax()) if count else None
    bounds=None
    if target is not None:
        mask=ids==target
        bounds=dict(south=float(lat[mask].min()),north=float(lat[mask].max()),
            west=float(lon[mask].min()),east=float(lon[mask].max()),
            touches_score_boundary=bool(mask[0].any() or mask[-1].any() or
                                       mask[:,0].any() or mask[:,-1].any()))
    return dict(valid_pixels=int(valid.sum()),missing_pixels=int((~valid).sum()),
        observed_frequency_35=frequency,no_event_brier=frequency,
        same_window_frequency_brier=frequency*(1-frequency),largest_object_bounds=bounds)

def main():
    root=W/'prepared/observations'
    meta=json.loads((root/'nowcast.json').read_text()); lattice=meta['lattice']
    lon,lat=np.meshgrid(lattice['lon0']+np.arange(0,lattice['nx'],3)*lattice['dlon'],
                       lattice['lat0']+np.arange(0,lattice['ny'],3)*lattice['dlat'])
    rows=[]
    for hour,frame in enumerate(meta['frames']):
        path=root/frame['file']
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=frame['sha256']: raise ValueError('observation bytes changed')
        observed=np.fromfile(path,dtype='<f4').reshape(frame['shape'])[0,::3,::3]
        rows.append(dict(lead_hour=hour,nominal=frame['nominal'],sha256=sha,
                         **summarize(observed,lat,lon)))
    out=W/'products/showcase'
    (out/'observation-reference.json').write_text(json.dumps(dict(frames=rows,
        scope='verification references computed from each observed hour; same-window frequency is not a deployable forecast'),indent=2)+'\n')
    with (out/'observation-reference.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['lead_hour','observed_frequency_35','no_event_brier',
                                         'same_window_frequency_brier','missing_pixels'])
        writer.writeheader()
        for row in rows: writer.writerow({key:row[key] for key in writer.fieldnames})

if __name__=='__main__': main()
