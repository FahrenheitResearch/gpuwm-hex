"""Closed-run CPU products and observational scores. No card lease."""
from pathlib import Path
import csv, datetime as dt, hashlib, json, os, subprocess, sys
import numpy as np
from gpuwm import rustwx
from gpuwm.ensemble.batch_products import FieldProducts
from gpuwm.ensemble.batch_product_output import NativeDiagnosticSpool
from score import match_history, scores, ensemble_scores
W = Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
R = W/'prepared'

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def retire(path):
    path = path.absolute()
    resolved = path.resolve()
    if path.is_symlink() or not (resolved.is_relative_to(W.resolve()) or resolved.is_relative_to(R.resolve())):
        raise ValueError(f'not an owned regular path: {path}')
    if not path.is_file(): return
    row = dict(path=str(path),bytes=path.stat().st_size)
    with (W/'deleted-files.log').open('a') as f: f.write(json.dumps(row)+'\n')
    path.unlink()

def window_frames(observations,start,hours):
    frames = [frame for frame in observations['frames'] if 0 <=
        (dt.datetime.fromisoformat(frame['nominal'].removesuffix('Z'))-start).total_seconds() <= hours*3600]
    leads = [(dt.datetime.fromisoformat(frame['nominal'].removesuffix('Z'))-start).total_seconds()/3600
             for frame in frames]
    if leads != list(range(round(hours)+1)):
        raise ValueError('observations do not cover every required lead hour exactly once')
    return frames

def verification_request(member,path,valid,root):
    # rw_verify's reusable-plane protocol names NetCDF sources and requires
    # an explicit owned output root, independently of the request workdir.
    return dict(schema='gpuwm.verify-visuals.request.v1',valid_time=valid.isoformat(),domain='d01',
        out_root=str(root),arms=[dict(label=member,kind='netcdf',path=str(path),
        fields={'composite_reflectivity':'REFL_10CM'})])

def main(run_name):
    if run_name not in ('pilot','showcase'): raise ValueError('unknown run')
    root = W/'runs'/run_name
    receipt = json.loads((root/'ensemble-receipt.json').read_text())
    if receipt['status']=='running': raise ValueError('forecast command must close first')
    observations = json.loads((R/'observations/nowcast.json').read_text())
    out = W/'products'/run_name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').is_file(): return
    renderer = Path(os.environ['GPUWM_RW_WRFBATCH'])
    rustwx.require_ensemble_diagnostic_reducer(renderer)
    successes = [job for job in receipt['jobs'] if job['returncode']==0]
    if not successes: raise ValueError('no passed member to score')
    manifest = json.loads((R/'ensemble.json').read_text())
    rows = {row['id']:row for row in manifest['members']}
    start = dt.datetime.strptime(manifest['start_time'],'%Y-%m-%d_%H:%M:%S')
    scored = json.loads((out/'member-scores.json').read_text()) if (out/'member-scores.json').exists() else []
    mapped = json.loads((out/'maps.json').read_text()) if (out/'maps.json').exists() else []
    completed = {(row['member'],row['lead_hour']) for row in scored}
    planes = {}
    frames = window_frames(observations,start,receipt['cycle_hours'])
    spool = None
    coordinates = out/'render-coordinates.npz'
    def make_spool(rlat,rlon):
        return NativeDiagnosticSpool(out/'aggregate',members=len(successes),
            requests=(FieldProducts('refl','dBZ',(35.,),postage_stamp=False),),
            latitude=rlat,longitude=rlon,renderer=renderer,gpu_replay=False,
            resume=True,export_coordinates=True)
    if len(successes)>1 and coordinates.exists():
        with np.load(coordinates) as data: spool=make_spool(data['lat'],data['lon'])
    for index, job in enumerate(successes):
        history_root = root/job['id']/'history'
        for frame in frames:
            valid = dt.datetime.fromisoformat(frame['nominal'].removesuffix('Z'))
            lead = int((valid-start).total_seconds()/3600)
            history = history_root/('cuda-history.'+valid.strftime('%Y-%m-%d_%H.%M.%S')+'.nc')
            plane_path = out/f'score-{job["id"]}-f{lead:03d}.npy'
            planes.setdefault(lead,[]).append(plane_path)
            if (job['id'],lead) in completed:
                if not plane_path.is_file(): raise ValueError('retained score plane missing before ensemble closure')
                retire(history)
                continue
            if not history.is_file(): raise ValueError(f'missing required history: {history}')
            identity = dict(path=str(history),bytes=history.stat().st_size,sha256=digest(history))
            model, observed, lat, lon, mapping = match_history(history,R/'observations',frame)
            result = dict(member=job['id'],lead_hour=lead,valid_time=frame['nominal'],history=identity,
                          mapping=mapping,**scores(model,observed,lat,lon))
            np.save(plane_path,model)
            cvt = out/'converted'/job['id']/f'f{lead:03d}'
            cvt.mkdir(parents=True,exist_ok=True)
            convert_receipt = out/f'convert-{job["id"]}-f{lead:03d}.json'
            command = [os.environ['GPUWM_HEX_RW_MPAS_CONVERT'],'--history',str(history),
                '--mesh',rows[job['id']].get('grid',manifest['grid']), '--init',rows[job['id']]['init'],
                '--out-dir',str(cvt),'--clobber','--window','mesh','--field-set','surface',
                '--simulation-start',manifest['start_time'],'--json',str(convert_receipt)]
            with (out/f'convert-{job["id"]}-f{lead:03d}.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            files = list(cvt.glob('wrfout*'))
            if len(files)!=1: raise ValueError('converter did not write one frame')
            maps, failures, skipped = rustwx.run_renderer(renderer,files[0],store_root=out/'stores'/job['id']/f'f{lead:03d}',
                out_dir=out/'maps'/job['id']/f'f{lead:03d}', products='composite_reflectivity',frames='all',
                source_label=f'gpuwm-hex {job["id"]} static 48 to 3 km, active LTS')
            if failures or not maps: raise RuntimeError(str(failures or skipped))
            native_root = out/'native-planes'/job['id']/f'f{lead:03d}'
            request = verification_request(job['id'],files[0],valid,native_root)
            prepared = rustwx.prepare_verification(request,workdir=native_root)
            npz = Path(prepared['arms'][0]['path'])
            with np.load(npz) as data:
                rlat, rlon = data['lat'], data['lon']
                # Exact f32 words transported through the Rust f64 snapshot.
                refl = np.ascontiguousarray(data['composite_reflectivity'], dtype=np.float32)
            if spool is None and len(successes)>1:
                np.savez(coordinates,lat=rlat,lon=rlon)
                spool = make_spool(rlat,rlon)
            if spool is not None:
                if not np.array_equal(rlat,spool.latitude) or not np.array_equal(rlon,spool.longitude):
                    raise ValueError('different member render lattices require Rust remapping before reduction')
                existing = spool.frames.get(valid.isoformat(),{}).get('members_received',[])
                if index not in existing:
                    spool.submit({'refl':refl[None]},member_ids=(index,),valid_time=valid.isoformat(),array_module=np)
            mapped.extend(dict(member=job['id'],lead_hour=lead,path=str(p),sha256=digest(p)) for p in maps)
            (out/'maps.json').write_text(json.dumps(mapped,indent=2)+'\n')
            scored.append(result)
            (out/'member-scores.json').write_text(json.dumps(scored,indent=2,allow_nan=False)+'\n')
            # Receipt and native plane are committed before raw-frame disposal.
            retire(history)
            for p in cvt.rglob('*'):
                if p.is_file(): retire(p)
            for p in npz.parent.rglob('*.npz'): retire(p)
    ensemble = []
    for lead, paths in sorted(planes.items()):
        frame = next(f for f in observations['frames'] if (dt.datetime.fromisoformat(f['nominal'].removesuffix('Z'))-start).total_seconds()==lead*3600)
        lattice = observations['lattice']
        observed = np.fromfile(R/'observations'/frame['file'],dtype='<f4').reshape(frame['shape'])[0,::3,::3]
        lon,lat = np.meshgrid(lattice['lon0']+np.arange(0,lattice['nx'],3)*lattice['dlon'],
                             lattice['lat0']+np.arange(0,lattice['ny'],3)*lattice['dlat'])
        models = np.stack([np.load(p) for p in paths])
        ensemble.append(dict(lead_hour=lead,valid_time=frame['nominal'],**ensemble_scores(models,observed,lat,lon)))
        if spool is not None and spool.frames[dt.datetime.fromisoformat(frame['nominal'].removesuffix('Z')).isoformat()]['status']!='complete':
            spool.finish(dt.datetime.fromisoformat(frame['nominal'].removesuffix('Z')).isoformat(),
                         available_bytes=32*2**30,array_module=np,render_products=('mean','max','prob'))
        (out/'ensemble-scores.json').write_text(json.dumps(ensemble,indent=2,allow_nan=False)+'\n')
    with (out/'member-scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['member','lead_hour','csi_35','fss_20km','fss_50km',
                             'position_error_km','max_reflectivity_bias_dbz'])
        writer.writeheader()
        for row in scored:
            writer.writerow({**{k:row[k] for k in ('member','lead_hour','csi_35','position_error_km','max_reflectivity_bias_dbz')},
                             'fss_20km':row['fss']['20'],'fss_50km':row['fss']['50']})
    from labels import label_aggregates
    label_aggregates(out,renderer)
    (out/'complete.json').write_text(json.dumps(dict(members=len(successes),member_frames=len(scored),
        ensemble_frames=len(ensemble),maps=len(mapped),score_method='score.py',render_route='gpuwm.rustwx -> rw_wrfbatch'))+'\n')
    for paths in planes.values():
        for p in paths: retire(p)
    if coordinates.exists(): retire(coordinates)

if __name__=='__main__': main(sys.argv[1])
