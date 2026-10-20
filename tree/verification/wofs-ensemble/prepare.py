"""Box-only CPU orchestration. No card lease, decoding or mesh numerics in Python."""
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, os, subprocess, urllib.request
from pathlib import Path
W = Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
V = W / 'hex/tree/verification/wofs-ensemble'
R = W / 'prepared'
case = json.loads((V / 'case.json').read_text())
created = []
def run(name, argv):
    log = W / 'logs' / (name + '.log')
    created.append(str(log))
    with log.open('w') as f:
        subprocess.run([str(x) for x in argv], stdout=f, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    return log
def fetch(url, path):
    if path.exists():
        raise RuntimeError(f'refusing unreceipted existing input {path}')
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as f:
        while block := response.read(4 << 20): f.write(block)
    with path.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
    created.append(str(path))
    return dict(url=url, path=str(path), bytes=path.stat().st_size, sha256=sha)
def observations():
    obs = case['observations']
    run('mrms', [os.environ['RWBIN']+'/rw_mrms', 'frames', '--issue', obs['start'],
                '--start', obs['start'], '--end', obs['end'], '--step-minutes', obs['step_minutes'],
                '--bbox', obs['bbox'], '--out', R/'observations'])
    (W/'evidence/observations-ready.json').write_text(json.dumps(dict(source='rw_mrms frames',
        command_log=str(W/'logs/mrms.log'), manifest=str(R/'observations/nowcast.json')))+'\n')
def mesh():
    from hexcore.mesh_point import ladder_spec
    p = case['patch']
    spec = ladder_spec(tuple(p['center_deg']), fine_dx_m=1000*p['fine_spacing_km'],
                       radius_km=p['core_radius_km'], background_km=p['background_spacing_km'],
                       transition_factor=p['transition_factor'], ring_factor=p['ring_factor'])
    (R/'mesh.spec.json').write_text(json.dumps(spec, indent=2)+'\n')
    run('mesh', [os.environ['GPUWM_HEX_RW_MPAS_MESH'], '--spec', R/'mesh.spec.json',
                '--out', R/'mesh.grid.nc', '--receipt', W/'evidence/mesh.json'])
def sources():
    receipt = []
    s = case['source']
    for member in case['members']:
        for part in ('a', 'b'):
            url = f"https://{s['bucket']}.s3.amazonaws.com/gefs.{s['cycle']}/{s['hour']}/atmos/pgrb2{part}p5/{member}.t{s['hour']}z.pgrb2{part}.0p50.f{s['forecast_hour']}"
            receipt.append(fetch(url, R/f'{member}.{part}.grib2'))
        (W/'evidence/gefs-sources.json').write_text(json.dumps(receipt, indent=2)+'\n')
def geography():
    run('geography', [os.environ['PATH'].split(':')[0]+'/python', '-m', 'gpuwm.cli',
         'fetch-geog', '--root', R/'geog', '--datasets',
         'topo_gmted2010_30s,modis_landuse_20class_30s_with_lakes,soiltype_top_30s,greenfrac_fpar_modis,albedo_modis,maxsnowalb_modis,soiltemp_1deg,soilgrids'])
def main():
    for p in (R, W/'logs', W/'evidence', W/'tmp', W/'cupy-cache', W/'cuda-cache'):
        p.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(4) as pool:
            jobs = [pool.submit(f) for f in (observations, mesh, sources, geography)]
            for job in jobs: job.result()
    finally:
        with (W/'MANIFEST.txt').open('a') as f:
            for p in sorted(set(created)): f.write(p+'\n')
    (W/'evidence/cpu-fetch-mesh-ready.json').write_text('{"ready": true}\n')
if __name__ == '__main__': main()
