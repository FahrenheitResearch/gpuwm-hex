"""Box CPU preparation. Forecast inputs are complete before a card is leased."""
from pathlib import Path
import hashlib, json, os, subprocess
from hexcore import mesh_rows
from hexcore.device_admission import KNOWN_CARDS, model_for_card
W = Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
R = W/'prepared'
V = Path(__file__).resolve().parent   # this checkout's case, not a fixed lane path
case = json.loads((V/'case.json').read_text())

def run(name, argv):
    path = W/'logs'/f'{name}.log'
    with path.open('w') as log:
        subprocess.run(list(map(str, argv)), check=True, stdout=log,
                       stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))

def retained_output(path,receipt):
    if not path.exists(): return False
    data=json.loads(receipt.read_text())
    record=data.get('output',data)
    with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    size=record['bytes'] if 'bytes' in record else record['output_bytes']
    if record['sha256']!=actual or size!=path.stat().st_size:
        raise ValueError(f'retained output changed: {path}')
    return True

def main():
    if not (W/'evidence/cpu-fetch-mesh-ready.json').is_file():
        raise RuntimeError('mesh, sources and observations must be ready first')
    for name in ('soilcomp', 'texture_layer1', 'texture_layer2', 'texture_layer3', 'texture_layer4'):
        p = R/'geog'/name
        if not p.exists(): p.symlink_to(R/'geog/soilgrids'/name, target_is_directory=True)
    static = R/'mesh.static.nc'
    if not retained_output(static,W/'evidence/static.json'):
        run('static', [os.environ['GPUWM_HEX_RW_MPAS_STATIC'], '--grid', R/'mesh.grid.nc',
                   '--out', static, '--geog', R/'geog', '--receipt', W/'evidence/static.json'])
    members = []
    for name in case['members']:
        d = R/name
        d.mkdir(exist_ok=True)
        met = d/('MET:' + case['start_time'][:13])
        if not met.exists():
            run(f'{name}-met', [os.environ['MET_INTERMEDIATE'], '--vtable', os.environ.get('HEX_VTABLE_GFS', 'Vtable.GFS'),
                           '--date', case['start_time'], '--map-source', 'ncep-gefs', '--out', met,
                           R/f'{name}.a.grib2', R/f'{name}.b.grib2'])
        init = d/'init.nc'
        # The ordinary init door owns the content-keyed vertical cache. The
        # same spec reuses its verified artifact without assuming a filename.
        vertical = ['--vertical-spec', V.parent/'vertical-specs/tc55-v1.json']
        if not retained_output(init,init.with_suffix('.nc.provenance.json')):
            run(f'{name}-init', ['python', '-m', 'hexcore.cli', 'init', '--met', met,
            '--grid', R/'mesh.grid.nc', '--static', static, *vertical, '--out', init,
            '--start-time', case['start_time'], '--nfglevels', '34', '--nfgsoillevels', '4',
            '--extrap-airtemp', 'constant', '--use-spechumd', 'no', '--theta-adv-order', '3',
            '--coef-3rd-order', '0.25', '--virtual-factor', 'reproduce-fortran',
            '--deep-soil-moisture', 'reproduce-fortran', '--landuse-table', 'MODIFIED_IGBP_MODIS_NOAH',
            '--frac-seaice', 'yes', '--tsk-seaice-threshold', '100', '--oned-underflow', 'preserve'])
        members.append(dict(id=name, init=str(init), source=f'GEFS {case["source"]["cycle"]} '
                            f'{case["source"]["hour"]}Z {name} f000; gefs-sources.json',
                            forecast_flags=case.get('member_physics', {}).get(name, [])))
        (W/'evidence/init-progress.json').write_text(json.dumps(dict(completed=[m['id'] for m in members]))+'\n')
    p = case['patch']
    row = mesh_rows.describe_generated(name='wofs-patch', grid=R/'mesh.grid.nc', static=static,
        spec_path=R/'mesh.spec.json', generator_receipt=W/'evidence/mesh.json',
        static_receipt=W/'evidence/static.json', point_deg=tuple(p['center_deg']),
        fine_dx_m=1000*p['fine_spacing_km'], core_radius_km=p['core_radius_km'],
        background_km=p['background_spacing_km'], dt_seconds=18.75, n_levels=55,
        admission={'scope': 'closed global patch, one-card full-step LTS; forecast door checks bytes and geometry'})
    mesh_rows.write_rows(R/'mesh-rows.json', [row])
    admission = {}
    for rollback in (False, True):
        model = model_for_card(KNOWN_CARDS['96gib-188sm'], levels=55, rollback_armed=rollback)
        admission[str(rollback)] = dict(required_mib=model.required_bytes(row.n_cells)/2**20,
            predicted_peak_mib=model.predict_bytes(row.n_cells)/2**20, provenance=model.provenance,
            rollback_armed=rollback)
    (W/'evidence/cpu-admission.json').write_text(json.dumps(dict(cells=row.n_cells, levels=55,
        card='96gib-188sm', variants=admission), indent=2)+'\n')
    from hexcore.ensemble import SCHEMA, validate_inputs
    manifest = dict(schema=SCHEMA, resolution_km=3, start_time=case['start_time'],
        mesh=row.name, grid=str(R/'mesh.grid.nc'), static=str(static), mesh_rows=str(R/'mesh-rows.json'),
        native_template=members[0]['init'], reference_init=members[0]['init'], patch=p,
        forecast_flags=case['forecast_flags'], members=members)
    (R/'ensemble.json').write_text(json.dumps(manifest, indent=2)+'\n')
    dimensions = validate_inputs(manifest)
    (W/'evidence/inputs-ready.json').write_text(json.dumps(dict(ready=True, dimensions=dimensions,
        members=[dict(id=m['id'], physical_state_sha256=m['physical_state_sha256']) for m in members]), indent=2)+'\n')
    with (W/'MANIFEST.txt').open('a') as f:
        for root in (W, R.resolve()):
            for path in root.rglob('*'):
                if '.git' not in path.parts: f.write(str(path)+'\n')

if __name__ == '__main__': main()
