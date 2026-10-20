"""Prepare an exact-byte forecast verification deck from a sealed native run.

No field decoding occurs here. The real verifier authenticates the publication
and reads every stream through its Rust comparator before invoking gpuwm-hex.
"""
import argparse, hashlib, json
from datetime import datetime
from pathlib import Path
from hexcore.mpasphys.driver_v842 import resolve_namelist

p = argparse.ArgumentParser(description=__doc__)
for name in ('native', 'library', 'reader', 'source', 'tables', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--input-directory', type=Path,
               help='Original native input directory when publication contains selected files')
p.add_argument('--id', required=True)
p.add_argument('--suite', required=True)
p.add_argument('--scheme', required=True)
a = p.parse_args()
digest = lambda path: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
manifest_path = a.native/'MANIFEST.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['ranks'] == 1
log = a.native/'log.atmosphere.0000.out'
assert 'Finished running the atmosphere core' in log.read_text()
publication = a.native/'PUBLICATION.json'
if not publication.exists():
    selection = ['MANIFEST.json', log.name, *manifest['outputs']]
    files = {**manifest['outputs'], 'MANIFEST.json':digest(manifest_path), log.name:digest(log)}
    publication.write_text(json.dumps(dict(schema='mpas.native-cache-publication/v1',
        description='Owned strict native forecast; metadata seal, no copied/synthesized fields',
        source=str(a.native.resolve()), selection=selection, files=files), indent=2)+'\n')
ref = lambda path: dict(path=str(path.resolve()), sha256=digest(path))
runtime = manifest['runtime']
input_directory = a.input_directory or a.native
namelist = a.native/'namelist.atmosphere'
streams = a.native/'streams.atmosphere'
if not namelist.is_file():
    namelist = input_directory/'namelist.atmosphere'
if not streams.is_file():
    streams = input_directory/'streams.atmosphere'
init = input_directory/runtime['init']
config = resolve_namelist(namelist.read_text(), backends={'surface_layer':'hex-cuda'})
assert config.suite == a.suite and config.selectors['surface_layer'] == a.scheme
auxiliary = {name:ref(input_directory/name) for name in manifest['inputs']
    if name not in ('namelist.atmosphere','streams.atmosphere',runtime['init'],'atmosphere_model')}
command = ['gpuwm-hex','forecast','--mesh',runtime['mesh'],
    '--physics-namelist','{namelist}','--streams','{streams}','--init','{init}',
    '--out','{out}','--hours','24','--physics-backend','mpasphys',
    '--physics',"config_sfclayer_backend='hex-cuda'",
    '--native-physics-library',str(a.library.resolve()),
    '--native-pool-reader',str(a.reader.resolve()), '--mpas-source',str(a.source.resolve()),
    '--grid',str((input_directory/(runtime['mesh']+'.grid.nc')).resolve()),
    '--stream-list-directory',str(input_directory.resolve()),
    '--native-physics-tables',str(a.tables.resolve())]
oro = list(input_directory.glob('*.ugwp_oro_data.nc'))
if oro:
    command += ['--orographic-statistics',str(oro[0].resolve())]
steps = 86400//runtime['dt']
assert steps*runtime['dt'] == 86400
deck = dict(schema='mpas-verification-deck-v1', pin_set=manifest['pin_set_hash'],
    mesh=runtime['mesh'],date=runtime['start'],id=a.id,suite=a.suite,
    role='scheme',physics='on',process_count=1,
    inputs=dict(namelist=ref(namelist),streams=ref(streams),init=ref(init)),
    auxiliary_inputs=auxiliary,native=ref(publication),hex_command=command,
    duration_steps=steps, expected_samples={},
    providers=dict(library=ref(a.library),reader=ref(a.reader)))
start = datetime.strptime(runtime['start'],'%Y-%m-%d_%H:%M:%S')
for filename in manifest['outputs']:
    stream, stamp = filename.split('.',1)
    time = datetime.strptime(stamp,'%Y-%m-%d_%H.%M.%S.nc')
    seconds = int((time-start).total_seconds())
    assert seconds >= 0 and seconds % runtime['dt'] == 0
    deck['expected_samples'].setdefault(stream,[]).append(seconds//runtime['dt'])
deck['expected_samples'] = {name:sorted(set(times)) for name,times in deck['expected_samples'].items()}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(deck,indent=2)+'\n')
print(a.id, 'exact input bytes; process_count=1; no model run')
