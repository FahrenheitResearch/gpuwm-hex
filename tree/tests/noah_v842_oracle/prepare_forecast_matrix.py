"""Prepare authenticated real-door decks; this is input preparation, not proof.

Output clocks below come from native publication metadata. The Rust comparator
must independently validate every NetCDF field/clock during the real forecast.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--matrix', type=Path, required=True)
p.add_argument('--template', type=Path, required=True)
p.add_argument('--resources', type=Path, required=True)
p.add_argument('--native-restored-root', type=Path,
               help='owned f0 byte-restored publications; all bytes are rechecked by verify-mpas')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
a = p.parse_args()
matrix = json.loads(a.matrix.read_text())
template = json.loads(a.template.read_text())
a.out.mkdir(parents=True, exist_ok=True)
rows = []
for row in matrix['rows']:
    native = Path(row['native_forecast_directory'])
    publication = native / 'PUBLICATION.json'
    result = {key: row[key] for key in ('mesh', 'date', 'selection')}
    if not publication.is_file():
        result['prepared'] = False
        result['reason'] = 'native publication not yet present'
        rows.append(result)
        continue
    assert sha(publication) == row['native_publication_sha256']
    published = json.loads(publication.read_text())
    manifest_path = native / 'MANIFEST.json'
    assert sha(manifest_path) == published['files']['MANIFEST.json']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['ranks'] == 1
    deck = copy.deepcopy(template)
    mesh = row['mesh']
    # The publication root is explicit; locate the common immutable ref root.
    ref = next(parent for parent in publication.parents if parent.name == 'ref')
    case = ref / 'cases' / mesh / row['date'] / 'init-modis20-v2'
    files = {'namelist': native/'namelist.atmosphere',
             'streams': native/'streams.atmosphere', 'init': case/manifest['runtime']['init']}
    names = {'namelist': 'namelist.atmosphere', 'streams': 'streams.atmosphere',
             'init': manifest['runtime']['init']}
    for key, path in files.items():
        assert sha(path) == manifest['inputs'][names[key]], (row, key)
        deck['inputs'][key] = {'path': str(path), 'sha256': sha(path)}
    grid = case / (mesh+'.grid.nc')
    assert sha(grid) == manifest['inputs'][grid.name]
    auxiliary = {}
    for name, item in deck['auxiliary_inputs'].items():
        if name.startswith(template['mesh']+'.'):
            name = mesh+name[len(template['mesh']):]
            item['path'] = str(case/name)
        expected = manifest['inputs'][name]
        assert sha(item['path']) == expected, (row['mesh'], row['date'], name)
        item['sha256'] = expected
        auxiliary[name] = item
    deck['auxiliary_inputs'] = auxiliary
    command = deck['hex_command']
    assert command[:2] == ['gpuwm-hex', 'forecast']
    for flag, value in {'--mesh': mesh, '--grid': str(grid), '--init': '{init}',
                        '--physics-namelist': '{namelist}', '--streams': '{streams}',
                        '--out': '{out}'}.items():
        command[command.index(flag)+1] = value
    command.extend(['--physics', "config_lsm_backend='hex-cuda'",
                    '--physics-resources', str(a.resources.resolve())])
    start = datetime.strptime(manifest['runtime']['start'], '%Y-%m-%d_%H:%M:%S')
    dt = manifest['runtime']['dt']
    clocks = {}
    for name in manifest['outputs']:
        match = re.fullmatch(r'(.+)\.(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})\.nc', name)
        assert match, name
        time = datetime.strptime(match[2], '%Y-%m-%d_%H.%M.%S')
        seconds = int((time-start).total_seconds())
        assert seconds % dt == 0
        clocks.setdefault(match[1], []).append(seconds//dt)
    clocks = {name: sorted(set(values)) for name, values in clocks.items()}
    duration = 86400//dt
    assert 86400 % dt == 0 and max(max(values) for values in clocks.values()) == duration
    assert manifest['runtime']['duration'] == '1_00:00:00'
    deck.update(id='p16-noah-'+native.name, mesh=mesh, date=row['date'],
                suite=row['explicit_options'].get('config_physics_suite', "'none'").strip("'"),
                duration_steps=duration, expected_samples=clocks,
                native={'path': str(publication), 'sha256': sha(publication)})
    output_root = native
    if not all((native/name).exists() for name in manifest['outputs']) and a.native_restored_root:
        output_root = a.native_restored_root / native.name
        restored_publication = output_root / 'PUBLICATION.json'
        assert sha(restored_publication) == sha(publication)
        assert sha(output_root/'MANIFEST.json') == sha(manifest_path)
        assert all((output_root/name).is_file() for name in manifest['outputs'])
        # f0's extractor already checks every restored output hash. The real
        # verifier independently seals those bytes before launching the model.
        # Preparation never declares them numerically proven.
        deck['native']['path'] = str(restored_publication)
    path = a.out / (native.name+'.json')
    path.write_text(json.dumps(deck, indent=2)+'\n')
    result.update(prepared=True, deck=str(path.resolve()), deck_sha256=sha(path),
                  native_publication_sha256=sha(publication), duration_steps=duration,
                  expected_samples=clocks, process_count=1,
                  native_output_payloads_present=all((output_root/name).exists() for name in manifest['outputs']),
                  native_output_directory=str(output_root))
    rows.append(result)
a.receipt.write_text(json.dumps({'scope': 'authenticated forecast input preparation; no numerical proof',
    'forecast_proven': False, 'matrix_sha256': sha(a.matrix),
    'template_sha256': sha(a.template), 'resources_sha256': sha(a.resources),
    'rows': rows}, indent=2)+'\n')
print('prepared', sum(row['prepared'] for row in rows), 'of', len(rows), 'native forecast decks')
