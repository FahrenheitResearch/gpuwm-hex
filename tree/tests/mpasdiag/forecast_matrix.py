"""Seal actual forecast decks from published native manifests, without case defaults."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seconds(interval):
    day, clock = interval.split('_')
    hour, minute, second = map(int, clock.split(':'))
    return int(day)*86400 + hour*3600 + minute*60 + second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', type=Path)
    parser.add_argument('forecasts', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    template = json.loads(args.template.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows, verified = [], {}
    for source in sorted(args.forecasts.glob('gnu-strict-*-rank1')):
        manifest = json.loads((source/'MANIFEST.json').read_text())
        runtime = manifest['runtime']
        if manifest['ranks'] != 1 or seconds(runtime['duration']) != 86400:
            raise ValueError('matrix requires published one-process 24h native decks')
        mesh, start = runtime['mesh'], runtime['start']
        case = args.cases/mesh/start.replace(':','.')/'init-modis20-v2'
        deck = copy.deepcopy(template)
        deck.update(id='o2-isobaric-'+source.name, mesh=mesh, date=start,
                    suite=manifest['deck'].removeprefix('suite-'))
        for name, path, input_name in (
                ('namelist',source/'namelist.atmosphere','namelist.atmosphere'),
                ('streams',source/'streams.atmosphere','streams.atmosphere'),
                ('init',case/(mesh+'.init.nc'),mesh+'.init.nc')):
            deck['inputs'][name] = dict(path=str(path.resolve()),sha256=manifest['inputs'][input_name])
        for name in tuple(deck['auxiliary_inputs']):
            if name.startswith(template['mesh']+'.'):
                deck['auxiliary_inputs'].pop(name)
                name = mesh+name[len(template['mesh']):]
                deck['auxiliary_inputs'][name] = dict(path=str((case/name).resolve()),
                                                     sha256=manifest['inputs'][name])
        for item in (*deck['inputs'].values(), *deck['auxiliary_inputs'].values()):
            path = Path(item['path'])
            digest = verified.setdefault(str(path), None)
            if digest is None:
                digest = verified[str(path)] = sha(path)
            if digest != item['sha256']:
                raise ValueError('native deck bytes differ: '+str(path))
        command = deck['hex_command']
        command[command.index('--mesh')+1] = mesh
        command[command.index('--grid')+1] = str((case/(mesh+'.grid.nc')).resolve())
        deck['native'] = dict(path=str((source/'PUBLICATION.json').resolve()),
                              sha256=sha(source/'PUBLICATION.json'))
        duration, dt = seconds(runtime['duration']), runtime['dt']
        if duration % dt:
            raise ValueError('native duration is not an integer number of model steps')
        deck['duration_steps'] = duration//dt
        beginning = datetime.strptime(start,'%Y-%m-%d_%H:%M:%S')
        samples = {}
        for name in manifest['outputs']:
            if not name.endswith('.nc'):
                continue
            stream, timestamp = name.split('.',1)
            ending = datetime.strptime(timestamp[:-3],'%Y-%m-%d_%H.%M.%S')
            offset = int((ending-beginning).total_seconds())
            if offset % dt:
                raise ValueError('native output is between model steps: '+name)
            samples.setdefault(stream,[]).append(offset//dt)
        deck['expected_samples'] = {name:sorted(values) for name,values in samples.items()}
        destination = args.out/(source.name+'.json')
        destination.write_text(json.dumps(deck,indent=2)+'\n')
        rows.append(dict(deck=str(destination.resolve()),sha256=sha(destination),
                         native=deck['native'], suite=deck['suite'], mesh=mesh,date=start,
                         process_count=1,steps=deck['duration_steps'],
                         expected_samples=deck['expected_samples'],executed=False))
    result = dict(schema='mpas-o2.forecast-matrix/v1',process_count=1,
                  prepared=True,passed=False,identity_checked=False,decks=rows)
    (args.out/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(prepared_decks=len(rows),verified_input_files=len(verified))),flush=True)


if __name__ == '__main__':
    main()
