"""Exercise actual CLI routing on every sealed YSU deck without a GPU.

Rust performs native input decoding. This proves routing only: preflight
does not initialize physics, call a scheme, or execute a forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'library', 'reader', 'source', 'stream-lists', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('CLI preflight needs a fresh output directory')
    args.output.mkdir(parents=True)
    summary = dict(schema='mpas-ysu-v842/real-cli-route-v1', host=platform.node(),
                   process_count=1, forecast_proven=False, scheme_executed=False,
                   inputs={name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for name, path in (('library', args.library), ('reader', args.reader),
                                              ('index', args.index))}, entries=[])
    for entry in json.loads(args.index.read_text())['entries']:
        out = args.output/entry['name']
        out.mkdir()
        case = Path(entry['case'])
        deck = Path(entry['deck'])
        # This is gpuwm-hex's actual project.scripts implementation. The
        # lane environment exposes the checkout but installs no console shim.
        command = [sys.executable, '-m', 'hexcore.cli', 'forecast', '--mesh', entry['mesh'],
                   '--grid', str(case/(entry['mesh']+'.grid.nc')),
                   '--init', str(case/(entry['mesh']+'.init.nc')),
                   '--physics-backend', 'mpasphys', '--physics-namelist',
                   str(deck/'namelist.atmosphere'), '--streams', str(deck/'streams.atmosphere'),
                   '--physics', "config_pbl_backend='hex-cuda'",
                   '--native-physics-library', str(args.library), '--native-pool-reader',
                   str(args.reader), '--mpas-source', str(args.source),
                   '--stream-list-directory', str(args.stream_lists), '--preflight',
                   '--out', str(out.resolve())]
        with (out/'CLI.log').open('w') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        row = dict(name=entry['name'], argv=command, returncode=status, passed=False)
        request = out/'native-physics-request.json'
        if status == 0 and request.exists():
            result = json.loads(request.read_text())
            config = result['physics']
            assert config['selectors']['pbl'] == 'bl_ysu'
            assert config['backends']['pbl'] == 'hex-cuda'
            assert config['options']['config_ysu_pblmix'] == entry['ysu_pblmix']
            assert all(value in ('native-cpu', 'off') for key, value in config['backends'].items()
                       if key != 'pbl')
            row.update(passed=True, request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
                       physics=config)
            shutil.rmtree(out/'scratch')
        summary['entries'].append(row)
        (args.output/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    summary['all_routes_passed'] = all(row['passed'] for row in summary['entries'])
    (args.output/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    return int(not summary['all_routes_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
