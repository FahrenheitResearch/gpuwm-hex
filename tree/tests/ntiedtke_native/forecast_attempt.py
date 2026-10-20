"""Attempt the real forecast door without changing a published native deck."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def attempt(args):
    deck, case = args.deck.resolve(), args.case.resolve()
    argv = [sys.executable, '-m', 'hexcore', 'forecast', '--mesh', args.mesh,
            '--grid', str(case / (args.mesh+'.grid.nc')),
            '--init', str(case / (args.mesh+'.init.nc')),
            '--physics-namelist', str(deck/'namelist.atmosphere'),
            '--physics-backend', 'mpasphys',
            '--physics', "config_convection_backend='hex-cuda'", '--hours', '24',
            '--out', str(args.output.resolve())]
    if args.streams:
        argv += ['--streams', str(deck/'streams.atmosphere')]
    if args.preflight:
        argv += ['--preflight']
    providers = {}
    for name, option in (('library', '--native-physics-library'),
                         ('pool_reader', '--native-pool-reader')):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve()
            argv += [option, str(path)]
            providers[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if args.mpas_source is not None:
        argv += ['--mpas-source', str(args.mpas_source.resolve())]
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (deck/'namelist.atmosphere', deck/'streams.atmosphere',
             case/(args.mesh+'.grid.nc'), case/(args.mesh+'.init.nc'))}
    completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    args.log.write_text(completed.stdout)
    tip = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[3]),
                                  'rev-parse', 'HEAD'], text=True).strip()
    result = dict(schema='mpas-ntiedtke-real-door-attempt/v1', acceptance=False,
                  process_count=1, requested_duration_hours=24, argv=argv, checkout=tip,
                  input_sha256=pins, exit_code=completed.returncode, log=str(args.log),
                  preflight_only=args.preflight, native_providers=providers)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return completed.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('deck', 'case', 'output', 'log', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--streams', action='store_true')
    for name in ('library', 'pool-reader', 'mpas-source'):
        parser.add_argument('--'+name, type=Path)
    sys.exit(attempt(parser.parse_args()))
