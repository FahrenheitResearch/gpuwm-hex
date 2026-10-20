"""Exercise actual installed CLI namelist routing; never certifies forecasts.

MPAS mpas_atmphys_driver_seaice.F:160-558 supplies the driver boundary.
All input decoding stays in the foundation Rust reader. This harness retains
only request metadata and logs, then deletes its own decoded pool buffers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    for variable in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
        assert int(os.environ.get(variable, '1')) == 1
    index = json.loads(args.prepared.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for number, item in enumerate(index['decks']):
        deck = Path(item['deck']).resolve()
        # Native publication layout: G2 / mesh / start-date / option.
        parent = Path(item['native_source'])
        mesh, date = parent.parent.parent.name, parent.parent.name
        case = args.case_root / mesh / date / 'init-modis20-v2'
        namelist, streams = deck / 'namelist.hex.atmosphere', deck / 'streams.atmosphere'
        assert sha(namelist) == item['hex_namelist_sha256']
        assert sha(streams) == item['streams_sha256']
        grid, init = case / (mesh + '.grid.nc'), case / (mesh + '.init.nc')
        output = args.out / ('preflight-' + str(number))
        command = [str(args.cli.resolve()), 'forecast', '--mesh', mesh,
                   '--grid', str(grid), '--init', str(init),
                   '--physics-namelist', str(namelist), '--streams', str(streams),
                   '--physics-backend', 'mpasphys', '--native-physics-library',
                   str(args.library.resolve()), '--native-pool-reader', str(args.reader.resolve()),
                   '--mpas-source', str(args.source.resolve()), '--stream-list-directory',
                   str(args.stream_list.resolve()), '--native-physics-tables',
                   str(args.tables.resolve()), '--out', str(output.resolve()), '--preflight']
        log = args.out / ('cli-' + str(number) + '.log')
        with log.open('w') as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        row = dict(deck=str(deck), command=command, returncode=result.returncode,
                   namelist_sha256=sha(namelist), streams_sha256=sha(streams),
                   log=str(log), model_execution=False, passed=False)
        request = output / 'native-physics-request.json'
        if result.returncode == 0 and request.is_file():
            plan = json.loads(request.read_text())
            physics = plan['physics']
            assert physics['backends']['seaice'] == 'hex-cuda'
            assert physics['selectors']['seaice'] in ('sf_noah', 'sf_noahmp')
            assert physics['options']['config_frac_seaice'] is item['fractional']
            assert all(value == 'native-cpu' for name, value in physics['backends'].items()
                       if name != 'seaice')
            assert plan['identity_checked'] is False
            row.update(passed=True, request=plan, request_sha256=sha(request))
        if output.exists():
            shutil.rmtree(output)
        rows.append(row)
        args.receipt.write_text(json.dumps(dict(
            kind='installed gpuwm-hex mpasphys preflight; not forecast proof',
            host=platform.node(), process_count=1, cli_sha256=sha(args.cli),
            prepared_sha256=sha(args.prepared), reader_sha256=sha(args.reader),
            native_library_sha256=sha(args.library), calls=len(rows),
            passed=all(row['passed'] for row in rows), decks=rows), indent=2) + '\n')
    print(json.dumps(dict(calls=len(rows), passed=all(row['passed'] for row in rows))))
    return int(not all(row['passed'] for row in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'prepared', 'case-root', 'library', 'reader', 'source',
                 'stream-list', 'tables', 'out', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
