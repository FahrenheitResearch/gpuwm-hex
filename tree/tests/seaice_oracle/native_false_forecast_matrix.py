"""Measure unchanged strict MPAS forecasts for every fractional-false deck.

MPAS mpas_atmphys_driver_seaice.F:390 reads cpm_sea, while
mpas_atmphys_driver_sfclayer.F:150-158 allocates it only for fractional ice.
Python records process status and byte hashes; it never processes weather data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = (args.out / 'native-runtime-inputs').resolve()
    runtime.mkdir(exist_ok=True)
    for directory in (args.build, *args.auxiliary):
        for source in directory.iterdir():
            if not source.is_file() or not (source.name == 'atmosphere_model' or
                    source.name.startswith('stream_list.') or
                    source.suffix in ('.TBL', '.DBL', '.tbl', '.nc') or 'DATA' in source.name):
                continue
            target = runtime / source.name
            if target.exists():
                assert sha(target) == sha(source), source.name
            elif not target.is_symlink():
                target.symlink_to(source.resolve())
    rows = []
    for item in json.loads(args.prepared.read_text())['decks']:
        if item['fractional']:
            continue
        source, deck = Path(item['native_source']), Path(item['deck']).resolve()
        mesh, date = source.parent.parent.name, source.parent.name
        assert sha(deck / 'namelist.atmosphere') == item['namelist_sha256']
        assert sha(deck / 'streams.atmosphere') == item['streams_sha256']
        index = deck.parent / 'INDEX.json'
        original = source.parent / 'INDEX.json'
        if index.exists():
            assert index.read_bytes() == original.read_bytes()
        else:
            index.write_bytes(original.read_bytes())
        name = '-'.join((mesh, date, source.name, 'frac-false-rank1'))
        output = (args.out / name).resolve()
        proof = args.out / (name + '.json')
        if proof.exists():
            row = json.loads(proof.read_text())
            assert row['namelist_sha256'] == item['namelist_sha256']
            assert row['streams_sha256'] == item['streams_sha256']
            assert row['binary_sha256'] == sha(args.build / 'atmosphere_model')
        else:
            command = [sys.executable, str(args.runner.resolve()), '--build',
                       str(runtime), '--deck', str(deck), '--case',
                       str((args.cases / mesh / date / 'init-modis20-v2').resolve()),
                       '--out', str(output), '--tables', str(args.tables.resolve()),
                       '--mpi', str(args.mpi.resolve()), '--ranks', '1',
                       '--pin-set-hash', args.pin_set]
            with (args.out / (name + '.log')).open('w') as handle:
                status = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                        check=False).returncode
            stdout = (output / 'native.stdout.log').read_text()
            log = (output / 'log.atmosphere.0000.out').read_text()
            row = dict(deck=name, process_count=1, host=platform.node(),
                       native_command=command, runner_exit=status,
                       completed='Finished running the atmosphere core' in log,
                       sigsegv='Segmentation fault' in stdout,
                       binary_sha256=sha(args.build / 'atmosphere_model'),
                       namelist_sha256=item['namelist_sha256'],
                       streams_sha256=item['streams_sha256'],
                       stdout=stdout, native_log_tail=log[-4000:],
                       outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                                for p in output.glob('*.nc') if not p.is_symlink()},
                       numerical_identity_claim=False, exclusions=[])
            proof.write_text(json.dumps(row, indent=2) + '\n')
            # Retire only this invocation's disposable fields, after the receipt.
            if not row['completed']:
                for filename in row['outputs']:
                    (output / filename).unlink()
        rows.append(row)
        args.receipt.write_text(json.dumps(dict(
            kind='strict native fractional-false whole-deck measurements; no CUDA tier',
            process_count=1, runs=rows), indent=2) + '\n')
        print(json.dumps({key: row[key] for key in
                          ('deck', 'runner_exit', 'completed', 'sigsegv')}), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'runner', 'build', 'cases', 'tables', 'mpi', 'out', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pin-set', required=True)
    parser.add_argument('--auxiliary', type=Path, action='append', default=[])
    raise SystemExit(run(parser.parse_args()))
