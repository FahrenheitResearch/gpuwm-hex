"""Measure original native forecast references on the CUDA proof host.

MPAS mpas_atmphys_driver_seaice.F:160-558 defines the subsystem boundary.
The unchanged published executable processes weather; sealed Rust compares
all stored types. Python handles metadata, paths and byte hashes only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seal(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def require_complete_comparison(comparison, previous):
    """Reject lost fields, wrong schedules, incompatible types or any bit."""
    rows = comparison['fields']
    key = lambda row: tuple(row[name] for name in ('stream', 'step', 'time', 'field'))
    expected = {key(row) for row in previous['comparison']['fields']}
    actual = {key(row) for row in rows}
    if not rows or len(actual) != len(rows) or actual != expected:
        raise ValueError('Native host comparison changed the full field/time inventory')
    if any(row['status'] != 'compared' or row['differing_words'] != 0 for row in rows):
        raise ValueError('Native host comparison has missing, incompatible or differing words')
    return sum(row['compared_words'] for row in rows)


def run(args):
    if platform.node() != args.host:
        raise ValueError('Native host closure must execute on the stated CUDA proof host')
    index = json.loads(args.index.read_text())
    def pinned(ref):
        path = Path(ref['path'])
        if digest(path) != ref['sha256']:
            raise ValueError('Changed pinned input: ' + str(path))
        return path
    comparator = pinned(index['comparator'])
    args.work.mkdir(parents=True, exist_ok=args.resume)
    args.receipts.mkdir(parents=True, exist_ok=args.resume)
    empty_tables = args.work / 'empty-extra-tables'
    empty_tables.mkdir(exist_ok=args.resume)
    records = []
    for number in args.number:
        ref = index['decks'][number]
        deck = json.loads(pinned(ref).read_text())
        original = pinned(ref['native_publication']).parent
        publication = json.loads((original / 'PUBLICATION.json').read_text())
        manifest_path = original / 'MANIFEST.json'
        assert digest(manifest_path) == publication['files']['MANIFEST.json']
        manifest = json.loads(manifest_path.read_text())
        assert manifest['ranks'] == deck['process_count'] == 1
        previous = json.loads(pinned(ref['inventory_receipt']).read_text())
        clocks = previous['provenance']['native_stream_clocks']
        assert {row['path'] for row in clocks} == set(manifest['outputs'])
        base = args.work / f'case-{number:02d}'
        runtime, inputs, output = base / 'runtime', base / 'deck', base / 'output'
        finished_native = args.resume and (output / 'MANIFEST.json').is_file()
        if not finished_native:
            runtime.mkdir(parents=True)
            inputs.mkdir()
        for key, filename in [('namelist', 'namelist.atmosphere'), ('streams', 'streams.atmosphere')]:
            source = pinned(deck['inputs'][key])
            assert digest(source) == manifest['inputs'][filename]
            if finished_native:
                assert digest(inputs / filename) == digest(source)
            else:
                shutil.copyfile(source, inputs / filename)
        init = pinned(deck['inputs']['init'])
        for name, pin in deck['auxiliary_inputs'].items():
            source = pinned(pin)
            assert pin['sha256'] == manifest['inputs'][name]
            if not finished_native and source.parent.resolve() != init.parent.resolve():
                (runtime / name).symlink_to(source.resolve())
        assert digest(args.binary) == manifest['binary_sha256']
        if not finished_native:
            (runtime / 'atmosphere_model').symlink_to(args.binary.resolve())
        command = [sys.executable, str(args.runner.resolve()), '--build', str(runtime.resolve()),
                   '--deck', str(inputs.resolve()), '--case', str(init.parent.resolve()),
                   '--out', str(output.resolve()), '--tables', str(empty_tables.resolve()),
                   '--mpi', str(args.mpi.resolve()), '--ranks', '1',
                   '--pin-set-hash', manifest['pin_set_hash']]
        log = args.receipts / f'native-{number:02d}.log'
        record = dict(schema='p18-native-host-closure/v1', deck=ref,
                      original_manifest=seal(manifest_path), host=platform.node(),
                      process_count=1, command=command, binary=seal(args.binary),
                      mpi=seal(args.mpi), runner=seal(args.runner), complete=False)
        attempt = args.receipts / f'native-{number:02d}.attempt.json'
        if finished_native:
            prior_attempt = json.loads(attempt.read_text())
            for name in ('host', 'process_count', 'command', 'binary', 'mpi', 'runner', 'original_manifest'):
                assert prior_attempt[name] == record[name], 'Changed completed native execution provenance'
            assert prior_attempt['log'] == seal(log)
            record = prior_attempt
        else:
            attempt.write_text(json.dumps(record, indent=2)+'\n')
            while next(int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()
                       if row.startswith('MemAvailable:')) < 16 * 1024 * 1024:
                time.sleep(10)
            print(json.dumps(dict(start=number, host=platform.node(), command=command)), flush=True)
            with log.open('w') as handle:
                record['exit_code'] = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
            record['log'] = seal(log)
            attempt.write_text(json.dumps(record, indent=2)+'\n')
        assert record['exit_code'] == 0, record
        actual_path = output / 'MANIFEST.json'
        actual = json.loads(actual_path.read_text())
        assert actual['ranks'] == 1 and actual['binary_sha256'] == manifest['binary_sha256']
        assert actual['inputs'] == manifest['inputs'], 'Native closure changed runtime input bytes'
        assert set(actual['outputs']) == set(manifest['outputs']), 'Native closure changed stream inventory'
        model_log = output / 'log.atmosphere.0000.out'
        assert 'Finished running the atmosphere core' in model_log.read_text()
        samples = []
        for clock in clocks:
            name = clock['path']
            assert digest(original / name) == manifest['outputs'][name]
            assert digest(output / name) == actual['outputs'][name]
            sample = dict(stream=clock['stream'], steps=clock['steps'], times=clock['times'],
                          strict=str(original / name), hex=str(output.resolve() / name))
            if 'clock_witness' in clock:
                sample['clock_witness'] = clock['clock_witness']
            samples.append(sample)
        request = args.receipts / f'native-{number:02d}.request.json'
        comparison_path = args.receipts / f'native-{number:02d}.words.json'
        request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1', samples=samples), indent=2)+'\n')
        subprocess.run([str(comparator), '--request', str(request), '--out', str(comparison_path)], check=True)
        comparison = json.loads(comparison_path.read_text())
        words = require_complete_comparison(comparison, previous)
        record.update(complete=True, actual_manifest=seal(actual_path), model_log=seal(model_log),
                      request=seal(request), comparison=seal(comparison_path), comparator=seal(comparator),
                      files=len(samples), compared_words=words, differing_words=0,
                      CUDA_acceptance_inferred=False, global_host_a_scope_extended=False)
        (args.receipts / f'native-{number:02d}.closure.json').write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
        args.receipt.write_text(json.dumps(dict(schema='p18-native-host-closure-matrix/v1',
            host=platform.node(), process_count=1, runs=records,
            complete=len(records)==len(args.number), CUDA_acceptance_inferred=False), indent=2)+'\n')
        print(json.dumps(dict(complete=number, words=words, differing_words=0)), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'work', 'receipts', 'receipt', 'runner', 'binary', 'mpi'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--number', type=int, action='append', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Compare authenticated completed native output before running unfinished decks')
    raise SystemExit(run(parser.parse_args()))
