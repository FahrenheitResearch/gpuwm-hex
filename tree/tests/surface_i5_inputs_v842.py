"""Prepare authentic i5 Rust inputs and compare every stored word in Rust.

This is input preparation, not execution of the shipping hex surface door or
the coupled forecast. Python handles commands, pins, and clock metadata only.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(engine, engine_sha256, comparator, decks, mesh, output, receipt):
    """mpas_init_atm_surface.F:29-101,104-394; Registry.xml:1161-1171.

    Invoke the existing i5 engine; native case-8 bytes remain read-only. Use
    the real Rust stream comparator for fields, primitive types and clocks.
    """
    if digest(engine) != engine_sha256:
        raise SystemExit('The i5 engine differs from its declared artifact pin.')
    abi = subprocess.run([str(engine), '--mode', 'surface-update', '--abi'],
                         capture_output=True, text=True, check=True).stdout.strip()
    if not abi.startswith('rw_mpas_init --mode surface-update --sfc-prefix PREFIX'):
        raise SystemExit('The i5 engine does not expose its surface-update ABI.')
    rows = [row for row in json.loads(decks.read_text())
            if row['label'].split('/')[0] == mesh]
    if not rows or len({row['start'] for row in rows}) != len(rows):
        raise SystemExit('Input decks must have distinct clocks for the requested mesh.')
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for number, row in enumerate(rows):
        directory = output / str(number)
        directory.mkdir()
        candidate = directory / 'sfc_update.nc'
        engine_receipt = directory / 'engine.json'
        native = Path(row['published_native'])
        static = Path(row['static_path'])
        submitted = {'native': digest(native), 'static': digest(static)}
        start = datetime.strptime(row['start'], '%Y-%m-%d_%H:%M:%S')
        stop = datetime.strptime(row['stop'], '%Y-%m-%d_%H:%M:%S')
        elapsed = int((stop - start).total_seconds())
        if elapsed < 0 or elapsed % row['interval']:
            raise SystemExit('Surface clock is not an inclusive integral record sequence.')
        clocks = [start + timedelta(seconds=offset)
                  for offset in range(0, elapsed + 1, row['interval'])]
        times = [now.strftime('%Y-%m-%d_%H:%M:%S') for now in clocks]
        intermediates = {str(Path(row['sfc_prefix'] + now.strftime(':%Y-%m-%d_%H'))):
                         digest(Path(row['sfc_prefix'] + now.strftime(':%Y-%m-%d_%H')))
                         for now in clocks}
        command = [str(engine), '--mode', 'surface-update', '--sfc-prefix', row['sfc_prefix'],
                   '--static', str(static), '--out', str(candidate),
                   '--start-time', row['start'], '--stop-time', row['stop'],
                   '--fg-interval', str(row['interval']),
                   '--surface-interval', str(row['interval']), '--receipt', str(engine_receipt)]
        executed = subprocess.run(command, capture_output=True, text=True)
        (directory / 'engine.log').write_text(executed.stdout + executed.stderr)
        executed.check_returncode()
        decoded = json.loads(engine_receipt.read_text())
        if (decoded['n_cells'] <= 0 or [record['xtime'] for record in decoded['records']] != times
            or {record['intermediate'] for record in decoded['records']} != set(intermediates)):
            raise SystemExit('The engine consumed different intermediate files or clocks.')
        described = subprocess.run([str(comparator), '--describe', str(native)],
                                   capture_output=True, text=True, check=True)
        metadata = json.loads(described.stdout)
        if metadata['times'] != times or metadata['records'] != len(times):
            raise SystemExit('The native case-8 tape differs from its deck clocks.')
        request = directory / 'comparison-request.json'
        request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1', samples=[dict(
            stream='surface', steps=list(range(len(times))), times=times,
            strict=str(native), hex=str(candidate))]), indent=2) + '\n')
        compared = directory / 'comparison.json'
        subprocess.run([str(comparator), '--request', str(request), '--out', str(compared)], check=True)
        fields = json.loads(compared.read_text())['fields']
        if not fields or any(item['status'] != 'compared' or item['differing_words'] != 0 for item in fields):
            raise SystemExit('The i5 input differs from native; retain the full Rust comparison.')
        if digest(native) != submitted['native'] or digest(static) != submitted['static']:
            raise SystemExit('An immutable native input changed during preparation.')
        if any(digest(Path(path)) != sha for path, sha in intermediates.items()):
            raise SystemExit('An immutable meteorological input changed during preparation.')
        results.append(dict(label=row['label'], start=row['start'], stop=row['stop'],
            interval_seconds=row['interval'], records=len(times), fields=metadata['fields'],
            native_file=str(native), native_sha256=submitted['native'],
            static_file=str(static), static_sha256=submitted['static'],
            rust_file=str(candidate), rust_sha256=digest(candidate),
            comparison=str(compared), comparison_sha256=digest(compared),
            stored_words=sum(item['compared_words'] for item in fields), differing_words=0,
            engine_receipt=str(engine_receipt), engine_receipt_sha256=digest(engine_receipt),
            intermediate_sha256=intermediates, n_cells=decoded['n_cells'],
            command=command, actual_engine_exit=executed.returncode, process_count=1))
    result = dict(schema='p25-i5-input-preparation/v1', process_count=1,
        tier='Rust input preparation and exact native-file substitution; no hex forecast claim',
        shipping_hex_surface_door_executed=False, hex_forecast_identity_claimed=False,
        engine=str(engine), engine_sha256=engine_sha256, comparator_sha256=digest(comparator),
        source_decks=str(decks), source_decks_sha256=digest(decks), rows=results)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(inputs=len(results), stored_words=sum(row['stored_words'] for row in results),
                          differing_words=0, hex_forecast_identity_claimed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'comparator', 'decks', 'output', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--engine-sha256', required=True)
    parser.add_argument('--mesh', required=True)
    arguments = parser.parse_args()
    prepare(arguments.engine.resolve(), arguments.engine_sha256, arguments.comparator.resolve(),
            arguments.decks.resolve(), arguments.mesh, arguments.output.resolve(), arguments.receipt.resolve())
