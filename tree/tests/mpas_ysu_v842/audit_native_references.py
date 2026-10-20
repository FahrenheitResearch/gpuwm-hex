"""Inventory completed native references without rerunning their forecasts.

Python checks receipts and schedules the Rust NetCDF stored-word reader.
Native references are prerequisites, never proof of a hex forecast.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, record):
    text = json.dumps(record, indent=2) + '\n'
    if len(text.encode()) >= 10_000_000:
        raise RuntimeError(f'Receipt exceeds delivery limit: {path}')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--outputs', type=Path)
    parser.add_argument('--index', type=Path)
    args = parser.parse_args()
    root = args.work.resolve()
    helper = args.helper.resolve()
    target = args.receipts.resolve()
    target.mkdir(parents=True, exist_ok=True)
    index = args.index.resolve() if args.index else root / 'native-acceptance/index.json'
    summary = {'schema': 'p11-native-reference-inventory/v1',
               'scope': 'native reference preparation, not hex acceptance',
               'host': platform.node(),
               'index_sha256': sha(index), 'helper_sha256': sha(helper),
               'entries': [], 'pending': []}
    for entry in json.loads(index.read_text())['entries']:
        output = (args.outputs.resolve() if args.outputs else root / 'native-acceptance/outputs') / entry['name']
        manifest = output / 'MANIFEST.json'
        if not manifest.is_file():
            summary['pending'].append(entry['name'])
            continue
        native = json.loads(manifest.read_text())
        assert native['ranks'] == entry['ranks']
        assert native['binary_sha256'] == sha(root / 'native-defects/build/atmosphere_model')
        for name, key in [('namelist.atmosphere', 'namelist_sha256'),
                          ('streams.atmosphere', 'streams_sha256')]:
            assert sha(output / name) == native[key] == sha(root / entry['deck'] / name)
        assert 'Finished running the atmosphere core' in (output / 'log.atmosphere.0000.out').read_text()
        if any(row.startswith('ERROR:') for row in (output / 'log.atmosphere.0000.out').read_text().splitlines()):
            raise RuntimeError('Native reference reported errors; do not inventory failed output as an oracle')
        start = datetime.strptime(entry['date'], '%Y-%m-%d_%H.%M.%S')
        end = start + timedelta(hours=entry['hours'])
        intervals = {}
        for declaration in ET.parse(output / 'streams.atmosphere').getroot():
            interval = declaration.get('output_interval', 'none')
            if 'output' not in declaration.get('type', '') or interval == 'none':
                continue
            days, clock = interval.split('_')
            hours, minutes, seconds = map(int, clock.split(':'))
            cadence = timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds)
            assert cadence.total_seconds() > 0
            prefix = declaration.attrib['filename_template'].split('.', 1)[0]
            intervals[prefix] = cadence
        streams = {}
        fields = 0
        words = 0
        for filename, digest in sorted(native['outputs'].items()):
            path = output / filename
            assert sha(path) == digest
            result = subprocess.run([str(helper), '--field-digests', str(path)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Rust native inventory failed: ' + result.stderr)
            inventory = json.loads(result.stdout)
            assert inventory['variables']
            stream, filename_clock = filename[:-3].split('.', 1)
            file_date = datetime.strptime(filename_clock, '%Y-%m-%d_%H.%M.%S')
            dates = [datetime.strptime(value, '%Y-%m-%d_%H:%M:%S')
                     for value in inventory['xtime']]
            # Native surface stream does not write xtime. Preserve its empty
            # stored clock; explicitly identify the filename clock instead.
            clock_source = 'stored xtime' if dates else 'native output filename'
            if dates:
                assert dates == [file_date], f'Stored clock/filename differ: {filename}'
            else:
                dates = [file_date]
            assert all(start <= clock <= end for clock in dates)
            assert stream
            streams.setdefault(stream, []).extend(date.strftime('%Y-%m-%d_%H:%M:%S') for date in dates)
            fields += len(inventory['variables'])
            words += sum(field['words'] for field in inventory['variables'])
            write(target / (entry['name'] + '--' + filename + '.json'), {
                'scope': summary['scope'], 'host': summary['host'], 'ranks': entry['ranks'],
                'native_manifest_sha256': sha(manifest), 'file_sha256': digest,
                'helper_sha256': summary['helper_sha256'], 'clock_source': clock_source,
                'filename_clock': filename_clock, **inventory})
        clocks = {}
        for stream, values in sorted(streams.items()):
            dates = sorted(datetime.strptime(value, '%Y-%m-%d_%H:%M:%S') for value in values)
            cadence = intervals[stream]
            assert dates[-1] == end, f'{stream} does not reach forecast end'
            assert len(dates) == len(set(dates)), f'Duplicate clock in {stream}'
            assert dates[0] in (start, start + cadence), f'Initial stream clock differs: {stream}'
            assert all(right - left == cadence
                       for left, right in zip(dates, dates[1:])), f'Stream cadence differs: {stream}'
            clocks[stream] = {'samples': len(dates), 'first': values[0], 'last': values[-1],
                              'interval_seconds': int(cadence.total_seconds())}
        summary['entries'].append({**entry, 'native_manifest_sha256': sha(manifest),
                                   'binary_sha256': native['binary_sha256'],
                                   'output_files': len(native['outputs']),
                                   'field_samples': fields, 'stored_words': words,
                                   'clocks': clocks})
        print(json.dumps({'inventoried': entry['name'], 'ranks': entry['ranks'],
                          'output_files': len(native['outputs']), 'stored_words': words}), flush=True)
        write(target / 'SUMMARY.json', summary)
    write(target / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
