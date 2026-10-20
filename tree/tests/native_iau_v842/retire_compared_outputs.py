"""Receipt and retire owned, completed outputs after a Rust comparison.

Field decoding and stored-word hashing belong to f0's published Rust reader.
Native controls needed by another comparison must not be passed to this tool.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
for name in ('run', 'clocks', 'reader', 'receipt'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
work = Path(__file__).resolve().parents[4]
run = args.run.resolve()
if not run.is_relative_to(work):
    raise SystemExit('only owned lane output directories can be retired')
expected = set(json.loads(args.clocks.read_text())['files'])
files = {p.name: p for p in run.glob('*.nc') if not p.is_symlink()}
if set(files) - expected:
    raise SystemExit('unexpected output files could be proof inputs; preserve all raw files')
records = []
for name, path in sorted(files.items()):
    digest = subprocess.check_output(['sha256sum', str(path)], text=True).split()[0]
    inventory = json.loads(subprocess.check_output(
        [str(args.reader), '--field-digests', str(path)], text=True))
    records.append(dict(file=name, bytes=path.stat().st_size,
                        sha256=digest, inventory=inventory))
receipt = dict(schema='mpas.iau-compared-output-retirement/v1',
    scope='only owned completed and compared outputs, never reference archives',
    run=str(run), files=records, expected_files=len(expected),
    absent_files=sorted(expected - set(files)),
    bytes=sum(r['bytes'] for r in records))
encoded = json.dumps(receipt, indent=2) + '\n'
if len(encoded.encode()) >= 10_000_000:
    raise SystemExit('receipt exceeds delivery size rule; preserve outputs')
args.receipt.write_text(encoded)
for path in files.values():
    path.unlink()
print(json.dumps(dict(retired_files=len(records), retired_bytes=receipt['bytes'])))
