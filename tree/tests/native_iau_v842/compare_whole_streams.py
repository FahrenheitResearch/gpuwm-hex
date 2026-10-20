"""Run f2's Rust stored-word comparison for a complete controlled experiment.

This helper orchestrates metadata only. Diagnostic comparisons never establish
acceptance, including a zero-word result against controlled increment forcing.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
for name in ('clocks', 'streams', 'strict', 'candidate', 'comparator', 'out'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--start', required=True)
parser.add_argument('--dt', type=int, required=True)
parser.add_argument('--summarize-only', action='store_true',
                    help='Summarize an existing authentic Rust request/result without changing it')
args = parser.parse_args()
if args.dt <= 0:
    parser.error('dt must be positive')
args.out.mkdir(parents=True, exist_ok=args.summarize_only)
clocks = json.loads(args.clocks.read_text())
templates = {s.attrib['name']: s.attrib['filename_template']
             for s in ET.parse(args.streams).getroot()}
start = datetime.strptime(args.start, '%Y-%m-%d_%H:%M:%S')
samples = []
for name, record in sorted(clocks['files'].items()):
    times = record['expected_clocks']
    steps = []
    for stamp in times:
        seconds = int((datetime.strptime(stamp, '%Y-%m-%d_%H:%M:%S') - start).total_seconds())
        if seconds < 0 or seconds % args.dt:
            raise SystemExit('native output clock is not an outer step: ' + name)
        steps.append(seconds // args.dt)
    stream = name.split('.', 1)[0]
    samples.append(dict(stream=stream, steps=steps, times=times,
        strict=str((args.strict / name).resolve()),
        hex=str((args.candidate / name).resolve()),
        clock_witness=dict(filename_template=templates[stream])))
request = args.out / 'request.json'
raw = args.out / 'comparison.json'
if args.summarize_only:
    recorded = json.loads(request.read_text())['samples']
    keys = ('stream', 'steps', 'times')
    if [{k: row[k] for k in keys} for row in recorded] != [
            {k: row[k] for k in keys} for row in samples]:
        raise SystemExit('stored comparison clock inventory differs')
else:
    request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1', samples=samples), indent=2) + '\n')
    subprocess.run([str(args.comparator), '--request', str(request), '--out', str(raw)], check=True)
rows = json.loads(raw.read_text())['fields']
inventory = defaultdict(lambda: dict(samples=0, requested_words=0, compared_words=0,
                                     differing_words=0, noncompared_words=0,
                                     charged_differing_words=0, statuses={}))
for row in rows:
    entry = inventory[(row['stream'], row['field'])]
    entry['samples'] += 1
    entry['requested_words'] += row['compared_words']
    entry['charged_differing_words'] += row['differing_words']
    if row['status'] == 'compared':
        entry['compared_words'] += row['compared_words']
        entry['differing_words'] += row['differing_words']
    else:
        entry['noncompared_words'] += row['compared_words']
    status = row['status']
    entry['statuses'][status] = entry['statuses'].get(status, 0) + 1
expected = {Path(s['hex']) for s in samples}
extras = sorted(str(p) for p in args.candidate.glob('*.nc')
                if p.resolve() not in expected and not p.is_symlink())
summary = dict(schema='mpas.iau-controlled-whole-comparison/v1',
    scope='diagnostic controlled forcing; no landing acceptance', acceptance=False,
    process_count=1, requested_files=len(samples), field_samples=len(rows),
    requested_words=sum(r['compared_words'] for r in rows),
    compared_words=sum(r['compared_words'] for r in rows if r['status'] == 'compared'),
    differing_words=sum(r['differing_words'] for r in rows if r['status'] == 'compared'),
    noncompared_words=sum(r['compared_words'] for r in rows if r['status'] != 'compared'),
    charged_differing_words=sum(r['differing_words'] for r in rows),
    statuses={status: sum(r['status'] == status for r in rows)
              for status in sorted({r['status'] for r in rows})}, extra_files=extras,
    clocks_sha256=hashlib.sha256(args.clocks.read_bytes()).hexdigest(),
    comparator_sha256=hashlib.sha256(args.comparator.read_bytes()).hexdigest(),
    fields=[dict(stream=s, field=f, **v) for (s, f), v in sorted(inventory.items())])
(args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'fields'}))
