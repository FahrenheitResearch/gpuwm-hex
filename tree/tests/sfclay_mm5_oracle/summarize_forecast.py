"""Summarize actual verifier evidence without counting missing fields as arithmetic.

The original receipt remains authoritative and must be retained. This script
reads comparison JSON only; weather file transport stays in the Rust comparator.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('receipt', type=Path)
parser.add_argument('--log', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
raw = args.receipt.read_bytes()
receipt = json.loads(raw)
assert receipt['schema'] == 'mpas-verification-v1'
rows = receipt.get('comparison', {}).get('fields', [])
streams, fields = {}, {}
first = None
for row in sorted(rows, key=lambda r: (r['step'], r['stream'], r['field'])):
    missing = 'missing' in row['status'].lower()
    inventory_failure = row['status'].lower().startswith('extra')
    for key, destination in ((row['stream'], streams),
                             (row['stream'] + '/' + row['field'], fields)):
        aggregate = destination.setdefault(key, dict(
            field_samples=0, missing_field_samples=0, missing_words=0,
            inventory_failure_samples=0,
            computed_field_samples=0, computed_words=0,
            computed_differing_words=0, first_computed_differing_step=None,
            max_computed_ulp=None, statuses=Counter()))
        aggregate['field_samples'] += 1
        aggregate['statuses'][row['status']] += 1
        if missing:
            aggregate['missing_field_samples'] += 1
            aggregate['missing_words'] += row['compared_words']
        elif inventory_failure:
            aggregate['inventory_failure_samples'] += 1
        else:
            aggregate['computed_field_samples'] += 1
            aggregate['computed_words'] += row['compared_words']
            aggregate['computed_differing_words'] += row['differing_words']
            if row['differing_words'] and aggregate['first_computed_differing_step'] is None:
                aggregate['first_computed_differing_step'] = row['step']
            if row.get('max_ulp') is not None:
                aggregate['max_computed_ulp'] = max(
                    aggregate['max_computed_ulp'] or 0, row['max_ulp'])
    if first is None and not missing and not inventory_failure and row['differing_words']:
        first = row
summary = dict(
    scope='Actual verifier evidence; missing output is separate from computed word differences',
    receipt=str(args.receipt.resolve()), receipt_sha256=hashlib.sha256(raw).hexdigest(),
    receipt_bytes=len(raw), passed=receipt['passed'], tier=receipt.get('tier'),
    evidence_kind=receipt.get('evidence_kind'), process_count=receipt['process_count'],
    wall_seconds=receipt['wall_seconds'], acceptance_failures=receipt.get('acceptance_failures', []),
    streams=streams, fields=fields, first_computed_difference=first,
    ratchet_written=False)
if args.log:
    log = args.log.read_bytes()
    summary['hex_log_sha256'] = hashlib.sha256(log).hexdigest()
    summary['hex_error_lines'] = [line for line in log.decode(errors='replace').splitlines()
                                if line.startswith('gpuwm-hex:')]
args.output.parent.mkdir(parents=True, exist_ok=True)
staging = args.output.with_name(args.output.name + '.writing')
staging.write_text(json.dumps(summary, indent=2) + '\n')
staging.replace(args.output)
print(json.dumps(dict(passed=summary['passed'], tier=summary['tier'],
    computed_words=sum(s['computed_words'] for s in streams.values()),
    computed_differing_words=sum(s['computed_differing_words'] for s in streams.values()),
    missing_field_samples=sum(s['missing_field_samples'] for s in streams.values()),
    inventory_failure_samples=sum(s['inventory_failure_samples'] for s in streams.values()),
    first_computed_difference=first)))
