"""Summarize every verify-mpas field without treating absent words as compared.

Consumes sealed comparator metadata only; no weather-field decoding or maths.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(path, receipt):
    data = json.loads(path.read_text())
    fields = data.get('comparison', {}).get('fields', [])
    aggregates = {}
    for row in fields:
        key = (row['stream'], row['field'])
        out = aggregates.setdefault(key, dict(stream=key[0], field=key[1],
            computed_words=0, computed_differing_words=0, missing_words=0,
            samples=0, statuses=Counter(), first_computed_differing_step=None))
        out['samples'] += 1
        out['statuses'][row['status']] += 1
        if row['status'] == 'compared':
            out['computed_words'] += row['compared_words']
            out['computed_differing_words'] += row['differing_words']
            if row['differing_words'] and out['first_computed_differing_step'] is None:
                out['first_computed_differing_step'] = row['step']
        elif 'missing' in row['status']:
            out['missing_words'] += row['compared_words']
    rows = [aggregates[key] for key in sorted(aggregates)]
    result = dict(schema='gf-whole-forecast-field-summary-v1',
        source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        deck=data['deck'], process_count=data['process_count'], evidence_kind=data['evidence_kind'],
        passed=data['passed'], samples=len(fields),
        computed_words=sum(r['computed_words'] for r in rows),
        computed_differing_words=sum(r['computed_differing_words'] for r in rows),
        missing_words=sum(r['missing_words'] for r in rows), fields=rows,
        scope='Every sealed stream/field; absence does not establish numerical equality')
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('receipt', type=Path)
    a = p.parse_args(); r = summarize(a.source, a.receipt)
    print(json.dumps({k: v for k, v in r.items() if k != 'fields'}))
