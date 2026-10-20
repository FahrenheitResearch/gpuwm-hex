"""Compare complete owned ORCL capture sets using the real Rust word reader.

Python handles metadata and receipts only. This also records diagnostic native
build differences without treating their flags or outputs as acceptance inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def compare(reference, candidate, reader, receipt):
    names = sorted(p.name for p in candidate.glob('*.post.*.bin'))
    expected = sorted(p.name for p in reference.glob('*.post.*.bin'))
    if not names or names != expected:
        raise ValueError('Complete POST capture sets differ')
    rows = []
    for name in names:
        left, right = reference/name, candidate/name
        result = subprocess.run([str(reader.resolve()), 'compare', str(left), str(right)],
                                check=True, capture_output=True, text=True)
        row = json.loads(result.stdout)
        if row.get('different_words') is None:
            raise ValueError('ORCL capture metadata differs: '+name)
        row.update(file=name,
                   reference_sha256=hashlib.sha256(left.read_bytes()).hexdigest(),
                   candidate_sha256=hashlib.sha256(right.read_bytes()).hexdigest())
        rows.append(row)
    result = dict(scope='complete ORCL POST capture comparison; no forecast or exclusion claim',
                  process_count=1, excluded_words=0, complete=True,
                  reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),
                  arrays=len(rows), words=sum(r['words'] for r in rows),
                  different_words=sum(r['different_words'] for r in rows), records=rows)
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
    return bool(result['different_words'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'reader', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(compare(args.reference, args.candidate, args.reader, args.receipt))
