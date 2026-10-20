"""Union branch identities from same-source, same-topology gcov receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def merge(inputs, output):
    rows = [json.loads(p.read_text()) for p in inputs]
    if len({(r['authority_sha256'], r['branches'], r['executable_lines']) for r in rows}) != 1:
        raise ValueError('coverage source or branch topology differs')
    key = lambda b: (b['routine'], b['line'], b['branch'], b['source'])
    misses = [{key(b): b for b in r['uncovered_branches']} for r in rows]
    common = set(misses[0]).intersection(*(set(m) for m in misses[1:]))
    result = dict(authority_sha256=rows[0]['authority_sha256'], branches=rows[0]['branches'],
                  branches_taken=rows[0]['branches']-len(common),
                  routines_reached=sorted({name for r in rows for name, n in r['routine_calls'].items() if n}),
                  uncovered_branches=[misses[0][k] for k in sorted(common)],
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  scope='Union of gcov branch outcomes; input call counts overlap and are not added; no forecast proof')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('uncovered_branches', 'inputs')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    merge(a.input, a.receipt)
