"""Authenticate the complete cold-core banks and launch production Rust replay.

MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1268-1309,1481-1584
at 1d6e370e. Python hashes files and launches; Rust reads and compares words.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def authenticate(contract, lane, stage):
    if contract['source_commit'] != '1d6e370e7b43a332074809f170532a216c2212fb':
        raise ValueError('cold-core contract is outside the pinned fork')
    for row in contract['rust_sources']:
        path = stage / row['staged']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('cold-core bridge source changed: ' + str(path))
    for case in contract['cases']:
        bank = lane / case['bank']
        actual = {path.name for path in bank.iterdir()}
        expected = {row['name'] for row in case['records']}
        if actual != expected:
            raise ValueError('cold-core record inventory changed: ' + str(bank))
        for row in case['records']:
            path = bank / row['name']
            if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('cold-core native record changed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    authenticate(contract, args.lane, args.stage)
    if args.validate_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    compiled = args.out.parent / 'compiled'
    rows = []
    for case in contract['cases']:
        target = args.out / (case['case'] + '.json')
        subprocess.run([str(compiled / 'replay_cold_core'), str(args.lane / case['bank']),
                        str(compiled / 'rw_mpas_rrfs_connectivity'),
                        str(compiled / 'rw_mpas_rrfs_damping'), str(target)], check=True)
        row = json.loads(target.read_text())
        for key in ('complete_calls', 'post_records', 'compared_words'):
            if row[key] != case[key]:
                raise ValueError('cold-core output inventory changed: ' + case['case'])
        if not row['passed'] or row['different_words']:
            raise ValueError('cold-core native words differ: ' + case['case'])
        rows.append(row)
    summary = dict(schema=contract['schema'], source_commit=contract['source_commit'],
                   rust_dependency_commit=contract['gpuwm_commit'], passed=True,
                   different_words=0, complete_calls=sum(r['complete_calls'] for r in rows),
                   post_records=sum(r['post_records'] for r in rows),
                   compared_words=sum(r['compared_words'] for r in rows),
                   scope='complete real cold pool boundaries; all getter fields, entry tails and dummy storage; reference-host MPI1 native controls')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
