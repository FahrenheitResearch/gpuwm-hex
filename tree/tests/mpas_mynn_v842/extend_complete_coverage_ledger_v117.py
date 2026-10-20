"""Extend the entire authenticated native coverage union, retaining every gap.

Only source/coverage metadata is processed. The scientific Fortran remains
unchanged. Reached branches do not certify numerical identity, and failed
whole-driver executions stay charged in the separately sealed run receipts.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from merge_science_coverage import main as merge, sha, parse, FILES


def main(a):
    prior = json.loads(a.prior_union.read_text())
    ledger = copy.deepcopy(json.loads(a.classified.read_text()))
    assert sha(a.prior_union) == ledger['coverage_sha256']
    roots = []
    for name in FILES:
        for row in prior['files'][name]['receipts']:
            path = Path(row['path'])
            assert sha(path) == row['sha256'], path
            if path.parent not in roots:
                roots.append(path.parent)
    roots.extend(a.new_coverage)
    merge(SimpleNamespace(sources=a.sources, coverage=roots, output=a.output))
    union_path = a.output / 'science-coverage-union.json'
    union = json.loads(union_path.read_text())
    newly_reached, contradictions = [], []
    for gap in ledger['uncovered_edges']:
        key = (gap['line'], gap['branch'])
        witnesses = []
        for root in a.new_coverage:
            path = root / (gap['file'] + '.gcov')
            branches = parse(path)[1]
            if branches[key]['taken']:
                witnesses.append(dict(path=str(path), sha256=sha(path),
                                      branch=branches[key]))
        if witnesses:
            if gap['classification'].startswith('UNREACHABLE'):
                contradictions.append([gap['file'], *key])
            gap['prior_classification'] = gap['classification']
            gap['classification'] = 'TAKEN in additional complete native driver attempt'
            gap['coverage_witnesses'] = witnesses
            newly_reached.append([gap['file'], *key])
    ledger['counts'] = {name: sum(row['classification'].startswith(name)
                         for row in ledger['uncovered_edges'])
                        for name in ('OPEN', 'UNREACHABLE', 'TAKEN')}
    ledger['additional_coverage_audit'] = dict(scope=__doc__, process_count=1,
        prior_union_sha256=sha(a.prior_union), prior_ledger_sha256=sha(a.classified),
        complete_union_sha256=sha(union_path), original_gap_count=len(ledger['uncovered_edges']),
        newly_reached=newly_reached, prior_caller_contract_contradictions=contradictions,
        source_roots=[str(root) for root in a.sources],
        excluded_words=[], numerical_identity_proven=False, forecast_proven=False)
    ledger['coverage_sha256'] = sha(union_path)
    ledger['full_branch_coverage'] = not ledger['counts']['OPEN'] and not contradictions
    (a.output / 'classified-gaps.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(ledger['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-union', 'classified', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('sources', 'new-coverage'):
        parser.add_argument('--' + name, type=Path, nargs='+', required=True)
    main(parser.parse_args())
