#!/usr/bin/env python3
"""Combine authenticated gcov observations without merging distinct builds.

This processes coverage metadata only. Native line mapping excludes oracle
instrumentation; branch identity includes its enclosing native function.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

if __package__:
    from .coverage import summarize
else:
    from coverage import summarize


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def merge(reports):
    groups = defaultdict(list)
    counts = set()
    for report in reports:
        count = report['process_count']
        if type(count) is not int or count < 1:
            raise ValueError('coverage matrix needs a stated positive process count')
        counts.add(count)
        for source in report['files']:
            key = (source['native_source'], source['source_sha256'],
                   source['instrumented_source_sha256'], report['notes_sha256'],
                   report['gcc_version'])
            groups[key].append((report, source))
    if len(counts) != 1:
        raise ValueError('coverage matrix cannot combine different process counts')
    results = []
    for key, entries in sorted(groups.items()):
        universe = None
        observed, line_observed = set(), set()
        witnesses = []
        for report, source in entries:
            branches, lines = set(), set()
            occurrence = defaultdict(int)
            for row in source['line_counters']:
                slot = (row['function'], row['line'])
                number = occurrence[slot]
                occurrence[slot] += 1
                line = (*slot, number)
                lines.add(line)
                if row['count'] > 0:
                    line_observed.add(line)
                for ordinal, branch in enumerate(row['branches']):
                    if 'source_block_id' not in branch or 'destination_block_id' not in branch:
                        # THE BREAKAGE THIS PREVENTS (release 0.4.0 lane tail, B8, 2026-10-14):
                        # gcov writes branch block ids only from GCC 14 (JSON format 2).  A
                        # GCC 12 or 13 report has none, and the merge died with an unnamed
                        # KeyError on the public CPU tier's gfortran.
                        raise ValueError(
                            'coverage matrix needs gcov branch block ids, which GCC 14 and later '
                            f"write (this report: gcc {report['gcc_version']}); without them two "
                            'branches of one line cannot be told apart')
                    identity = (*line, ordinal, branch['source_block_id'],
                                branch['destination_block_id'], branch['fallthrough'],
                                branch['throw'])
                    branches.add(identity)
                    if branch['count'] > 0:
                        observed.add(identity)
            current = (branches, lines)
            if universe is not None and universe != current:
                raise ValueError('same gcno binding has changed native counter topology')
            universe = current
            witnesses.append(report['witness'])
        branches, lines = universe
        results.append(dict(native_source=key[0], source_sha256=key[1],
            instrumented_source_sha256=key[2], notes_sha256=key[3], gcc_version=key[4],
            captures=len(entries), branches_total=len(branches), branches_taken=len(observed),
            lines_total=len(lines), lines_executed=len(line_observed),
            uncovered_branches=[dict(function=b[0], line=b[1], occurrence=b[2],
                branch_ordinal=b[3], source_block_id=b[4], destination_block_id=b[5],
                fallthrough=b[6], throw=b[7])for b in sorted(branches-observed)],
            unmapped_native_lines=entries[0][1]['unmapped_native_lines'],
            allocation_control_mappings=entries[0][1].get('allocation_control_mappings',[]),
            witnesses=witnesses))
    unmapped=[dict(row,witness=report['witness']) for report in reports
              for row in report.get('unmapped_files',[])]
    return dict(schema='mpas-native-coverage-matrix-v1', process_count=counts.pop(),
        files=results, unmapped_files=unmapped,
        coverage_complete=bool(results) and not unmapped and all(r['branches_taken']==r['branches_total']
            and not r['unmapped_native_lines'] for r in results),
        forecast_proven=False, proof_bar=False,
        scope='Union of observed native branches within each exact source/gcno/compiler binding; independent builds remain separate.')


def load_collection(index, source, build):
    data = json.loads(index.read_text())
    if data.get('schema') != 'mpas-native-coverage-collection-v1':
        raise ValueError('native coverage collection required')
    count = data['process_count']
    for execution in data['executions']:
        path = index.parent.parent / Path(execution['receipt']).name
        if sha(path) != execution['sha256']:
            raise ValueError('coverage replay witness changed')
        replay = json.loads(path.read_text())
        if (not replay.get('comparison_complete') or replay.get('replay_recipe') != 'coverage'
                or replay.get('replay_process_count') != count
                or replay['dump_root_sha256'] != execution['dump_root_sha256']
                or replay['executable_sha256'] != execution['executable_sha256']):
            raise ValueError('coverage replay binding differs from collection')
    result = []
    for member in data['files']:
        recorded = Path(member['receipt'])
        path = index.parent / recorded.parent.name / recorded.name
        if sha(path) != member['sha256']:
            raise ValueError('coverage source receipt changed')
        sealed = json.loads(path.read_text())
        if sealed['process_count'] != count or sealed['replay_process_counts'] != data['replay_process_counts']:
            raise ValueError('coverage source process-count binding changed')
        raw = path.with_name(Path(sealed['counter']).stem + '.gcov.json.gz')
        if sha(raw) != sealed['gcov_sha256']:
            raise ValueError('raw gcov metadata changed')
        observed = summarize(json.loads(gzip.decompress(raw.read_bytes())), source, build,
                             Path(sealed['compiled_build_root']), include_counters=True,
                             retain_unmapped_sources=sealed.get('retain_unmapped_sources',False))
        # Recompute the original complete native mapping before adding counters.
        plain = dict(observed, files=[{k:v for k,v in row.items() if k!='line_counters'}
                                     for row in observed['files']])
        if any(plain[k] != sealed[k] for k in ('gcc_version','files','unmapped_files')):
            raise ValueError('native mapping changed from sealed coverage receipt')
        # Authenticate the original sealed mapping first, then recover only
        # the source-verified native conditional allocation counters.
        observed = summarize(json.loads(gzip.decompress(raw.read_bytes())), source, build,
                             Path(sealed['compiled_build_root']), include_counters=True,
                             map_allocation_controls=True,
                             retain_unmapped_sources=sealed.get('retain_unmapped_sources',False))
        observed.update(process_count=count, notes_sha256=sealed['notes_sha256'],
            witness=dict(collection=str(index), collection_sha256=sha(index),
                         receipt=str(path), sha256=sha(path), gcov_sha256=sha(raw)))
        result.append(observed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, action='append', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports = [r for p in args.collection for r in load_collection(p,args.source_root,args.build_root)]
    payload = (json.dumps(merge(reports), indent=2)+'\n').encode()
    if len(payload) >= 10_000_000:
        raise ValueError('coverage matrix receipt exceeds delivery limit')
    args.out.write_bytes(payload)


if __name__ == '__main__':
    main()
