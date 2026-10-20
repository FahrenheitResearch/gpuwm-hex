"""Replay a complete native capture matrix into isolated gcov counters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dump', type=Path, action='append', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--legacy-process-authority', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve()
    shutil.copyfile(args.notes, root / 'sf_mynn.gcno')
    shutil.copyfile(args.source, root / 'sf_mynn.F90')
    env = dict(os.environ, GCOV_PREFIX=str(root), GCOV_PREFIX_STRIP='6',
               MYNN_REPLAY_BACKEND='host', MYNN_NATIVE_LIBRARY=str(args.library.resolve()),
               MYNN_ORACLE_RECORD=str(Path('real-mynn/oracle_record.so').resolve()))
    parts = []
    authority = json.loads(args.legacy_process_authority.read_text()) if args.legacy_process_authority else None
    if authority:
        for entry in authority['sources']:
            assert sha(Path(entry['source'])) == entry['sha256']
            assert entry['native_process_count'] == 1
    for index, dump in enumerate(args.dump):
        receipt = root / f'replay-{index}.json'
        command = [sys.executable, 'real-mynn/column_oracle_replay.py', '--scheme', 'sf_mynn',
                   '--backend', 'native-replay', '--dump', str(dump),
                   '--executable', 'hex/tree/tests/mpas_sf_mynn/replay_capture.py',
                   '--words', 'real-mynn/oracle_words', '--results', str(root / f'results-{index}'),
                   '--receipt', str(receipt), '--recipe', 'gnu-strict', '--process-count', '1']
        subprocess.run(command, env=env, check=True)
        result = json.loads(receipt.read_text())
        assert result['comparison_complete'] and result['all_outputs_bitwise_equal']
        assert result['replay_process_count'] == 1
        if result['capture_process_count'] is None:
            assert authority and result['dump_root_sha256'] in authority['capture_roots']
            assert authority['capture_roots'][result['dump_root_sha256']] == 1
        else:
            assert result['capture_process_count'] == 1
        parts.append(dict(dump=str(dump), root_sha256=result['dump_root_sha256'],
                          calls=len(result['captures']), differing_words=0,
                          words=sum(a['words'] for c in result['captures']
                                    for a in c['arrays'].values()), receipt_sha256=sha(receipt)))
    coverage = subprocess.run(['gcov', '-b', '-c', 'sf_mynn.gcno'], cwd=root,
                              capture_output=True, text=True, check=True)
    assert not coverage.stderr, coverage.stderr
    (root / 'gcov-summary.txt').write_text(coverage.stdout)
    measures = {}
    for key in ('Lines executed', 'Branches executed', 'Taken at least once', 'Calls executed'):
        match = re.search(re.escape(key) + r':([\d.]+)% of (\d+)', coverage.stdout)
        assert match, coverage.stdout
        measures[key] = dict(percent=float(match[1]), denominator=int(match[2]))
    args.receipt.write_text(json.dumps(dict(
        schema='p10-mynn-complete-real-native-coverage-v1', parts=parts,
        process_count=1, total_calls=sum(p['calls'] for p in parts),
        total_words=sum(p['words'] for p in parts), differing_words=0,
        library_sha256=sha(args.library), notes_sha256=sha(args.notes),
        source_sha256=sha(args.source), isolated_gcov_prefix=str(root),
        gcov_prefix_strip=6, coverage=measures, whole_forecast_proven=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
