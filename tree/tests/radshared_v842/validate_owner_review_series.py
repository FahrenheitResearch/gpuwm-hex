"""Check a whole unapplied review series against its original native evidence.

Transient metadata copies are used only for git apply --check. They are never
supplied to a forecast and are removed after validation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from owner_context_packets import load_packet, encoded, LIMIT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'qualification', 'baseline',
                 'work-root', 'scratch', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--original-packet', type=Path,
                        help='Optional independent single packet for a byte-identity check.')
    args = parser.parse_args()

    def local(path):
        return args.work_root / str(path).split('/p21-rad-shared/', 1)[1]

    packet = load_packet(args.packet, local)
    if args.original_packet is not None:
        assert encoded(packet) == args.original_packet.read_bytes()
    qualification = json.loads(args.qualification.read_text())
    assert qualification['packet_sha256'] == sha(args.packet)
    assert qualification['baseline_sha256'] == sha(args.baseline)
    assert qualification['proposal_only'] and not qualification['owner_approved']
    assert not qualification['forecast_acceptance']
    series_path = args.qualification.parent / 'owner-contexts-proposed-series.json'
    assert sha(series_path) == qualification['proposed_diff_series_sha256']
    series = json.loads(series_path.read_text())
    assert series['baseline_sha256'] == sha(args.baseline)
    assert series['proposal_only'] and not series['owner_approved']
    assert not series['usable_as_baseline']
    parts = series['ordered_parts']
    assert parts == qualification['ordered_proposed_diffs']
    contexts = {row['request_index']: row for row in packet['completed_contexts']}
    indices = [index for part in parts for index in part['request_indices']]
    assert indices == list(contexts)
    original = args.baseline.read_bytes()
    current = original
    proposed = json.loads(original)
    original_entries = copy.deepcopy(proposed['entries'])
    assert not args.receipt.exists()
    args.scratch.mkdir(parents=True, exist_ok=False)
    transient = args.scratch / 'receipts' / 'ratchet.json'
    transient.parent.mkdir()
    rows = []
    try:
        for part in parts:
            patch = args.qualification.parent / part['path']
            assert sha(patch) == part['sha256'] and patch.stat().st_size < LIMIT
            assert hashlib.sha256(current).hexdigest() == part['input_text_sha256']
            transient.write_bytes(current)
            run = subprocess.run(['git', 'apply', '--check', str(patch.resolve())],
                                 cwd=args.scratch.resolve(), capture_output=True, text=True)
            assert run.returncode == 0, run.stderr
            assert transient.read_bytes() == current
            for index in part['request_indices']:
                proposed['entries'].extend(contexts[index]['all_actual_aggregate_entries'])
            current = encoded(proposed)
            assert hashlib.sha256(current).hexdigest() == part['output_text_sha256']
            assert proposed['entries'][:len(original_entries)] == original_entries
            rows.append(dict(patch_sha256=sha(patch), request_indices=part['request_indices'],
                             git_apply_check_exit=run.returncode,
                             diagnostic_stdout=run.stdout, diagnostic_stderr=run.stderr))
    finally:
        if transient.exists():
            transient.unlink()
        transient.parent.rmdir()
        args.scratch.rmdir()
    assert args.baseline.read_bytes() == original
    value = dict(schema='p21-complete-owner-review-series-check/v1',
        completed_contexts=len(contexts), process_count=packet['process_count'],
        packet_sha256=sha(args.packet),
        original_packet_sha256=sha(args.original_packet) if args.original_packet else None,
        qualification_sha256=sha(args.qualification), series_sha256=sha(series_path),
        baseline_sha256=sha(args.baseline),
        lossless_original_packet=True if args.original_packet else None,
        original_owner_entries_unchanged=True, all_diffs_dry_checked=True,
        transient_review_copies_retired=True, baseline_supplied_to_forecast=False,
        proposal_only=True, owner_approved=False, usable_as_baseline=False,
        forecast_acceptance=False, rows=rows)
    args.receipt.write_bytes(encoded(value))
    print(json.dumps(dict(completed_contexts=len(contexts),
        patches=len(rows), all_diffs_dry_checked=True, forecast_acceptance=False)))


if __name__ == '__main__':
    main()
