"""Retain small original driver metadata before verified outputs are retired.

This observes owned live forecasts without executing CUDA or reading weather
arrays. An original verifier hash authenticates a snapshot when available;
unhashed metadata remains explicit supporting logs. A missed final snapshot
never becomes a complete trace or changes forecast acceptance.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def pin(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_rows(packet, plan, actual):
    """Close only this collection's authenticated requests after a retry.

    Historical matrices retain failed attempts and completed requests that
    another packet supersedes. Neither can satisfy a selected pending request.
    """
    deck_ids = []
    for entry in plan['decks']:
        path = Path(entry['path'])
        assert pin(path) == entry['sha256']
        deck_ids.append(json.loads(path.read_text())['id'])
    assert len(deck_ids) == len(set(deck_ids))
    requested = packet.get('only_ids', deck_ids)
    assert len(requested) == len(set(requested))
    assert set(requested) <= set(deck_ids)
    by_id = {row['deck']: row for row in actual['rows']}
    assert len(by_id) == len(actual['rows'])
    assert set(by_id) <= set(deck_ids)
    return ([by_id[name] for name in requested if name in by_id],
            set(requested) <= set(by_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, action='append', required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--collection-config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.receipts.mkdir(parents=True, exist_ok=args.resume)
    roots = {path.name: path for path in args.work_root}
    assert len(roots) == len(args.work_root)
    records = {}
    if args.resume and (args.receipts / 'index.json').exists():
        previous = json.loads((args.receipts / 'index.json').read_text())
        assert previous['schema'] == 'p21-retained-original-driver-metadata/v1'
        for row in previous['rows']:
            assert pin(Path(row['retained_path'])) == row['sha256']
            records[row['work'], row['deck'], row['file']] = row
    fingerprints = {}
    verified_receipts = {}
    config = json.loads(args.collection_config.read_text())
    while True:
        for name, root in roots.items():
            for output in root.glob('*/hex-output'):
                for filename in ('physics-calls.jsonl', 'native-physics-receipt.json'):
                    source = output / filename
                    try:
                        stat = source.stat()
                        stamp = (stat.st_size, stat.st_mtime_ns)
                        key = name, output.parent.name, filename
                        if fingerprints.get(key) == stamp:
                            continue
                        assert stat.st_size < 10_000_000
                        raw = source.read_bytes()
                        if filename.endswith('.jsonl'):
                            values = [json.loads(line) for line in raw.splitlines()]
                            assert values and all('step' in v and 'calls' in v for v in values)
                        else:
                            value = json.loads(raw)
                            assert value['schema'] == 'mpas-native-forecast-v1'
                        target = args.receipts / name / output.parent.name / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix('.partial')
                        temporary.write_bytes(raw)
                        temporary.replace(target)
                        fingerprints[key] = stamp
                        records[key] = dict(work=name, deck=output.parent.name,
                            file=filename, retained_path=str(target.resolve()),
                            sha256=hashlib.sha256(raw).hexdigest(),
                            authenticated_by_original_verifier=False)
                    except (OSError, json.JSONDecodeError, AssertionError):
                        # A live write or retirement can race observation. It
                        # never creates a false complete/authenticated record.
                        continue
        complete = True
        for packet in config['plans']:
            matrix = Path(packet['matrix'])
            if matrix.parent.name not in roots:
                continue
            plan = json.loads(Path(packet['plan']).read_text())
            actual = json.loads(matrix.read_text()) if matrix.exists() else {'rows': []}
            rows, packet_complete = selected_rows(packet, plan, actual)
            if not packet_complete:
                complete = False
            for row in rows:
                receipt = Path(row['receipt'])
                stat = receipt.stat()
                stamp = stat.st_size, stat.st_mtime_ns, row['receipt_sha256']
                cached = verified_receipts.get(receipt)
                if cached is None or cached[0] != stamp:
                    assert pin(receipt) == row['receipt_sha256']
                    cached = stamp, json.loads(receipt.read_text())
                    verified_receipts[receipt] = cached
                original = cached[1]
                expected = original.get('provenance', {}).get('hex_output_sha256', {})
                directory = Path(row['work_directory']) if 'work_directory' in row else roots[matrix.parent.name] / row['deck']
                work_name = directory.parent.name
                assert work_name in roots and directory.name == row['deck']
                assert directory.resolve() == (roots[work_name] / row['deck']).resolve()
                for filename in ('physics-calls.jsonl', 'native-physics-receipt.json'):
                    key = work_name, row['deck'], filename
                    if key not in records:
                        continue
                    if records[key].get('checked_terminal_receipt_sha256') == row['receipt_sha256']:
                        continue
                    records[key]['authenticated_by_original_verifier'] = (
                        expected.get(filename) == records[key]['sha256'])
                    records[key]['original_receipt_sha256'] = row['receipt_sha256']
                    records[key]['original_passed'] = original['passed']
                    records[key]['verifier_metadata_hash_present'] = filename in expected
                    if original.get('model_execution') and original['evidence_kind'] == 'FORECAST':
                        log = directory / 'hex.log'
                        values = [json.loads(line) for line in log.read_text().splitlines()
                                  if line.startswith('{')]
                        terminal = [v for v in values
                                    if v.get('schema') == 'mpas-native-forecast-v1']
                        if len(terminal) == 1 and terminal[0]['completed']:
                            model = terminal[0]
                            target = Path(records[key]['retained_path'])
                            assert pin(target) == records[key]['sha256']
                            records[key]['model_log_sha256'] = pin(log)
                            if filename.endswith('.jsonl'):
                                calls = [json.loads(line) for line in target.read_text().splitlines()]
                                records[key]['complete_step_trace'] = (
                                    [v['step'] for v in calls] ==
                                    list(range(1, model['steps_completed'] + 1)))
                                flattened = [call for step in calls for call in step['calls']]
                                cloud = [call for call in flattened if call['slot'] == 'cloudiness']
                                records[key]['cloudiness_dispatches'] = len(cloud)
                                records[key]['only_cloudiness_cuda'] = (
                                    bool(cloud) and all(call['backend'] == 'hex-cuda' for call in cloud)
                                    and all(call['backend'] == 'native-cpu' for call in flattened
                                            if call['slot'] != 'cloudiness'))
                                records[key]['call_step_numbers_match'] = all(
                                    call['itimestep'] == step['step']
                                    for step in calls for call in step['calls'])
                            else:
                                records[key]['matches_printed_completion_record'] = (
                                    json.loads(target.read_text()) == model)
                            records[key]['checked_terminal_receipt_sha256'] = row['receipt_sha256']
        index = dict(schema='p21-retained-original-driver-metadata/v1',
            observer_complete=complete, weather_arrays_read=False,
            forecast_acceptance_unchanged=True,
            rows=[records[key] for key in sorted(records)])
        temporary = args.receipts / 'index.partial'
        temporary.write_text(json.dumps(index, indent=2) + '\n')
        temporary.replace(args.receipts / 'index.json')
        if complete:
            print(json.dumps(dict(observer_complete=True, files=len(records))), flush=True)
            return
        time.sleep(1)


if __name__ == '__main__':
    main()
