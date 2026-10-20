"""Bind every real CUDA output payload to complete native wrapper receipts.

This compares metadata hashes emitted by Rust after complete raw-word grading.
Encoded-record hashes are not used. Same-payload evidence is a column diagnostic;
it does not certify the captured parent forecast or explain native variation.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError('Complete original receipt changed: ' + str(path))
    return json.loads(path.read_text())


def main(args):
    cuda = load(args.cuda / 'summary.json')
    if not cuda['comparison_complete'] or len(cuda['captures']) != 30:
        raise ValueError('Every original CUDA call must complete before comparison')
    rows = []
    for specification in args.native:
        label, directory = specification.split('=', 1)
        root = Path(directory)
        native = load(root / 'summary.json')
        if not native['comparison_complete'] or len(native['captures']) != 30:
            raise ValueError('Every original native call must complete')
        by_call = {(r['label'], r['prefix']): r for r in native['captures']}
        if len(by_call) != 30:
            raise ValueError('Native call identities are not unique')
        calls = []
        for capture in cuda['captures']:
            original = by_call[(capture['label'], capture['prefix'])]
            if capture['archive_sha256'] != original['archive_sha256']:
                raise ValueError('Original native capture differs')
            recipe = original['comparisons']['native']
            if not capture['complete'] or not recipe['complete']:
                raise ValueError('Original comparison is incomplete')
            device_path = args.cuda / capture['receipt']
            native_path = root / recipe['receipt']
            device = load(device_path, capture['receipt_sha256'])
            host = load(native_path, recipe['receipt_sha256'])
            if device['excluded_words'] or host['excluded_words']:
                raise ValueError('An original word was excluded')
            device_fields = {r['name']: r for r in device['fields']}
            host_fields = {r['name']: r for r in host['fields']}
            if set(device_fields) != set(host_fields) or len(host_fields) != 47:
                raise ValueError('Complete wrapper output inventory differs')
            fields = []
            for name, left in device_fields.items():
                right = host_fields[name]
                if any(left[k] != right[k] for k in
                       ('shape', 'width', 'words', 'reference_sha256')):
                    raise ValueError('Raw payload domains or original references differ')
                fields.append(dict(name=name, words=left['words'],
                    same_payload=left['candidate_sha256'] == right['candidate_sha256'],
                    cuda_payload_sha256=left['candidate_sha256'],
                    native_payload_sha256=right['candidate_sha256'],
                    reference_payload_sha256=left['reference_sha256'],
                    cuda_differing_words=left['differing_words'],
                    native_differing_words=right['differing_words']))
            words = sum(r['words'] for r in fields)
            if words != device['words'] or words != host['words']:
                raise ValueError('The complete raw payload word count differs')
            calls.append(dict(label=capture['label'], prefix=capture['prefix'],
                archive_sha256=capture['archive_sha256'], words=words,
                cuda_receipt_sha256=sha(device_path), native_receipt_sha256=sha(native_path),
                native_executable_sha256=recipe['executable_sha256'],
                same_payload=all(r['same_payload'] for r in fields), fields=fields,
                parent_forecast_proven=False,
                diagnostic_limitation=capture['diagnostic_limitation']))
        words = sum(r['words'] for r in calls)
        expected_words = sum(r['comparisons']['native']['expected_words']
                             for r in native['captures'])
        if words != expected_words or words != cuda['words']:
            raise ValueError('The complete thirty-call word inventory differs')
        rows.append(dict(native_recipe=label, native_summary_sha256=sha(root/'summary.json'),
            all_payloads_identical=all(r['same_payload'] for r in calls),
            calls_with_different_payloads=sum(not r['same_payload'] for r in calls),
            compared_payload_words=words, calls=calls))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope=__doc__, process_count=1,
        cuda_summary_sha256=sha(args.cuda/'summary.json'), native_comparisons=rows,
        comparison_complete=True, excluded_words=[], forecast_proven=False,
        native_variation_cause_proven=False), indent=2)+'\n')
    return int(any(not r['all_payloads_identical'] for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--native', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
