"""Seal complete column-replay receipts without assigning forecast acceptance.

Every recorded array and nonfinite word stays in the word comparator's result.
This utility handles JSON evidence only; Rust compares the weather records.
"""
import argparse
import hashlib
import json
from pathlib import Path


def reference(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


def seal(replays, artifacts, process_count):
    cases = []
    pins = set()
    for label, path in replays:
        replay = json.loads(path.read_text())
        if replay['scheme'] != 'thompson_aero':
            raise ValueError('matrix contains another scheme: ' + label)
        captures = replay['captures']
        arrays = [array for capture in captures
                  for array in capture['arrays'].values()]
        if not captures or not arrays:
            raise ValueError('matrix contains an empty replay: ' + label)
        differences = sum(array['different_words'] for array in arrays)
        equal = all(array['bitwise_equal'] for array in arrays)
        if equal != (differences == 0) or equal != replay['all_outputs_bitwise_equal']:
            raise ValueError('inconsistent word-comparison receipt: ' + label)
        pins.add(replay['pin_set_sha256'])
        cases.append(dict(label=label, receipt=reference(path),
            backend=replay['backend'], captures=len(captures), arrays=len(arrays),
            words=sum(array['words'] for array in arrays), different_words=differences,
            different_nonfinite_words=sum(array['different_nonfinite_words']
                                          for array in arrays),
            dump_root_sha256=replay['dump_root_sha256'],
            hidden_state_complete=replay['hidden_state_complete'],
            forecast_proven=replay['forecast_proven']))
    if len(pins) != 1 or len({case['label'] for case in cases}) != len(cases):
        raise ValueError('matrix labels must be unique and use one pin set')
    return dict(schema='mpas.aerosol-column-matrix/v1', acceptance=False,
        forecast_proven=False, process_count=process_count,
        scope='recorded column output/state words; no forecast identity tier',
        pin_set_sha256=pins.pop(), cases=cases,
        totals={key: sum(case[key] for case in cases) for key in
                ('captures', 'arrays', 'words', 'different_words',
                 'different_nonfinite_words')},
        artifacts=[reference(path) for path in artifacts])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='append', required=True, metavar='LABEL=JSON')
    parser.add_argument('--artifact', action='append', type=Path, default=[])
    parser.add_argument('--process-count', required=True, type=int, choices=range(1, 257))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    replays = []
    for value in args.replay:
        label, separator, path = value.partition('=')
        if not label or not separator or not path:
            parser.error('--replay requires LABEL=JSON')
        replays.append((label, Path(path)))
    result = seal(replays, args.artifact, args.process_count)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['totals']))
