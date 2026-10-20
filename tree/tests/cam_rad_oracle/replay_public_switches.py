"""Replay both complete switch fixtures and both literal native STOP prefixes.

MPAS module_ra_cam_support.F:2651 stops the forcing-enabled native process.
Every preceding output word is compared; no missing final state is invented.
Automatic PRE bytes are independently observed inputs, never native POST seeds.
"""
import argparse
import json
from pathlib import Path
import shutil

from compact_compare import boundary
from observe_original_entry import digest, replay, write
from replay_cuda import replay as replay_capture


def qualify(captures, output):
    observation = json.loads((captures / 'summary.json').read_text())
    assert observation['complete'] and observation['original_native_outputs_unchanged']
    assert observation['full_fixture_masks'] == [0, 2]
    assert observation['stopped_fixture_masks'] == [1, 3]
    assert observation['records'] == 60
    output.mkdir(parents=True, exist_ok=False)
    complete_rc = replay(captures, output / 'full')
    full = json.loads((output / 'full/summary.json').read_text())
    assert full['complete'] and full['records'] == 60
    stops = []
    for row in observation['native_stops']:
        assert row['native_stop_source'] == 'module_ra_cam_support.F:2651'
        assert row['expected_records'] == 4 and row['complete_after_records'] == 3
        assert row['original_native_outputs_unchanged']
        native = captures / row['label']
        destination = output / row['label']
        destination.mkdir()
        for name, expected in row['native_files'].items():
            original = Path(row['baseline']) / name
            assert digest(original) == digest(native / name) == expected
            shutil.copy2(original, destination / name)
        for name, expected in row['entry_files'].items():
            assert digest(native / ('entry-' + name + '.bin')) == expected
        receipt = destination / 'prefix.json'
        try:
            replay_capture(destination, Path(row['inventory']), receipt,
                           rust_transport=True, native_entry=native)
        except RuntimeError as error:
            assert str(error) == 'MPAS CAM fatal source line 2651', str(error)
        else:
            raise AssertionError('CUDA continued beyond the unconditional native STOP')
        differences = boundary(destination / 'cam-after.bin', destination / 'cam-cuda-after.bin',
                               Path(row['inventory']), receipt)
        result = json.loads(receipt.read_text())
        assert len(result['records']) == 3
        stops.append(dict(mask=int(row['label'].split('-')[1]), native_calls=4,
            complete_output_records=3, native_stop_source=row['native_stop_source'],
            cuda_stop_source_line=2651, words=result['words'], differing_words=differences,
            prefix_receipt_sha256=digest(receipt), native_observation_sha256=digest(native / 'observation.json'),
            final_state_written=False))
    record = dict(scope='two complete native-supported switch replays and two literal native STOP prefix replays',
        complete=True, process_count=1, whole_model_acceptance=False, excluded_words=0,
        observed_automatic_pre_test_only=True, original_unseeded_failures_superseded=False,
        complete_calls=full['records'], complete_words=full['words'],
        complete_differing_words=full['differing_words'], native_stops=stops,
        differing_words=full['differing_words'] + sum(row['differing_words'] for row in stops),
        words=full['words'] + sum(row['words'] for row in stops),
        observation_summary_sha256=digest(captures / 'summary.json'),
        full_summary_sha256=digest(output / 'full/summary.json'),
        orchestration_sha256=digest(Path(__file__)))
    write(output / 'summary.json', record)
    print(json.dumps({key: record[key] for key in ('complete_calls', 'words', 'differing_words')}), flush=True)
    return int(bool(complete_rc or record['differing_words']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('captures', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(qualify(args.captures, args.output))
