"""Observe automatic PRE storage in all four unchanged CAM switch fixtures.

MPAS module_ra_cam.F:364-365,678-679. This supplements the original unseeded
comparisons; it neither excludes words nor qualifies a production forecast.
The Rust tracer reads an owned child without allocating inside that child.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess

from compact_compare import boundary, slices
from observe_original_entry import digest, offsets, write


def observe(fixtures, source, tracer, output):
    seal = json.loads((fixtures / 'native-seal.json').read_text())
    binary = source / 'cam_columns_capture'
    assert digest(binary) == seal['binary_sha256']
    inventory = fixtures / 'preparation/translation-inventory.json'
    schema = json.loads(inventory.read_text())['schema']
    switches = json.loads((fixtures / 'preparation/public-switch-caller.json').read_text())
    assert set(switches['masks']) == {'0', '1', '2', '3'}
    assert switches['radiation_replacements'] == []
    assert switches['production_defaults_changed'] is False
    assert digest(source / 'run_columns_capture.F90') == switches['caller_sha256']
    output.mkdir(parents=True, exist_ok=False)
    results, stops = [], []
    names = ('cam-before.bin', 'cam-after.bin', 'cam-state.bin', 'cam-state-after.bin')
    for mask in range(4):
        label = f'mask-{mask}'
        baseline = fixtures / label
        captured_names = names if mask % 2 == 0 else names[:3]
        for name in captured_names:
            record = seal['files'][f'{label}/{name}']
            assert (baseline / name).stat().st_size == record['bytes']
            assert digest(baseline / name) == record['sha256']
        destination = output / label
        destination.mkdir()
        for name in ('cam_columns_capture', 'CAM_ABS_DATA.DBL', 'CAM_AEROPT_DATA.DBL',
                     'OZONE_DAT.TBL', 'OZONE_LAT.TBL', 'OZONE_PLEV.TBL'):
            shutil.copy2(source / name, destination / name)
        copied = destination / binary.name
        assert digest(copied) == seal['binary_sha256']
        entry, ci, cl = offsets(copied)
        command = [str(tracer.resolve()), str(copied.resolve()), hex(entry),
                   hex(ci['return_offset']), hex(cl['return_offset']),
                   str(destination.resolve()), 'native.bin', '6', 'unused', '1',
                   'diagnostic', str(mask)]
        with (destination / 'native-trace.log').open('w') as log:
            result = subprocess.run(command, cwd=destination, stdout=log,
                                    stderr=subprocess.STDOUT)
        assert result.returncode == 0, (mask, result.returncode)
        expected_records = 30 if mask % 2 == 0 else 4
        trace = (destination / 'native-trace.log').read_text()
        if mask % 2:
            assert 'CAM_NATIVE_ENTRY_TERMINAL calls_returned=3 terminal_call=4 exit=0' in trace
            assert '--- enter subroutine get_rf_scales:' in trace
            assert not (baseline / names[3]).exists() and not (destination / names[3]).exists()
        else:
            assert 'CAM_NATIVE_ENTRY_COMPLETE calls=30 exit=0' in trace
        comparisons = {}
        for name in names[:2]:
            receipt = destination / (name + '.comparison.json')
            comparisons[name] = boundary(baseline / name, destination / name, inventory, receipt)
            assert len(json.loads(receipt.read_text())['records']) == (expected_records if name == 'cam-before.bin' else (3 if mask % 2 else 30))
        for name in captured_names[2:]:
            comparisons[name] = slices(baseline / name, destination / name,
                [('Complete native module state', schema['globals'], 1)],
                destination / (name + '.comparison.json'),
                'unchanged public-switch native executable traced versus original capture')
        row = dict(label=label, baseline=str(baseline.resolve()),
                   inventory=str(inventory.resolve()), expected_records=expected_records,
                   native_executable_sha256=digest(copied), tracer_sha256=digest(tracer),
                   tracer_source_sha256=digest(Path(__file__).with_name('trace_entry.rs')),
                   host=socket.gethostname(), process_count=1, switches=switches['masks'][str(mask)],
                   comparison_differences=comparisons,
                   original_native_outputs_unchanged=not any(comparisons.values()),
                   native_files={name: digest(destination / name) for name in captured_names},
                   entry_files={name: digest(destination / ('entry-' + name + '.bin'))
                                for name in ('tauxci', 'tauxcl')},
                   no_injected_library=True, no_child_allocation=True,
                   no_post_seed=True, excluded_words=0, whole_model_acceptance=False)
        if mask % 2:
            row.update(native_stop_source='module_ra_cam_support.F:2651', native_stop_exitcode=0,
                       complete_after_records=3, final_state_written=False)
            stops.append(row)
        else:
            results.append(row)
        write(destination / 'observation.json', row)
        write(output / 'summary.json', dict(complete=len(results) == len(stops) == 2,
            records=sum(r['expected_records'] for r in results), fixtures=results, native_stops=stops,
            full_fixture_masks=[0, 2], stopped_fixture_masks=[1, 3],
            original_native_outputs_unchanged=all(r['original_native_outputs_unchanged'] for r in results + stops),
            original_failures_superseded=False, whole_model_acceptance=False,
            fixture_native_seal_sha256=digest(fixtures / 'native-seal.json'),
            observation_orchestration_sha256=digest(Path(__file__))))
        print(label, comparisons, flush=True)
    return int(any(not row['original_native_outputs_unchanged'] for row in results + stops))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'source', 'tracer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(observe(args.fixtures, args.source, args.tracer, args.output))
