#!/usr/bin/env python3
"""Map authenticated real archived replay counters to whole native modules.

The counters accumulate all selected cases and sites. Shared module counts are
not attributed to individual schemes, and absent runtime counters remain named.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_synthetic_coverage import compilation_directory
from coverage import summarize, validate_object_sources
from reproduce_archived_replay import authority, sha


def validate(replay, index, dependency_root=None):
    data = authority(index, dependency_root)
    measured = json.loads((replay / 'reproduction.json').read_text())
    if (data.get('recipe') != 'coverage' or measured.get('recipe') != 'coverage'
            or not measured.get('comparison_complete') or measured['process_count'] != 1
            or measured['differing_words'] or measured['input_index_sha256'] != sha(index)
            or measured['selected_sites'] != data['sites']
            or [r['id'] for r in measured['cases']] != [r['id'] for r in data['cases']]):
        raise ValueError('complete bound coverage replay required')
    counters = measured.get('coverage_counters_sha256', {})
    if not counters:
        raise ValueError('coverage replay has no runtime counters')
    for relative, expected in counters.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'profiles':
            raise ValueError('coverage counter escapes replay profile directory')
        if sha(replay / path) != expected:
            raise ValueError('coverage replay counter changed')
    if set(counters) != {str(p.relative_to(replay)) for p in (replay / 'profiles').rglob('*.gcda')}:
        raise ValueError('coverage runtime counter inventory differs')
    return data, measured


def collect(replay, index, source, out, dependency_root=None):
    data, measured = validate(replay, index, dependency_root)
    return collect_verified(data, measured, replay, index, source, out,
                            replay / 'reproduction.json')


def collect_verified(data, measured, replay, index, source, out, replay_receipt):
    """Collect native counters after the caller authenticates its receipt format.

    Sealed column banks and archived forecasts use different producer receipts.
    Both must validate complete comparisons and the exact counter inventory
    before calling this shared native-source/object mapping implementation.
    """
    build = Path(data['build'])
    inventory_path = build / 'oracle-inventory.json'
    native = json.loads(inventory_path.read_text())
    from pin_sets import provenance
    selected = provenance(source, native['pin_set'])
    if any(native.get(key) != value for key, value in selected.items()):
        raise ValueError('native coverage pinned source identity differs')
    audit = json.loads(Path(data['audit']).read_text())
    if not audit['all_ok'] or audit['recipe'] != 'coverage':
        raise ValueError('passing native coverage object audit required')
    for relative, expected in native['generated_sha256'].items():
        if sha(build / relative) != expected:
            raise ValueError('inventoried native coverage source changed')
    sites = [s for s in native['sites'] if s['site'] in data['sites']]
    if len(sites) != len(data['sites']):
        raise ValueError('coverage site is absent from native inventory')
    modules = {name for site in sites for name in site['state_modules']}
    sources = {m['path'] for m in native['module_state'] if m['name'] in modules}
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for relative in sorted(sources):
        candidates = list((build / 'src').rglob(Path(relative).stem + '.gcno'))
        if len(candidates) != 1:
            raise ValueError('native source has no unique coverage notes: ' + relative)
        note = candidates[0]
        obj = note.with_suffix('.o')
        rows = [r for r in audit['objects'] if r['object'] == str(obj.relative_to(build))]
        if len(rows) != 1 or sha(obj) != rows[0]['object_sha256']:
            raise ValueError('native coverage object is not audit-bound')
        compiled = compilation_directory(note)
        suffix = note.relative_to(build).parent
        if compiled.parts[-len(suffix.parts):] != suffix.parts:
            raise ValueError('native coverage compile directory differs')
        compiled_root = compiled
        for _ in suffix.parts:
            compiled_root = compiled_root.parent
        counter = replay / 'profiles' / str(compiled / note.with_suffix('.gcda').name).lstrip('/')
        row = dict(native_source=relative, notes_sha256=sha(note), object_sha256=sha(obj),
                   compilation_directory=str(compiled), counter=str(counter), missing_runtime_counter=True)
        if counter.is_file():
            key = str(counter.relative_to(replay))
            if measured['coverage_counters_sha256'].get(key) != sha(counter):
                raise ValueError('native coverage counter is not replay-bound')
            row.update(counter_sha256=sha(counter), missing_runtime_counter=False)
            stage = out / ('stage-' + str(len(reports)))
            stage.mkdir()
            shutil.copy2(note, stage / note.name)
            shutil.copy2(counter, stage / counter.name)
            run = subprocess.run(['gcov', '-j', '-b', '-c', note.name], cwd=stage,
                                 text=True, capture_output=True)
            (out / (note.stem + '.log')).write_text(run.stdout + run.stderr)
            run.check_returncode()
            compressed = stage / (note.stem + '.gcov.json.gz')
            report = summarize(json.loads(gzip.decompress(compressed.read_bytes())), source, build,
                               compiled_root, native['generated_sha256'])
            validate_object_sources(report, relative, source, build, native['source_sha256'])
            report.update(counters_cumulative=True, process_count=1,
                          branch_closure_proven=False, forecast_proven=False)
            target = out / (note.stem + '.json')
            payload = (json.dumps(report, indent=2) + '\n').encode()
            if len(payload) >= 10_000_000:
                raise ValueError('native coverage receipt exceeds delivery cap')
            target.write_bytes(payload)
            row.update(report=target.name, report_sha256=sha(target), gcov_sha256=sha(compressed),
                       branches_total=sum(f['branches_total'] for f in report['files']),
                       branches_taken=sum(f['branches_taken'] for f in report['files']))
            shutil.rmtree(stage)
        reports.append(row)
    result = dict(schema='real-native-archived-replay-coverage-v1', files=reports,
                  selected_sites=data['sites'], case_ids=[r['id'] for r in measured['cases']],
                  cumulative_shared_module_counts=True, process_count=1,
                  native_replay_host=measured.get('native_replay_host'),
                  replay_sha256=sha(replay_receipt), index_sha256=sha(index),
                  inventory_sha256=sha(inventory_path), audit_sha256=sha(data['audit']),
                  collector_sha256=sha(Path(__file__)), branch_closure_proven=False,
                  forecast_proven=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'index', 'source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path,
                        help='Portable metadata copies named by each original dependency SHA256')
    raise SystemExit(collect(**vars(parser.parse_args())))
