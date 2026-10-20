#!/usr/bin/env python3
"""Seal native branch coverage only after every authenticated call compares exactly.

RRFS 1d6e370e physics/physics_wrf/module_sf_mynn.F:223-388 and its unchanged
physics_mmm/sf_mynn.F90 and physics_wrf/sf_mynn_pre.F callees. Coverage belongs
to an owned standalone library, never to a rebuilt native reference forecast.
The eight-call MYNN-SFC option observes the full native land/water/ice driver.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(build, replay, out, scheme='sf_mynn'):
    authority = json.loads((build / 'build.json').read_text())
    comparison = json.loads((replay / 'summary.json').read_text())
    output_records, words = (176, 356272) if scheme == 'sf_mynn' else (360, 371552)
    if (not authority['coverage_instrumented'] or not comparison['passed']
            or comparison['backend'] != 'native-replay'
            or comparison['output_records'] != output_records or comparison['compared_words'] != words
            or comparison['different_words'] != 0
            or comparison['native_library_sha256'] != sha(build / 'native.so')
            or authority['library_sha256'] != sha(build / 'native.so')):
        raise ValueError('Coverage requires the complete exact real-call native comparison')
    files = []
    for item in authority['sources']:
        source = build / Path(item['fork_path']).name
        if sha(source) != item['sha256']:
            raise ValueError('Native source changed before gcov collection')
        if scheme == 'sf_mynnsfclay' and source.stem not in (
                'module_sf_mynnsfc_land', 'module_sf_mynnsfc_water',
                'module_sf_mynnsfc_ice', 'module_sf_mynnsfc_driver'):
            # The pristine kind/constants/common modules have no executed scheme branches.
            continue
        stem = source.stem
        if not (build / (stem + '.gcda')).is_file():
            raise ValueError('Missing executed native coverage: ' + stem)
        run = subprocess.run(['gcov', '-b', '-c', '-j', stem + '.gcno'], cwd=build,
                             capture_output=True, text=True, check=True)
        data = json.loads(gzip.decompress((build / (stem + '.gcov.json.gz')).read_bytes()))
        if len(data['files']) != 1 or Path(data['files'][0]['file']).name != source.name:
            raise ValueError('Unexpected native gcov source inventory')
        native = data['files'][0]
        lines = source.read_text().splitlines()
        branches = [dict(line=line['line_number'], branch=index, count=branch['count'],
                         source=lines[line['line_number'] - 1].strip())
                    for line in native['lines']
                    for index, branch in enumerate(line.get('branches', []))]
        files.append(dict(fork_path=item['fork_path'], source_sha256=item['sha256'],
            gcno_sha256=sha(build / (stem + '.gcno')), gcda_sha256=sha(build / (stem + '.gcda')),
            branches=len(branches), taken=sum(value['count'] > 0 for value in branches),
            instrumented_lines=len(native['lines']),
            executed_lines=sum(line['count'] > 0 for line in native['lines']),
            functions=native['functions'], unreached=[b for b in branches if b['count'] == 0],
            gcov_log=run.stdout + run.stderr))
    result = dict(scope=('four authentic sf_mynn calls' if scheme == 'sf_mynn' else 'eight authentic MYNN-SFC calls')
        + ', standalone strict fork library on captured PRE records', scheme=scheme,
        fork_pin=authority['fork_pin'], forecast_coverage=False,
        call_set=comparison['authority']['captures'] if scheme == 'sf_mynn' else comparison['cases'], process_count=1,
        replay_host=comparison['replay_host'], compared_words=comparison['compared_words'], different_words=0,
        build_sha256=sha(build / 'build.json'), comparison_sha256=sha(replay / 'summary.json'),
        library_sha256=authority['library_sha256'],
        branches=sum(file['branches'] for file in files), taken=sum(file['taken'] for file in files), files=files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('branches', 'taken', 'compared_words', 'different_words')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build', 'replay', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--scheme', choices=('sf_mynn', 'sf_mynnsfclay'), default='sf_mynn')
    args = parser.parse_args()
    collect(args.build.resolve(), args.replay.resolve(), args.out.resolve(), args.scheme)
