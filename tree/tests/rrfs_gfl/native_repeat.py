"""Repeat every available full instrumented GFL profile, comparing with Rust.

This standalone measurement is not the published-model defect-policy witness.
No automatic word is substituted, ignored, or used as an accepted exclusion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from prepare_capture_profiles import PROFILES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-mpas-bounds', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists():
        raise ValueError('requires at least two repeats and a new owned capture directory')
    args.output.mkdir(parents=True)
    rows = []
    profiles = list(PROFILES)
    if args.include_mpas_bounds:
        profiles.extend(profile+'_mpas_bounds' for profile in PROFILES)
    for profile in profiles:
        executable = (args.root/f'capture_{profile}').resolve()
        source = executable.with_suffix('.F90')
        folders = []
        for number in range(args.repeats):
            folder = (args.output/profile/str(number)).resolve()
            folder.mkdir(parents=True)
            with (folder/'native.log').open('w') as log:
                subprocess.run([str(executable)], check=True, timeout=90,
                               env={**os.environ, 'MPAS_ORACLE_DIR':str(folder)},
                               stdout=log, stderr=subprocess.STDOUT)
            folders.append(folder)
        files = {p.name for p in folders[0].glob('*.bin')}
        if not files:
            raise ValueError((profile, 'native captured no records'))
        measurements = []
        for number, folder in enumerate(folders[1:], 1):
            other = {p.name for p in folder.glob('*.bin')}
            if files != other:
                raise ValueError((profile, number, 'native capture inventory varied'))
            differing = []
            compared = 0
            for name in sorted(files):
                left, right = folders[0]/name, folder/name
                result = json.loads(subprocess.check_output([
                    str((args.root/'oracle_words').resolve()), 'compare', str(left), str(right)],
                    text=True))
                compared += result['words']
                if result['different_words']:
                    differing.append({'record':name, **result,
                                      'reference_sha256':digest(left), 'repeat_sha256':digest(right)})
            measurements.append({'repeat':number, 'compared_words':compared,
                                 'differing_records':differing,
                                 'outer_driver_differing_words':sum(
                                     r['different_words'] for r in differing
                                     if '__driver_convection__L670' in r['record']),
                                 'differing_words':sum(r['different_words'] for r in differing)})
        rows.append({'profile':profile, 'executable_sha256':digest(executable),
                     'source_sha256':digest(source), 'records_per_run':len(files),
                     'production_vertical_bounds':profile.endswith('_mpas_bounds'),
                     'synthetic_bounds':{'kms':1,'kme':56 if profile.endswith('_mpas_bounds') else 55,
                                         'kds':1,'kde':56 if profile.endswith('_mpas_bounds') else 55,
                                         'kts':1,'kte':55},
                     'measurements':measurements})
    result = {'scope':'repeated full instrumented standalone GFL wrapper; not published-model proof',
              'host':socket.gethostname(), 'process_count':1, 'repeats':args.repeats,
              'cpu_affinity':sorted(os.sched_getaffinity(0)), 'invocation':sys.argv,
              'native_defect_policy_witness':False,
              'source_pin':'1d6e370e/GFL bf6a4b0',
              'comparator_sha256':digest(args.root/'oracle_words'), 'profiles':rows,
              'differing_words':sum(m['differing_words'] for r in rows for m in r['measurements']),
              'outer_driver_differing_words':sum(m['outer_driver_differing_words']
                                                for r in rows for m in r['measurements'])}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    # Every measured word and its binary hashes are in the receipt; keep only
    # small metadata, with no archive of disposable standalone dumps.
    shutil.rmtree(args.output)
    print(json.dumps({'profiles':len(rows), 'repeats':args.repeats,
                      'differing_words':result['differing_words']}))
    return bool(result['differing_words'])


if __name__ == '__main__':
    raise SystemExit(main())
