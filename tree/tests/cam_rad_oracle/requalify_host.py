"""Replay complete stored native boundaries through one actual host binary.

Every boundary and persistent-state word counts. Python orchestrates compiled
test capture replay and comparison; this is standalone evidence, not a forecast.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

from compare_boundary import compare as boundary
from compare_slices import compare as state


def replay(executable, fixtures, work, receipts):
    executable = executable.resolve()
    fixtures = [(directory.resolve(), inventory.resolve())
                for directory, inventory in fixtures]
    required = [executable]
    for directory, inventory in fixtures:
        required.extend(directory / name for name in
                        ('cam-state.bin', 'cam-before.bin', 'cam-after.bin',
                         'cam-state-after.bin'))
        required.append(inventory)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Incomplete native captures: ' + ', '.join(missing))
    for _, inventory in fixtures:
        schema = json.loads(inventory.read_text())['schema']
        if not schema['arguments'] or not schema['globals']:
            raise ValueError(f'Incomplete native ABI inventory: {inventory}')
    work.mkdir(parents=True, exist_ok=False)
    receipts.mkdir(parents=True, exist_ok=False)
    results = []
    for index, (directory, inventory) in enumerate(fixtures):
        label = f'{index:02d}-{directory.name}'
        output = work / (label + '.bin')
        with (receipts / (label + '.log')).open('w') as log:
            run = subprocess.run([str(executable), str(directory / 'cam-state.bin'),
                                  str(directory / 'cam-before.bin'), str(output)],
                                 stdout=log, stderr=subprocess.STDOUT)
        row = {'fixture': str(directory), 'inventory': str(inventory),
               'process_count': 1, 'returncode': run.returncode}
        if run.returncode == 0:
            schema = json.loads(inventory.read_text())['schema']['globals']
            row['boundary_differing_words'] = boundary(
                directory / 'cam-after.bin', output, inventory,
                receipts / (label + '-boundary.json'))
            row['state_differing_words'] = state(
                directory / 'cam-state-after.bin', Path(str(output) + '.state'),
                [('complete CAM module state', schema, 1)],
                receipts / (label + '-state.json'),
                'complete strict standalone CAM persistent state; process count1')
            row['result'] = 'PASS' if (row['boundary_differing_words'] == 0
                                      and row['state_differing_words'] == 0) else 'FAIL'
            # Receipt retains both complete capture hashes. Native inputs stay
            # available for CUDA; generated host output is disposable.
            output.unlink()
            Path(str(output) + '.state').unlink()
        else:
            row['result'] = 'FAIL'
        results.append(row)
        print(json.dumps(row), flush=True)
    result = {'scope': 'complete standalone host boundary/state replay',
              'process_count': 1, 'excluded_words': 0,
              'host_artifact_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
              'whole_model_acceptance': False, 'fixtures': results,
              'passed': all(row['result'] == 'PASS' for row in results)}
    (receipts / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'work', 'receipts'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fixture', nargs=2, type=Path, action='append', required=True,
                        metavar=('CAPTURE_DIRECTORY', 'INVENTORY_JSON'))
    args = parser.parse_args()
    raise SystemExit(replay(args.executable, args.fixture, args.work, args.receipts))
