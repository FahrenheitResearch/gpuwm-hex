"""Run a Python command under the untimed NVRTC witness and exec census.

Must be invoked inside the program's normal GPU admission helper for GPU work.
The target uses this interpreter: run.py [options] -- -B target.py arguments.
"""
import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from summarize import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--require-zero', action='store_true')
    parser.add_argument('python_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command_args = args.python_args
    if command_args[:1] == ['--']:
        command_args = command_args[1:]
    if not command_args:
        parser.error('a target Python command is required after --')
    if os.environ.get('HEX_NVRTC_WITNESS_DIR'):
        parser.error('do not nest witness runners')
    witness = Path(__file__).resolve().parent
    search_path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != witness]
    existing = importlib.machinery.PathFinder.find_spec('sitecustomize', search_path)
    if existing is not None:
        parser.error('an existing sitecustomize would be shadowed: ' + str(existing.origin))
    strace = shutil.which('strace')
    if strace is None:
        parser.error('strace is required for independent Python process coverage')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment['HEX_NVRTC_WITNESS_DIR'] = str(output)
    environment['PYTHONPATH'] = str(witness) + os.pathsep + environment.get('PYTHONPATH', '')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [strace, '-ff', '-e', 'trace=execve', '-o', str(output / 'exec'),
               sys.executable, *command_args]
    started = time.perf_counter()
    with (output / 'command.log').open('w') as log:
        completed = subprocess.run(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT)
    result = summarize(output, output / 'exec')
    result.update(command=command, target_returncode=completed.returncode,
                  elapsed_seconds=time.perf_counter() - started,
                  timing_valid=False, require_zero=args.require_zero,
                  witness_sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(witness.glob('*.py'))})
    result['check_passed'] = completed.returncode == 0 and result['coverage_valid'] and (
        not args.require_zero or result['zero_nvrtc_observed'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('check_passed', 'target_returncode',
        'coverage_valid', 'nvrtc_calls', 'zero_nvrtc_observed', 'issues')}, indent=2))
    return 0 if result['check_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
