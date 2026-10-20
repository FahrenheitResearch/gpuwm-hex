"""Measure unchanged strict native outcomes for the undefined-index fixture.

cu_ntiedtke.F90:2616-2650. This is a diagnostic, never a parity acceptance.
Python runs processes and hashes opaque output bytes; it constructs no fields.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def measure(args):
    binary = args.binary.resolve()
    args.scratch.mkdir(parents=True, exist_ok=False)
    runs = []
    for repeat in range(args.repeats):
        work = args.scratch / str(repeat)
        work.mkdir()
        with (work / 'native.log').open('w') as log:
            outcome = subprocess.run([str(binary), '0', '1', '0', '162'],
                                     cwd=work, stdout=log, stderr=subprocess.STDOUT)
        runs.append(dict(repeat=repeat, exit_code=outcome.returncode,
                         files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in work.iterdir() if p.is_file()}))
    args.receipt.write_text(json.dumps(dict(
        schema='mpas-ntiedtke-native-defect-measurement/v1', acceptance=False,
        native_process_count=1, native_host=args.native_host, diagnostic_build=args.diagnostic,
        defect='native-cuddrafn-unassigned-itopde', source='cu_ntiedtke.F90:2616-2650',
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        fixture='33 levels, pressure exponent 0.70, whole pristine native driver',
        runs=runs), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'scratch', 'receipt'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--native-host', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--diagnostic', action='store_true')
    measure(parser.parse_args())
