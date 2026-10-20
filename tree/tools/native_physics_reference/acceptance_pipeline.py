#!/usr/bin/env python3
"""Continue queued real dycore acceptance; build the pinned fallback on mismatch.

Explicit paths bind this orchestration to existing native runs and copied inputs.
GPU and compiler commands retain their required slot wrappers.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from hexcore import site_runner


def run(argv, log):
    print(json.dumps({'command': argv, 'log': str(log)}), flush=True)
    with log.open('w') as stream:
        subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)


def await_receipt(path, exit_path):
    while True:
        if path.exists():
            try:
                receipt = json.loads(path.read_text())
                if receipt.get('schema') == 'gpuwm-hex.identity-extend/v1' and not receipt.get('partial'):
                    return receipt
            except json.JSONDecodeError:
                pass
        if exit_path.exists():
            raise RuntimeError(f'comparison exited without a completed receipt: {exit_path.read_text()}')
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'init', 'native-one', 'native-day', 'source', 'mpi', 'pnetcdf'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--await-existing', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    tools = Path(__file__).resolve().parent
    dry = tools.parent / 'native_dry_reference'
    receipts = work / 'receipts'
    def compare(native, output, log):
        attempt = 0
        while True:
            attempt += 1
            try:
                run(site_runner.gpu_command(['taskset', '-c', '0-15', 'python', str(dry/'identity_compare.py'),
                     '--init', str(args.init), '--native', str(native), '--out', str(output),
                     '--cache-root', str(work/'cache/acceptance')], 10, tag='f0-native-ref-acceptance', build=True),
                    log.with_name(log.stem+f'-attempt-{attempt}'+log.suffix))
                return json.loads(output.read_text())
            except subprocess.CalledProcessError as error:
                # An interrupted queue has no numerics result. Keep slot wrappers
                # on retry and retain each launch log; never turn it into a pass.
                if error.returncode not in (-15, 143):
                    raise
                print(json.dumps({'queue_interrupted': error.returncode, 'attempt': attempt}), flush=True)
                time.sleep(15)
    one = receipts/'host-dry-onestep.json'
    if args.await_existing:
        host = await_receipt(one, receipts/'hex-onestep-background.exit')
    else:
        host = compare(args.native_one, one, receipts/'hex-onestep-background.log')
    if host['all_identical']:
        host = compare(args.native_day, receipts/'host-dry-24h.json', receipts/'hex-24h-background.log')
    if host['all_identical']:
        (receipts/'acceptance-pipeline.json').write_text(json.dumps({'host_acceptance': True,
            'one_step': 'host-dry-onestep.json', 'day': 'host-dry-24h.json'}, indent=2)+'\n')
        return
    # No tolerance: immediately compile GNU13.3/glibc2.39 from the local image.
    fallback = work/'scratch/fallback'
    command = ('export PATH='+shlex.quote(str(args.mpi/'bin'))+':/opt/views/view/bin:$PATH; '
               'export LD_LIBRARY_PATH='+shlex.quote(str(args.mpi/'lib'))+':/opt/views/view/lib; '
               'export NETCDF_PREFIX=/opt/views/view; bash '+shlex.join([
                   str(tools/'build_reference.sh'), str(fallback), str(args.source),
                   'gnu-strict', 'atmosphere', str(args.mpi), str(args.pnetcdf)]))
    container = ['podman', 'run', '--rm', '--network', 'none', '--entrypoint', '/bin/bash',
                 '-v', f'{work}:{work}:rw', '-w', str(work), args.container, '-lc']
    run(['taskset', '-c', '0-15', *site_runner.build_prefix(), *container, command], receipts/'fallback-build-launch.log')
    build = fallback/'native/v8.4.2-gnu-strict-atmosphere'
    comparisons = []
    for label, original in (('onestep', args.native_one), ('24h', args.native_day)):
        directory = fallback/label
        directory.mkdir(parents=True, exist_ok=False)
        for path in original.iterdir():
            if path.name in ('namelist.atmosphere', 'streams.atmosphere'):
                shutil.copy2(path, directory/path.name)
            elif path.is_symlink() and path.name != 'atmosphere_model':
                (directory/path.name).symlink_to(path.resolve(strict=True))
        (directory/'atmosphere_model').symlink_to(build/'atmosphere_model')
        native_command = ('export LD_LIBRARY_PATH='+shlex.quote(str(args.mpi/'lib'))+':/opt/views/view/lib; '
                          'cd '+shlex.quote(str(directory))+'; '+shlex.join([
                              str(args.mpi/'bin/mpirun'), '-np', '16', './atmosphere_model']))
        run(['taskset', '-c', '0-15', *container, native_command], receipts/f'fallback-native-{label}.log')
        if 'Finished running the atmosphere core' not in (directory/'log.atmosphere.0000.out').read_text():
            raise RuntimeError('fallback native did not complete')
        receipt = compare(directory, receipts/f'fallback-dry-{label}.json', receipts/f'fallback-hex-{label}.log')
        comparisons.append({'receipt': f'fallback-dry-{label}.json', 'all_identical': receipt['all_identical']})
        if not receipt['all_identical']:
            break
    (receipts/'acceptance-pipeline.json').write_text(json.dumps({'host_acceptance': False,
        'fallback': comparisons, 'container': args.container}, indent=2)+'\n')


if __name__ == '__main__':
    main()
