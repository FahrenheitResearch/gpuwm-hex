#!/usr/bin/env python3
"""Run every sealed original forecast through the given checkout's real door.

MPAS authority: mpas_atmphys_driver.F:202-373. The independent whole-stream
verifier and its ratchet are preserved. Only the integrator's documented M0
written-field scope is checked separately, at a fixed process count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from hexcore import site_runner

from check_written_forecast import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    args.work_root.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.index.read_text())['entries']
    if not entries:
        parser.error('sealed index contains no forecast decks')
    env = dict(os.environ)
    env['PYTHONPATH'] = ':'.join(map(str, (checkout / 'tree/src', checkout / 'tree/tools',
                                         Path(os.environ.get('GPUWM_ENGINE_SRC') or os.environ.get('MPAS_ENGINE') or '.'))))
    results = []
    for number, entry in enumerate(entries):
        result = {'number': number, 'passed': False}
        try:
            deckpath = args.index.parent / entry['deck']
            inventorypath = args.index.parent / entry['inventory']
            for path, sha in ((deckpath, entry['deck_sha256']),
                              (inventorypath, entry['inventory_sha256'])):
                if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                    raise ValueError(f'sealed reproduction input changed: {path}')
            deck = json.loads(deckpath.read_text())
            work = args.work_root / str(number)
            work.mkdir()
            receiptpath = work / 'verify.json'
            command = site_runner.gpu_command(['taskset', '-c', '8-23', sys.executable,
                       str(checkout / 'tree/tools/native_physics_reference/backend/verify_forecast.py'),
                       'verify-mpas', '--deck', str(deckpath), '--receipt', str(receiptpath),
                       '--work-root', str(work / 'model'), '--comparator', str(args.comparator)],
                       120, proof=True, build=True)
            with (work / 'model.log').open('w') as log:
                run = subprocess.run(command, cwd=checkout / 'tree', env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            # Exit1 may retain exactly the deferred missing-field inventory.
            # Any execution failure, extra field loss or nonzero word fails check.
            if run.returncode not in (0, 1):
                raise ValueError(f'model/verifier command exited {run.returncode}')
            receipt = json.loads(receiptpath.read_text())
            completion = json.loads(receiptpath.with_name('verify-completion.json').read_text())
            result = check(receipt, completion, deck, json.loads(inventorypath.read_text()),
                           deck_sha256=entry['deck_sha256'])
            result.update(number=number, receipt=str(receiptpath),
                          verifier_exit=run.returncode, deck_sha256=entry['deck_sha256'],
                          inventory_sha256=entry['inventory_sha256'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['failure'] = str(error)
        results.append(result)
        (args.work_root / 'scope-results.json').write_text(json.dumps(dict(
            schema='mpas-written-forecast-reproduction-v1', checkout=str(checkout),
            passed=len(results) == len(entries) and all(r['passed'] for r in results),
            process_count=1, rows=results), indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
