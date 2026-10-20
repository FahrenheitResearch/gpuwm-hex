#!/usr/bin/env python3
"""Run the unchanged whole-model verifier and preserve native completion evidence.

The verifier launches the actual model door and retains its complete word
comparison and acceptance policy. This test-only wrapper saves the door's
completion receipt before retiring any output. Failed comparisons retain
output for debugging; no words, fields or acceptance requirements are removed.
Native driver authority: mpas_atmphys_driver.F:202-373.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from hexcore import verify_mpas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify_mpas.add_verify_mpas_parser(commands)
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.hex_output is not None:
        parser.error('this whole-forecast runner requires an actual model execution')
    start = time.monotonic()
    args._owns_work = False
    args._context = {}
    try:
        # These are immutable test-provider binaries, separate from weather
        # inputs. Reject a stale provider selection before launching the door.
        deck = verify_mpas.read_json(args.deck)
        for reference in deck.get('hex_support_inputs', {}).values():
            path = Path(reference['path'])
            if not path.is_absolute():
                path = args.deck.resolve().parent / path
            if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
                raise verify_mpas.VerificationRefusal(f'test-provider digest differs: {path}')
        receipt = verify_mpas.verify(args)
    except (verify_mpas.VerificationRefusal, OSError, KeyError, TypeError,
            ValueError, IndexError, ArithmeticError, ET.ParseError) as error:
        receipt = {'schema': 'mpas-verification-v1', 'passed': False, 'tier': None,
                   **args._context, 'process_count': args._context.get('process_count'),
                   'failures': [str(error)], 'evidence_kind': 'REFUSAL'}
    receipt['wall_seconds'] = time.monotonic() - start
    receipt['test_runner_sha256'] = runner_sha256
    verify_mpas.write_json(args.receipt, receipt)
    output = args.work_root.resolve() / 'hex-output'
    completion = output / 'native-physics-receipt.json'
    if args._owns_work and completion.is_file():
        target = args.receipt.with_name(args.receipt.stem + '-completion.json')
        # Preserve exactly the model's evidence, without inventing completion.
        shutil.copyfile(completion, target)
    if receipt['passed'] and args._owns_work and output.is_dir():
        shutil.rmtree(output)
    print(json.dumps({'receipt': str(args.receipt), 'passed': receipt['passed'],
                      'evidence_kind': receipt['evidence_kind'],
                      'retained_failed_output': str(output) if output.is_dir() else None}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
