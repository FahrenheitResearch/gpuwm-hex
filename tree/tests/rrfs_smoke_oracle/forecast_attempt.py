"""Run the landed verifier and retain its fresh output for difference sheets.

This calls the same verify() used by verify-mpas: that implementation launches
the real checkout CLI and compares every stored word with the pinned native
publication. Only output retirement is deferred to the enclosing reproducer.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import hexcore.verify_mpas as verifier
from hexcore.verify_mpas import VerificationRefusal, verify, write_json
from proof_mode import require_exact_verifier_deck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout', 'deck', 'receipt', 'work-root', 'ratchet', 'comparator'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    assert Path(verifier.__file__).resolve() == checkout / 'tree/src/hexcore/verify_mpas.py', 'verifier must come from the proof checkout'
    assert not subprocess.check_output([
        'git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'
    ]).strip(), 'proof requires committed code'
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    os.environ['PYTHONPATH'] = os.pathsep.join([
        str(checkout / 'tree/src'), str(checkout / 'tree/tools'),
        os.environ.get('PYTHONPATH', '')
    ])
    args.hex_output, args.render = None, False
    args._owns_work, args._context = False, {}
    start = time.monotonic()
    try:
        require_exact_verifier_deck(verifier.read_json(args.deck), args.deck.resolve().parent)
        receipt = verify(args)
    except (VerificationRefusal, OSError, KeyError, TypeError, ValueError,
            IndexError, ArithmeticError, ET.ParseError) as error:
        receipt = dict(schema='mpas-verification-v1', passed=False, tier=None,
                       **args._context, failures=[str(error)], evidence_kind='REFUSAL')
    receipt['wall_seconds'] = time.monotonic() - start
    receipt['r14_checkout_commit'] = head
    receipt['r14_retained_output'] = str(args.work_root.resolve() / 'hex-output')
    write_json(args.receipt, receipt)
    print(f"verify-mpas: {'PASSED' if receipt['passed'] else 'REFUSED'}; receipt {args.receipt}")
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
