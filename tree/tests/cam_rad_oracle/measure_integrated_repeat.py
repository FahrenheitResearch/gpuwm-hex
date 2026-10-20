"""Compare every native integrated-driver word across independent executions.

This measures the standalone test executable, not f0's pinned atmosphere_model.
It cannot authorize exclusions or extend any whole-model identity claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket

from compact_codec import codec


def measure(first, repeats, inventory_path, receipt):
    inventory = json.loads(inventory_path.read_text())
    transport = codec()
    binary = hashlib.sha256((first / 'integrated_columns').read_bytes()).hexdigest()
    comparisons = []
    for repeat in repeats:
        if hashlib.sha256((repeat / 'integrated_columns').read_bytes()).hexdigest() != binary:
            raise ValueError('Independent native run used a different executable')
        case = dict(directory=str(repeat))
        for filename, fields, records in (
            ('integrated-before.bin', inventory['fields'], inventory['records']),
            ('integrated-after.bin', inventory['fields'], inventory['records']),
            ('cam-state.bin', inventory['schema']['globals'], 1),
            ('cam-state-after.bin', inventory['schema']['globals'], 1),
        ):
            left, right = first / filename, repeat / filename
            result = transport.compare(left, right, [dict(name=filename, fields=fields, records=records)])
            result.update(first_sha256=hashlib.sha256(left.read_bytes()).hexdigest(),
                          repeat_sha256=hashlib.sha256(right.read_bytes()).hexdigest())
            case[filename] = result
        comparisons.append(case)
    result = dict(scope='native standalone integrated CAM driver run-to-run measurement',
                  complete=True, process_count=1, host=socket.gethostname(),
                  native_binary_sha256=binary, comparisons=comparisons,
                  transport_library_sha256=hashlib.sha256(transport.path.read_bytes()).hexdigest(),
                  excluded_words=0, exclusions_authorized=False,
                  pinned_atmosphere_model_witness=False, whole_model_acceptance=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, action='append', required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    measure(args.first, args.repeat, args.inventory, args.receipt)
