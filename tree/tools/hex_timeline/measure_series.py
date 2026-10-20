"""Run the cold arms and three warm ABBA repeats through the C3 controller."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=14)
    args = parser.parse_args()
    lane = args.lane.resolve()
    summary = lane / 'series.json'
    rows = json.loads(summary.read_text()) if summary.exists() else []
    sequence = [('B', 'COLD'), ('A', 'COLD')] + [(arm, 'WARM') for arm in 'ABBA' * 3]
    if not 1 <= args.stop_after <= len(sequence):
        raise ValueError('stop-after must be within the fixed measurement sequence')
    for index, (arm, temperature) in enumerate(sequence):
        if index >= args.stop_after:
            break
        if index < len(rows):
            if rows[index]['arm'] != arm or rows[index]['temperature'] != temperature:
                raise ValueError('saved series differs from the fixed ABBA sequence')
            continue
        source = lane / ('base' if arm == 'A' else 'src') / 'tree'
        env = dict(os.environ, PYTHONPATH=str(source / 'src'), BENCH_REFERENCE='1' if arm == 'A' else '0')
        command = [sys.executable, str(args.controller / 'tools/bench/protocol.py'), 'run', args.case,
            temperature, '--cards', '1', '--est', '1500', '--wait', '3600', '--cache-id', 'hex-' + arm.lower()]
        existing = set((lane / 'receipts').glob(args.case + '-' + temperature + '-*'))
        result = subprocess.run(command, cwd=source, env=env)
        created = set((lane / 'receipts').glob(args.case + '-' + temperature + '-*')) - existing
        if len(created) != 1:
            raise RuntimeError('controller did not create exactly one measurement receipt')
        output = created.pop()
        if result.returncode:
            raise RuntimeError(f'{arm} {temperature} failed: {output}/receipt.json')
        receipt = json.loads((output / 'receipt.json').read_text())
        case = json.loads((output / 'case-summary.json').read_text())
        rows.append(dict(index=index, arm=arm, temperature=temperature, output=str(output),
            door_wall_seconds=case['door_wall_seconds'], steps=case['steps'],
            final_state_sha256=case['final_state_sha256'], history_sha256=case['history_sha256'],
            proxy_ff=receipt['proxy_ff'], proxy_wall_seconds=receipt['proxy_wall_seconds'],
            integrate_union_seconds=receipt['stepping_wall_seconds']))
        summary.write_text(json.dumps(rows, indent=2) + '\n')
        print(f'SERIES {index + 1}/{len(sequence)} {arm} {temperature} complete: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
