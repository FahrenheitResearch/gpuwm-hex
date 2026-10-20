"""Preserve actual run-door JSON before verify-mpas retires owned output.

Only orchestration metadata is copied. Weather files remain verifier-owned.
Partial JSON writes are retried; missing metadata never becomes a proof pass.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def preserve(source, target, stopped, parent):
    target.mkdir(parents=True, exist_ok=True)
    observed = {}
    while True:
        for name in ('native-physics-request.json', 'native-physics-receipt.json'):
            try:
                data = (source/name).read_bytes()
                json.loads(data)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            digest = hashlib.sha256(data).hexdigest()
            if observed.get(name) != digest:
                pending = target/(name+'.pending')
                pending.write_bytes(data)
                pending.replace(target/name)
                observed[name] = digest
        if stopped.exists():
            break
        try:
            os.kill(parent, 0)
        except ProcessLookupError:
            break
        time.sleep(0.25)
    (target/'preservation.json').write_text(json.dumps(dict(
        acceptance=False, source=str(source), observed_sha256=observed,
        scope='unchanged actual run-door JSON; last complete observed write'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--stopped', type=Path, required=True)
    parser.add_argument('--parent', type=int, required=True)
    args = parser.parse_args()
    preserve(args.source.resolve(), args.target.resolve(), args.stopped.resolve(), args.parent)
