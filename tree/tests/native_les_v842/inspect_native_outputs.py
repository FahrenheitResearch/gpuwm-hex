"""Inspect completed native output storage with the existing Rust reader.

Python orchestrates files and JSON receipts only. Finiteness is a native
reference diagnostic, separate from the required whole-model word comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ('3d_smagorinsky', 'prognostic_1.5_order'):
        directory = args.root/model
        assert 'Finished running the atmosphere core' in (directory/'log.atmosphere.0000.out').read_text()
        for file in sorted(directory.glob('*.nc')):
            output = args.out/(model+'-'+file.name+'.json')
            subprocess.run([str(args.reader.resolve()), str(file.resolve()), str(output.resolve())],
                           check=True)
            result = json.loads(output.read_text())
            rows.append(dict(model=model, file=file.name,
                             sha256=hashlib.sha256(file.read_bytes()).hexdigest(), **result))
    record = dict(schema='d2-les-native-finite-v1', host=socket.gethostname(),
                  process_count=1, whole_forecast_proven=False,
                  reader_sha256=hashlib.sha256(args.reader.read_bytes()).hexdigest(),
                  floating_words=sum(row['floating_words'] for row in rows),
                  nonfinite=sum(row['nonfinite'] for row in rows), files=rows)
    (args.out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(files=len(rows), floating_words=record['floating_words'],
                          nonfinite=record['nonfinite'])))


if __name__ == '__main__':
    main()
