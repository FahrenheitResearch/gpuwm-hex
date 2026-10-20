"""Seal an existing f0 native forecast; this is never hex acceptance.

No compilation, field decoding, output mutation or tolerance is involved.
Run under the lane cpuset after run_forecast.py finishes successfully.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('publication', type=Path)
parser.add_argument('exit_receipt', type=Path)
parser.add_argument('receipt', type=Path)
args = parser.parse_args()
if args.exit_receipt.read_text().strip() != '0':
    raise ValueError('native forecast did not finish successfully')
manifest_path = args.run / 'MANIFEST.json'
manifest = json.loads(manifest_path.read_text())
for category in ('inputs', 'outputs'):
    for filename, expected in manifest[category].items():
        if Path(filename).name != filename:
            raise ValueError('manifest filename escapes run directory')
        if sha(args.run / filename) != expected:
            raise ValueError(f'native {category} file hash changed: {filename}')
log = args.run / 'log.atmosphere.0000.out'
text = log.read_text()
for category in ('Error', 'Critical error'):
    counts = [line.split('=')[1].strip() for line in text.splitlines()
              if line.strip().startswith(category + ' messages =')]
    if counts != ['0']:
        raise ValueError(f'native log lacks zero {category} messages')
timesteps = [line.strip() for line in text.splitlines()
             if 'Begin timestep' in line]
if not timesteps:
    raise ValueError('native forecast has no timesteps')
receipt = dict(
    status='PASS', scope='strict native baseline only; no hex forecast acceptance',
    run_directory=str(args.run.resolve()), manifest_sha256=sha(manifest_path),
    case_publication_sha256=sha(args.publication), native_manifest=manifest,
    output_files=len(manifest['outputs']),
    stream_files=dict(collections.Counter(name.split('.')[0]
                                          for name in manifest['outputs'])),
    exit=0, last_timestep=timesteps[-1], log_sha256=sha(log),
    native_source_rebuilt=False, forecast_hex_proven=False)
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS native baseline', receipt['output_files'], 'sealed files')
