"""Validate the real runner's opaque large-file transport using a native artifact.

This tests file transport and the Rust whole-file decoder, not a hex forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('receipt', type=Path)
a = p.parse_args()
root = a.directory.resolve()
metadata = json.loads((root / 'source.json').read_text())
subprocess.run([sys.executable, str(root / 'hydrate_remote.py'), str(root)], check=True)
artifact = root / 'artifact.nc'
h = hashlib.sha256()
with artifact.open('rb') as stream:
    for block in iter(lambda: stream.read(1048576), b''):
        h.update(block)
assert h.hexdigest() == metadata['native_file_sha256']
assert artifact.stat().st_size == metadata['native_file_bytes']
binary = root / 'rw_mpas_verify'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == metadata['comparator_sha256']
described = json.loads(subprocess.check_output([str(binary), '--describe', str(artifact)], text=True))
assert described['times'] and len(described['times']) == 1
copy = root / 'copied-artifact.nc'
shutil.copyfile(artifact, copy)
request = dict(schema='mpas-stream-compare-v1', samples=[dict(
    stream=metadata['stream'], steps=[metadata['step']], times=described['times'],
    strict=str(artifact), hex=str(copy))])
request_path = root / 'request.json'
request_path.write_text(json.dumps(request, indent=2) + '\n')
comparison = a.receipt.with_name(a.receipt.stem + '-comparison.json')
subprocess.run([str(binary), '--request', str(request_path), '--out', str(comparison)], check=True)
rows = json.loads(comparison.read_text())['fields']
assert rows and all(row['status'] == 'compared' and row['differing_words'] == 0 for row in rows)
r = dict(scope='Native byte transport and full Rust file comparison; no GPU/model/forecast acceptance',
         original_host_b_native_file=metadata['native_file'],
         native_file_sha256=metadata['native_file_sha256'], bytes=artifact.stat().st_size,
         chunk_transport=True, numeric_words=sum(row['compared_words'] for row in rows),
         differing_words=0, fields=len(rows), process_count=1,
         forecast_acceptance=False, source_native_reference_removed=False,
         comparator_sha256=metadata['comparator_sha256'],
         comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest(),
         owned_transport_retired=True)
a.receipt.write_text(json.dumps(r, indent=2) + '\n')
shutil.rmtree(root)
print(json.dumps(r), flush=True)
