"""Rust word validation of actual CUDA records already emitted by f1 replay.
MPAS: mpas_atmphys_driver_radiation_lw.F:865-885. Does not run physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser()
p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
p.add_argument('words',type=Path);p.add_argument('receipt',type=Path)
a=p.parse_args();rows={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for candidate in sorted(a.candidate.glob('*.post.*.bin')):
    reference=a.reference/candidate.name
    row=json.loads(subprocess.check_output([str(a.words),'compare',str(reference),str(candidate)],text=True))
    row.update(reference_sha256=sha(reference),candidate_sha256=sha(candidate))
    rows[candidate.name]=row
assert rows,'no actual emitted records'
result={'scope':'Rust verification of actual CUDA-generated f1 replay records; not forecast acceptance',
        'arrays':rows,'compared_words':sum(x['words'] for x in rows.values()),
        'different_words':sum(x['different_words'] for x in rows.values()),
        'all_outputs_bitwise_equal':all(x['bitwise_equal'] for x in rows.values())}
a.receipt.write_text(json.dumps(result,indent=2)+'\n')
assert result['all_outputs_bitwise_equal']
print(json.dumps({k:v for k,v in result.items() if k!='arrays'}))
