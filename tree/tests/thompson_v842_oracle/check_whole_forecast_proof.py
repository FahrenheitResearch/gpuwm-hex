"""Check a complete actual Thompson forecast against the canonical owner policy.

Orchestration only: field words come from the real Rust verifier. Existing
native-only omissions require the integrator's unchanged recorded inventory;
BITWISE-WRITTEN-FIELDS is never described as complete native BITWISE identity.
This helper does not execute a model, normalize output, or alter the baseline.
"""
import hashlib,json,sys
from pathlib import Path
receipt_path=Path(sys.argv[1]);receipt=json.loads(receipt_path.read_text());metadata=Path(sys.argv[2])
assert receipt['passed'] and receipt['evidence_kind']=='FORECAST', 'completed passing FORECAST required'
deck=json.loads(Path(sys.argv[3]).read_text());baseline_path=Path(sys.argv[4])
baseline=json.loads(baseline_path.read_text())
from hexcore.verify_mpas import identity, aggregate, ratchet_failures, recorded_missing_failures
seal=json.loads((metadata/'preservation.json').read_text())
for name,expected in seal['observed_sha256'].items():
    assert hashlib.sha256((metadata/name).read_bytes()).hexdigest()==expected
request=json.loads((metadata/'native-physics-request.json').read_text())
progress=json.loads((metadata/'native-physics-receipt.json').read_text())
assert receipt['passed']
fields=receipt['comparison']['fields'];assert fields
context={**identity(deck), 'process_count': 1}
assert all(receipt[key]==value for key,value in context.items())
assert not ratchet_failures(aggregate(fields,context),baseline)
assert any(row['status']=='compared' for row in fields)
assert all(row['differing_words']==0 for row in fields if row['status']=='compared')
accepted=receipt['acceptance_tier']
if accepted=='BITWISE':
    assert receipt['tier']=='BITWISE' and receipt['all_native_fields_bitwise']
    assert all(row['status']=='compared' for row in fields)
else:
    # Integrator's binding 09:22Z landing note and unchanged foundation policy.
    # This tier certifies written fields only; never label it full native identity.
    assert accepted=='BITWISE-WRITTEN-FIELDS'
    policy=receipt['recorded_missing_policy']
    assert policy['ruling']=='integrator-2026-10-13T07:45Z'
    assert policy['applied'] and not policy['failures'] and policy['process_count']==1
    assert policy['baseline_sha256']==hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert not recorded_missing_failures(receipt['comparison'],context,baseline)
print(receipt['deck'],accepted,'process_count=1')
assert receipt['model_execution'] and receipt['evidence_kind']=='FORECAST'
assert not receipt['ratchet_regressions'] and not receipt['declared_libm_residue']
assert receipt['process_count']==1
assert progress['completed'] and progress['steps_completed']==progress['steps_requested']==request['steps']
assert progress['physics']==request['physics']
backends=request['physics']['backends'];assert backends['microphysics']=='hex-cuda'
assert all(v=='native-cpu' for k,v in backends.items() if k!='microphysics'),backends
assert request['physics']['selectors']['microphysics']=='mp_thompson'
