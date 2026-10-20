"""Receipt all standalone CAM words using the qualified Rust comparator."""
import hashlib
import json
from compact_codec import codec


def boundary(reference,candidate,inventory,receipt):
    fields=json.loads(inventory.read_text())['schema']['arguments']
    reader=codec()
    result=reader.compare(reference,candidate,[dict(name='CAM boundary',fields=fields,records=None,indirect=True)])
    total=result['differing_words']
    out=dict(scope='standalone whole CAM boundary',
             reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
             candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
             total_differing_words=total,defined_differing_words=result['defined_differing_words'],
             words=result['words'],groups=result['groups'],
             records=result['groups'][0]['boundary_records'],whole_model_acceptance=False,
             process_count=1,excluded_words=0,comparison_result='PASS' if total==0 else 'FAIL',
             transport='rw-mpas Rust compact ABI1',
             transport_library_sha256=hashlib.sha256(reader.path.read_bytes()).hexdigest(),
             native_unwritten_classification='diagnostic only; no approved exclusion')
    receipt.write_text(json.dumps(out,indent=2)+'\n')
    return total


def slices(native,port,groups,receipt,scope):
    reader=codec()
    result=reader.compare(native,port,[dict(name=n,fields=f,records=r) for n,f,r in groups])
    result.update(scope=scope,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                  port_sha256=hashlib.sha256(port.read_bytes()).hexdigest(),
                  transport='rw-mpas Rust compact ABI1',
                  transport_library_sha256=hashlib.sha256(reader.path.read_bytes()).hexdigest(),
                  process_count=1,whole_model_acceptance=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    return result['differing_words']
