"""Classify complete signed native/CUDA masks without removing missing words."""
import hashlib,json,subprocess
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    root=Path('n1out/full-tkeadvect-native-cuda-masks-v1')
    out=Path('n1out/full-tkeadvect-native-cuda-sets-v1');out.mkdir(exist_ok=False)
    tool=Path('n1out/native-mask-sets-tool-v2/p12-native-mask-sets')
    receipt=out/'complete-sets.json'
    result=subprocess.run([str(tool),str(root/'native-repeat/summary.json'),str(root/'cuda/summary.json'),str(receipt)],capture_output=True,text=True)
    (out/'comparison.log').write_text(result.stdout+result.stderr)
    if result.returncode not in [0,1]:raise ValueError('Full native/CUDA mask comparison failed: '+result.stderr)
    data=json.loads(receipt.read_text())
    status={r['file']+'.'+r['field']:r['comparison_kind'] for r in json.loads((root/'cuda/summary.json').read_text())['groups']}
    by_status={}
    # This consumes signed receipt metadata; all word/bitmap processing is Rust.
    rows=data['affected_groups']
    for chunk in data.get('affected_group_chunks',[]):
        path=Path(chunk['file'])
        if sha(path)!=chunk['sha256']:raise ValueError('Affected-set chunk changed')
        rows.extend(json.loads(path.read_text()))
    for row in rows:
        kind=status[row['array_deck_call']]
        counts=by_status.setdefault(kind,dict(native_affected_words=0,cuda_affected_words=0,intersection_words=0,symmetric_difference_words=0,cuda_only_words=0,native_only_words=0))
        counts['native_affected_words']+=row['measured_affected_words']
        counts['cuda_affected_words']+=row['diagnostic_affected_words']
        counts['intersection_words']+=row['intersection_words']
        counts['symmetric_difference_words']+=row['symmetric_difference_words']
        counts['cuda_only_words']+=row['diagnostic_affected_words']-row['intersection_words']
        counts['native_only_words']+=row['measured_affected_words']-row['intersection_words']
    summary=dict(scope=__doc__,process_count=1,words=data['words'],comparison_returncode=result.returncode,
        full_comparison_sha256=sha(receipt),tool_sha256=sha(tool),by_candidate_status=by_status,
        excluded_words=[],landing_proof=False,acceptance='failed; all missing and numerical words remain in the primary grade')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
