"""Authenticate every final output hash before trusting a whole-bank diagnostic."""
from pathlib import Path
import argparse,hashlib,json

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ledger(root,summary):
 s=json.loads(summary.read_text()); rows={}
 assert s['grade_records']==49152 and s['words']==13892608 and s['excluded_words']==[]
 for c in s['grade_chunks']:
  p=root/Path(c['path']).name
  assert sha(p)==c['sha256']
  chunk=json.loads(p.read_text());assert len(chunk)==c['records']
  for r in chunk:
   k=(r['deck'],r['call'],r['field']);assert k not in rows
   rows[k]=r
 assert len(rows)==49152
 return s,rows

def main(a):
 ls,left=ledger(a.left,a.left_summary);rs,right=ledger(a.right,a.right_summary)
 assert set(left)==set(right)
 mismatch=[]
 for k,l in left.items():
  r=right[k]
  if any(l[f]!=r[f] for f in ('words','reference_sha256','candidate_sha256','differing_words','difference_mask_sha256','excluded_words')):mismatch.append(k)
 a.output.write_text(json.dumps(dict(scope=__doc__,process_count=1,excluded_words=[],
  left_summary_sha256=sha(a.left_summary),right_summary_sha256=sha(a.right_summary),
  all_final_payloads_and_masks_identical=not mismatch,grade_records=len(left),words=ls['words'],
  differing_ledger_records=mismatch,diagnostic_only=True,forecast_proven=False),indent=2)+'\n')
 return int(bool(mismatch))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['left','right','left-summary','right-summary','output']:p.add_argument('--'+n,type=Path,required=True)
 raise SystemExit(main(p.parse_args()))
