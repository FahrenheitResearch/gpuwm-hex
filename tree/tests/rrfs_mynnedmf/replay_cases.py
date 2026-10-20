#!/usr/bin/env python3
"""Run both complete real calls of both archived cases; fail on any word."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--selections',required=True);p.add_argument('--scratch',required=True);p.add_argument('--snapshot');a=p.parse_args()
root=Path(a.selections);scratch=Path(a.scratch);tool=Path(__file__).with_name('replay_capture.py');results=[]
for case,bank in [('rap','real'),('gfs','real-gfs')]:
 for selection in sorted((root/case).glob('mynnedmf*.json')):
  receipt=scratch/'results'/case/(selection.stem+'.json')
  command=[sys.executable,str(tool),'--bank',str(scratch/bank),'--selection',str(selection),'--image',str(scratch/'mynn.cubin'),'--io',str(scratch/'oracle_io'),'--words',str(scratch/'oracle_words'),'--out',str(scratch/'replay-output'/case/selection.stem),'--receipt',str(receipt)]
  if a.snapshot:command.extend(['--snapshot',a.snapshot])
  result=subprocess.run(command);results.append({'case':case,'capture':selection.stem,'exit':result.returncode,'receipt':str(receipt)})
if len(results)!=4:raise RuntimeError('exactly four stated real invocations are required')
path=scratch/'results/summary.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'schema':'rrfs-mynnedmf-real-case-matrix-v1','calls':results,'bitwise':all(r['exit']==0 for r in results)},indent=2)+'\n')
raise SystemExit(0 if all(r['exit']==0 for r in results) else 1)
