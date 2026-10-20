"""Orchestrate f1 Rust word comparisons of captured hidden state.
MPAS: module_ra_rrtmg_lw.F:1-12535. This audits data; it does not run physics.
"""
import argparse
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser()
p.add_argument('prefix',type=Path)
p.add_argument('words',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
root=a.prefix.parent;base=a.prefix.name
result={'scope':'native pre/post hidden-state audit; not CUDA replay','arrays':{}}
for pre in sorted(root.glob(base+'.pre.*.bin')):
    name=pre.name[len(base+'.pre.'):-4]
    if not name.startswith(('state.','local.')):continue
    post=root/(base+'.post.'+name+'.bin')
    row=json.loads(subprocess.check_output([str(a.words), 'compare',str(pre),str(post)],text=True))
    result['arrays'][name]=row
result['mutated']={name:row for name,row in result['arrays'].items() if not row['bitwise_equal']}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'arrays':len(result['arrays']),'mutated':list(result['mutated'])}))
