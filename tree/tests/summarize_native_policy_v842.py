"""Word evidence for policy items 2/3; no weather processing in runtime code."""
from pathlib import Path
import hashlib,json
import numpy as np

import os
root=Path(os.environ.get('MPAS_POLICY_INPUT','n1out/policy'))
runs={}
for run in sorted(root.iterdir()):
    frames={}
    for manifest in sorted(run.glob('*/fields.json')):
        frame={}
        for f in json.loads(manifest.read_text()):
            words=np.fromfile(manifest.parent/(f['field']+'.raw'),dtype='<u4')
            item=dict(sha256=hashlib.sha256(words.tobytes()).hexdigest(),words=int(words.size),shape=f['shape'])
            if f['shape'][-1]==55:
                top=words.reshape(-1,55)[:,-1]
                unique,count=np.unique(top,return_counts=True)
                item['top_words']={hex(int(k)):int(n) for k,n in zip(unique,count)}
            if f['field']=='iLev_DT':
                unique,count=np.unique(words,return_counts=True)
                item['level_counts']={str(int(k)):int(n) for k,n in zip(unique,count)}
            frame[f['field']]=item
        frames[manifest.parent.name]=frame
    runs[run.name]=frames
comparisons=[]
for prefix in ('n1','n2','n1-step','n2-step'):
    a,b=runs[prefix+'-r1-a'],runs[prefix+'-r1-b']
    comparisons.append(dict(left=prefix+'-r1-a',right=prefix+'-r1-b',identical=a==b))
for prefix in ('','-step'):
    for ranks in ('r1-a','r1-b','r4'):
        a,b=runs['n1'+prefix+'-'+ranks],runs['n2'+prefix+'-'+ranks]
        comparisons.append(dict(left='n1'+prefix+'-'+ranks,right='n2'+prefix+'-'+ranks,identical=a==b))
rankdiff=[]
for prefix in ('n1','n2','n1-step','n2-step'):
    for frame in runs[prefix+'-r1-a']:
        for field,a in runs[prefix+'-r1-a'][frame].items():
            b=runs[prefix+'-r4'][frame][field]
            if a['sha256']==b['sha256']:continue
            wa=np.fromfile(root/(prefix+'-r1-a')/frame/(field+'.raw'),dtype='<u4')
            wb=np.fromfile(root/(prefix+'-r4')/frame/(field+'.raw'),dtype='<u4')
            indices=np.flatnonzero(wa!=wb)
            rankdiff.append(dict(host_run=prefix,time=frame,field=field,different_words=int(indices.size),
                first_indices=indices[:16].tolist(),level_counts={str(int(k)):int(n) for k,n in zip(*np.unique(indices%55+1,return_counts=True))} if a['shape'][-1]==55 else None,
                any_nonfinite=bool(np.any(~np.isfinite(wa[indices].view('<f4'))) or np.any(~np.isfinite(wb[indices].view('<f4'))))))
receipt=dict(tier='strict real-native defect measurement',whole_forecast_proof=False,
    native_sha256='a71feef594ae020f86aa9450608e21b8e2074415479de521438a7701e5b1d479',runs=runs,
    comparisons=comparisons,rank_differences=rankdiff,undefined_exclusions=[])
out=Path('n1out/native-policy-summary.json');out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(comparisons=comparisons,rank_differences=rankdiff),indent=2))
