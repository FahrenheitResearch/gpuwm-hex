"""Reproduce a source-location diagnostic of the entire CAM numeric module.

Only source locations are reported. The diagnostic observes the first active
column and adds host-visible writes/fences; it supplies no identity proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


HELPER='''
#ifndef CAM_HOST
extern "C" __device__ unsigned long long cam_diagnostic_progress=0;
__device__ __noinline__ void cam_diag(int module,int line) {
 auto p=(volatile int*)cam_diagnostic_progress;
 if(p && threadIdx.x==0 && blockIdx.x==0) {p[0]=module;p[1]=line;__threadfence_system();}
}
#else
#define cam_diag(module,line) ((void)0)
#endif

'''


def prepare(source,output,receipt):
    text=source.read_text();lines=text.splitlines(keepends=True)
    result=[];inside=False;depth=0;count=0
    for line in lines:
        if line.rstrip()=='struct CamColumn {':
            assert not inside
            result.append(HELPER);inside=True;depth=0
        result.append(line)
        match=re.fullmatch(r'// MPAS (module_ra_cam(?:_support)?\.F):(\d+)\n',line)
        if inside and depth>=2 and match:
            module=2 if 'support' in match[1] else 1
            result.append(f'cam_diag({module},{match[2]});\n');count+=1
        if inside:
            depth+=line.count('{')-line.count('}')
            if depth==0:inside=False
    assert count>0 and not inside
    candidate=''.join(result)
    # Match the diagnostic generator's final CUDA literal newline exactly.
    ending='\n*error=cam.error;}\n\n"""'
    assert candidate.count(ending)==1
    candidate=candidate.replace(ending,'\n*error=cam.error;}\n"""')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(candidate)
    receipt.write_text(json.dumps(dict(scope='whole CAM source-location diagnostic; no identity proof',
        observed_column='first active CUDA column only',writes=count,
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        whole_model_acceptance=False,identity_checked=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','receipt'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.output,a.receipt)
