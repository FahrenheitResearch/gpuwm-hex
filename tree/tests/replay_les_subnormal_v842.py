"""Strict native subnormal witness, including an explicit FTZ failure control."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from hexcore import mixing_v842_les as les
from replay_les_v842 import subnormal_fixtures


def run(reference,receipt):
    import cupy as cp
    from cupy.cuda import compiler
    expected=np.load(reference);original=les._module;rows=[]
    for ftz in [False,True]:
        if ftz:
            options=tuple('--ftz=true' if opt=='--ftz=false' else opt for opt in les.CUDA_OPTIONS)
            binary,_=compiler.compile_using_nvrtc(les._cuda_source(),options=options,
                arch=cp.cuda.Device().compute_capability)
            forced=cp.cuda.function.Module();forced.load(binary)
            les._module=lambda device_id:forced
        try:
            for routine,case,values in subnormal_fixtures():
                spec=les.ROUTINES[routine]
                device={n:cp.array(v,order='F') if spec['fields'][n]['dims'] or spec['fields'][n]['type']=='float' and spec['fields'][n]['intent']!='in' else v.item() for n,v in values.items()}
                les.LesDissipationV842().call(routine,**device)
                cp.cuda.get_current_stream().synchronize()
                want=expected[f'{routine}/{case}/tend_u_euler'].ravel(order='F').view('u4')
                got=cp.asnumpy(device['tend_u_euler']).ravel(order='F').view('u4')
                bad=int(np.count_nonzero(got!=want))
                subnormal=int(np.count_nonzero(((want&0x7fffffff)>0) & ((want&0x7fffffff)<0x800000)))
                rows.append({'forced_ftz':ftz,'case':case,'compared_words':int(want.size),
                             'native_subnormal_words':subnormal,'different_words':bad})
        finally:
            les._module=original
    assert all(r['native_subnormal_words'] for r in rows)
    strict=sum(r['different_words'] for r in rows if not r['forced_ftz'])
    control=sum(r['different_words'] for r in rows if r['forced_ftz'])
    result={'kind':'native subnormal tendency input regression with explicit FTZ failure control',
        'process_count':1,'MPI_used':False,'whole_forecast_proven':False,
        'different_words':strict,'forced_ftz_different_words':control,'cases':rows,
        'native_reference_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(les._cuda_source().encode()).hexdigest(),
        'options':les.CUDA_OPTIONS}
    Path(receipt).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return int(strict!=0 or control==0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('receipt');a=p.parse_args()
    raise SystemExit(run(a.reference,a.receipt))
