"""Regenerate the complete backend from the three pristine fork modules."""
import argparse,ast,json
from pathlib import Path
from translate_module import generate


def lower_gnu_operand_order(text):
    """Preserve GNU 15.2 strict option-loop ordering at native F90:1250,2869,3005,3015,3122-3124."""
    for old,new in (
        ('qbuoy1(k) = mpas_add(mpas_add(mpas_mul(elq, mpas_add(mpas_mul(sh(k), gh(k)), gamv)), mpas_mul(p5, tkeprod_dn(k))), mpas_mul(p5, tkeprod_up(k)));',
         'qbuoy1(k) = mpas_add(mpas_add(mpas_mul(elq, mpas_add(mpas_mul(gh(k), sh(k)), gamv)), mpas_mul(p5, tkeprod_dn(k))), mpas_mul(p5, tkeprod_up(k)));'),
        ('mynn_dmul(mynn_dmul(q3sq, mpas_mul(a2, a2fac)), mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel)))',
         'mynn_dmul((tke_budget == 1 ? mynn_dmul(q3sq, mpas_mul(a2, a2fac)) : mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel))), (tke_budget == 1 ? mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel)) : mynn_dmul(q3sq, mpas_mul(a2, a2fac))))'),
        ('mpas_div(r6_neg(mpas_mul(mynn_ipow<3>(ust), zet)), mpas_mul(karman, dz1(k)))',
         'r6_neg(mpas_div(mpas_mul(mynn_ipow<3>(ust), zet), mpas_mul(karman, dz1(k))))'),
        ('mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel))',
         'mynn_dadd(mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel), e2)'),
        ('gamt = mynn_ddiv(r6_neg(mynn_dmul(e1, mynn_enum)), eden);',
         'gamt = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));'),
        ('gamq = mynn_ddiv(r6_neg(mynn_dmul(e1, mynn_enum)), eden);',
         'gamq = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));'),
    ):
        assert old in text, old
        text = text.replace(old, new)
    for name in ("add","sub","mul","div"):
        text=text.replace("mpas_"+name+"(","r6_"+name+"(")
    return text


def emit(root,target,receipt):
    module=root/'src/core_atmosphere/physics/physics_noaa/MYNN-EDMF'
    paths=[module/'MPAS/module_bl_mynnedmf_common.F90',module/'module_bl_mynnedmf.F90',module/'MPAS/module_bl_mynnedmf_driver.F90']
    temporary=receipt.with_suffix('.cuh')
    text,routines=generate(paths,temporary,receipt);temporary.unlink()
    run=routines['mynnedmf_driver']
    params=['int batch_first','int batch_count','unsigned char* workspace','unsigned long long workspace_stride','int* error_flags']
    bindings=[];values=[]
    for n in run.arguments:
        v=run.variables[n]
        if n=='errmsg':values.append('message');continue
        if n=='errflg':values.append('error_flag');continue
        kind='int' if v.kind=='bool' else v.kind
        params.append(f'{kind}{"*" if v.dimensions else ""} {n}')
        if v.dimensions:
            bindings.append(f'MynnArray<{v.kind},{len(v.dimensions)}> a_{n};a_{n}.data={n};')
            stride='1'
            for d,dim in enumerate(v.dimensions):
                edges=dim.split(':');lo=edges[0] if len(edges)==2 else '1';hi=edges[-1]
                bindings.append(f'a_{n}.lo[{d}]={lo};a_{n}.hi[{d}]={hi};a_{n}.stride[{d}]={stride};')
                stride+=f'*({hi}-{lo}+1)'
            values.append('a_'+n)
        elif v.optional:
            # All optional scalar values are part of the MPAS actual argument list.
            params.append('int present_'+n)
            values.append(f'(present_{n} ? MynnOptional<{v.kind}>({n}) : MynnOptional<{v.kind}>())')
        elif v.kind=='bool':values.append(f'({n}!=0)')
        else:values.append(n)
    kernel='''
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653
extern "C" __global__ void rrfs_mynnedmf_run('''+','.join(params)+''') {
int thread=blockIdx.x*blockDim.x+threadIdx.x;
if(thread>=batch_count) return;
int index=batch_first+thread;
MynnArena arena{workspace+thread*workspace_stride,0};
MynnColumn column{arena};
column.active_i=its+index%(ite-its+1);
column.active_j=jts+index/(ite-its+1);
MynnText message;int error_flag=0;
'''+ '\n'.join(bindings)+'\ncolumn.f_mynnedmf_driver('+','.join(values)+');\nerror_flags[index]=error_flag;\n}\n'
    schema={n:{k:getattr(v,k) for k in ('kind','dimensions','intent','optional')} for n in run.arguments for v in [run.variables[n]]}
    # Sum of all automatic arrays is a conservative allocation bound: frames
    # restore the cursor on return, so no call path can exceed this sum.
    arrays=[{'kind':v.kind,'dimensions':v.dimensions} for r in routines.values() for n,v in r.variables.items() if n not in r.arguments and v.dimensions]
    old=target.read_text()
    prefix=old.split('# BEGIN GENERATED RRFS MYNN-EDMF')[0]
    suffix=old.split('# END GENERATED RRFS MYNN-EDMF')[-1] if '# END GENERATED RRFS MYNN-EDMF' in old else ''
    result=prefix+'# BEGIN GENERATED RRFS MYNN-EDMF\nMODULE_CUDA = r\'\'\'\n'+lower_gnu_operand_order(text+kernel)+"'''\n\nARGUMENTS = "+repr(schema)+'\nWORKSPACE_ARRAYS = '+repr(arrays)+'\n# END GENERATED RRFS MYNN-EDMF'+suffix
    target.write_text(result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--target',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args();emit(a.root,a.target,a.receipt)
