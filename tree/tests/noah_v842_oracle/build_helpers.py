"""Generate supplemental entry wrappers; never replace native numerical routines."""
from pathlib import Path
import json
import re
import os
import sys
from hexcore.mpasphys.noah_v842 import _CORE_CUDA

base=Path(__file__).resolve().parent
def extract_private():
    ref=Path(os.environ["MPAS_REF"])/"src/core_atmosphere/physics/physics_wrf/module_sf_noahlsm_glacial_only.F"
    text=ref.read_text();blocks=[]
    for routine in ('SNOW_NEW','ALCALC'):
        start=text.index('  SUBROUTINE '+routine+' (');end=text.index('  END SUBROUTINE '+routine,start)+len('  END SUBROUTINE '+routine)
        block=text[start:end]
        block=block.replace('    IMPLICIT NONE','    USE module_sf_noahlsm, ONLY: EMISSI_S, LVCOEF_DATA\n    IMPLICIT NONE')
        blocks.append(block)
    (base/"private_helpers.F90").write_text("! Unchanged private numerical bodies exposed only to supplemental tests.\n"+'\n'.join(blocks)+"\n")


if '--private-only' in sys.argv:
    extract_private()
    raise SystemExit(0)

names=('noah_fac2mit','noah_frh2o','gl_snow_new','noah_evapo','noah_nopac','noah_snopac','noah_penman','noah_tdfcnd','noah_alcalc','gl_alcalc','noah_devap_hydro','noah_snow_new')
cuda=[];native=[];layouts={}
for name in names:
    signature=re.search(r'__device__ void '+name+r'\(([^;]+?)\);',_CORE_CUDA)[1]
    fs=[];ints=[];ca=[];fa=[];pre=[];post=[];fb=[]
    for spec in signature.split(', ')[1:]:
        typ,sep,nm=re.fullmatch(r'(float|int|bool)([&*]?) (\w+)',spec).groups()
        if nm.startswith('present_'):ca.append('true');continue
        group=fs if typ=='float' else ints;buf='x' if typ=='float' else 'ix';at=len(group)
        if sep=='*':
            group.extend(nm+f'[{k}]' for k in range(4));ca.append(f'x+{at}');fa.append(f'x({at+1}:{at+4},c)')
        else:
            group.append(nm);arg=nm+'_arg';pre.append(f'{typ} {arg}={buf}[{at}];');post.append(f'{buf}[{at}]={arg};');ca.append(arg)
            if typ=='bool':
                fb.append(nm);fa.append(arg)
            else:fa.append(f'{buf}({at+1},c)')
    key=name;layouts[key]={'float':fs,'int':ints}
    cuda.append(f'''extern "C" __global__ void helper_{key}(float* xs,int* ixs,const NoahTables* ts,int* status,int n){{
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*{len(fs)};int* ix=ixs+c*{len(ints)};NoahTables tables=*ts;tables.error=0;
{''.join(pre)}
{name}(tables,{','.join(ca)});
{''.join(post)}
status[c]=tables.error;
}}
''')
    mod='module_sf_noahlsm_glacial_only' if name.startswith('gl_') else 'module_sf_noahlsm';routine=name.split('_',1)[1].upper()
    bdecl='logical :: '+','.join(n+'_arg' for n in fb) if fb else ''
    bpre='\n'.join(f'{n}_arg=ix({ints.index(n)+1},c)/=0' for n in fb)
    bpost='\n'.join(f'ix({ints.index(n)+1},c)=merge(1,0,{n}_arg)' for n in fb)
    joined=', &\n'.join(','.join(fa[k:k+4]) for k in range(0,len(fa),4))
    native.append(f'''subroutine helper_{key}(x,ix,n) bind(C,name="helper_{key}")
use iso_c_binding
{'' if name.startswith('gl_') else ('use '+mod+', only: '+routine)}
implicit none
{('external :: '+routine) if name.startswith('gl_') else ''}
integer(c_int),value :: n
real(c_float) :: x({len(fs)},n)
integer(c_int) :: ix({max(1,len(ints))},n)
integer :: c
{bdecl}
do c=1,n
{bpre}
call {routine}({joined})
{bpost}
enddo
end subroutine
''')
(base/'helper_entry.cu').write_text('\n'.join(cuda));(base/'helper_replay.F90').write_text('\n'.join(native));(base/'helper_layouts.json').write_text(json.dumps(layouts,indent=2))

extract_private()
