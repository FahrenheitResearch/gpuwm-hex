"""Expose MPAS-private leaf routines only in a test copy, preserving numerics."""
from pathlib import Path
import importlib.util
import json
import sys
root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('transcribe',Path(__file__).with_name('transcribe.py'))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
source=(root/'sf_mynn.F90').read_text();t.parse(source)
leaves=[r for name,r in t.ROUTINES.items() if name not in ('sf_mynn_init','sf_mynn_finalize','sf_mynn_run','psi_init')]
meta=[]
fcode='''subroutine native_leaves(data,which,psi,n) bind(c)
use iso_c_binding
use sf_mynn
implicit none
integer(c_int),value :: n,which,psi
real(c_float),intent(inout) :: data(n,11)
integer :: i
 do i=1,n
 select case(which)
'''
cuda='''extern "C" __global__ void sf_mynn_leaves(float *data,int which,int psi,int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
switch(which) {
'''
for number,r in enumerate(leaves):
    floats=[a for a in r.args if r.variables[a][0]=='float'];ints=[a for a in r.args if r.variables[a][0]=='int'];bools=[a for a in r.args if r.variables[a][0]=='bool']
    meta.append({'name':r.name,'float_args':floats,'int_args':ints,'bool_args':bools})
    args=[];ca=[]
    for a in r.args:
        if a in floats:
            j=floats.index(a);args.append('data(i,'+str(j+1)+')');ca.append('data['+str(j)+'*n+i]')
        elif a in ints:args.append('psi');ca.append('psi')
        elif a in bools:args.append('data(i,10)>0.5');ca.append('data[9*n+i]>0.5f')
    ca+=['true']*len(r.optional)
    fcall=r.name+'('+','.join(args)+')'
    ccall='m_'+r.name+'('+','.join(ca)+');'
    fcode+=' case('+str(number)+')\n'
    fcode+=('data(i,11)='+fcall if r.result else 'call '+fcall)+'\n'
    cuda+='case '+str(number)+': '+('data[10*n+i]=' if r.result else '')+ccall+' break;\n'
fcode+=' end select\n enddo\nend subroutine\n'
cuda+='}\n}\n'
# Visibility-only patch; retain an exact diff for review.
exports='\n'.join('public :: '+r.name for r in leaves)+'\npublic :: psi_init\n'
(root/'sf_mynn_leaves.F90').write_text(source.replace('  contains','  '+exports+'\n  contains',1))
fcode+='''subroutine native_leaf_init(psi) bind(c)
use iso_c_binding
use sf_mynn
integer(c_int),value :: psi
call psi_init(psi)
end subroutine
'''
(root/'leaf_bridge.F90').write_text(fcode)
(root/'leaf-kernel.cu').write_text(cuda)
(root/'leaf-meta.json').write_text(json.dumps(meta,indent=2)+'\n')
# Generated host witness shares the same scalar transcription and dispatch.
cpp=(root/'witness.cpp').read_text()
cpp+=cuda
cpp+='''extern "C" void witness_leaves(float* data,int which,int psi,int n) {
blockDim.x=1;blockIdx.x=0;
for(int i=0;i<n;++i) { threadIdx.x=i; sf_mynn_leaves(data,which,psi,n); }
threadIdx.x=0;
}
extern "C" void witness_leaf_init(int psi) { m_psi_init(psi); }
'''
(root/'leaf-witness.cpp').write_text(cpp)
(root/'LeafMakefile').write_text('''FFLAGS = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -fprofile-arcs -ftest-coverage -ffree-line-length-none
CFLAGS = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -std=c++17
all: leaf-native.so leaf-witness.so
kinds.o: kinds.F90
\tgfortran $(FFLAGS) -c $<
mynn_shared.o: mynn_shared.F90 kinds.o
\tgfortran $(FFLAGS) -c $<
sf_mynn_leaves.o: sf_mynn_leaves.F90 mynn_shared.o
\tgfortran $(FFLAGS) -c $<
leaf_bridge.o: leaf_bridge.F90 sf_mynn_leaves.o
\tgfortran $(FFLAGS) -c $<
leaf_columns.o: bridge.F90 sf_mynn_leaves.o
\tgfortran $(FFLAGS) -c $< -o $@
leaf-native.so: sf_mynn_leaves.o leaf_bridge.o kinds.o mynn_shared.o leaf_columns.o
\tgfortran -shared -fprofile-arcs $^ -o $@
leaf-witness.so: leaf-witness.cpp
\tg++ $(CFLAGS) -shared $< -o $@
''')
