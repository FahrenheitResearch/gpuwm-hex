"""Trace complete1088-profile native/host driver calls at their first divergence.

Diagnostic copies retain complete calls. Only trace output is selected; the
unchanged complete replay is the referee, with no omitted acceptance words.
"""
from pathlib import Path
import argparse,re,shutil,json
from translate_module import parse_modules


def main(a):
 root=a.output;root.mkdir(exist_ok=False)
 if not a.decks or any(d<a.first_deck or d>=a.first_deck+a.profiles for d in a.decks):raise ValueError('Trace selection must belong to the complete unchanged bank')
 condition=' .or. '.join(f'(p12_trace_context>={(d-a.first_deck)*12+1}.and.p12_trace_context<={(d-a.first_deck)*12+12})' for d in a.decks)
 c_condition=' || '.join(f'(p12_trace_context>={(d-a.first_deck)*12+1}&&p12_trace_context<={(d-a.first_deck)*12+12})' for d in a.decks)
 for kind in ['native','host']:
  src=a.native if kind=='native' else a.host;dest=root/kind;dest.mkdir()
  for p in src.iterdir():
   if p.suffix in ['.F90','.F','.cpp','.cuh'] or p.name=='Makefile':shutil.copyfile(p,dest/p.name)
  original=a.native;routines=parse_modules([original/'mynn_shared.F90',original/'bl_mynn_subroutines.F90',original/'bl_mynn.F90'])[1]
  selected={}
  for name,limits in [(name,(0,100000)) for name in a.routines]:
   r=routines[name];assignments={}
   lines=(original/r.source).read_text().splitlines()
   for number,stmt in r.body:
    m=re.match(r'(\w+)(\(k\))?\s*=([^=].*)',stmt)
    if not m or not limits[0]<=number<=limits[1]:continue
    var=r.variables.get(m[1]);
    if var is None or var.kind not in ['float','double'] or bool(var.dimensions)!=bool(m[2]):continue
    end=number
    while lines[end-1].split('!',1)[0].rstrip().endswith('&'):end+=1
    assignments[end]=(number,m[1]+(m[2] or ''),var.kind)
   selected[name]=dict(routine=r,assignments=assignments)
  if kind=='native':
   for filename in ['bl_mynn_subroutines.F90','bl_mynn.F90']:
    p=dest/filename;lines=p.read_text().splitlines();out=[]
    for number,line in enumerate(lines,1):
     if filename=='bl_mynn_subroutines.F90' and number==413:
      out.extend(['  integer, save, public :: p12_trace_context=0'])
     out.append(line)
     if filename=='bl_mynn.F90' and number==437:
      assert line.strip().lower()=='do i = its,ite'
      out.append(' p12_trace_context=p12_trace_context+1')
     for name,data in selected.items():
      if data['routine'].source==filename and number in data['assignments']:
       n,expr,kind=data['assignments'][number]
       expr=expr.replace('mynn_enum','enum')
       coordinate='k' if '(' in expr else '0'
       out.append(f"    if({condition}) call p12_trace_{kind}('{name}_line{n}',p12_trace_context,{coordinate},{expr})")
    p.write_text('\n'.join(out)+'\n')
   p=dest/'framework_stubs.F90';p.write_text(p.read_text()+'''
subroutine p12_trace_float(name,count,k,x)
implicit none
character(len=*) :: name
integer :: count,k
real :: x
logical, save :: opened=.false.
if(.not.opened) then
open(unit=89,file='stability.words',status='replace')
opened=.true.
endif
write(89,'(a,4(1x,i0),1x,z8.8)') 'R out:'//trim(name),count,1,k,0,transfer(x,0)
end subroutine
subroutine p12_trace_double(name,count,k,x)
implicit none
character(len=*) :: name
integer :: count,k,bits(2)
double precision :: x
real :: lo,hi
bits=transfer(x,bits)
lo=transfer(bits(1),lo);hi=transfer(bits(2),hi)
call p12_trace_float(name//'_lo',count,k,lo)
call p12_trace_float(name//'_hi',count,k,hi)
end subroutine
''')
  else:
   p=dest/'translation.cpp';text=p.read_text();marker='// Generated from pinned MPAS MMM; see translation receipt.'
   helper=r'''
#include <cstdio>
static FILE* p12_trace_file=nullptr;
static int p12_trace_context=0;
template<class T> static void p12_trace(const char* name,int count,int k,T x) {
if(!p12_trace_file)p12_trace_file=std::fopen("stability.words","w");
uint32_t bits[2]={0,0};std::memcpy(bits,&x,sizeof(T));
if(sizeof(T)==4)std::fprintf(p12_trace_file,"R out:%s %d 1 %d 0 %08X\n",name,count,k,bits[0]);
else {std::fprintf(p12_trace_file,"R out:%s_lo %d 1 %d 0 %08X\n",name,count,k,bits[0]);std::fprintf(p12_trace_file,"R out:%s_hi %d 1 %d 0 %08X\n",name,count,k,bits[1]);}
std::fflush(p12_trace_file);
}
'''
   text=text.replace(marker,helper+marker,1);out=[];active=None;source_line=0
   for line in text.splitlines():
    out.append(line)
    m=re.match(r'__device__ __noinline__ \w+ f_(\w+)\(',line)
    if m:
     active=selected.get(m[1]);

    if line.startswith('for (i=') and source_line==437 and active and active['routine'].name=='bl_mynn_run':
     out.append('++p12_trace_context;')
    if active:
     m=re.match(r'// '+re.escape(active['routine'].source)+r':(\d+)',line)
     if m:source_line=int(m[1])
     for n,expr,kind in active['assignments'].values():
      # Compiler-order branches can wrap the original assignment on one line.
      # Record its resulting value after either branch, just as Fortran does.
      if n==source_line and re.search(r'(?<!\w)'+re.escape(expr)+r'\s*=',line):
       coordinate='k' if '(' in expr else '0'
       out.append(f'if({c_condition})p12_trace("{active["routine"].name}_line{n}",p12_trace_context,{coordinate},{expr});')
   p.write_text('\n'.join(out)+'\n')
 (root/'scope.json').write_text(json.dumps(dict(scope=__doc__,decks=a.decks,all_levels=True,profiles=a.profiles,first_deck=a.first_deck,selected_routines=a.routines,complete_driver_bank_executed=True),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--native',type=Path,default=Path('scratch/native-branch-1088-v1'))
 p.add_argument('--host',type=Path,default=Path('scratch/host-ieee-1088-v8'))
 p.add_argument('--output',type=Path,default=Path('scratch/complete-source-trace-v1'))
 p.add_argument('--decks',type=int,nargs='+',default=[577,593])
 p.add_argument('--first-deck',type=int,default=0)
 p.add_argument('--profiles',type=int,default=1088)
 p.add_argument('--routines',nargs='+',default=['mym_initialize','mym_length','mym_turbulence','mym_predict','mym_condensation','mynn_tendencies','bl_mynn_run'])
 main(p.parse_args())
