"""Trace full576-profile native/host calls at the first stability divergence.

Diagnostic copies retain complete calls. Only trace output is selected; the
unchanged complete replay is the referee, with no omitted acceptance words.
"""
from pathlib import Path
import argparse,re,shutil,json
from translate_module import parse_modules


def main(a):
 root=a.output;root.mkdir(exist_ok=False)
 condition=' .or. '.join(f'(p12_trace_count>={d*12+1}.and.p12_trace_count<={d*12+12})' for d in a.decks)
 c_condition=' || '.join(f'(p12_trace_count>={d*12+1}&&p12_trace_count<={d*12+12})' for d in a.decks)
 for kind in ['native','host']:
  src=a.native if kind=='native' else a.host;dest=root/kind;dest.mkdir()
  for p in src.iterdir():
   if p.suffix in ['.F90','.F','.cpp','.cuh'] or p.name=='Makefile':shutil.copyfile(p,dest/p.name)
  original=a.native;routines=parse_modules([original/'mynn_shared.F90',original/'bl_mynn_subroutines.F90',original/'bl_mynn.F90'])[1]
  selected={}
  for name,limits in [('mym_length',(818,1230)),('mym_turbulence',(1685,1980)),('mym_condensation',(2640,3100))]:
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
   p=dest/'bl_mynn_subroutines.F90';lines=p.read_text().splitlines();out=[]
   for number,line in enumerate(lines,1):
    for name,data in selected.items():
     r=data['routine']
     if number==r.body[0][0]:out.extend(['    integer, save :: p12_trace_count=0','    p12_trace_count=p12_trace_count+1'])
    out.append(line)
    for name,data in selected.items():
     if number in data['assignments']:
      n,expr,kind=data['assignments'][number]
      expr=expr.replace('mynn_enum','enum')
      coordinate='k' if '(' in expr else '0'
      out.append(f"    if({condition}) call p12_trace_{kind}('{name}_line{n}',p12_trace_count,{coordinate},{expr})")
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
     if active:out.append('static int p12_trace_count=0; ++p12_trace_count;')
    if active:
     m=re.match(r'// bl_mynn_subroutines.F90:(\d+)',line)
     if m:source_line=int(m[1])
     for n,expr,kind in active['assignments'].values():
      if n==source_line and line.startswith(expr+' = '):
       coordinate='k' if '(' in expr else '0'
       out.append(f'if({c_condition})p12_trace("{active["routine"].name}_line{n}",p12_trace_count,{coordinate},{expr});')
   p.write_text('\n'.join(out)+'\n')
 (root/'scope.json').write_text(json.dumps(dict(scope=__doc__,decks=a.decks,all_levels=True,profiles=576),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--native',type=Path,default=Path('scratch/native-branch-576'))
 p.add_argument('--host',type=Path,default=Path('scratch/host-branch-576'))
 p.add_argument('--output',type=Path,default=Path('scratch/stability-trace-v3'))
 p.add_argument('--decks',type=int,nargs='+',default=[448,450])
 main(p.parse_args())
