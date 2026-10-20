"""Instrument complete original and host replays to locate a cloud-water difference.

This is a diagnostic build. Unmodified whole-module replay remains the referee.
Trace writes preserve local binary32/binary64 words and source-line addresses.
"""
from pathlib import Path
import argparse,re,shutil


def prepare(args):
    for kind,origin in [('native',args.native),('host',args.host)]:
        dest=args.output/kind;dest.mkdir(parents=True,exist_ok=False)
        for p in origin.iterdir():
            if p.suffix in {'.F90','.F','.cpp','.cuh'} or p.name=='Makefile':shutil.copyfile(p,dest/p.name)
        if kind=='native':
            p=dest/'bl_mynn_subroutines.F90';lines=p.read_text().splitlines();out=[];inside=False
            for number,line in enumerate(lines,1):
                if re.match(r'\s*SUBROUTINE\s+mym_condensation\b',line,re.I):inside=True
                if inside and number==2632:
                    out.extend(['    integer,save :: p12_trace_count=0', '    p12_trace_count=p12_trace_count+1', "    if(p12_trace_count==2323) open(unit=89,file='condensation.words',status='replace')"])
                out.append(line)
                if inside and 2652<=number<=2713:
                    m=re.match(r'\s*([a-zA-Z_][\w]*(?:\(k\))?)\s*=([^=].*)',line)
                    if m:
                        expr=m[1];name=f'line{number}'
                        if expr.lower() in {'t3sq','r3sq','c3sq'}:
                            out.append(f"    if(p12_trace_count==2323.and.k==kts) write(89,'(a,4(1x,i0),1x,z8.8)') 'R out:{name}_lo',0,1,0,0,transfer({expr},[0,0])(1)")
                            # A helper avoids nonstandard indexing of TRANSFER expressions.
                            out[-1]=f"    if(p12_trace_count==2323.and.k==kts) call p12_trace_double('{name}',{expr})"
                        else:out.append(f"    if(p12_trace_count==2323.and.k==kts) write(89,'(a,4(1x,i0),1x,z8.8)') 'R out:{name}',0,1,0,0,transfer({expr},0)")
                if inside and re.match(r'\s*END\s+SUBROUTINE',line,re.I):inside=False
            p.write_text('\n'.join(out)+'\n')
            p=dest/'framework_stubs.F90';p.write_text(p.read_text()+'''
subroutine p12_trace_double(name,x)
implicit none
character(len=*) :: name
double precision :: x
integer :: bits(2)
bits=transfer(x,bits)
write(89,'(a,4(1x,i0),1x,z8.8)') 'R out:'//trim(name)//'_lo',0,1,0,0,bits(1)
write(89,'(a,4(1x,i0),1x,z8.8)') 'R out:'//trim(name)//'_hi',0,1,0,0,bits(2)
end subroutine
''')
        else:
            p=dest/'translation.cpp';text=p.read_text();marker='// Generated from pinned MPAS MMM; see translation receipt.'
            helper='''
#include <cstdio>
static int p12_trace_count=0;
static FILE* p12_trace_file=nullptr;
template<class T> static void p12_trace_value(const char* name,T x) {
if(!p12_trace_file)p12_trace_file=std::fopen("condensation.words","w");
FILE* f=p12_trace_file;
uint32_t w[2]={0,0};std::memcpy(w,&x,sizeof(T));
if(sizeof(T)==4)std::fprintf(f,"R out:%s 0 1 0 0 %08X\\n",name,w[0]);
else {std::fprintf(f,"R out:%s_lo 0 1 0 0 %08X\\n",name,w[0]);std::fprintf(f,"R out:%s_hi 0 1 0 0 %08X\\n",name,w[1]);}
std::fflush(f);
}
'''
            assert marker in text;text=text.replace(marker,helper+marker,1)
            out=[];inside=False;number=0
            for line in text.splitlines():
                out.append(line)
                if line.startswith('__device__ __noinline__ void f_mym_condensation('):inside=True;out.append('++p12_trace_count;')
                if inside:
                    m=re.match(r'// bl_mynn_subroutines.F90:(\d+)',line)
                    if m:number=int(m[1])
                    if 2652<=number<=2713:
                        m=re.match(r'([a-zA-Z_][\w]*(?:\(k\))?) = ',line)
                        if m:out.append(f'if(p12_trace_count==2323 && k==kts)p12_trace_value("line{number}",{m[1]});')
                if inside and line.startswith('__device__ __noinline__') and 'f_mym_condensation(' not in line:inside=False
            p.write_text('\n'.join(out)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['native','host','output']:p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
