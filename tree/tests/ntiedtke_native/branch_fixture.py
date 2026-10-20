"""Auxiliary native flux edge states, not full-driver acceptance fixtures.

Expose private procedures in a scratch copy by changing accessibility only.
All native numerical statements remain unchanged. Test the complete module
first; these states additionally reach rare flux-repair and downdraft arms.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


LEVEL = ("pten", "pqen", "pqsen", "ptenh", "pqenh", "paph", "pap",
         "pgeoh", "pmfu", "pmfd", "pmfus", "pmfds", "pmfuq", "pmfdq",
         "pmful", "plude", "pdmfup", "pdmfdp", "pdpmel", "plglac",
         "pmfdde_rate", "pmflxr", "pmflxs")
SURFACE_I = ("lndj", "kcbot", "kctop", "kdtop", "ktype")
SURFACE_L = ("ldcum", "lddraf")


def generate(source: Path, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    assert text.count(" private\n") == 1
    (directory / "cu_ntiedtke.F90").write_text(text.replace(" private\n", " public\n"))
    dumps = []
    for name in LEVEL:
        dumps.append(f"call dump2('{name}',{name},phase)")
    dumps += [f"call dump1('{n}',{n},phase)" for n in SURFACE_I]
    dumps += [f"call dump1('{n}',merge(1,0,{n}),phase)" for n in SURFACE_L]
    dumps.append("call dump2('prain',reshape(prain,(/nc,1/)),phase)")
    body = '''program branch_flux
use cu_ntiedtke
implicit none
integer,parameter::nc=4,nz=12
integer::j,k,phase,ktopm2,errflg
character(len=256)::errmsg
real(kind_phys)::prain(nc)
'''
    body += "real(kind_phys)::" + ",".join(f"{n}(nc,nz+1)" for n in LEVEL) + "\n"
    body += "integer::" + ",".join(f"{n}(nc)" for n in SURFACE_I) + "\n"
    body += "logical::" + ",".join(f"{n}(nc)" for n in SURFACE_L) + "\n"
    body += '''call cu_ntiedtke_init(1004.5,287.,461.6,2.5e6,2.85e6,350000.,9.80616,errmsg,errflg)
'''
    body += "; ".join(f"{n}=0." for n in LEVEL) + "\n"
    body += '''prain=0.; ktopm2=2; lndj=(/1,0,1,0/)
ldcum=.true.; lddraf=.true.; ktype=(/1,3,1,3/)
kcbot=6; kctop=2; kdtop=3
do k=1,nz+1
 paph(:,k)=20000.+6000.*k
 pap(:,k)=paph(:,k)+3000.
 pgeoh(:,k)=1000.*(nz+1-k)
 pten(:,k)=280.; ptenh(:,k)=280.
 pqen(:,k)=0.01; pqenh(:,k)=0.01; pqsen(:,k)=0.01
 pmfu(:,k)=0.1; pmfus(:,k)=30000.; pmfuq(:,k)=0.001
 if(k>=3.and.k<=7)then
  pmfd(:,k)=-0.03; pmfds(:,k)=-9000.; pmfdq(:,k)=-0.0003
 endif
enddo
! Positive liquid rain, then a negative frozen increment: snow repair.
pdmfup(:,2)=0.02
pten(3:4,3)=250.; ptenh(3:4,3)=250.
pdmfdp(3:4,3)=-0.002
! Positive snow, then a negative liquid increment: rain repair.
pten(1:2,2)=250.; ptenh(1:2,2)=250.
pdmfdp(1:2,3)=-0.002
open(97,file='branch-flux.csv',status='replace')
do phase=0,1
'''
    body += "\n".join(dumps) + "\n"
    body += '''if(phase==0)call cuflxn(nc,nz,60.,pten,pqen,pqsen,ptenh,pqenh, &
 paph,pap,pgeoh,lndj,ldcum,kcbot,kctop,kdtop,ktopm2,ktype,lddraf, &
 pmfu,pmfd,pmfus,pmfds,pmfuq,pmfdq,pmful,plude,pdmfup,pdmfdp, &
 pdpmel,plglac,prain,pmfdde_rate,pmflxr,pmflxs)
enddo
close(97)
contains
subroutine dump2(name,a,phase)
character(len=*),intent(in)::name
real(kind_phys),intent(in)::a(:,:)
integer,intent(in)::phase
integer::i,k
do k=1,size(a,2)
 do i=1,size(a,1)
  write(97,'(i0,",",a,",",i0,",",i0,",",z8.8)')phase,name,k,i,transfer(a(i,k),0)
 enddo
enddo
end subroutine
subroutine dump1(name,a,phase)
character(len=*),intent(in)::name
integer,intent(in)::a(:),phase
integer::i
do i=1,size(a)
 write(97,'(i0,",",a,",",i0,",",i0,",",z8.8)')phase,name,1,i,a(i)
enddo
end subroutine
end program
'''
    (directory / "branch_flux.F90").write_text(body)


def compare(fixture: Path, receipt: Path, library: Path | None):
    from hexcore.mpasphys.ntiedtke_v842 import Constants, _pipeline
    from gpuwm.core import ntiedtke as nt
    if library is None:
        import cupy as cp
        pipeline = _pipeline(4, 12, 60., 1, 0, Constants())
    else:
        cp = np
        try:
            from .host import HostPipeline
        except ImportError:  # Direct script invocation.
            from host import HostPipeline
        pipeline = HostPipeline(library, 4, 12, 60., 1, 0, Constants())
    stage = "ntiedtke_cuflxn"
    rows = list(csv.reader(fixture.open()))
    for phase,name,k,j,word in rows:
        if phase != "0":
            continue
        a = pipeline.w.bind(nt.NT_STAGE_ALIASES.get((stage,name),name),1)
        value = np.array(int(word,16),dtype=np.uint32).view(a.dtype).item()
        if a.ndim == 2:
            a[int(k),int(j)-1] = value
        else:
            a[int(j)-1] = value
    pipeline.run_stage(stage) if library is not None else pipeline.stages.launch(stage,pipeline.args_for(stage))
    results = {}
    for phase,name,k,j,word in rows:
        if phase != "1":
            continue
        a = pipeline.w.bind(nt.NT_STAGE_ALIASES.get((stage,name),name),1)
        a = cp.asnumpy(a) if library is None else a
        got = int(a.view(np.uint32)[int(k),int(j)-1] if a.ndim==2 else a.view(np.uint32)[int(j)-1])
        result = results.setdefault(name,dict(words=0,differing_words=0,samples=[]))
        result['words'] += 1
        if got != int(word,16):
            result['differing_words'] += 1
            if len(result['samples']) < 3:
                result['samples'].append(dict(level=k,column=j,got=hex(got),native=word))
    data = dict(scope="auxiliary-native-private-flux-edge-states",forecast_proof=False,
                accessibility_only_source_change=True,results=results,
                execution="hex-cuda" if library is None else "host-glibc-witness",
                differing_words=sum(v['differing_words'] for v in results.values()))
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(data['differing_words'])
    if data['differing_words']:
        raise SystemExit(1)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command",required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("source",type=Path)
    generate_parser.add_argument("directory",type=Path)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("fixture",type=Path)
    compare_parser.add_argument("receipt",type=Path)
    compare_parser.add_argument("--library",type=Path)
    args = parser.parse_args()
    if args.command == "generate": generate(args.source,args.directory)
    else: compare(args.fixture,args.receipt,args.library)
