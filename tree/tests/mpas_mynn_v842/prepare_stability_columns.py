"""Whole576-profile native replay, retaining all448 prior inputs unchanged.

The added128 profiles span strong/weak turbulence, shear and buoyancy under
restart, targeting the untouched stability-limit edges with original science.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

PROFILES='''
    if(deck>=448) then
      do_restart=.true.
      bl_mynn_closure=2.+0.5*real(mod(deck/4,3))
      bl_mynn_edmf_dd=0;bl_mynn_edmf=0
      bl_mynn_mixlength=mod(deck/12,3)
      bl_mynn_topdown=mod(deck,2)
      bl_mynn_stfunc=mod(deck/16,2)+1
      bl_mynn_cloudpdf=mod(deck,4)-1
      bl_mynn_tkeadvect=0
      do c=1,ncol
        hfx(c,1)=0.;qfx(c,1)=0.;ust(c,1)=0.1
        pblh(c,1)=100.+200.*real(mod(deck+c,8))
        do k=1,nz+1
          dz(c,k,1)=500.
          if(k<20) dz(c,k,1)=10.+10.*real(mod(deck+c,5))
          th(c,k,1)=280.+(-1.)**mod(deck+c,2)*10.**real(mod(deck/8+c,4)-2)*real(k)
          if(k>=30) th(c,k,1)=300.+real(k-30)
          qv(c,k,1)=0.002;qc(c,k,1)=0.;qi(c,k,1)=0.;qs(c,k,1)=0.
          if(mod(deck/32,2)==1) qc(c,k,1)=1.e-4
          qke(c,k,1)=10.**real(mod(deck+c,21)-10)
          qke_adv(c,k,1)=qke(c,k,1)
          tsq(c,k,1)=10.**real(mod(deck+3*c,13)-6)
          qsq(c,k,1)=10.**real(mod(deck+3*c,13)-12)
          cov(c,k,1)=(-1.)**mod(deck+c,2)*sqrt(tsq(c,k,1)*qsq(c,k,1))
          u(c,k,1)=10.**real(mod(deck+c,11)-6)*real(k)
          v(c,k,1)=-u(c,k,1)
          if(mod(deck/32,4)==0) then
            th(c,k,1)=280.;qv(c,k,1)=0.002;qc(c,k,1)=0.
            u(c,k,1)=0.;v(c,k,1)=0.;cov(c,k,1)=0.
            if(k>=30) th(c,k,1)=300.+real(k-30)
          endif
          tt(c,k,1)=th(c,k,1)*exner(c,k,1)
          rho(c,k,1)=p(c,k,1)/(287.*tt(c,k,1))
          qc_bl(c,k,1)=qc(c,k,1);qi_bl(c,k,1)=0.;cldfra_bl(c,k,1)=0.
        enddo
      enddo
    endif
'''


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def extend(source,target):
    target.mkdir(parents=True,exist_ok=False)
    for path in source.iterdir():
        if path.suffix in ['.F90','.F','.cpp','.cuh'] or path.name in ['Makefile','translation.json','source-receipt.json']:
            shutil.copyfile(path,target/path.name)
    harness=target/('run_adapter.F90' if (target/'run_adapter.F90').exists() else 'run_host_adapter.F90')
    text=harness.read_text()
    if text.count('do deck=0,447')!=1 or text.count('    do step=1,2\n')!=1:raise ValueError('Whole448-profile driver template changed')
    harness.write_text(text.replace('do deck=0,447','do deck=0,575').replace('    do step=1,2\n',PROFILES+'    do step=1,2\n'))
    (target/'branch-profiles.json').write_text(json.dumps(dict(scope=__doc__,profiles=576,calls=1152,original_harness_sha256=sha(source/harness.name),extended_harness_sha256=sha(harness),scientific_sources={p.name:sha(p) for p in target.iterdir() if p.suffix in ['.F','.F90'] and not p.name.startswith('run_')}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','target']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();extend(a.source,a.target)
