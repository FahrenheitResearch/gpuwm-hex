"""Extend the whole576 original driver with512 stability/restart profiles.

All earlier inputs remain unchanged. The original Fortran initializes the
synthetic inputs and runs the whole scheme; Python only prepares source files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


PROFILES = '''
    if(deck>=576) then
      do_restart=.true.
      bl_mynn_closure=3.
      bl_mynn_edmf=0;bl_mynn_edmf_dd=0
      bl_mynn_cloudpdf=mod(deck,4)-1
      bl_mynn_mixlength=mod(deck/4,3)
      bl_mynn_stfunc=mod(deck/12,2)+1
      bl_mynn_topdown=mod(deck/24,2)
      bl_mynn_tkeadvect=mod(deck/48,2)
      do c=1,ncol
        hfx(c,1)=real(mod(31*deck+17*c,101)-50)*20.
        qfx(c,1)=real(mod(deck+c,3)-1)*1.e-5
        ust(c,1)=10.**real(mod(deck+c,8)-5)
        pblh(c,1)=10.+real(mod(deck+c,25))*100.
        do k=1,nz+1
          dz(c,k,1)=10.**real(mod(deck+c,5)-2)*(1.+0.01*real(mod(13*k+17*deck,97)))
          th(c,k,1)=280.+(-1.)**mod(deck+c,2)*10.**real(mod(deck/8+c,6)-4)*real(k)
          if(mod(deck/64,8)==0) th(c,k,1)=280.
          qv(c,k,1)=0.002;qc(c,k,1)=0.;qi(c,k,1)=0.;qs(c,k,1)=0.
          qke(c,k,1)=10.**real(mod(deck+c,23)-13)*(1.+0.01*real(mod(deck+13*k,97)))
          if(mod(deck/64,8)==1) qke(c,k,1)=0.
          qke_adv(c,k,1)=qke(c,k,1)
          tsq(c,k,1)=10.**real(mod(deck+3*c,15)-8)
          qsq(c,k,1)=10.**real(mod(deck+3*c,15)-14)
          cov(c,k,1)=(-1.)**mod(deck+c,2)*sqrt(tsq(c,k,1)*qsq(c,k,1))
          u(c,k,1)=10.**real(mod(deck+c,13)-8)*real(k)
          v(c,k,1)=-u(c,k,1)
          if(mod(deck/64,8)==2) then
            u(c,k,1)=0.;v(c,k,1)=0.
          endif
          if(mod(deck/64,8)==3) then
            th(c,k,1)=280.;u(c,k,1)=0.;v(c,k,1)=0.;cov(c,k,1)=0.
          endif
          if(mod(deck/64,8)==4) dz(c,k,1)=0.01
          if(mod(deck/64,8)==5) tsq(c,k,1)=0.
          if(mod(deck/64,8)==6) qsq(c,k,1)=0.
          if(k>=30) th(c,k,1)=300.+real(k-30)
          tt(c,k,1)=th(c,k,1)*exner(c,k,1)
          rho(c,k,1)=p(c,k,1)/(287.*tt(c,k,1))
          qc_bl(c,k,1)=qc(c,k,1);qi_bl(c,k,1)=0.;cldfra_bl(c,k,1)=0.
        enddo
      enddo
    endif
'''


def sha(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def extend(source,target):
    target.mkdir(parents=True,exist_ok=False)
    for path in source.iterdir():
        if path.suffix in ['.F90','.F','.cpp','.cuh'] or path.name in ['Makefile','translation.json','source-receipt.json']:
            shutil.copyfile(path,target/path.name)
    name='run_adapter.F90' if (target/'run_adapter.F90').exists() else 'run_host_adapter.F90'
    harness=target/name;text=harness.read_text()
    if text.count('do deck=0,575')!=1 or text.count('    do step=1,2\n')!=1:
        raise ValueError('Complete576 original driver template changed')
    harness.write_text(text.replace('do deck=0,575','do deck=0,1087').replace(
        '    do step=1,2\n',PROFILES+'    do step=1,2\n'))
    (target/'branch-profiles.json').write_text(json.dumps(dict(scope=__doc__,profiles=1088,calls=2176,
        original_harness_sha256=sha(source/name),extended_harness_sha256=sha(harness),
        scientific_sources={p.name:sha(p) for p in target.iterdir() if p.suffix in ['.F','.F90'] and not p.name.startswith('run_')}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','target']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();extend(a.source,a.target)
