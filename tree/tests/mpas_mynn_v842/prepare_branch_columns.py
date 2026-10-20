"""Extend the complete original MYNN driver replay from 320 to 448 decks.

The first 320 profiles remain unchanged. Additional full profiles target the
moisture recovery, neutral-gradient, shear and level-three covariance paths.
Native science is copied unchanged and always invoked through its wrapper.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


PROFILES = '''
    ! Full-driver branch profiles; input defects are deliberate recovery tests.
    if(deck>=320) then
      bl_mynn_closure=2.+0.5*real(mod(deck/4,3))
      bl_mynn_mixlength=mod(deck/12,3)
      bl_mynn_cloudpdf=mod(deck,4)-1
      bl_mynn_edmf_dd=mod(deck/8,2)
      bl_mynn_mixqt=1;bl_mynn_cloudmix=1;bl_mynn_mixscalars=1
      bl_mynn_stfunc=mod(deck/16,2)+1
      do c=1,ncol
        pblh(c,1)=50.+100.*real(mod(deck+c,16))
        hfx(c,1)=real(mod(deck+c,7)-3)*300.
        qfx(c,1)=real(mod(deck+c,3))*1.e-5
        ust(c,1)=0.01+0.15*real(mod(deck+c,8))
        do k=1,nz+1
          dz(c,k,1)=500.
          if(k<20) dz(c,k,1)=5.+20.*real(mod(deck+c,6))
          th(c,k,1)=280.+0.1*real(k)
          qv(c,k,1)=0.004
          qc(c,k,1)=0.;qi(c,k,1)=0.;qs(c,k,1)=0.
          u(c,k,1)=0.01*real(mod(deck+c,8))*real(k)
          v(c,k,1)=-u(c,k,1)
          qke(c,k,1)=10.**real(mod(deck+c,9)-6)
          tsq(c,k,1)=10.**real(mod(deck+c,7)-4)
          qsq(c,k,1)=10.**real(mod(deck+c,7)-10)
          cov(c,k,1)=(-1.)**mod(deck+c,2)*sqrt(tsq(c,k,1)*qsq(c,k,1))
          select case(mod(deck/32,4))
          case(0)
            ! A surface deficit with enough upper-column water to repay it.
            if(k==1) qv(c,k,1)=-1.e-4*real(1+mod(deck+c,8))
          case(1)
            ! Deficits exceed available moisture; exercise conservation limits.
            if(k<20) qv(c,k,1)=-1.e-3
            if(k>=20) qv(c,k,1)=1.e-6
          case(2)
            ! Exact neutral thermodynamic and momentum gradients.
            th(c,k,1)=280.;u(c,k,1)=0.;v(c,k,1)=0.
            cov(c,k,1)=0.
          case(3)
            ! Strong unstable lapse rate with weak or large momentum shear.
            if(k<20) th(c,k,1)=290.-0.8*real(k)
            u(c,k,1)=real(mod(deck+c,4))*2.*real(k)
            v(c,k,1)=-u(c,k,1)
          end select
          if(k>=30) th(c,k,1)=300.+real(k-30)
          tt(c,k,1)=th(c,k,1)*exner(c,k,1)
          rho(c,k,1)=p(c,k,1)/(287.*tt(c,k,1))
          qke_adv(c,k,1)=qke(c,k,1)
          qc_bl(c,k,1)=qc(c,k,1);qi_bl(c,k,1)=qi(c,k,1)
          cldfra_bl(c,k,1)=0.
        enddo
      enddo
    endif
'''


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(a):
    a.target.mkdir(parents=True, exist_ok=False)
    for path in a.source.iterdir():
        if path.suffix in ['.F90', '.F'] or path.name in ['Makefile', 'source-receipt.json']:
            shutil.copyfile(path, a.target/path.name)
    harness = a.target/'run_adapter.F90'
    text = harness.read_text()
    if text.count('do deck=0,319') != 1 or text.count('    do step=1,2\n') != 1:
        raise ValueError('The complete 320-deck original driver template changed')
    harness.write_text(text.replace('do deck=0,319', 'do deck=0,447').replace(
        '    do step=1,2\n', PROFILES+'    do step=1,2\n'))
    proof = dict(scope=__doc__, original_harness_sha256=sha(a.source/'run_adapter.F90'),
                 extended_harness_sha256=sha(harness), profiles=448, calls=896,
                 scientific_sources={p.name: sha(p) for p in a.target.iterdir()
                                     if p.suffix in ['.F90', '.F'] and not p.name.startswith('run_')})
    (a.target/'branch-profiles.json').write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ['source', 'target']:
        p.add_argument('--'+n, type=Path, required=True)
    run(p.parse_args())
