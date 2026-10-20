"""Add 512 whole-driver profiles spanning every nine-species presence mask.

All previous banks remain unchanged. The additional profiles vary initialization,
restart, stochastic forcing, cloud, shear, transport and closure together. Native
defects are measured without excluding words or suppressing scheme options.
"""
from pathlib import Path
import hashlib
import json
import shutil


OVERRIDE = """
    ! Whole-driver species/regime bank, additional to all previous inputs.
    f_qc=btest(deck-1344,0); f_qi=btest(deck-1344,1)
    f_qs=btest(deck-1344,2); f_qoz=btest(deck-1344,3)
    f_nc=btest(deck-1344,4); f_ni=btest(deck-1344,5)
    f_nifa=btest(deck-1344,6); f_nwfa=btest(deck-1344,7)
    f_nbca=btest(deck-1344,8)
    do_restart=btest(deck-1344,0); do_dacycling=btest(deck-1344,1)
    spp_pbl=mod(deck/4,2); icloud_bl=mod(deck/8,2)
    bl_mynn_closure=2.+0.5*real(mod(deck/16,3))
    bl_mynn_cloudpdf=mod(deck/3,4)-1
    bl_mynn_mixlength=mod(deck/5,3)
    bl_mynn_stfunc=1+mod(deck/7,2)
    bl_mynn_topdown=mod(deck/9,2); bl_mynn_scaleaware=mod(deck/11,2)
    bl_mynn_dheat_opt=mod(deck/13,2); bl_mynn_tkebudget=mod(deck/15,2)
    bl_mynn_edmf=mod(deck/17,2); bl_mynn_edmf_dd=mod(deck/19,2)
    bl_mynn_edmf_mom=mod(deck/21,2); bl_mynn_edmf_tke=mod(deck/23,2)
    bl_mynn_output=mod(deck/25,2); bl_mynn_mixscalars=mod(deck/27,2)
    bl_mynn_cloudmix=mod(deck/29,2); bl_mynn_mixqt=mod(deck/31,2)
    bl_mynn_tkeadvect=btest(deck-1344,2)
    do c=1,ncol
      xland(c,1)=real(1+mod(deck+c,2))
      hfx(c,1)=real(mod(deck+c,9)-3)*100.
      qfx(c,1)=real(mod(deck+c,5)-1)*1.e-5
      ust(c,1)=0.01+0.1*real(mod(deck+c,10))
      dx(c,1)=100.+500.*real(mod(deck+c,20))
      ts(c,1)=270.+5.*real(mod(deck+c,8))
      ps(c,1)=100000.; pblh(c,1)=100.+200.*real(mod(deck+c,12))
      kpbl(c,1)=2+mod(deck+c,8)
      qsfc(c,1)=0.005+0.002*real(mod(deck+c,6))
      ch(c,1)=0.01; znt(c,1)=0.01+0.1*real(mod(deck+c,3))
      rmol(c,1)=real(mod(deck+c,5)-2)*0.001
      z=0.
      do k=1,nz+1
        dz(c,k,1)=400.+real(mod(13*k+17*deck,97))
        zm=z+0.5*dz(c,k,1); z=z+dz(c,k,1)
        p(c,k,1)=100000.*exp(-zm/8000.)
        exner(c,k,1)=(p(c,k,1)/100000.)**(287./1004.5)
        th(c,k,1)=ts(c,1)+0.005*zm
        if(k<6.and.mod(deck+c,3)==0) th(c,k,1)=ts(c,1)-0.2*real(k)
        if(k<6.and.mod(deck+c,3)==1) th(c,k,1)=ts(c,1)
        if(k>=6.and.k<=9) th(c,k,1)=th(c,k,1)+real(mod(deck+c,4))*5.
        tt(c,k,1)=th(c,k,1)*exner(c,k,1)
        rho(c,k,1)=p(c,k,1)/(287.*tt(c,k,1))
        qv(c,k,1)=(0.002+0.003*real(mod(deck+c,6)))*exp(-zm/2000.)
        qc(c,k,1)=0.; qi(c,k,1)=0.; qs(c,k,1)=0.
        if(k>=3.and.k<=8) then
          qc(c,k,1)=1.e-4*real(mod(deck+c,5))
          qi(c,k,1)=1.e-5*real(mod(deck+2*c,5))
          qs(c,k,1)=1.e-5*real(mod(deck+3*c,5))
        endif
        qke(c,k,1)=10.**real(mod(deck+c,9)-5)
        qke_adv(c,k,1)=qke(c,k,1)
        tsq(c,k,1)=10.**real(mod(deck+c,7)-5)
        qsq(c,k,1)=10.**real(mod(deck+2*c,7)-11)
        cov(c,k,1)=0.5*(-1.)**mod(deck+c,2)*sqrt(tsq(c,k,1)*qsq(c,k,1))
        u(c,k,1)=real(mod(deck+c,10))*0.2*real(k)
        v(c,k,1)=real(mod(deck+2*c,7)-3)*0.1*real(k)
        w(c,k,1)=real(mod(deck+c,3)-1)*0.01
        nc(c,k,1)=1.e6; ni(c,k,1)=1.e5
        nifa(c,k,1)=1.e5; nwfa(c,k,1)=1.e6; nbca(c,k,1)=1.e4
        qoz(c,k,1)=1.e-6; rthraten(c,k,1)=-real(mod(deck+c,4))*1.e-4
        pattern_spp(c,k,1)=0.1*real(mod(deck+c+k,5)-2)
        qc_bl(c,k,1)=qc(c,k,1); qi_bl(c,k,1)=qi(c,k,1)
        cldfra_bl(c,k,1)=0.1*real(mod(deck+c+k,10))
      enddo
      wspd(c,1)=sqrt(u(c,1,1)**2+v(c,1,1)**2)
    enddo
"""


def main():
    root = Path('scratch/species-regime-bank-v1')
    root.mkdir(exist_ok=False)
    for kind in ['native', 'host', 'coverage', 'bounds']:
        source = Path('scratch/defined-branch-bank-v1/' + ('host' if kind == 'host' else 'native'))
        target = root / kind
        target.mkdir()
        for p in source.iterdir():
            if p.suffix in ['.F90', '.F', '.cpp', '.cuh'] or p.name == 'Makefile':
                shutil.copy2(p, target / p.name)
        harness = target / ('run_host_adapter.F90' if kind == 'host' else 'run_adapter.F90')
        s = harness.read_text()
        assert s.count('do deck=1088,1343') == 1 and s.count('    do step=1,2\n') == 1
        s = s.replace('do deck=1088,1343', 'do deck=1344,1855')
        harness.write_text(s.replace('    do step=1,2\n', OVERRIDE + '    do step=1,2\n'))
        if kind in ['coverage', 'bounds']:
            make = target / 'Makefile'
            flags = '-fprofile-arcs -ftest-coverage' if kind == 'coverage' else '-fcheck=bounds -fbacktrace'
            make.write_text(make.read_text().replace('-O3 -cpp', '-O3 ' + flags + ' -cpp'))
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()}
    (root / 'scope.json').write_text(json.dumps(dict(scope=__doc__, process_count=1,
        additional_profiles=512, whole_driver_calls=1024, species_masks=512,
        previous_banks_retained=True, excluded_words=[], files=files), indent=2) + '\n')


if __name__ == '__main__':
    main()
