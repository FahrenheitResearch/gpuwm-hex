"""Feed Rust-extracted real profiles into the pristine standalone CAM oracle.

Common preparation uses native arithmetic slices, not Python data processing.
Cloud fraction/emissivity and unavailable condensates remain explicit synthetic
companions. This is not an f1 driver tap or a complete initialized forecast.
"""
from pathlib import Path
import argparse
import hashlib
import json

FIELDS = ('latCell','lonCell','rho','theta','rho_base','theta_base','qv','qc','qr',
          'zz','zgrid','skintemp','xland','xice','snow','sfc_albbck')

PREP = '''
  ! MPAS mpas_atm_time_integration.F:7538-7550,7605-7651
  rcv_r=rgas/(cp-rgas)
  do i=1,nc
    do k=1,nv
      theta_m_r(k,i)=theta_r(k,i)*(1._RKIND+rvord*qv_r(k,i))
      rho_zz_r(k,i)=rho_r(k,i)/zz_r(k,i)
      rho_p_r(k,i)=rho_zz_r(k,i)-rho_base_r(k,i)
      rtheta_base_r(k,i)=theta_base_r(k,i)*rho_base_r(k,i)
      rtheta_p_r(k,i)=theta_m_r(k,i)*rho_p_r(k,i) &
                    +rho_base_r(k,i)*(theta_m_r(k,i)-theta_base_r(k,i))
      exner_r(k,i)=(zz_r(k,i)*(rgas/1.e5)*(rtheta_p_r(k,i)+rtheta_base_r(k,i)))**rcv_r
      exner_base_r(k,i)=(zz_r(k,i)*(rgas/1.e5)*rtheta_base_r(k,i))**rcv_r
      pressure_p_r(k,i)=zz_r(k,i)*rgas*(exner_r(k,i)*rtheta_p_r(k,i) &
                       +rtheta_base_r(k,i)*(exner_r(k,i)-exner_base_r(k,i)))
      pressure_base_r(k,i)=zz_r(k,i)*rgas*exner_base_r(k,i)*rtheta_base_r(k,i)
      pressure(k,i)=pressure_base_r(k,i) ! camradinit requests pressure_base.
      ! MPAS mpas_atmphys_interface.F:328-352
      qv(i,k,1)=max(0.,qv_r(k,i))
      qc(i,k,1)=max(0.,qc_r(k,i));qr(i,k,1)=max(0.,qr_r(k,i))
      rho(i,k,1)=zz_r(k,i)*rho_zz_r(k,i)
      rho(i,k,1)=rho(i,k,1)*(1._RKIND+qv(i,k,1))
      t(i,k,1)=theta_m_r(k,i)/(1._RKIND+rvord*qv(i,k,1))
      t(i,k,1)=t(i,k,1)*exner_r(k,i)
      pi(i,k,1)=exner_r(k,i)
      p(i,k,1)=pressure_p_r(k,i)+pressure_base_r(k,i)
      z(i,k,1)=0.5*(zgrid_r(k+1,i)+zgrid_r(k,i))
      dz(i,k,1)=zgrid_r(k+1,i)-zgrid_r(k,i)
    enddo
    ! MPAS mpas_atmphys_interface.F:480-505 (interface top pressure)
    k=nv+1
    z0_r=zgrid_r(k,i)
    z1_r=0.5*(zgrid_r(k,i)+zgrid_r(k-1,i))
    z2_r=0.5*(zgrid_r(k-1,i)+zgrid_r(k-2,i))
    w1_r=(z0_r-z2_r)/(z1_r-z2_r);w2_r=1.-w1_r
    p8w(i,k,1)=exp(w1_r*log(p(i,k-1,1))+w2_r*log(p(i,k-2,1)))
    ! MPAS mpas_atmphys_interface.F:531-550 (native moist hydrostatic pressure)
    do k=nv,1,-1
      p8w(i,k,1)=p8w(i,k+1,1)+gravity*rho(i,k,1)*dz(i,k,1)
    enddo
    do k=nv,1,-1
      p(i,k,1)=0.5*(p8w(i,k+1,1)+p8w(i,k,1))
    enddo
    ps(i)=p8w(i,1,1)
    ! CAM ignores level nv+1 inputs except interface pressure. Defined padding.
    t(i,nv+1,1)=t(i,nv,1);qv(i,nv+1,1)=qv(i,nv,1)
    qc(i,nv+1,1)=0.;qr(i,nv+1,1)=0.
    p(i,nv+1,1)=p8w(i,nv+1,1);pi(i,nv+1,1)=pi(i,nv,1)
    rho(i,nv+1,1)=rho(i,nv,1);z(i,nv+1,1)=zgrid_r(nv+1,i)
    dz(i,nv+1,1)=dz(i,nv,1)
  enddo
'''

def prepare(directory, stored, year, julian, receipt):
    manifest=json.loads(stored.with_suffix('.json').read_text())
    assert [f['name'] for f in manifest['fields']]==list(FIELDS)
    assert len(manifest['cells_zero_based'])==6
    assert all(f['words_per_cell']==55 for f in manifest['fields'][2:10])
    text=(directory/'run_columns_capture.F90').read_text()
    text=text.replace('  use cam_capture','  use cam_capture\n  use mpas_constants, only: rgas,cp,rvord,gravity\n  use cam_driver_oracle, only: radconst',1)
    decl='  real :: rcv_r,z0_r,z1_r,z2_r,w1_r,w2_r,solar_decl,solar_flux\n'
    names=('rho','theta','rho_base','theta_base','qv','qc','qr','zz','theta_m',
           'rho_zz','rho_p','rtheta_base','rtheta_p','exner','exner_base','pressure_p','pressure_base')
    decl+='  real :: '+','.join(n+'_r(nv,nc)' for n in names)+'\n  real :: zgrid_r(nv+1,nc)\n'
    text=text.replace('  character(len=256) :: filename',decl+'  character(len=256) :: filename',1)
    a=text.index('  do i=1,nc\n    lat(i)=');b=text.index('  aerp=0.; aern=0.',a)
    reads=['  open(newunit=u,file='+repr(str(stored.resolve()))+",form='unformatted',access='stream',status='old')"]
    destinations=['lat','lon']+[n+'_r' for n in FIELDS[2:11]]+['tsk','xland','xice','snow','albedo']
    reads+=['  read(u) '+n for n in destinations];reads+=['  close(u)']
    text=text[:a]+'\n'.join(reads)+'\n'+PREP+text[b:]
    text=text.replace('  year=yrdata(1)+50','  year='+str(year))
    a=text.index('  xland(:,1)=');b=text.index('  do m=1,nm',a)
    text=text[:a]+'  emiss=0.97 ! Synthetic companion: native surface emissivity is not in this init file.\n'+text[b:]
    a=text.index('  do k=1,nv+1\n    p8w');b=text.index('  fice=0.6',a)
    text=text[:a]+text[b:]
    a=text.index('  variant_count=6');b=text.index('  call get_command_argument(1,filename)',a)
    text=text[:a]+'  variant_count=1;mode_count=8\n'+text[b:]
    text=text.replace('    qc=0.; qr=0.; qi=0.; qs=0.; qg=0.; cld=0.',
        '    qi=0.; qs=0.; qg=0.; cld=0. ! Explicit synthetic unavailable companions.')
    text=text.replace('    julian=real((variant-1)*70)','    julian='+str(float(julian)))
    text=text.replace('    do mode=1,mode_count',
        '    call radconst(solar_decl,degrad,360./365.,julian,solar_flux,1370.)\n    do mode=1,mode_count')
    text=text.replace('300.,xtime,0.2,1370.,30.', '300.,xtime,solar_decl,solar_flux,30.')
    (directory/'run_real_capture.F90').write_text(text)
    driver=(directory/'driver_oracle.F90').read_text()
    solar=driver[:driver.index('program driver_columns')]
    (directory/'real_solar.F90').write_text(solar)
    make=(directory/'Makefile').read_text().split('\nreal_solar.o:',1)[0]
    make+='\nreal_solar.o: real_solar.F90\n\t$(FC) $(FFLAGS) -c $< -o $@\n'
    objects=[p.name for p in directory.glob('*.o') if p.name not in
             ('run_columns.o','run_columns_capture.o','run_real_capture.o','real_solar.o')]
    make+='run_real_capture.o: run_real_capture.F90 real_solar.o\n\t$(FC) $(FFLAGS) -c $< -o $@\n'
    make+='cam_real_capture: run_real_capture.o real_solar.o '+' '.join(objects)+'\n\t$(FC) $(FFLAGS) $^ -o $@\n'
    (directory/'Makefile').write_text(make)
    receipt.write_text(json.dumps({'scope':__doc__,'stored_columns':manifest,'year':year,'julian':julian,
        'oracle_source_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'synthetic_companions':['cloud fraction zero','surface emissivity0.97','unavailable qi/qs/qg zero',
                                'padding','mixed cache and day/night invocation modes'],
        'whole_model_acceptance':False},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('directory','stored','receipt'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--year',type=int,required=True);p.add_argument('--julian',type=float,required=True)
    a=p.parse_args();prepare(a.directory,a.stored,a.year,a.julian,a.receipt)
