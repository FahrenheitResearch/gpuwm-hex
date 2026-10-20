! Native-init-derived thermodynamics; companion radiation flags are test choices.
! MPAS: mpas_atmphys_interface.F:331-341,468-545; module_ra_rrtmg_lw.F:11574-12305.
program real_columns
use module_ra_rrtmg_lw,only:rrtmg_lwrad
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_dump_state,only:dump_all_state
use lw_binio
use mpas_constants,only:rgas,p0,cp,cv,rvord,gravity
implicit none
integer,parameter::km=128,nz=7
integer::nm,ncolumns,magic,cell,u_in
real,allocatable::rho_native(:),theta_native(:),qv_native(:),qc_native(:),z_native(:)
real::theta_m_native,pressure_native(km),z0,z1,z2,w1,w2,tem1,fzm,fzp,rho_moist
character(1024)::input_path
real::p(1,km,1),pw(1,km,1),pi(1,km,1),t(1,km,1),tw(1,km,1),dz(1,km,1)
real::qv(1,km,1),qc(1,km,1),qi(1,km,1),qs(1,km,1),cf(1,km,1),rc(1,km,1),ri(1,km,1),rs(1,km,1)
real::skin(1,1),emiss(1,1),land(1,1),xice(1,1),snow(1,1),lat(1,1),pin(nz),ozn(1,nz,1)
real::glw(1,1),olr(1,1),lwcf(1,1),hr(1,km,1),rcloud(1,km,1),rice(1,km,1),rsnow(1,km,1),flux(11)
real::lwupt(1,1),lwuptc(1,1),lwdnt(1,1),lwdntc(1,1),lwupb(1,1),lwupbc(1,1),lwdnb(1,1),lwdnbc(1,1)
real,allocatable::model(:,:),interfaces(:,:)
real::surface(6)
real::up(1,km,1),up_clear(1,km,1),down(1,km,1),down_clear(1,km,1)
integer::scenario,k,overlap,idcor,day,icloud,reqc,reqi,reqs,o3input,flags(9),nout
character(20)::prefix
character(1024)::outdir
 type(dm_info)::info
call get_command_argument(1,outdir)
call get_command_argument(2,input_path)
open(newunit=u_in,file=trim(input_path),access="stream",form="unformatted",convert="little_endian",status="old")
read(u_in)magic,nm,ncolumns,day
if(magic/=int(z"4C574331").or.nm<4.or.nm>120)error stop "invalid native input shape"
allocate(model(12,nm),interfaces(2,nm+1),rho_native(nm),theta_native(nm),qv_native(nm),qc_native(nm),z_native(nm+1))
call rrtmg_initlw_forMPAS(info)
call bio_open(trim(outdir)//'/native-real-columns.bin')
pin=[.05,.5,5.,50.,250.,700.,1200.]
 do scenario=0,ncolumns-1
 read(u_in)cell,rho_native,theta_native,qv_native,qc_native,z_native
 read(u_in)skin(1,1),land(1,1),xice(1,1),snow(1,1),lat(1,1)
 lat=lat*(180./acos(-1.));emiss=.975
 overlap=2;idcor=0;icloud=1;reqc=0;reqi=0;reqs=0;o3input=0
 p=0.;pw=0.;t=0.;tw=0.;dz=0.;pi=0.;qv=0.;qc=0.;qi=0.;qs=0.;cf=0.;rc=0.;ri=0.;rs=0.
 ! Init-derived ideal-gas state, then the MPAS interface's vertical preparation.
 ! These inputs are not claimed to be an f1 dump of the real driver boundary.
 do k=1,nm
 qv(1,k,1)=max(0.,qv_native(k));qc(1,k,1)=max(0.,qc_native(k))
 theta_m_native=theta_native(k)*(1.+rvord*qv(1,k,1))
 pi(1,k,1)=((rgas/p0)*rho_native(k)*theta_m_native)**(rgas/cv)
 t(1,k,1)=theta_native(k)*pi(1,k,1)
 pressure_native(k)=rgas*rho_native(k)*theta_m_native*pi(1,k,1)
 dz(1,k,1)=z_native(k+1)-z_native(k)
 enddo
 do k=2,nm
 tem1=1./(z_native(k+1)-z_native(k-1))
 fzm=(z_native(k)-z_native(k-1))*tem1;fzp=(z_native(k+1)-z_native(k))*tem1
 tw(1,k,1)=fzm*t(1,k,1)+fzp*t(1,k-1,1)
 enddo
 z0=z_native(nm+1);z1=.5*(z_native(nm+1)+z_native(nm));z2=.5*(z_native(nm)+z_native(nm-1))
 w1=(z0-z2)/(z1-z2);w2=1.-w1
 tw(1,nm+1,1)=w1*t(1,nm,1)+w2*t(1,nm-1,1)
 pw(1,nm+1,1)=exp(w1*log(pressure_native(nm))+w2*log(pressure_native(nm-1)))
 z0=z_native(1);z1=.5*(z_native(1)+z_native(2));z2=.5*(z_native(2)+z_native(3))
 w1=(z0-z2)/(z1-z2);w2=1.-w1;tw(1,1,1)=w1*t(1,1,1)+w2*t(1,2,1)
 do k=nm,1,-1
 rho_moist=rho_native(k)*(1.+qv(1,k,1))
 pw(1,k,1)=pw(1,k+1,1)+gravity*rho_moist*dz(1,k,1)
 p(1,k,1)=.5*(pw(1,k+1,1)+pw(1,k,1))
 enddo
 do k=1,nz
 ozn(1,k,1)=real(k)*1.e-6
 enddo
 glw=0.;olr=0.;lwcf=0.;hr=0.;rcloud=0.;rice=0.;rsnow=0.
 call rrtmg_lwrad(p3d=p,p8w=pw,pi3d=pi,t3d=t,t8w=tw,dz8w=dz,qv3d=qv,qc3d=qc,qi3d=qi,qs3d=qs, &
 cldfra3d=cf,tsk=skin,emiss=emiss,xland=land,xice=xice,snow=snow,xlat=lat,julday=day,icloud=icloud, &
 cldovrlp=overlap,idcor=idcor,o3input=o3input,noznlevels=nz,pin=pin,o3clim=ozn,glw=glw,olr=olr,lwcf=lwcf, &
 rthratenlw=hr,has_reqc=reqc,has_reqi=reqi,has_reqs=reqs,re_cloud=rc,re_ice=ri,re_snow=rs, &
 rre_cloud=rcloud,rre_ice=rice,rre_snow=rsnow,lwupt=lwupt,lwuptc=lwuptc,lwdnt=lwdnt,lwdntc=lwdntc, &
 lwupb=lwupb,lwupbc=lwupbc,lwdnb=lwdnb,lwdnbc=lwdnbc, &
 lwupflx=up,lwupflxc=up_clear,lwdnflx=down,lwdnflxc=down_clear, &
 ids=1,ide=1,jds=1,jde=1,kds=1,kde=nm+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=km, &
 its=1,ite=1,jts=1,jte=1,kts=1,kte=nm)
 model(1,:)=p(1,1:nm,1);model(2,:)=t(1,1:nm,1);model(3,:)=dz(1,1:nm,1);model(4,:)=qv(1,1:nm,1)
 model(5,:)=qc(1,1:nm,1);model(6,:)=qi(1,1:nm,1);model(7,:)=qs(1,1:nm,1);model(8,:)=cf(1,1:nm,1)
 model(9,:)=rc(1,1:nm,1);model(10,:)=ri(1,1:nm,1);model(11,:)=rs(1,1:nm,1);model(12,:)=pi(1,1:nm,1)
 interfaces(1,:)=pw(1,1:nm+1,1);interfaces(2,:)=tw(1,1:nm+1,1)
 surface=[skin(1,1),emiss(1,1),land(1,1),xice(1,1),snow(1,1),lat(1,1)]
 flags=[icloud,overlap,idcor,day,reqc,reqi,reqs,o3input,nz]
 flux=[glw(1,1),olr(1,1),lwcf(1,1),lwupt(1,1),lwuptc(1,1),lwdnt(1,1),lwdntc(1,1), &
 lwupb(1,1),lwupbc(1,1),lwdnb(1,1),lwdnbc(1,1)]
 write(prefix,'(i3.3)')scenario
 call wr_r2(trim(prefix)//'/model',model)
 call wr_r2(trim(prefix)//'/interfaces',interfaces)
 call wr_r1(trim(prefix)//'/surface',surface)
 call wr_i1(trim(prefix)//'/flags',flags)
 call wr_r1(trim(prefix)//'/pin',pin)
 call wr_r1(trim(prefix)//'/ozone',ozn(1,:,1))
 call wr_r1(trim(prefix)//'/flux',flux)
 call wr_r1(trim(prefix)//'/heating',hr(1,1:nm,1))
 call wr_r1(trim(prefix)//'/rre_cloud',rcloud(1,1:nm,1))
 call wr_r1(trim(prefix)//'/rre_ice',rice(1,1:nm,1))
 call wr_r1(trim(prefix)//'/rre_snow',rsnow(1,1:nm,1))
 nout=nm+max(nint(pw(1,nm+1,1)/100./4.),1)+1
 call wr_r1(trim(prefix)//'/uflx',up(1,1:nout,1))
 call wr_r1(trim(prefix)//'/uflxc',up_clear(1,1:nout,1))
 call wr_r1(trim(prefix)//'/dflx',down(1,1:nout,1))
 call wr_r1(trim(prefix)//'/dflxc',down_clear(1,1:nout,1))
 enddo
 close(u_in)
 call bio_close()
end program
