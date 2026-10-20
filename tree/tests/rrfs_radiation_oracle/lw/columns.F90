! Strict full pinned fork LW wrapper oracle; synthetic cases adapted from p22.
program columns
use module_ra_rrtmg_lw,only:rrtmg_lwrad
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_dump_state,only:dump_all_state
use lw_binio
implicit none
integer,parameter::nm=8,km=128,nz=7
real::p(1,km,1),pw(1,km,1),pi(1,km,1),t(1,km,1),tw(1,km,1),dz(1,km,1)
real::qv(1,km,1),qc(1,km,1),qi(1,km,1),qs(1,km,1),cf(1,km,1),rc(1,km,1),ri(1,km,1),rs(1,km,1)
real::skin(1,1),emiss(1,1),land(1,1),xice(1,1),snow(1,1),lat(1,1),pin(nz),ozn(1,nz,1)
real::glw(1,1),olr(1,1),lwcf(1,1),hr(1,km,1),rcloud(1,km,1),rice(1,km,1),rsnow(1,km,1),flux(11)
real::lwupt(1,1),lwuptc(1,1),lwdnt(1,1),lwdntc(1,1),lwupb(1,1),lwupbc(1,1),lwdnb(1,1),lwdnbc(1,1)
real::model(12,nm),interfaces(2,nm+1),surface(6)
real::up(1,km,1),up_clear(1,km,1),down(1,km,1),down_clear(1,km,1)
integer::scenario,k,overlap,idcor,day,icloud,reqc,reqi,reqs,o3input,flags(9),nout
character(20)::prefix
character(1024)::outdir
 type(dm_info)::info
call get_command_argument(1,outdir)
call rrtmg_initlw_forMPAS(info)
call dump_all_state(trim(outdir)//'/native-tables.bin')
call bio_open(trim(outdir)//'/native-columns.bin')
pin=[.05,.5,5.,50.,250.,700.,1200.]
 do scenario=0,299
 pin=[.05,.5,5.,50.,250.,700.,1200.]
 overlap=mod(scenario,5)+1;idcor=mod(scenario/5,2);day=80+mod(scenario/10,2)*220
 if(scenario>=80.and.scenario<90)overlap=0
 reqc=mod(scenario/20,2);reqi=mod(scenario/10,2);reqs=mod(scenario/40,2)
 icloud=1;if(mod(scenario,13)==0)icloud=0
 o3input=0;if(mod(scenario,2)==1)o3input=2
 pw=0.;tw=0.
 pw(1,1:nm+1,1)=[100000.,85000.,70000.,55000.,40000.,25000.,10000.,5000.,1100.]
 pw(1,9,1)=real(1+mod(scenario,7))*200.+real(mod(scenario,3))*100.
 tw(1,1:nm+1,1)=[290.,278.,267.,253.,240.,229.,218.,208.,202.]
 do k=1,nm
 p(1,k,1)=.5*(pw(1,k,1)+pw(1,k+1,1))+real(k)*.125
 t(1,k,1)=285.-real(k)*10.+real(mod(k,3))*.3
 pi(1,k,1)=.9-real(k)*.05
 dz(1,k,1)=200.+real(k)*75.25
 qv(1,k,1)=.009/real(k);if(k==6)qv(1,k,1)=-.001
 qc(1,k,1)=real(mod(k,3))*.0001
 qi(1,k,1)=real(mod(k,2))*.00007
 qs(1,k,1)=real(mod(k+1,2))*.00003
 cf(1,k,1)=real(mod(k,4))*.24
 rc(1,k,1)=real(mod(k,4))*4.e-6
 ri(1,k,1)=real(mod(k,3))*20.e-6
 rs(1,k,1)=real(mod(k,4))*60.e-6
 enddo
 if(scenario>=270.and.scenario<276)then
 reqc=1;reqi=1;reqs=1;icloud=1
 endif
 if(scenario>=276.and.scenario<282)pw(1,nm+1,1)=.5
 if(scenario>=282.and.scenario<288)pw(1,1,1)=150000.
 if(scenario>=90)then
 do k=1,nm
 t(1,k,1)=max(185.,315.-real(k)*8.-real(mod(scenario,6))*13.)
 tw(1,k,1)=t(1,k,1)+2.
 qv(1,k,1)=10.**(-7+mod(scenario/6,6))/real(k)
 if(mod(scenario,9)==0)cf(1,k,1)=1.
 if(mod(scenario,11)==0)cf(1,k,1)=0.
 enddo
 if(scenario>=270.and.scenario<276)then
 cf=1.;rc=0.;ri=0.;rs=0.
 endif
 if(scenario>=288)then
 do k=1,nm
 t(1,k,1)=155.+real(mod(scenario,2))*195.
 tw(1,k,1)=t(1,k,1)+1.
 enddo
 endif
 tw(1,nm+1,1)=tw(1,nm,1)-3.
 if(mod(scenario,3)==1)pin=[5.,10.,50.,100.,250.,500.,700.]
 if(mod(scenario,3)==2)pin=[10.,20.,50.,100.,250.,500.,700.]
 endif
 skin=290.;emiss=.975;land=1.+real(mod(scenario,3))*.5
 xice=real(mod(scenario,4))*.2;snow=real(mod(scenario,3))*35.;lat=-60.+real(mod(scenario,7))*20.
 if(scenario>=270.and.scenario<276)land=2.-real(mod(scenario,3))*.5
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
 call bio_close()
end program
