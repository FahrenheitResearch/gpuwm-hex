! MPAS: module_ra_rrtmg_lw.F:10705-11194. Direct solver boundary oracle.
program solver
use rrtmg_lw_rad,only:rrtmg_lw
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_binio
implicit none
integer,parameter::nl=8,ng=140
type(dm_info)::info
real::play(1,nl),plev(1,nl+1),tlay(1,nl),tlev(1,nl+1),tsfc(1),emis(1,16),gas(10,nl)
real::cloud(ng,1,nl),ice(ng,1,nl),liquid(ng,1,nl),snow(ng,1,nl),tau(ng,1,nl)
real::rei(1,nl),rel(1,nl),res(1,nl),aerosol(1,nl,16)
real::u(1,nl+1),d(1,nl+1),uc(1,nl+1),dc(1,nl+1),hr(1,nl),hrc(1,nl),clean_u(1,nl+1),clean_d(1,nl+1)
integer::scenario,k,ig,ib,inflag,iceflag,liqflag,icloud
character(20)::prefix
character(1024)::outdir
call get_command_argument(1,outdir)
call rrtmg_initlw_forMPAS(info)
call bio_open(trim(outdir)//'/native-solver.bin')
do scenario=0,157
 inflag=2;if(scenario>=144)inflag=0
 iceflag=mod(scenario,6);liqflag=mod(scenario/6,2);icloud=1
 if(mod(scenario,17)==0)icloud=0
 if(scenario>=156)then
  inflag=2;iceflag=0;liqflag=1;icloud=1
 endif
 plev(1,:)=[1000.,800.,550.,300.,140.,50.,5.,.001,.000001]
 if(scenario>=150)plev(1,1)=4000.
 do k=1,nl
  play(1,k)=.5*(plev(1,k)+plev(1,k+1))
  tlay(1,k)=155.+real(mod(scenario/12,8))*25.-real(k)*2.
  tlev(1,k)=tlay(1,k)+1.
  gas(1,k)=10.**(-7+mod(scenario/12+k,7))
  gas(2,k)=10.**(-8+mod(scenario/24,6))
  gas(3,k)=.0004;if(mod(scenario/12,2)==1)gas(3,k)=.004
  gas(4,k)=.0000018;if(mod(scenario/24,2)==1)gas(4,k)=.0001
  gas(5,k)=.00000032;if(mod(scenario/36,2)==1)gas(5,k)=.0001
  gas(6,k)=.209
  gas(7:10,k)=0.;if(mod(scenario/12,2)==1)gas(7:10,k)=1.e-7
  ! Reach native zero-gas floors and the low-N2O upper-atmosphere paths.
  if(scenario>=156)gas(2:5,k)=0.
  if(scenario==157)gas(5,k)=1.e-30
  rei(1,k)=30.+real(k);rel(1,k)=10.+real(k);res(1,k)=60.+real(k)
  if(k==nl)then
   rei(1,k)=130.;if(iceflag==2)rei(1,k)=131.
   if(iceflag>=3)rei(1,k)=140.
   rel(1,k)=60.;res(1,k)=140.
  endif
  if(scenario>=156)rel(1,k)=2.5
  do ig=1,ng
   cloud(ig,1,k)=1.;if(mod(ig+k,7)==0)cloud(ig,1,k)=0.
   ice(ig,1,k)=real(k)*7.;liquid(ig,1,k)=real(k)*9.;snow(ig,1,k)=real(k)*3.
   if(mod(k+scenario,3)==0)ice(ig,1,k)=0.
   if(mod(k+scenario,4)==0)liquid(ig,1,k)=0.
   if(mod(k+scenario,5)==0)snow(ig,1,k)=0.
   if(mod(k+scenario,6)==0)then
    ice(ig,1,k)=0.;liquid(ig,1,k)=0.;snow(ig,1,k)=0.
   endif
   tau(ig,1,k)=.01+real(ig)*.0001
  enddo
  do ib=1,16
   aerosol(1,k,ib)=0.;if(mod(scenario,2)==1)aerosol(1,k,ib)=real(k*ib)*.00001
  enddo
 enddo
 tlev(1,nl+1)=tlay(1,nl)-2.;tsfc=288.
 do ib=1,16
  emis(1,ib)=.9+real(mod(ib,3))*.05
 enddo
 write(prefix,'(i3.3)')scenario
 call wr_i1(trim(prefix)//'/flags',[icloud,inflag,iceflag,liqflag])
 call wr_r2(trim(prefix)//'/play',play);call wr_r2(trim(prefix)//'/plev',plev)
 call wr_r2(trim(prefix)//'/tlay',tlay);call wr_r2(trim(prefix)//'/tlev',tlev)
 call wr_r1(trim(prefix)//'/tsfc',tsfc);call wr_r2(trim(prefix)//'/emis',emis)
 call wr_r2(trim(prefix)//'/gas',gas)
 call wr_r3(trim(prefix)//'/cloud',cloud);call wr_r3(trim(prefix)//'/ice',ice)
 call wr_r3(trim(prefix)//'/liquid',liquid);call wr_r3(trim(prefix)//'/snow',snow)
 call wr_r3(trim(prefix)//'/tau',tau)
 call wr_r2(trim(prefix)//'/re_ice',rei);call wr_r2(trim(prefix)//'/re_liquid',rel);call wr_r2(trim(prefix)//'/re_snow',res)
 call wr_r3(trim(prefix)//'/aerosol',aerosol)
 call rrtmg_lw(1,nl,icloud,play,plev,tlay,tlev,tsfc, &
  reshape(gas(1,:),[1,nl]),reshape(gas(2,:),[1,nl]),reshape(gas(3,:),[1,nl]), &
  reshape(gas(4,:),[1,nl]),reshape(gas(5,:),[1,nl]),reshape(gas(6,:),[1,nl]), &
  reshape(gas(7,:),[1,nl]),reshape(gas(8,:),[1,nl]),reshape(gas(9,:),[1,nl]),reshape(gas(10,:),[1,nl]), &
  emis,inflag,iceflag,liqflag,cloud,tau,ice,liquid,snow,rei,rel,res,aerosol,u,d,hr,uc,dc,hrc,clean_u,clean_d,0)
 call wr_r2(trim(prefix)//'/uflx',u);call wr_r2(trim(prefix)//'/dflx',d)
 call wr_r2(trim(prefix)//'/hr',hr);call wr_r2(trim(prefix)//'/uflxc',uc)
 call wr_r2(trim(prefix)//'/dflxc',dc);call wr_r2(trim(prefix)//'/hrc',hrc)
enddo
call bio_close()
end program solver
