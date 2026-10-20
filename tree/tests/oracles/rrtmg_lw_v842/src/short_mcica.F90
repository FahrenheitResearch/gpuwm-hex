! MPAS: module_ra_rrtmg_lw.F:2099-2697. Short-layer native defect probe.
! Source module is copied verbatim; every call gets its exact declared extent.
program short_mcica
use mcica_subcol_gen_lw,only:mcica_subcol_lw
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_binio
implicit none
integer,parameter::ng=140
integer::nl,nc,scenario,k,ib,c,overlap,rng,flags(5),rank
real,parameter::p0(8)=[900.123,700.234,500.345,300.456,100.567,30.678,10.789,.987]
real,parameter::h0(8)=[50.,150.,1000.,1700.,3000.,4100.,9500.,17000.]
real,parameter::cf0(8,3)=reshape([.5,.72,.96,.2,0.,.24,1.,.52, &
 .2,1.,0.,.6,.72,.5,1.e-21,-1.e-21,1.,0.,.1,.6,.72,.5,1.e-21,0.],[8,3])
type(dm_info)::info
real,allocatable::play(:,:),height(:,:),cf(:,:),liq(:,:),ice(:,:),snow(:,:)
real,allocatable::tau(:,:,:),cloud_out(:,:,:),liq_out(:,:,:),ice_out(:,:,:),snow_out(:,:,:),tau_out(:,:,:)
real,allocatable::rei(:,:),rel(:,:),res(:,:),rei_out(:,:),rel_out(:,:),res_out(:,:)
real,allocatable::a(:,:,:),surface(:,:),radii(:,:,:)
character(1024)::outdir,arg
character(16)::rankstr
call get_command_argument(1,outdir)
call get_command_argument(2,arg);read(arg,*)nl
call get_command_argument(3,arg);read(arg,*)nc
call get_command_argument(4,arg);read(arg,*)scenario
rankstr='0';call get_environment_variable('PMI_RANK',arg)
if(len_trim(arg)==0)call get_environment_variable('OMPI_COMM_WORLD_RANK',arg)
if(len_trim(arg)>0)then
 read(arg,*)rank;write(rankstr,'(i0)')rank
endif
if(nl<1.or.nl>4.or.nc<1.or.nc>3.or.scenario<0.or.scenario>35)stop 2
allocate(play(nc,nl),height(nc,nl),cf(nc,nl),liq(nc,nl),ice(nc,nl),snow(nc,nl))
allocate(tau(16,nc,nl),cloud_out(ng,nc,nl),liq_out(ng,nc,nl),ice_out(ng,nc,nl),snow_out(ng,nc,nl),tau_out(ng,nc,nl))
allocate(rei(nc,nl),rel(nc,nl),res(nc,nl),rei_out(nc,nl),rel_out(nc,nl),res_out(nc,nl))
allocate(a(nc,12,nl),surface(nc,6),radii(nc,3,nl))
call rrtmg_initlw_forMPAS(info)
overlap=mod(scenario,6);rng=scenario/12
flags=[overlap,rng,mod(scenario/6,2),80+mod(scenario,2)*220,150]
a=0.;surface=0.
! Establish the caller's returned-clear state explicitly; not native seed padding.
cloud_out=0.;liq_out=0.;ice_out=0.;snow_out=0.;tau_out=0.
rei_out=0.;rel_out=0.;res_out=0.
do c=1,nc
 play(c,:)=p0(:nl)+real(c-1)*.125
 height(c,:)=h0(:nl)*real(c);cf(c,:)=cf0(:nl,c)
 do k=1,nl
  liq(c,k)=real(k+c)*10.25;ice(c,k)=real(k+c)*7.75;snow(c,k)=real(k+c)*3.25
  rei(c,k)=20.+real(k+c);rel(c,k)=8.+real(k+c);res(c,k)=50.+real(k+c)
  do ib=1,16
   tau(ib,c,k)=real(ib+c)*.003+real(k)*.015
  enddo
 enddo
 a(c,1,:)=play(c,:);a(c,3,:)=height(c,:);a(c,6,:)=cf(c,:)
 a(c,7,:)=liq(c,:);a(c,8,:)=ice(c,:);a(c,9,:)=snow(c,:)
 a(c,10,:)=rel(c,:);a(c,11,:)=rei(c,:);a(c,12,:)=res(c,:)
 surface(c,6)=-60.+real(mod(scenario,7))*20.
enddo
call bio_open(trim(outdir)//'/rank'//trim(rankstr)//'-input.bin')
call wr_i1('flags',flags);call wr_r3('a',a);call wr_r2('surface',surface);call wr_r3('tau',tau)
call bio_close()
call mcica_subcol_lw(1,nc,nl,overlap,150,rng,play,cf,ice,liq,snow,rei,rel,res,tau, &
 height,flags(3),flags(4),surface(1,6),cloud_out,ice_out,liq_out,snow_out,rei_out,rel_out,res_out,tau_out)
radii(:,1,:)=rel_out;radii(:,2,:)=rei_out;radii(:,3,:)=res_out
call bio_open(trim(outdir)//'/rank'//trim(rankstr)//'-output.bin')
call wr_i0('rng_out',rng)
call wr_r3('cloud',cloud_out);call wr_r3('liquid',liq_out);call wr_r3('ice',ice_out)
call wr_r3('snow',snow_out);call wr_r3('optical_depth',tau_out);call wr_r3('radii',radii)
call bio_close()
end program short_mcica
