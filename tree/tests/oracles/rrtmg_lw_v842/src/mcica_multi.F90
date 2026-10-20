! MPAS: module_ra_rrtmg_lw.F:2099-2697. Global MT stream ordering oracle.
program mcica_multi
use mcica_subcol_gen_lw,only:mcica_subcol_lw
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_binio
implicit none
integer,parameter::nl=8,ng=140,nc=3
type(dm_info)::info
real::play(nc,nl),height(nc,nl),cf(nc,nl),liq(nc,nl),ice(nc,nl),snow(nc,nl)
real::tau(16,nc,nl),cloud_out(ng,nc,nl),liq_out(ng,nc,nl),ice_out(ng,nc,nl),snow_out(ng,nc,nl),tau_out(ng,nc,nl)
real::rei(nc,nl),rel(nc,nl),res(nc,nl),rei_out(nc,nl),rel_out(nc,nl),res_out(nc,nl)
real::a(nc,12,nl),surface(nc,6),radii(nc,3,nl)
integer::scenario,k,ib,c,overlap,rng,flags(5)
character(20)::prefix
character(1024)::outdir
call get_command_argument(1,outdir)
call rrtmg_initlw_forMPAS(info)
call bio_open(trim(outdir)//'/native-mcica-multi.bin')
do scenario=0,29
 overlap=mod(scenario,5)+1;rng=scenario/10
 a=0.;surface=0.
 do c=1,nc
  play(c,:)=[900.123,700.234,500.345,300.456,100.567,30.678,10.789,.987]+real(c-1)*.125
  height(c,:)=[50.,150.,1000.,1700.,3000.,4100.,9500.,17000.]*real(c)
  cf(c,:)=[-1.e-21,1.e-21,.72,.96,0.,.24,1.,.52]
  if(c==2)cf(c,:)=[.2,1.,0.,.6,.72,.5,1.e-21,-1.e-21]
  if(c==3)cf(c,:)=[1.,0.,.1,.6,.72,.5,1.e-21,0.]
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
 flags=[overlap,rng,mod(scenario/5,2),80+mod(scenario,2)*220,150]
 call mcica_subcol_lw(1,nc,nl,overlap,150,rng,play,cf,ice,liq,snow,rei,rel,res,tau, &
                     height,flags(3),flags(4),surface(1,6),cloud_out,ice_out,liq_out,snow_out,rei_out,rel_out,res_out,tau_out)
 radii(:,1,:)=rel_out;radii(:,2,:)=rei_out;radii(:,3,:)=res_out
 write(prefix,'(i3.3)')scenario
 call wr_i1(trim(prefix)//'/flags',flags)
 call wr_i0(trim(prefix)//'/rng_out',rng)
 call wr_r3(trim(prefix)//'/a',a)
 call wr_r2(trim(prefix)//'/surface',surface)
 call wr_r3(trim(prefix)//'/tau',tau)
 call wr_r3(trim(prefix)//'/cloud',cloud_out)
 call wr_r3(trim(prefix)//'/liquid',liq_out)
 call wr_r3(trim(prefix)//'/ice',ice_out)
 call wr_r3(trim(prefix)//'/snow',snow_out)
 call wr_r3(trim(prefix)//'/optical_depth',tau_out)
 call wr_r3(trim(prefix)//'/radii',radii)
enddo
call bio_close()
end program mcica_multi
