! MPAS: module_ra_rrtmg_lw.F:2099-2737. Exercise both stochastic engines.
program mcica
use mcica_subcol_gen_lw,only:mcica_subcol_lw
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_binio
implicit none
integer,parameter::nl=8,ng=140
type(dm_info)::info
real::play(1,nl),height(1,nl),cf(1,nl),liq(1,nl),ice(1,nl),snow(1,nl),radius(3,nl)
real::tau(16,1,nl),cloud_out(ng,1,nl),liq_out(ng,1,nl),ice_out(ng,1,nl),snow_out(ng,1,nl)
real::tau_out(ng,1,nl),rei(1,nl),rel(1,nl),res(1,nl),a(12,nl),surface(6)
integer::scenario,k,ib,overlap,rng,flags(5)
character(20)::prefix
character(1024)::outdir
call get_command_argument(1,outdir)
call rrtmg_initlw_forMPAS(info)
call bio_open(trim(outdir)//'/native-mcica.bin')
do scenario=0,29
 overlap=mod(scenario,5)+1;rng=scenario/10
 play(1,:)=[900.123,700.234,500.345,300.456,100.567,30.678,10.789,.987]
 height(1,:)=[50.,150.,1000.,1700.,3000.,4100.,9500.,17000.]
 cf(1,:)=[-1.e-21,1.e-21,.72,.96,0.,.24,1.,.52]
 do k=1,nl
  liq(1,k)=real(k)*10.25;ice(1,k)=real(k)*7.75;snow(1,k)=real(k)*3.25
  rei(1,k)=20.+real(k);rel(1,k)=8.+real(k);res(1,k)=50.+real(k)
  do ib=1,16
   tau(ib,1,k)=real(ib)*.003+real(k)*.015
  enddo
 enddo
 flags=[overlap,rng,mod(scenario/5,2),80+mod(scenario,2)*220,150]
 a=0.;a(1,:)=play(1,:);a(3,:)=height(1,:);a(6,:)=cf(1,:)
 a(7,:)=liq(1,:);a(8,:)=ice(1,:);a(9,:)=snow(1,:)
 a(10,:)=rel(1,:);a(11,:)=rei(1,:);a(12,:)=res(1,:)
 surface=0.;surface(6)=-60.+real(mod(scenario,7))*20.
 call mcica_subcol_lw(1,1,nl,overlap,150,rng,play,cf,ice,liq,snow,rei,rel,res,tau, &
                     height,flags(3),flags(4),surface(6),cloud_out,ice_out,liq_out,snow_out,rei,rel,res,tau_out)
 radius(1,:)=rel(1,:);radius(2,:)=rei(1,:);radius(3,:)=res(1,:)
 write(prefix,'(i3.3)')scenario
 call wr_i1(trim(prefix)//'/flags',flags)
 call wr_i0(trim(prefix)//'/rng_out',rng)
 call wr_r2(trim(prefix)//'/a',a)
 call wr_r1(trim(prefix)//'/surface',surface)
 call wr_r2(trim(prefix)//'/tau',tau(:,1,:))
 call wr_r2(trim(prefix)//'/cloud',cloud_out(:,1,:))
 call wr_r2(trim(prefix)//'/liquid',liq_out(:,1,:))
 call wr_r2(trim(prefix)//'/ice',ice_out(:,1,:))
 call wr_r2(trim(prefix)//'/snow',snow_out(:,1,:))
 call wr_r2(trim(prefix)//'/optical_depth',tau_out(:,1,:))
 call wr_r2(trim(prefix)//'/radii',radius)
enddo
call bio_close()
end program mcica
