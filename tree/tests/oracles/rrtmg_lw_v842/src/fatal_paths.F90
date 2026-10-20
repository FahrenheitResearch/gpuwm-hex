! MPAS: module_ra_rrtmg_lw.F:2195-2196,2433-2434,2922-3017.
! Exercise native invalid-input stops separately; no numerical replacement.
program fatal_paths
use mcica_subcol_gen_lw,only:mcica_subcol_lw
use rrtmg_lw_cldprmc,only:cldprmc
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
implicit none
integer,parameter::nl=8,ng=140
type(dm_info)::info
real::play(1,nl),height(1,nl),cf(1,nl),liq(1,nl),ice(1,nl),snow(1,nl),rei(1,nl),rel(1,nl),res(1,nl)
real::tc(16,1,nl),co(ng,1,nl),lo(ng,1,nl),io(ng,1,nl),so(ng,1,nl),to(ng,1,nl)
real::reiout(1,nl),relout(1,nl),resout(1,nl)
integer::scenario,overlap,rng,iflag,iceflag,liqflag,ncb
character(16)::arg
call get_command_argument(1,arg);read(arg,*)scenario
call rrtmg_initlw_forMPAS(info)
play(1,:)=[900.123,700.234,500.345,300.456,100.567,30.678,10.789,.987]
height=100.;cf=.5;liq=10.;ice=10.;snow=10.;rei=30.;rel=10.;res=60.;tc=.1
rng=0;overlap=1
if(scenario<=1)then
 if(scenario==0)overlap=6
 if(scenario==1)play(1,1)=600.
 call mcica_subcol_lw(1,1,nl,overlap,150,rng,play,cf,ice,liq,snow,rei,rel,res,tc, &
       height,0,100,0.,co,io,lo,so,reiout,relout,resout,to)
else
 co=1.;io=10.;lo=10.;so=10.;to=.1;iflag=2;iceflag=3;liqflag=1
 select case(scenario)
 case(2)
 iflag=1
 case(3)
 iceflag=0;rei=9.
 case(4)
 iceflag=1;rei=12.
 case(5)
 iceflag=1;rei=131.
 case(6)
 iceflag=2;rei=4.
 case(7)
 iceflag=2;rei=132.
 case(8)
 rei=4.
 case(9)
 rei=141.
 case(10)
 iceflag=5;res=4.
 case(11)
 iceflag=5;res=141.
 case(12)
 rel=2.
 case(13)
 rel=61.
 end select
 call cldprmc(nl,iflag,iceflag,liqflag,co(:,1,:),io(:,1,:),lo(:,1,:),so(:,1,:), &
       rei(1,:),rel(1,:),res(1,:),ncb,to(:,1,:))
endif
error stop 'oracle expected native invalid-input termination'
end program fatal_paths
