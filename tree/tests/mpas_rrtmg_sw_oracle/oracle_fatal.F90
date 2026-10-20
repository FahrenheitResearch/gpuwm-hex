! Native MPAS module_ra_rrtmg_sw.F:2209-2421, cloud fatal boundaries.
program oracle_fatal
 use rrtmg_sw_cldprmc,only:cldprmc_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer,parameter::nl=4,ng=112
 integer::caseid,ice,inflag
 character(16)::argument
 real::cf(ng,nl),ci(ng,nl),cl(ng,nl),cs(ng,nl),tau(ng,nl),ssa(ng,nl),asy(ng,nl),fsf(ng,nl),taor(ng,nl)
 real::rei(nl),rel(nl),res(nl)
 call get_command_argument(1,argument);read(argument,*)caseid
 call rrtmg_initsw_forMPAS(dminfo)
 cf=1.;ci=1.;cl=1.;cs=0.;tau=.2;ssa=.91;asy=.71;fsf=.12
 rei=40.;rel=12.;res=60.;ice=3;inflag=2
 select case(caseid)
 case(0);inflag=1
 case(1);ice=2;rei=4.999
 case(2);ice=2;rei=131.001
 case(3);rei=4.999
 case(4);rei=140.001
 case(5);ice=5;cs=1.;res=4.999
 case(6);ice=5;cs=1.;res=140.001
 case(7);rel=1.499
 case(8);rel=60.001
 case default;error stop 'unknown fatal witness'
 end select
 call cldprmc_sw(nl,inflag,ice,1,cf,ci,cl,cs,rei,rel,res,taor,tau,ssa,asy,fsf)
 error stop 'native unexpectedly returned from fatal boundary'
end program oracle_fatal
