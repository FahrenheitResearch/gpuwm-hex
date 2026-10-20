! Pristine MPAS module_ra_rrtmg_sw_aerosols.F:70-451 rejection paths.
program oracle_aerosol_fatal
 use module_ra_rrtmg_sw_aerosols, only: calc_aerosol_rrtmg_sw
 implicit none
 integer :: mode,opts(4),typ(1,1)
 real :: vals(4),p(1,4,1),t(1,4,1),q(1,4,1),dz(1,4,1),ht(1,1)
 real :: tau(1,4,1,14),ssa(1,4,1,14),asy(1,4,1,14)
 real :: aod(1,1),ang(1,1),sad(1,1),gad(1,1)
 character(len=16) :: argument
 call get_command_argument(1,argument)
 read(argument,*)mode
 opts=[1,3,3,3];vals=[.12,1.3,.85,.9]
 p=90000.;t=280.;q=.005;dz=300.;ht=0.;typ=1
 aod=.12;ang=1.3;sad=.85;gad=.9;tau=0.;ssa=1.;asy=0.
 select case(mode)
 case(0)
  vals(1)=-.01
 case(1,2)
  opts(1)=2;aod=-.01
 case(3)
  opts(2)=2
 case(4,5)
  opts(3)=1;vals(3)=merge(-.01,1.01,mode==4)
 case(6,7,8)
  opts(3)=2;sad=merge(-.01,1.01,mode==7)
 case(9,10)
  opts(4)=1;vals(4)=merge(-.01,1.01,mode==9)
 case(11,12,13)
  opts(4)=2;gad=merge(-1.01,1.01,mode==12)
 case(14:17)
  opts(mode-13)=0
 end select
 select case(mode)
 case(1)
  call invoke(ang=ang,sad=sad,gad=gad)
 case(3)
  call invoke(aod=aod,sad=sad,gad=gad)
 case(6)
  call invoke(aod=aod,ang=ang,gad=gad)
 case(11)
  call invoke(aod=aod,ang=ang,sad=sad)
 case default
  call invoke(aod=aod,ang=ang,sad=sad,gad=gad)
 end select
 error stop 'native unexpectedly returned'
contains
 subroutine invoke(aod,ang,sad,gad)
  real,optional,intent(inout) :: aod(1,1),ang(1,1),sad(1,1),gad(1,1)
  call calc_aerosol_rrtmg_sw(ht,dz,p,t,q,typ,opts(1),opts(2),opts(3),opts(4), &
   vals(1),vals(2),vals(3),vals(4),aod,ang,sad,gad,1,1,1,1,1,4,1,1,1,1,1,4,tau,ssa,asy)
 end subroutine
end program
