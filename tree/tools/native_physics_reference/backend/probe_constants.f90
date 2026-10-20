! Independent direct native constants and post-dynamics RKIND time addition.
! MPAS source: src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103;
! src/framework/mpas_constants.F:43-56;
! src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1847-1852.
program probe_constants
use iso_c_binding
use mpas_kind_types, only: RKIND
use mpas_atmphys_constants, only: cp,gravity,R_d,rcp,ep_1,ep_2,xlv,xls,svp1,svp2,svp3,svpt0,karman,stbolt
use mpas_constants, only: omega
implicit none
real(kind=RKIND) :: old_time,dt,new_time
call word('cp',cp)
call word('gravity',gravity)
call word('r_d',R_d)
call word('rcp',rcp)
call word('ep_1',ep_1)
call word('ep_2',ep_2)
call word('xlv',xlv)
call word('xls',xls)
call word('svp1',svp1)
call word('svp2',svp2)
call word('svp3',svp3)
call word('svpt0',svpt0)
call word('karman',karman)
call word('stbolt',stbolt)
call word('omega',omega)
old_time=16777216.0_RKIND
dt=1.0_RKIND
new_time=old_time+dt
call word('rounded_time',new_time)
contains
subroutine word(name,value)
 character(len=*),intent(in) :: name
 real(kind=RKIND),intent(in) :: value
 if(storage_size(value)/=32) stop 1
 write(*,'(a,1x,z8.8)') name,transfer(value,0_c_int32_t)
end subroutine
end program
