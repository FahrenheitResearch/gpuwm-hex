module mpas_kind_types
 integer, parameter :: RKIND=selected_real_kind(6)
end module
module mpas_atm_dimensions
 integer :: nVertLevels=8, maxEdges=3, num_scalars=6
end module
module mpas_derived_types
 type MPAS_Clock_type
 integer :: unused=0
 end type
 type MPAS_Time_type
 integer :: unused=0
 end type
 integer, parameter :: MPAS_NOW=1,MPAS_LOG_CRIT=1
end module
module mpas_log
contains
 subroutine mpas_log_write(message)
 character(len=*),intent(in)::message
 end subroutine
end module
module mpas_timekeeping
 use mpas_derived_types
contains
 function mpas_get_clock_time(clock,which,ierr) result(time)
 type(MPAS_Clock_type)::clock
 integer::which,ierr
 type(MPAS_Time_type)::time
 ierr=0
 end function
 subroutine mpas_get_time(curr_time,H,M,S,S_n,S_d)
 type(MPAS_Time_type)::curr_time
 integer::H,M,S,S_n,S_d
 H=0;M=0;S=0;S_n=0;S_d=1
 end subroutine
end module
