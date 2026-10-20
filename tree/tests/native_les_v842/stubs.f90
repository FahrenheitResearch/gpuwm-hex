module mpas_kind_types
 integer, parameter :: RKIND=kind(1.0)
end module
module mpas_derived_types
 type MPAS_Clock_type
 integer :: unused
 end type
 type MPAS_Time_type
 integer :: unused
 end type
 integer, parameter :: MPAS_NOW=0,MPAS_LOG_CRIT=0
end module
module mpas_timekeeping
contains
 subroutine mpas_get_clock_time()
 end subroutine
 subroutine mpas_get_time()
 end subroutine
end module
module mpas_log
contains
 subroutine mpas_log_write(message)
 character(*) :: message
 end subroutine
end module
module mpas_atm_dimensions
 integer :: nVertLevels=5,maxEdges=2,num_scalars=3
end module
