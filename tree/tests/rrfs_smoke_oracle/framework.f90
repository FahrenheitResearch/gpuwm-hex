module mpas_kind_types
 implicit none
 integer, parameter :: RKIND=kind(1.0)
end module
module mpas_pool_routines
 implicit none
contains
 subroutine mpas_log_write(message)
 character(*),intent(in)::message
 end subroutine
end module
module mpas_constants
 use mpas_kind_types
 implicit none
 real(RKIND),parameter::cp=1004.5
end module
module mpas_timer
 implicit none
contains
 subroutine mpas_timer_start(name)
 character(*),intent(in)::name
 end subroutine
 subroutine mpas_timer_stop(name)
 character(*),intent(in)::name
 end subroutine
end module
