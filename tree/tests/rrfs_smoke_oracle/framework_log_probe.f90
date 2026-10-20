! Framework-only test adapter; no native SMOKE numerical routine is called.
! MPAS-Model-rrfs/src/framework/mpas_log.F:49-65,565-595 @1d6e370e.
module r14_framework_log_probe
use iso_c_binding, only: c_int
use mpas_log, only: mpas_log_info, mpas_log_write
! MPAS-Model-rrfs/src/framework/mpas_timer.F:45,88-229,243-310 @1d6e370e.
use mpas_timer, only: timer_root, mpas_timer_start, mpas_timer_stop
implicit none
contains
function r14_timer_associated() result(value) bind(C)
integer(c_int) :: value
value = 0
if (associated(timer_root)) value = 1
end function
subroutine r14_timer_reference(site) bind(C)
integer(c_int),value :: site
select case(site)
case(1)
call mpas_timer_start('r14 parent')
case(2)
! Original smoke driver starts with clear_timer absent.
call mpas_timer_start('mpas_smoke')
case(3)
call mpas_timer_stop('mpas_smoke')
case(4)
call mpas_timer_stop('r14 parent')
case default
error stop "unknown timer fixture site"
end select
end subroutine
subroutine r14_log_active(active) bind(C)
integer(c_int),value :: active
mpas_log_info%outputLog%isActive = active /= 0
end subroutine
function r14_log_counter() result(value) bind(C)
integer(c_int) :: value
value = mpas_log_info%outputMessageCount
end function
subroutine r14_log_reference(site) bind(C)
integer(c_int),value :: site
select case(site)
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:262 @1d6e370e.
case(262)
call mpas_log_write(' Beginning Aerosol Driver')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:290 @1d6e370e.
case(290)
call mpas_log_write(' Calling smoke prep')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:312 @1d6e370e.
case(312)
call mpas_log_write(' Initializing dry deposition parameterss ')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:314 @1d6e370e.
case(314)
call mpas_log_write(' Initializing radiation feedback parameterss ')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:321 @1d6e370e.
case(321)
call mpas_log_write(' Calling module wildfire smoke emissions ')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:452 @1d6e370e.
case(452)
call mpas_log_write(' Calling ebu_driver')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:477 @1d6e370e.
case(477)
call mpas_log_write(' Calling add_emis_burn')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:513 @1d6e370e.
case(513)
call mpas_log_write(' Calling dust driver')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:538 @1d6e370e.
case(538)
call mpas_log_write(' Calling anthro emis driver')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:555 @1d6e370e.
case(555)
call mpas_log_write(' Calling dry_dep_driver_emerson')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:577 @1d6e370e.
case(577)
call mpas_log_write('Not performing dry deposition')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:583 @1d6e370e.
case(583)
call mpas_log_write(' Calling wetdep_ls')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:595 @1d6e370e.
case(595)
call mpas_log_write(' Calculating AOD ')
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:600 @1d6e370e.
case(600)
call mpas_log_write(' Calculating VIS ')
case default
error stop "unknown original log site"
end select
end subroutine
end module
