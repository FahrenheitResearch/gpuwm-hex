! Framework-only stubs; all numerical routines are pristine MPAS sources.
module mpas_kind_types
  integer, parameter :: RKIND=kind(1.0), R4KIND=kind(1.0), R8KIND=kind(1.0d0)
end module
module ccpp_kind_types
  integer, parameter :: kind_phys=kind(1.0)
end module
module mpas_derived_types
  integer, parameter :: MPAS_LOG_ERR=1
end module
module mpas_log
contains
  subroutine mpas_log_write(message, messageType, realArgs, intArgs)
    character(*), intent(in) :: message
    integer, optional, intent(in) :: messageType, intArgs(:)
    real, optional, intent(in) :: realArgs(:)
    write(0,*) trim(message)
  end subroutine
end module
module mpas_atmphys_utilities
contains
  subroutine physics_error_fatal(message)
    character(*), intent(in) :: message
    write(0,*) trim(message)
    error stop 1
  end subroutine
end module
module mpas_io_units
contains
  subroutine mpas_new_unit(unit, unformatted)
    integer, intent(out) :: unit
    logical, optional, intent(in) :: unformatted
    unit=71
  end subroutine
  subroutine mpas_release_unit(unit)
    integer, intent(in) :: unit
  end subroutine
end module
