! Minimal framework kinds for the standalone pristine MPAS scheme only.
module mpas_kind_types
  use iso_c_binding, only: c_float
  implicit none
  integer, parameter :: RKIND = c_float
end module
module ccpp_kind_types
  use mpas_kind_types, only: kind_phys => RKIND
  implicit none
end module
