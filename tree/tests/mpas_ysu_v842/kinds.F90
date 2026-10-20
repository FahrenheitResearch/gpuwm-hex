! Standalone framework stubs: campaign single precision, no executable physics.
module ccpp_kind_types
  implicit none
  integer, parameter :: kind_phys = selected_real_kind(6)
end module
module mpas_kind_types
  implicit none
  integer, parameter :: RKIND = selected_real_kind(6)
end module
