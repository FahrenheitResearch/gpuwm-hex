! Test-only ABI storage access. MPAS mpas_atmphys_o3climatology.F:90-98,146-187.
! The complete pristine module and native pool stubs allocate this storage.
subroutine ozone_callback_storage(ncells,nlevels,latitude,pressure,climatology) bind(C)
 use iso_c_binding
 use oracle_state, only: nc,nl,lat,pin,monthly
 implicit none
 integer(c_int),intent(out)::ncells,nlevels
 type(c_ptr),intent(out)::latitude,pressure,climatology
 ncells=nc;nlevels=nl
 latitude=c_loc(lat);pressure=c_loc(pin);climatology=c_loc(monthly)
end subroutine
