! Native correct-mode referee mpas_atmphys_lsm_shared.F:60-103.
! Test-only C ABI; arithmetic stays in the copied cluster referee module.
subroutine rp3_recover(tsk,sst,ice,tsea,tice,threshold,n) bind(c)
use iso_c_binding
use mpas_atmphys_lsm_shared
implicit none
integer(c_int),value :: n
real(c_float),value :: threshold
real(c_float),intent(in) :: tsk(n,1),sst(n,1),ice(n,1)
real(c_float),intent(out) :: tsea(n,1),tice(n,1)
call seaice_component_tsk(1,n,1,1,1,n,1,1,threshold,ice,sst,tsk,tsea,tice)
end subroutine
