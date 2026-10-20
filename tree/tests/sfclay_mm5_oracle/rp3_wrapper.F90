! RP3 referee: mpas_atmphys_lsm_shared.F:62-105 in the cluster native build 42d7e5f2.
subroutine rp3_components(tsk,sst,xice,sea,ice,n,threshold) bind(c)
use iso_c_binding
use mpas_atmphys_lsm_shared, only: seaice_component_tsk
implicit none
integer(c_int),value:: n
real(c_float),value:: threshold
real(c_float),intent(in):: tsk(n,1),sst(n,1),xice(n,1)
real(c_float),intent(out):: sea(n,1),ice(n,1)
call seaice_component_tsk(1,n,1,1,1,n,1,1,threshold,xice,sst,tsk,sea,ice)
end subroutine
