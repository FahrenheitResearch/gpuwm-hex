! Direct MPAS initialization / table-copy comparison, without generated table ABI.
! MPAS source: mpas_atmphys_rrtmg_lwinit.F:45-59; mpas_atmphys_rrtmg_swinit.F:48-62.
program probe_tables
use iso_c_binding
use mpas_kind_types
use mpas_derived_types, only: dm_info
use mpas_atmphys_rrtmg_lwinit
use mpas_atmphys_rrtmg_swinit
use hex_native_runtime, only: native_runtime_create,native_runtime_destroy,native_runtime_object
use rrlw_ref, only: lw_pref=>pref
use rrsw_ref, only: sw_pref=>pref
use rrlw_tbl, only: lw_tau=>tau_tbl
use rrsw_tbl, only: sw_tau=>tau_tbl
use rrlw_kg01, only: lw_kao=>kao
use rrsw_cld, only: sw_extliq=>extliq1
type(c_ptr) :: runtime
type(dm_info),pointer :: dminfo
integer :: i
runtime=native_runtime_create()
call c_f_pointer(native_runtime_object(runtime,1_c_int),dminfo)
call rrtmg_initlw_forMPAS(dminfo)
call rrtmg_initsw_forMPAS(dminfo)
write(*,'(a)') 'TABLE_WORDS_BEGIN'
! The pinned parkind rb=kind(1.0) is four bytes in the single recipe.
if(storage_size(lw_pref)/=32.or.storage_size(sw_pref)/=32) stop 1
write(*,'(z8.8)') transfer(lw_pref,[(0_c_int32_t,i=1,size(lw_pref))])
write(*,'(z8.8)') transfer(sw_pref,[(0_c_int32_t,i=1,size(sw_pref))])
write(*,'(z8.8)') transfer(lw_tau,[(0_c_int32_t,i=1,size(lw_tau))])
write(*,'(z8.8)') transfer(sw_tau,0_c_int32_t)
write(*,'(z8.8)') transfer(lw_kao,[(0_c_int32_t,i=1,size(lw_kao))])
write(*,'(z8.8)') transfer(sw_extliq,[(0_c_int32_t,i=1,size(sw_extliq))])
write(*,'(a)') 'TABLE_WORDS_END'
call native_runtime_destroy(runtime)
end program
