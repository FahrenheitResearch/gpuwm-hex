program probe
 use mpas_derived_types
 use mpas_atmphys_rrtmg_lwinit
 use parkind
 use mpas_kind_types
 use mpas_constants,only:cp,gravity
 use rrlw_tbl
 use rrlw_kg01
 type(dm_info)::dminfo
 call rrtmg_initlw_forMPAS(dminfo)
 print *, 'kinds',RKIND,kind_rb,storage_size(ka(1,1,1))
 print *, 'constants', cp,gravity
 print *, 'init',ka(1,1,1), exp_tbl(10),tau_tbl(10)
end program
