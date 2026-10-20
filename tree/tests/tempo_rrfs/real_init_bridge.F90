! Test ABI only: published pristine fork modules supply all initialization.
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182 @ 1d6e370e.
subroutine tempo_real_init(flags, outputs, version) bind(c)
 use iso_c_binding
 use module_mp_tempo_driver, only: tempo_init
 use module_mp_tempo_cfgs, only: ty_tempo_cfgs
 use module_mp_tempo_params, only: tempo_version
 implicit none
 integer(c_int), intent(in) :: flags(4)
 integer(c_int), intent(out) :: outputs(17)
 character(c_char), intent(out) :: version(11)
 type(ty_tempo_cfgs) :: cfg
 integer :: i
 call tempo_init(aerosolaware_flag=flags(1)/=0, hailaware_flag=flags(2)/=0, &
   ml_for_bl_nc_flag=flags(3)/=0, refl10cm_from_melting_flag=flags(4)/=0, tempo_cfgs=cfg)
 outputs = [merge(1,0,cfg%aerosolaware_flag), merge(1,0,cfg%hailaware_flag), &
   merge(1,0,cfg%ml_for_bl_nc_flag), merge(1,0,cfg%ml_for_nc_flag), &
   merge(1,0,cfg%semi_sedi_flag), merge(1,0,cfg%refl10cm_from_melting_flag), &
   merge(1,0,cfg%turn_off_micro_flag), merge(1,0,cfg%cloud_condensation_flag), &
   merge(1,0,cfg%verbose), merge(1,0,cfg%refl10cm_flag), merge(1,0,cfg%re_cloud_flag), &
   merge(1,0,cfg%re_ice_flag), merge(1,0,cfg%re_snow_flag), merge(1,0,cfg%max_hail_diameter_flag), &
   merge(1,0,cfg%rain_med_vol_diam_flag), merge(1,0,cfg%graupel_med_vol_diam_flag), &
   merge(1,0,cfg%cloud_number_mixing_ratio_flag)]
 do i=1,11
   version(i)=tempo_version(i:i)
 end do
end subroutine
