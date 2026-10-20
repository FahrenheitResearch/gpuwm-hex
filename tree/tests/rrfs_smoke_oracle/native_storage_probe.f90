! Read-only adapter to original published modules; no science/accessor calls.
! Public constants are the original compiler module parameters, not parsed literals.
module r14_native_storage_probe
use iso_c_binding, only: c_int
use mpas_smoke_config, only: v1 => epsilc
use mpas_smoke_config, only: v2 => pi
use mpas_smoke_config, only: v3 => seas_opt
use mpas_smoke_config, only: v4 => addsmoke_flag
use mpas_smoke_config, only: v5 => n_dbg_lines
use mpas_smoke_config, only: v6 => nfire_types
use mpas_smoke_config, only: v7 => chem_opt_gocart
use mpas_smoke_config, only: v8 => num_moist
use mpas_smoke_config, only: v9 => num_emis_seas
use mpas_smoke_config, only: v10 => num_emis_dust
use mpas_smoke_config, only: v11 => p_dust_1
use mpas_smoke_config, only: v12 => p_dust_2
use mpas_smoke_config, only: v13 => p_dust_3
use mpas_smoke_config, only: v14 => p_dust_4
use mpas_smoke_config, only: v15 => p_dust_5
use mpas_smoke_config, only: v16 => p_edust1
use mpas_smoke_config, only: v17 => p_edust2
use mpas_smoke_config, only: v18 => p_edust3
use mpas_smoke_config, only: v19 => p_edust4
use mpas_smoke_config, only: v20 => p_edust5
use mpas_smoke_config, only: v21 => ndvel
use mpas_smoke_config, only: v22 => numgas
use rad_data_mod, only: v23 => sc_eff
use rad_data_mod, only: v24 => ab_eff
use seas_data_mod, only: v25 => number_ss_bins
use seas_data_mod, only: v26 => ra
use seas_data_mod, only: v27 => rb
use seas_data_mod, only: v28 => emission_scale
use seas_data_mod, only: v29 => den_seas
use seas_data_mod, only: v30 => reff_seas
use dust_data_mod, only: v31 => ndust
use dust_data_mod, only: v32 => ndcls
use dust_data_mod, only: v33 => ndsrc
use dust_data_mod, only: v34 => maxstypes
use dust_data_mod, only: v35 => nsalt
use dust_data_mod, only: v36 => dyn_visc
use dust_data_mod, only: v37 => den_dust
use dust_data_mod, only: v38 => reff_dust
use dust_data_mod, only: v39 => frac_s
use dust_data_mod, only: v40 => lo_dust
use dust_data_mod, only: v41 => up_dust
use dust_data_mod, only: v42 => fengsha_maxstypes
use dust_data_mod, only: v43 => dust_uthres
use dust_data_mod, only: v44 => dust_calcdrag
use dust_data_mod, only: v45 => dust_moist_opt
use dust_data_mod, only: v46 => spoint
use dust_data_mod, only: v47 => reff_salt
use dust_data_mod, only: v48 => den_salt
use dust_data_mod, only: v49 => frac_salt
use dust_data_mod, only: v50 => max_soiltyp
use dust_data_mod, only: v51 => maxsmc
use dep_data_mod, only: v52 => max_dep_vel
use dep_data_mod, only: v53 => dep_ref_hgt
use dep_data_mod, only: v54 => pi
use dep_data_mod, only: v55 => threepi
use dep_data_mod, only: v56 => gravity
use dep_data_mod, only: v57 => grav
use dep_data_mod, only: v58 => boltzmann
use dep_data_mod, only: v59 => rgasuniv
use dep_data_mod, only: v60 => avo
use dep_data_mod, only: v61 => cb
use dep_data_mod, only: v62 => cim
use dep_data_mod, only: v63 => alpha
use dep_data_mod, only: v64 => cin
use dep_data_mod, only: v65 => vv
use dep_data_mod, only: v66 => a_for
use dep_data_mod, only: v67 => a_grs
use dep_data_mod, only: v68 => a_wat
use dep_data_mod, only: v69 => eps0_for
use dep_data_mod, only: v70 => eps0_grs
use dep_data_mod, only: v71 => eps0_wat
use dep_data_mod, only: v72 => one3
use dep_data_mod, only: v73 => two3
use dep_data_mod, only: v74 => sqrt2
use dep_data_mod, only: v75 => sqrtpi
use dep_data_mod, only: v76 => karman
use dep_data_mod, only: v77 => conmin
use dep_data_mod, only: v78 => pirs
use dep_data_mod, only: v79 => f6dpi
use dep_data_mod, only: v80 => f6dpim9
use dep_data_mod, only: v81 => sigma1
use dep_data_mod, only: v82 => mean_diameter1
use dep_data_mod, only: v83 => fact_wfa
use dep_data_mod, only: v84 => dgmin
use dep_data_mod, only: v85 => densmin
use dep_data_mod, only: v86 => aero_dens
use dep_data_mod, only: v87 => aero_diam
use mpas_smoke_init, only: v88 => p_smoke_fine
use mpas_smoke_init, only: v89 => p_smoke_coarse
use mpas_smoke_init, only: v90 => p_dust_fine
use mpas_smoke_init, only: v91 => p_dust_coarse
use mpas_smoke_init, only: v92 => p_polp_all
use mpas_smoke_init, only: v93 => p_polp_tree
use mpas_smoke_init, only: v94 => p_polp_grass
use mpas_smoke_init, only: v95 => p_polp_weed
use mpas_smoke_init, only: v96 => p_pols_tree
use mpas_smoke_init, only: v97 => p_pols_grass
use mpas_smoke_init, only: v98 => p_pols_weed
use mpas_smoke_init, only: v99 => p_pols_all
use mpas_smoke_init, only: v100 => p_unspc_fine
use mpas_smoke_init, only: v101 => p_unspc_coarse
use mpas_smoke_init, only: v102 => p_ssalt_fine
use mpas_smoke_init, only: v103 => p_ssalt_coarse
use mpas_smoke_init, only: v104 => p_no3_a_fine
use mpas_smoke_init, only: v105 => p_so4_a_fine
use mpas_smoke_init, only: v106 => p_nh4_a_fine
use mpas_smoke_init, only: v107 => p_so2
use mpas_smoke_init, only: v108 => p_nh3
use mpas_smoke_init, only: v109 => p_ch4
use mpas_smoke_init, only: v110 => p_mold_fine
use mpas_smoke_init, only: v111 => p_bact_fine
use mpas_smoke_init, only: v112 => p_plastic_fine
use mpas_smoke_init, only: v113 => p_dms
use module_zero_plumegen_coms, only: v114 => nkp
use module_zero_plumegen_coms, only: v115 => ntime
use mpas_smoke_config, only: v116 => dbg_opt
use mpas_smoke_config, only: v117 => aero_ind_fdb
use mpas_log, only: observed_log => mpas_log_info
implicit none
contains
subroutine r14_observe_storage(slot,capacity,words,count,lower,upper,status) bind(C)
integer(c_int),value :: slot,capacity
integer(c_int),intent(out) :: words(*),count,lower(5),upper(5),status
status = 0
lower = 0
upper = 0
select case(slot)
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:14 @1d6e370e.
case(1)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v1, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:15 @1d6e370e.
case(2)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v2, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:17 @1d6e370e.
case(3)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v3, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:18 @1d6e370e.
case(4)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v4, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:19 @1d6e370e.
case(5)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v5, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:20 @1d6e370e.
case(6)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v6, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:24 @1d6e370e.
case(7)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v7, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:25 @1d6e370e.
case(8)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v8, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:26 @1d6e370e.
case(9)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v9, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:26 @1d6e370e.
case(10)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v10, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:27 @1d6e370e.
case(11)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v11, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:27 @1d6e370e.
case(12)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v12, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:27 @1d6e370e.
case(13)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v13, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:27 @1d6e370e.
case(14)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v14, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:27 @1d6e370e.
case(15)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v15, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:28 @1d6e370e.
case(16)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v16, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:28 @1d6e370e.
case(17)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v17, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:28 @1d6e370e.
case(18)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v18, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:28 @1d6e370e.
case(19)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v19, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:28 @1d6e370e.
case(20)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v20, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:29 @1d6e370e.
case(21)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v21, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:30 @1d6e370e.
case(22)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v22, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/rad_data_mod.F90:13 @1d6e370e.
case(23)
count = size(v23)
lower(1) = lbound(v23,1)
upper(1) = ubound(v23,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v23, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/rad_data_mod.F90:13 @1d6e370e.
case(24)
count = size(v24)
lower(1) = lbound(v24,1)
upper(1) = ubound(v24,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v24, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90:9 @1d6e370e.
case(25)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v25, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90:11 @1d6e370e.
case(26)
count = size(v26)
lower(1) = lbound(v26,1)
upper(1) = ubound(v26,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v26, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90:12 @1d6e370e.
case(27)
count = size(v27)
lower(1) = lbound(v27,1)
upper(1) = ubound(v27,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v27, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90:15 @1d6e370e.
case(28)
count = size(v28)
lower(1) = lbound(v28,1)
upper(1) = ubound(v28,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v28, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90:17 @1d6e370e.
case(29)
count = size(v29)
lower(1) = lbound(v29,1)
upper(1) = ubound(v29,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v29, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90:19 @1d6e370e.
case(30)
count = size(v30)
lower(1) = lbound(v30,1)
upper(1) = ubound(v30,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v30, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:10 @1d6e370e.
case(31)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v31, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:11 @1d6e370e.
case(32)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v32, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:12 @1d6e370e.
case(33)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v33, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:13 @1d6e370e.
case(34)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v34, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:14 @1d6e370e.
case(35)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v35, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:16 @1d6e370e.
case(36)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v36, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:20 @1d6e370e.
case(37)
count = size(v37)
lower(1) = lbound(v37,1)
upper(1) = ubound(v37,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v37, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:21 @1d6e370e.
case(38)
count = size(v38)
lower(1) = lbound(v38,1)
upper(1) = ubound(v38,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v38, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:22 @1d6e370e.
case(39)
count = size(v39)
lower(1) = lbound(v39,1)
upper(1) = ubound(v39,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v39, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:23 @1d6e370e.
case(40)
count = size(v40)
lower(1) = lbound(v40,1)
upper(1) = ubound(v40,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v40, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:24 @1d6e370e.
case(41)
count = size(v41)
lower(1) = lbound(v41,1)
upper(1) = ubound(v41,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v41, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:46 @1d6e370e.
case(42)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v42, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:48 @1d6e370e.
case(43)
count = size(v43)
lower(1) = lbound(v43,1)
upper(1) = ubound(v43,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v43, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:64 @1d6e370e.
case(44)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v44, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:66 @1d6e370e.
case(45)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v45, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:74 @1d6e370e.
case(46)
count = size(v46)
lower(1) = lbound(v46,1)
upper(1) = ubound(v46,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v46, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:75 @1d6e370e.
case(47)
count = size(v47)
lower(1) = lbound(v47,1)
upper(1) = ubound(v47,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v47, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:77 @1d6e370e.
case(48)
count = size(v48)
lower(1) = lbound(v48,1)
upper(1) = ubound(v48,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v48, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:79 @1d6e370e.
case(49)
count = size(v49)
lower(1) = lbound(v49,1)
upper(1) = ubound(v49,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v49, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:84 @1d6e370e.
case(50)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v50, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90:85 @1d6e370e.
case(51)
count = size(v51)
lower(1) = lbound(v51,1)
upper(1) = ubound(v51,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v51, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:12 @1d6e370e.
case(52)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v52, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:13 @1d6e370e.
case(53)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v53, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:14 @1d6e370e.
case(54)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v54, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:16 @1d6e370e.
case(55)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v55, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:17 @1d6e370e.
case(56)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v56, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:19 @1d6e370e.
case(57)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v57, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:20 @1d6e370e.
case(58)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v58, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:22 @1d6e370e.
case(59)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v59, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:24 @1d6e370e.
case(60)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v60, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:27 @1d6e370e.
case(61)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v61, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:27 @1d6e370e.
case(62)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v62, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:27 @1d6e370e.
case(63)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v63, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:27 @1d6e370e.
case(64)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v64, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:27 @1d6e370e.
case(65)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v65, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:28 @1d6e370e.
case(66)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v66, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:29 @1d6e370e.
case(67)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v67, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:30 @1d6e370e.
case(68)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v68, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:31 @1d6e370e.
case(69)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v69, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:32 @1d6e370e.
case(70)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v70, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:33 @1d6e370e.
case(71)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v71, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:34 @1d6e370e.
case(72)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v72, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:35 @1d6e370e.
case(73)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v73, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:37 @1d6e370e.
case(74)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v74, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:39 @1d6e370e.
case(75)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v75, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:40 @1d6e370e.
case(76)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v76, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:41 @1d6e370e.
case(77)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v77, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:42 @1d6e370e.
case(78)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v78, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:43 @1d6e370e.
case(79)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v79, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:44 @1d6e370e.
case(80)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v80, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:46 @1d6e370e.
case(81)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v81, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:47 @1d6e370e.
case(82)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v82, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:48 @1d6e370e.
case(83)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v83, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:50 @1d6e370e.
case(84)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v84, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:52 @1d6e370e.
case(85)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v85, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:55 @1d6e370e.
case(86)
count = size(v86)
lower(1) = lbound(v86,1)
upper(1) = ubound(v86,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v86, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:55 @1d6e370e.
case(87)
count = size(v87)
lower(1) = lbound(v87,1)
upper(1) = ubound(v87,1)
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v87, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(88)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v88, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(89)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v89, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(90)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v90, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(91)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v91, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(92)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v92, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(93)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v93, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(94)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v94, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(95)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v95, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(96)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v96, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(97)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v97, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(98)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v98, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(99)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v99, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(100)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v100, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(101)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v101, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(102)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v102, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(103)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v103, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(104)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v104, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(105)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v105, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(106)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v106, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(107)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v107, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(108)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v108, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(109)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v109, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(110)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v110, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(111)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v111, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(112)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v112, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13 @1d6e370e.
case(113)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v113, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:9 @1d6e370e.
case(114)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v114, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:9 @1d6e370e.
case(115)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v115, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:21-22 @1d6e370e.
case(116)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v116, words(1:count))
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:21-22 @1d6e370e.
case(117)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(v117, words(1:count))
! MPAS-Model-rrfs/src/framework/mpas_log.F:49-65,565-595 @1d6e370e.
case(118)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(observed_log%outputlog%isactive, words(1:count))
! MPAS-Model-rrfs/src/framework/mpas_log.F:49-65,565-595 @1d6e370e.
case(119)
count = 1
if (capacity < count) then
status = 2
return
endif
words(1:count) = transfer(observed_log%outputmessagecount, words(1:count))
case default
count = 0
status = 1
end select
end subroutine
end module
