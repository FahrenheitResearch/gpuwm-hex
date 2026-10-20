! Write the aerosol-aware Thompson auxiliary arrays native initialisation computes,
! for the MPAS hex-cuda mp_thompson_aerosols slot (hexcore.mpasphys.resources_build).
! MPAS v8.4.2 module_mp_thompson.F:433-993 (thompson_init with is_aerosol_aware) and
! mpas_atmphys_init_microphysics.F:251-310 (table_ccnAct reads CCN_ACTIVATE_DATA into
! tnccn_act). The order is the slot loader's (thompson_aero_v842.load_resources).
! No numerical code here: thompson_init is the pristine MPAS routine.
program thompson_aero_auxiliary_export
use module_mp_thompson
implicit none
is_aerosol_aware = .true.
call thompson_init(.true.)
open(72,file='CCN_ACTIVATE_DATA',form='unformatted',status='old')
read(72) tnccn_act
close(72)
open(73,file='auxiliary.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
write(73) t_Efrw,t_Efsw,tpc_wev,tnc_wev,tnr_rev,tnccn_act
close(73)
end program
