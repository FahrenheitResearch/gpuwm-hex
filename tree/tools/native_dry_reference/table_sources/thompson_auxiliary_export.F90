! Write the Thompson auxiliary arrays native initialisation computes, for the
! MPAS hex-cuda Thompson slot (hexcore.mpasphys.resources_build).
! MPAS v8.4.2 module_mp_thompson.F:433-993 (thompson_init); the arrays are
! the six the slot's resource loader reads (table_resources_v842.py:82-84).
! No numerical code here: thompson_init is the pristine MPAS routine.
program thompson_auxiliary_export
use module_mp_thompson
implicit none
call thompson_init(.true.)
open(72,file='auxiliary.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
write(72) t_Efrw,t_Efsw,tnr_rev,tpc_wev,tnc_wev,tnccn_act
close(72)
end program
