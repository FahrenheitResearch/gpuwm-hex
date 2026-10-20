! Pristine MPAS mpas_atmphys_driver_radiation_sw.F:905-916 driver slice.
program oracle_driver_controls
 implicit none
 character(len=32) :: radt_cld_overlap, radt_cld_dcorrlen
 character(len=32), parameter :: overlaps(6) = [character(len=32) :: &
  'none', 'random', 'maximum_random', 'maximum', 'exponential', 'exponential_random']
 character(len=32), parameter :: decorators(4) = [character(len=32) :: &
  'constant', 'latitude_varying', '', 'unexpected']
 integer :: c,d,cldovrlp,idcor,u
 open(newunit=u,file='driver_control_options.bin',access='stream',form='unformatted',status='replace')
 do c=1,6
 do d=1,4
 radt_cld_overlap=overlaps(c)
 radt_cld_dcorrlen=decorators(d)
       if(trim(radt_cld_overlap) == "none"              ) cldovrlp = 0
       if(trim(radt_cld_overlap) == "random"            ) cldovrlp = 1
       if(trim(radt_cld_overlap) == "maximum_random"    ) cldovrlp = 2
       if(trim(radt_cld_overlap) == "maximum"           ) cldovrlp = 3
       if(trim(radt_cld_overlap) == "exponential"       ) cldovrlp = 4
       if(trim(radt_cld_overlap) == "exponential_random") cldovrlp = 5

       idcor = 0
       if(trim(radt_cld_overlap)=="exponential" .or. trim(radt_cld_overlap)=="exponential_random") then
          if(trim(radt_cld_dcorrlen) == "constant"        ) idcor = 0
          if(trim(radt_cld_dcorrlen) == "latitude_varying") idcor = 1
       endif
 write(u)c,d,cldovrlp,idcor
 enddo
 enddo
 close(u)
end program oracle_driver_controls
