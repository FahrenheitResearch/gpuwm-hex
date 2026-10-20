! Strict MPAS driver arithmetic: mpas_atmphys_driver_radiation_lw.F:490-492.
! Include normal inputs producing subnormal outputs and subnormal inputs.
program subnormal_pressure
  use lw_binio
  implicit none
  integer :: i
  integer :: words(12)
  real :: pres_hyd(12), pres(12)
  words = [int(z'00000100'), int(z'80000100'), int(z'007fffff'), &
           int(z'807fffff'), int(z'00800000'), int(z'80800000'), &
           int(z'00000001'), int(z'80000001'), int(z'00000000'), &
           int(z'80000000'), int(z'3f800000'), int(z'bf800000')]
  pres_hyd = transfer(words, pres_hyd)
  do i = 1, size(pres)
    pres(i) = pres_hyd(i) / 100.0
  end do
  call bio_open('native-subnormal-pressure.bin')
  call wr_r1('pres_hyd', pres_hyd)
  call wr_r1('pres', pres)
  call bio_close()
end program subnormal_pressure
