! Test-only argument adapter; source procedure and expressions remain pristine.
! MPAS module_cu_gf.mpas.F:3580-3786.
subroutine gf_native_output(nz,a,s,e,ints) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value :: nz
 real(c_float),intent(inout) :: a(4,nz,13),s(4,18),e(4,1,16,2)
 integer(c_int),intent(inout) :: ints(4,4)
 call cup_output_ens_3d(e(:,:,:,1),ints(:,1),a(:,:,1:1),a(:,:,2:2),a(:,:,3:3), &
  a(:,:,4:4),a(:,:,5:5),a(:,:,6),a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10), &
  a(:,:,11),a(:,:,12),s(:,1),a(:,:,13:13),s(:,2),ints(:,2), &
  1,'deep',1,1,1,ints(:,3),ints(:,4),e(:,:,:,2),16,16, &
  s(:,3),s(:,4),s(:,5),s(:,6),s(:,7),s(:,8),s(:,9),s(:,10),s(:,11), &
  s(:,12),s(:,13),s(:,14),s(:,15),s(:,16),s(:,17),s(:,18),0, &
  1,1,4,1,nz,1,4,1,1,1,nz)
end subroutine
