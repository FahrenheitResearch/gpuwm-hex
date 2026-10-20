! Test-only adapters; pristine module owns all numerical expressions.
! MPAS module_cu_gf.mpas.F:2400-2499,2502-2984,3176-3228.
subroutine gf_native_extra(nz,mode,ens4,maxens,choice,deeps,a,s,e,ints) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value :: nz,mode,ens4,maxens,choice,deeps
 real(c_float) :: a(4,nz,15),s(4,14),e(4,1,48,2)
 integer(c_int) :: ints(4,6)
 real :: mbdt(3)
 character(5) :: name
 mbdt=120.
 name='deeps'
 if(deeps==0)name='deep '
 if(mode==0)then
  call cup_env_clev(a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5),a(:,:,6),a(:,:,7), &
   a(:,:,8),a(:,:,9),a(:,:,10),a(:,:,11),a(:,:,12),a(:,:,13),a(:,:,14),a(:,:,15), &
   s(:,1),ints(:,1),s(:,2),2500000.,461.6,1004.5,4,1,nz,1,4,1,1,1,nz)
 else if(mode==1)then
  call cup_maximi(a(:,:,1),2,ints(:,2),ints(:,3),ints(:,1),4,1,nz,1,4,1,1,1,nz)
 else
  call cup_forcing_ens_3d(s(:,1),s(:,2),s(:,3),s(:,4),s(:,6:5+maxens), &
   mbdt(1:maxens),120.,ints(:,1),ints(:,2),ints(:,3),e(:,:,:,1),1,name, &
   s(:,9:8+ens4),maxens,1,1,1,16,s(:,12:11+ens4),a(:,:,1),ints(:,4), &
   a(:,:,4:3+ens4),a(:,:,2),ints(:,5),a(:,:,3),e(:,:,:,2),s(:,5),ints(:,6), &
   48,choice,1,1,4,1,nz,1,4,1,1,1,nz,ens4,2)
 endif
end subroutine
