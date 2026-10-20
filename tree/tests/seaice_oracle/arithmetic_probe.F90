! Scalar IEEE-word diagnostic for the operators used at:
! module_sf_noah_seaice.F:222-1292; module_sf_noah_seaice_drv.F:189-511.
! This is a compiler probe; whole MPAS source remains the numeric oracle.
subroutine seaice_arithmetic_probe(n, words, output) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),value :: n
  integer(c_int32_t),intent(in) :: words(n,2)
  integer(c_int32_t),intent(out) :: output(n,7)
  real(c_float) :: a,b,r
  integer :: i
  do i=1,n
    a=transfer(words(i,1),a); b=transfer(words(i,2),b)
    r=a+b; output(i,1)=transfer(r,output(i,1))
    r=a-b; output(i,2)=transfer(r,output(i,2))
    r=a*b; output(i,3)=transfer(r,output(i,3))
    r=a/b; output(i,4)=transfer(r,output(i,4))
    r=-a; output(i,5)=transfer(r,output(i,5))
    r=exp(a); output(i,6)=transfer(r,output(i,6))
    r=a**b; output(i,7)=transfer(r,output(i,7))
  enddo
end subroutine
