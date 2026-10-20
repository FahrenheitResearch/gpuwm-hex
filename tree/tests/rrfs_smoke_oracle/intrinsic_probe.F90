! Diagnostic for binary32 primitives used by the complete pinned SMOKE graph.
! These are synthetic arithmetic inputs, never native forecast evidence.
subroutine smoke_intrinsic_probe(a,b,out) bind(c)
use iso_c_binding
implicit none
real(c_float), value :: a,b
real(c_float), intent(out) :: out(17)
out(1)=a+b
out(2)=a-b
out(3)=a*b
out(4)=a/b
out(5)=sqrt(a)
out(6)=exp(a)
out(7)=log(a)
out(8)=a**b
out(9)=sin(a)
out(10)=cos(a)
out(11)=erf(a)
out(12)=-a
out(13)=abs(a)
out(14)=sign(a,b)
out(15)=aint(a)
out(16)=mod(a,b)
out(17)=log10(a)
end subroutine
