! MPAS-A v8.4.2 mp_wsm6.F90:1471-1494,1751-1994,1997-2272.
! Test-only C ABI; prepare --expose-private changes only helper visibility.
function gamma_oracle(x) result(y) bind(C)
 use iso_c_binding
 use mp_wsm6, only: rgmma
 implicit none
 real(c_float),value::x
 real(c_float)::y
 y=rgmma(x)
end function

subroutine sedimentation_oracle(nz,iterations,category,dt,c,a,precip) bind(C)
 use iso_c_binding
 use mp_wsm6
 implicit none
 integer(c_int),value::nz,iterations,category
 real(c_float),value::dt
 real(c_float)::c(26),a(nz,7),precip(2)
 real(c_float)::dz(1,nz),rho(1,nz),fac(1,nz),t(1,nz),w(1,nz),q(1,nz),q2(1,nz)
 character(len=256)::msg
 integer::err
 call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),0,msg,err)
 dz(1,:)=a(:,1);rho(1,:)=a(:,2);fac(1,:)=a(:,3);t(1,:)=a(:,4)
 w(1,:)=a(:,5);q(1,:)=a(:,6);q2(1,:)=a(:,7)
 if(category==1)then
  call nislfv_rain_plm(1,nz,rho,fac,t,dz,w,q,precip(1:1),dt,1,iterations)
 else
  call nislfv_rain_plm6(1,nz,rho,fac,t,dz,w,q,q2,precip(1:1),precip(2:2),dt,1,iterations)
 endif
 a(:,5)=w(1,:);a(:,6)=q(1,:);a(:,7)=q2(1,:)
end subroutine

! MPAS-A v8.4.2 mp_radar.F90:536-589; private valid/error result coverage.
subroutine maxwell_oracle(volumes,inclusion,out,error) bind(C)
 use iso_c_binding
 use mp_radar, only: radar_init,m_complex_maxwellgarnett,m_w_0,m_i_0
 implicit none
 real(c_double)::volumes(3),out(2)
 integer(c_int),value::inclusion
 integer(c_int)::error
 character(len=11)::name
 complex(c_double_complex)::answer
 call radar_init
 select case(inclusion)
 case(0);name='spherical'
 case(1);name='spheroidal'
 case default;name='invalid'
 end select
 answer=m_complex_maxwellgarnett(volumes(1),volumes(2),volumes(3), &
          m_w_0,m_i_0,cmplx(1.0d0,0.0d0,kind=c_double),name,error)
 out(1)=real(answer,c_double);out(2)=aimag(answer)
end subroutine
