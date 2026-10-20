! Test-only C bridge to pristine MPAS source; no reimplementation of Kessler.
subroutine oracle(t,qv,qc,qr,rho,pii,z,dz,rain,rainv,dt,constants,nz,nc) bind(C)
 use iso_c_binding
 use module_mp_kessler
 implicit none
 integer(c_int),value:: nz,nc
 real(c_float):: t(nc,nz,1),qv(nc,nz,1),qc(nc,nz,1),qr(nc,nz,1)
 real(c_float):: rho(nc,nz,1),pii(nc,nz,1),z(nc,nz,1),dz(nc,nz,1)
 real(c_float):: rain(nc,1),rainv(nc,1),constants(8)
 real(c_float),value:: dt
 call kessler(t,qv,qc,qr,rho,pii,dt,z,constants(1),constants(2), &
              constants(3),constants(4),constants(5),constants(6),constants(7),constants(8), &
              dz,rain,rainv, &
              1,nc,1,1,1,nz,1,nc,1,1,1,nz,1,nc,1,1,1,nz)
end subroutine
