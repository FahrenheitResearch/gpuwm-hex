! MPAS-A v8.4.2 mp_wsm6.F90:73-189,213-1468,2275-2444;
! mp_wsm6_effectRad.F90:59-193. Test-only pristine-module C bridge.
subroutine wsm6_oracle(nz,dt,hail,c,v,s,r,dbz,prod) bind(C)
 use iso_c_binding
 use mp_wsm6
 use mp_wsm6_effectrad
 implicit none
 integer(c_int),value:: nz,hail
 real(c_float),value:: dt
 real(c_float):: c(26),v(nz,10),s(7),r(nz,3),dbz(nz),prod(nz,2)
 character(len=256):: msg
 integer:: err
 real(c_float):: rrc(1,nz),rri(1,nz),rrs(1,nz)
 call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),hail,msg,err)
 call run_column
 call mp_wsm6_effectrad_run(.true.,transpose(v(:,1:1)),transpose(v(:,3:3)), &
      transpose(v(:,4:4)),transpose(v(:,6:6)),transpose(v(:,8:8)),c(12),c(6), &
      c(21),c(22),c(23),c(24),c(25),c(26),rrc,rri,rrs,1,1,1,nz,msg,err)
 r(:,1)=rrc(1,:);r(:,2)=rri(1,:);r(:,3)=rrs(1,:)
 call refl10cm_wsm6(v(:,2),v(:,5),v(:,6),v(:,7),v(:,1),v(:,9),dbz,1,nz)
contains
 subroutine run_column
 real(c_float):: t(1,nz),q(1,nz),qc(1,nz),qi(1,nz),qr(1,nz),qs(1,nz),qg(1,nz)
 real(c_float):: den(1,nz),p(1,nz),dz(1,nz),rp(1,nz),ep(1,nz)
 t(1,:)=v(:,1);q(1,:)=v(:,2);qc(1,:)=v(:,3);qi(1,:)=v(:,4)
 qr(1,:)=v(:,5);qs(1,:)=v(:,6);qg(1,:)=v(:,7)
 den(1,:)=v(:,8);p(1,:)=v(:,9);dz(1,:)=v(:,10)
 call mp_wsm6_run(t,q,qc,qi,qr,qs,qg,den,p,dz,dt, &
      c(1),c(2),c(3),c(4),c(5),c(6),c(7),c(8),c(12),c(13), &
      c(14),c(15),c(16),c(17),c(18),c(19),c(20), &
      s(1:1),s(2:2),s(7:7),s(3:3),s(4:4),s(5:5),s(6:6), &
      rp,ep,1,1,1,nz,msg,err)
 v(:,1)=t(1,:);v(:,2)=q(1,:);v(:,3)=qc(1,:);v(:,4)=qi(1,:)
 v(:,5)=qr(1,:);v(:,6)=qs(1,:);v(:,7)=qg(1,:)
 prod(:,1)=rp(1,:);prod(:,2)=ep(1,:)
 end subroutine
end subroutine

subroutine radar_constants_oracle(out) bind(C)
 use iso_c_binding
 use mp_wsm6
 use mp_radar
 implicit none
 real(c_double)::out(32)
 character(len=256)::msg
 integer::err
 call mp_wsm6_init(1.28,1000.,100.,4190.,1846.4,0,msg,err)
 out(1)=pi5;out(2)=lamda4;out(3)=k_w;out(4)=real(m_w_0,8);out(5)=aimag(m_w_0)
 out(6)=real(m_i_0,8);out(7)=aimag(m_i_0)
 out(8:11)=xcre;out(12:15)=xcrg;out(16:19)=xcse;out(20:23)=xcsg
 out(24:27)=xcge;out(28:31)=xcgg;out(32)=xam_s
end subroutine

! MPAS-A v8.4.2 mp_radar.F90:192-292. Alternate configuration branch oracle.
subroutine radar_branch_oracle(a, opts, result) bind(C)
 use iso_c_binding
 use mp_radar
 implicit none
 real(c_double):: a(6),result
 integer(c_int)::opts(6)
 character(len=20),parameter::names(13)=[character(len=20):: &
 'maxwellgarnett','invalid','air','ice','water','spherical','spheroidal', &
 'none','icewater','airwater','airice','iceair','waterair']
 call radar_init
 call rayleigh_soak_wetgraupel(a(1),a(2),a(3),a(4),a(5),m_w_0,m_i_0,a(6),result, &
 names(opts(1)),names(opts(2)),names(opts(3)),names(opts(4)),names(opts(5)),names(opts(6)))
end subroutine

! MPAS-A v8.4.2 mp_wsm6_effectRad.F90:22-193, mp_wsm6.F90:191-211.
subroutine radii_boundary_oracle(nz, enabled, c, v, r, status) bind(C)
 use iso_c_binding
 use mp_wsm6
 use mp_wsm6_effectrad
 implicit none
 integer(c_int),value::nz,enabled
 real(c_float)::c(26),v(nz,10),r(nz,3)
 integer(c_int)::status(3)
 real(c_float)::rc(1,nz),ri(1,nz),rs(1,nz)
 character(len=1)::msg
 integer::err
 call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),0,msg,err)
 call mp_wsm6_effectrad_init(msg,err)
 status(1)=err
 rc(1,:)=r(:,1);ri(1,:)=r(:,2);rs(1,:)=r(:,3)
 call mp_wsm6_effectrad_run(enabled/=0,transpose(v(:,1:1)),transpose(v(:,3:3)), &
 transpose(v(:,4:4)),transpose(v(:,6:6)),transpose(v(:,8:8)),c(12),c(6), &
 c(21),c(22),c(23),c(24),c(25),c(26),rc,ri,rs,1,1,1,nz,msg,err)
 r(:,1)=rc(1,:);r(:,2)=ri(1,:);r(:,3)=rs(1,:)
 call mp_wsm6_effectrad_finalize(msg,err)
 status(2)=err
 call mp_wsm6_finalize(msg,err)
 status(3)=err
end subroutine

! MPAS-A v8.4.2 mp_wsm6.F90:213-1468; optional diagnostic argument coverage.
subroutine optional_surface_oracle(nz,dt,hail,mask,c,v,s,prod) bind(C)
 use iso_c_binding
 use mp_wsm6
 implicit none
 integer(c_int),value::nz,hail,mask
 real(c_float),value::dt
 real(c_float)::c(26),v(nz,10),prod(nz,2)
 real(c_float),target::s(7)
 real(c_float),pointer::snow(:),snowncv(:),graup(:),graupncv(:)
 real(c_float)::t(1,nz),q(1,nz),qc(1,nz),qi(1,nz),qr(1,nz),qs(1,nz),qg(1,nz)
 real(c_float)::den(1,nz),p(1,nz),dz(1,nz),rp(1,nz),ep(1,nz)
 character(len=256)::msg
 integer::err
 nullify(snow,snowncv,graup,graupncv)
 if(btest(mask,0))snow=>s(3:3)
 if(btest(mask,1))snowncv=>s(4:4)
 if(btest(mask,2))graup=>s(5:5)
 if(btest(mask,3))graupncv=>s(6:6)
 t(1,:)=v(:,1);q(1,:)=v(:,2);qc(1,:)=v(:,3);qi(1,:)=v(:,4)
 qr(1,:)=v(:,5);qs(1,:)=v(:,6);qg(1,:)=v(:,7)
 den(1,:)=v(:,8);p(1,:)=v(:,9);dz(1,:)=v(:,10)
 call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),hail,msg,err)
 call mp_wsm6_run(t,q,qc,qi,qr,qs,qg,den,p,dz,dt, &
 c(1),c(2),c(3),c(4),c(5),c(6),c(7),c(8),c(12),c(13),c(14),c(15), &
 c(16),c(17),c(18),c(19),c(20),s(1:1),s(2:2),s(7:7), &
 snow,snowncv,graup,graupncv,rp,ep,1,1,1,nz,msg,err)
 v(:,1)=t(1,:);v(:,2)=q(1,:);v(:,3)=qc(1,:);v(:,4)=qi(1,:)
 v(:,5)=qr(1,:);v(:,6)=qs(1,:);v(:,7)=qg(1,:)
 prod(:,1)=rp(1,:);prod(:,2)=ep(1,:)
end subroutine
