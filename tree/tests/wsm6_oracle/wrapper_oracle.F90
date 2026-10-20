! MPAS-A v8.4.2 module_mp_wsm6.F:18-239; mpas_atmphys_driver_microphysics.F:690-740.
subroutine wsm6_wrapper_oracle(nz,dt,hail,c,v,s,r,dbz,prod,pi,radii_on) bind(C)
 use iso_c_binding
 use mp_wsm6
 use module_mp_wsm6
 implicit none
 integer(c_int),value::nz,hail,radii_on
 real(c_float),value::dt
 real(c_float)::c(26),v(nz,10),s(7),r(nz,3),dbz(nz),prod(nz,2),pi(nz)
 real(c_float)::th(1,nz,1),q(1,nz,1),qc(1,nz,1),qi(1,nz,1),qr(1,nz,1),qs(1,nz,1),qg(1,nz,1)
 real(c_float)::den(1,nz,1),p(1,nz,1),dz(1,nz,1),pii(1,nz,1),rc(1,nz,1),ri(1,nz,1),rs(1,nz,1)
 real(c_float)::ss(1,1,7)
 character(len=256)::msg
 integer::err,k
 call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),hail,msg,err)
 th(1,:,1)=v(:,1);q(1,:,1)=v(:,2);qc(1,:,1)=v(:,3);qi(1,:,1)=v(:,4)
 qr(1,:,1)=v(:,5);qs(1,:,1)=v(:,6);qg(1,:,1)=v(:,7)
 den(1,:,1)=v(:,8);p(1,:,1)=v(:,9);dz(1,:,1)=v(:,10);pii(1,:,1)=pi
 rc(1,:,1)=r(:,1);ri(1,:,1)=r(:,2);rs(1,:,1)=r(:,3)
 ss(1,1,:)=s
 call wsm6(th,q,qc,qr,qi,qs,qg,den,pii,p,dz,dt,c(1),c(2),c(3),c(4),c(5),c(6),c(7),c(8),c(12), &
 c(13),c(14),c(15),c(16),c(17),c(18),c(19),c(20), &
 rain=ss(:,:,1),rainncv=ss(:,:,2), &
 snow=ss(:,:,3),snowncv=ss(:,:,4), &
 graupel=ss(:,:,5),graupelncv=ss(:,:,6),sr=ss(:,:,7), &
 has_reqc=radii_on,has_reqi=radii_on,has_reqs=radii_on, &
 re_qc_bg=c(21),re_qi_bg=c(22),re_qs_bg=c(23),re_qc_max=c(24),re_qi_max=c(25),re_qs_max=c(26), &
 re_cloud=rc,re_ice=ri,re_snow=rs,ids=1,ide=1,jds=1,jde=1,kds=1,kde=nz, &
 ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz,its=1,ite=1,jts=1,jte=1,kts=1,kte=nz,errmsg=msg,errflg=err)
 s=ss(1,1,:)
 v(:,1)=th(1,:,1);v(:,2)=q(1,:,1);v(:,3)=qc(1,:,1);v(:,4)=qi(1,:,1)
 v(:,5)=qr(1,:,1);v(:,6)=qs(1,:,1);v(:,7)=qg(1,:,1)
 r(:,1)=rc(1,:,1);r(:,2)=ri(1,:,1);r(:,3)=rs(1,:,1)
 do k=1,nz
  th(1,k,1)=th(1,k,1)*pi(k)
 enddo
 call refl10cm_wsm6(v(:,2),v(:,5),v(:,6),v(:,7),th(1,:,1),v(:,9),dbz,1,nz)
end subroutine

! MPAS-A v8.4.2 module_mp_wsm6.F:18-239; mpas_atmphys_driver_microphysics.F:690-740.
subroutine wsm6_wrapper_flags_oracle(nz,dt,hail,c,v,s,r,dbz,prod,pi,cloud,ice,snow) bind(C)
 use iso_c_binding
 use mp_wsm6
 use module_mp_wsm6
 implicit none
 integer(c_int),value::nz,hail,cloud,ice,snow
 real(c_float),value::dt
 real(c_float)::c(26),v(nz,10),s(7),r(nz,3),dbz(nz),prod(nz,2),pi(nz)
 real(c_float)::th(1,nz,1),q(1,nz,1),qc(1,nz,1),qi(1,nz,1),qr(1,nz,1),qs(1,nz,1),qg(1,nz,1)
 real(c_float)::den(1,nz,1),p(1,nz,1),dz(1,nz,1),pii(1,nz,1),rc(1,nz,1),ri(1,nz,1),rs(1,nz,1)
 real(c_float)::ss(1,1,7)
 character(len=256)::msg
 integer::err,k
 call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),hail,msg,err)
 th(1,:,1)=v(:,1);q(1,:,1)=v(:,2);qc(1,:,1)=v(:,3);qi(1,:,1)=v(:,4)
 qr(1,:,1)=v(:,5);qs(1,:,1)=v(:,6);qg(1,:,1)=v(:,7)
 den(1,:,1)=v(:,8);p(1,:,1)=v(:,9);dz(1,:,1)=v(:,10);pii(1,:,1)=pi
 rc(1,:,1)=r(:,1);ri(1,:,1)=r(:,2);rs(1,:,1)=r(:,3)
 ss(1,1,:)=s
 call wsm6(th,q,qc,qr,qi,qs,qg,den,pii,p,dz,dt,c(1),c(2),c(3),c(4),c(5),c(6),c(7),c(8),c(12), &
 c(13),c(14),c(15),c(16),c(17),c(18),c(19),c(20), &
 rain=ss(:,:,1),rainncv=ss(:,:,2), &
 snow=ss(:,:,3),snowncv=ss(:,:,4), &
 graupel=ss(:,:,5),graupelncv=ss(:,:,6),sr=ss(:,:,7), &
 has_reqc=cloud,has_reqi=ice,has_reqs=snow, &
 re_qc_bg=c(21),re_qi_bg=c(22),re_qs_bg=c(23),re_qc_max=c(24),re_qi_max=c(25),re_qs_max=c(26), &
 re_cloud=rc,re_ice=ri,re_snow=rs,ids=1,ide=1,jds=1,jde=1,kds=1,kde=nz, &
 ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz,its=1,ite=1,jts=1,jte=1,kts=1,kte=nz,errmsg=msg,errflg=err)
 s=ss(1,1,:)
 v(:,1)=th(1,:,1);v(:,2)=q(1,:,1);v(:,3)=qc(1,:,1);v(:,4)=qi(1,:,1)
 v(:,5)=qr(1,:,1);v(:,6)=qs(1,:,1);v(:,7)=qg(1,:,1)
 r(:,1)=rc(1,:,1);r(:,2)=ri(1,:,1);r(:,3)=rs(1,:,1)
 do k=1,nz
  th(1,k,1)=th(1,k,1)*pi(k)
 enddo
 call refl10cm_wsm6(v(:,2),v(:,5),v(:,6),v(:,7),th(1,:,1),v(:,9),dbz,1,nz)
end subroutine
