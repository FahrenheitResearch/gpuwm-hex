! MPAS module_cu_gf.mpas.F:54-453, mpas_atmphys_driver_convection.F:482-516.
! Test-only C binding; authority procedure and arithmetic are unmodified.
subroutine gf_native(nz,lev,surface,isurface,outlev,outsurface,outint,control,constants) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int), value :: nz
 real(c_float), intent(in) :: lev(nz,15),surface(8),constants(4)
 integer(c_int), intent(in) :: isurface(1),control(4)
 real(c_float), intent(inout) :: outlev(nz,6),outsurface(6)
 integer(c_int), intent(inout) :: outint(3)
 real(c_float) :: paddedlev(nz+1,15),paddedout(nz+1,6)
 ! Real MPAS manager bounds include the unused final interface level.
 paddedlev(1:nz,:)=lev
 paddedlev(nz+1,:)=0.
 paddedout(1:nz,:)=outlev
 paddedout(nz+1,:)=-9876.
 call cu_grell_freitas( &
 itimestep=control(1),dt=surface(8),dxcell=surface(6:6),areacell=surface(7:7), &
 u=paddedlev(:,1),v=paddedlev(:,2),t=paddedlev(:,4),w=paddedlev(:,3),q=paddedlev(:,5),p=paddedlev(:,6), &
 pi=paddedlev(:,7),rho=paddedlev(:,8),dz8w=paddedlev(:,9),p8w=paddedlev(:,10), &
 xland=surface(4:4),ht=surface(1:1),hfx=surface(2:2),qfx=surface(3:3),gsw=surface(5:5), &
 rqvften=paddedlev(:,12),rthften=paddedlev(:,11),rthblten=paddedlev(:,14),rqvblten=paddedlev(:,15),rthraten=paddedlev(:,13), &
 kpbl=isurface,xlv=constants(1),cp=constants(2),g=constants(3),r_v=constants(4), &
 ichoice_deep=control(2),ichoice_shallow=control(3),ishallow_g3=control(4), &
 htop=outsurface(1:1),hbot=outsurface(2:2),k22_shallow=outint(1:1), &
 kbcon_shallow=outint(2:2),ktop_shallow=outint(3:3),xmb_total=outsurface(3:3), &
 xmb_shallow=outsurface(4:4),raincv=outsurface(5:5),pratec=outsurface(6:6), &
 gdc=paddedout(:,5),gdc2=paddedout(:,6),rthcuten=paddedout(:,1),rqvcuten=paddedout(:,2), &
 rqccuten=paddedout(:,3),rqicuten=paddedout(:,4), &
 ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz+1, &
 ids=1,ide=2,jds=1,jde=2,kds=1,kde=nz+1, &
 its=1,ite=1,jts=1,jte=1,kts=1,kte=nz)
 outlev=paddedout(1:nz,:)
 if(any(paddedout(nz+1,:)/=-9876.))error stop 'GF changed unused top interface level'
end subroutine

! MPAS mpas_atmphys_driver_convection.F:722-728, unchanged scalar expressions.
subroutine gf_native_surface(n,ht,hfx,qfx,xland,gsw,density,area,len_disp,dt,packed) bind(C)
 use iso_c_binding
 implicit none
 integer(c_int),value:: n
 real(c_float),intent(in):: ht(n),hfx(n),qfx(n),xland(n),gsw(n),density(n),area(n)
 real(c_float),value:: len_disp,dt
 real(c_float),intent(out):: packed(8,n)
 integer:: i
 do i=1,n
   packed(1,i)=ht(i)
   packed(2,i)=hfx(i)
   packed(3,i)=qfx(i)
   packed(4,i)=xland(i)
   packed(5,i)=gsw(i)
   packed(6,i)=len_disp / density(i)**0.25
   packed(7,i)=area(i)
   packed(8,i)=dt
 enddo
end subroutine
