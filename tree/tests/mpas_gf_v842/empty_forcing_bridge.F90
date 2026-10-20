! Whole GF empty active-row domain: initialization still runs; no forcing read.
! MPAS module_cu_gf.mpas.F:54-453, mpas_atmphys_driver_convection.F:482-516.
! Test-only C binding; authority procedure and arithmetic are unmodified.
subroutine gf_native_empty_forcing(nz,mask,lev,surface,isurface,outlev,outsurface,outint,control,constants) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int), value :: nz,mask
 real(c_float), intent(in) :: lev(nz,15),surface(8),constants(4)
 integer(c_int), intent(in) :: isurface(1),control(4)
 real(c_float), intent(inout) :: outlev(nz,6),outsurface(6)
 integer(c_int), intent(inout) :: outint(3)
 real(c_float),target :: paddedlev(nz+1,15)
 real(c_float),target :: paddedout(nz+1,6)
 real(c_float),pointer :: gdc(:,:,:),gdc2(:,:,:),rth(:,:,:),rqv(:,:,:)
 ! Real MPAS manager bounds include the unused final interface level.
 paddedlev(1:nz,:)=lev
 paddedlev(nz+1,:)=0.
 paddedout(1:nz,:)=outlev
 paddedout(nz+1,:)=-9876.
 nullify(gdc,gdc2,rth,rqv)
 if(iand(mask,4)/=0)rth(1:1,1:nz+1,1:1)=>paddedlev(:,11)
 if(iand(mask,8)/=0)rqv(1:1,1:nz+1,1:1)=>paddedlev(:,12)
 if(iand(mask,1)/=0)gdc(1:1,1:nz+1,1:1)=>paddedout(:,5)
 if(iand(mask,2)/=0)gdc2(1:1,1:nz+1,1:1)=>paddedout(:,6)
 call cu_grell_freitas( &
 itimestep=control(1),dt=surface(8),dxcell=surface(6:6),areacell=surface(7:7), &
 u=paddedlev(:,1),v=paddedlev(:,2),t=paddedlev(:,4),w=paddedlev(:,3),q=paddedlev(:,5),p=paddedlev(:,6), &
 pi=paddedlev(:,7),rho=paddedlev(:,8),dz8w=paddedlev(:,9),p8w=paddedlev(:,10), &
 xland=surface(4:4),ht=surface(1:1),hfx=surface(2:2),qfx=surface(3:3),gsw=surface(5:5), &
 rqvften=rqv,rthften=rth,rthblten=paddedlev(:,14),rqvblten=paddedlev(:,15),rthraten=paddedlev(:,13), &
 kpbl=isurface,xlv=constants(1),cp=constants(2),g=constants(3),r_v=constants(4), &
 ichoice_deep=control(2),ichoice_shallow=control(3),ishallow_g3=control(4), &
 htop=outsurface(1:1),hbot=outsurface(2:2),k22_shallow=outint(1:1), &
 kbcon_shallow=outint(2:2),ktop_shallow=outint(3:3),xmb_total=outsurface(3:3), &
 xmb_shallow=outsurface(4:4),raincv=outsurface(5:5),pratec=outsurface(6:6), &
 gdc=gdc,gdc2=gdc2,rthcuten=paddedout(:,1),rqvcuten=paddedout(:,2), &
 rqccuten=paddedout(:,3),rqicuten=paddedout(:,4), &
 ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz+1, &
 ids=1,ide=2,jds=1,jde=1,kds=1,kde=nz+1, &
 its=1,ite=1,jts=1,jte=1,kts=1,kte=nz)
 outlev=paddedout(1:nz,:)
 if(any(paddedout(nz+1,:)/=-9876.))error stop 'GF changed unused top interface level'
end subroutine
