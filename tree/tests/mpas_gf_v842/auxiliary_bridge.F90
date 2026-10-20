! Test-only access to full-module procedures and driver-fixed alternate modes.
! MPAS module_cu_gf.mpas.F:3106-3173,3353-3473,4060-4121.
subroutine gf_native_init(nz,scheme,restart_i,qc,qi,a,s) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,scheme,restart_i,qc,qi
 real(c_float),intent(inout)::a(2,nz,2,11),s(2,2,9)
 logical::restart
 restart=restart_i/=0
 if(scheme==0)then
 call g3init(a(:,:,:,1),a(:,:,:,2),a(:,:,:,3),a(:,:,:,4),s(:,:,1),1004.5,restart, &
 qc,qi,2,a(:,:,:,5),a(:,:,:,6),s(:,:,2),s(:,:,3),s(:,:,4),s(:,:,5),s(:,:,6), &
 s(:,:,7),s(:,:,8),s(:,:,9),a(:,:,:,7),a(:,:,:,8),a(:,:,:,9),a(:,:,:,10),a(:,:,:,11), &
 .false.,1,3,1,3,1,nz+1,1,2,1,2,1,nz,1,2,1,2,1,nz)
 else
 call gdinit(a(:,:,:,1),a(:,:,:,2),a(:,:,:,3),a(:,:,:,4),s(:,:,1),1004.5,restart, &
 qc,qi,2,a(:,:,:,5),a(:,:,:,6),s(:,:,2),s(:,:,3),s(:,:,4),s(:,:,5),s(:,:,6), &
 s(:,:,7),s(:,:,8),s(:,:,9),.false.,1,3,1,3,1,nz+1,1,2,1,2,1,nz,1,2,1,2,1,nz)
 endif
end subroutine

! MPAS module_cu_gf.mpas.F:3788-4058. Driver-fixed excess/Berry modes.
subroutine gf_native_up(nz,shallow,excess,autoconv,a,s,ints) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,shallow,excess,autoconv
 real(c_float),intent(inout)::a(4,nz,17),s(4,5)
 integer(c_int),intent(inout)::ints(4,4)
 character(7)::name
 name='deep'
 if(shallow/=0)name='shallow'
 call cup_up_moisture(name,ints(:,1),a(:,:,5),a(:,:,1),a(:,:,2),a(:,:,3),s(:,1), &
 100.,a(:,:,6),ints(:,2),ints(:,3),a(:,:,7),a(:,:,8),a(:,:,4),a(:,:,9),a(:,:,10), &
 a(:,:,11),a(:,:,12),a(:,:,13),ints(:,4),a(:,:,14),2500000.,s(:,4),excess,s(:,5), &
 a(:,:,15),a(:,:,16),a(:,:,17),s(:,2),s(:,3),autoconv,1,0,4,1,nz,1,0,0,1,4,1,1,1,nz)
end subroutine

! MPAS module_cu_gf.mpas.F:2127-2263. Both initial buoyancy signs.
subroutine gf_native_down(nz,iloop,a,s,ints,text) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,iloop
 real(c_float),intent(inout)::a(4,nz,15),s(4,2)
 integer(c_int),intent(inout)::ints(4,2)
 character(c_char),intent(inout)::text(50,4)
 character(50)::error_text(4)
 integer::i,k
 do i=1,4
 do k=1,50
 error_text(i)(k:k)=text(k,i)
 enddo
 enddo
 call cup_dd_moisture_new(error_text,a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5), &
 a(:,:,6),a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10),ints(:,1),ints(:,2),a(:,:,11), &
 s(:,1),s(:,2),a(:,:,12),a(:,:,13),a(:,:,14),a(:,:,15),iloop,2500000.,4,1,nz,1,4,1,1,1,nz)
 do i=1,4
 do k=1,50
 text(k,i)=error_text(i)(k:k)
 enddo
 enddo
end subroutine

! MPAS module_cu_gf.mpas.F:1989-2124. Alternate aerosol modes are
! source-fixed to one in the MPAS driver, but remain part of the module.
subroutine gf_native_edt(nz,aero,a,s,e,ints) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,aero
 real(c_float),intent(inout)::a(4,nz,6),s(4,6),e(4,3)
 integer(c_int),intent(inout)::ints(4,3)
 call cup_dd_edt(ints(:,1),a(:,:,1),a(:,:,2),a(:,:,3),ints(:,2),ints(:,3), &
 s(:,1),a(:,:,4),s(:,2),a(:,:,5),s(:,4),s(:,3),.95,.01,3,e,s(:,5),s(:,6), &
 100.,a(:,:,6),aero,4,1,nz,1,0,0,1,4,1,1,1,nz)
end subroutine

! MPAS module_cu_gf.mpas.F:2987-3103. All search/excess modes.
subroutine gf_native_kbcon(nz,iloop,excess,a,s,ints,text) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,iloop,excess
 real(c_float),intent(inout)::a(4,nz,3),s(4,5)
 integer(c_int),intent(inout)::ints(4,4)
 character(c_char),intent(inout)::text(50,4)
 character(50)::error_text(4)
 integer::i,k
 do i=1,4
 do k=1,50
 error_text(i)(k:k)=text(k,i)
 enddo
 enddo
 call cup_kbcon(error_text,s(:,3),iloop,ints(:,1),ints(:,2),a(:,:,1),a(:,:,2), &
 s(:,1),ints(:,4),ints(:,3),a(:,:,3),s(:,2),2500000.,1004.5,s(:,5),s(:,4), &
 excess,4,1,nz,1,4,1,1,1,nz)
 do i=1,4
 do k=1,50
 text(k,i)=error_text(i)(k:k)
 enddo
 enddo
end subroutine

! MPAS module_cu_gf.mpas.F:2265-2397.
subroutine gf_native_env(nz,itest,a,s,ierr) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,itest
 real(c_float),intent(inout)::a(4,nz,7),s(4,2)
 integer(c_int),intent(inout)::ierr(4)
 call cup_env(a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5),a(:,:,6),a(:,:,7), &
 s(:,2),s(:,1),ierr,258.,itest,2500000.,1004.5,4,1,nz,1,4,1,1,1,nz)
end subroutine

! MPAS module_cu_gf.mpas.F:3106-3173.
subroutine gf_native_ktop(nz,ilo,dby,kbcon,ktop,ierr,text) bind(C)
 use iso_c_binding
 use module_cu_gf
 implicit none
 integer(c_int),value::nz,ilo
 real(c_float),intent(inout)::dby(4,nz)
 integer(c_int),intent(in)::kbcon(4)
 integer(c_int),intent(out)::ktop(4)
 integer(c_int),intent(inout)::ierr(4)
 character(c_char),intent(inout)::text(50,4)
 character(50)::error_text(4)
 integer::i,k
 do i=1,4
 do k=1,50
 error_text(i)(k:k)=text(k,i)
 enddo
 enddo
 call cup_ktop(error_text,ilo,dby,kbcon,ktop,ierr,4,1,nz,1,4,1,1,1,nz)
 do i=1,4
 do k=1,50
 text(k,i)=error_text(i)(k:k)
 enddo
 enddo
end subroutine
