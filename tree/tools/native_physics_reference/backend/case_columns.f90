! Test fixture: decode native initialized columns; all conversion arithmetic is pristine MPAS.
! MPAS source: src/core_atmosphere/physics/mpas_atmphys_interface.F:88-167,235-561.
module native_case_columns
use mpas_kind_types
use mpas_pool_routines
use mpas_atmphys_vars
use mpas_atmphys_interface, only: allocate_forall_physics,MPAS_to_physics
use netcdf
implicit none
contains
subroutine prepare_case(configs,mesh,diag_physics,land,init_path,state_path,order,owned)
 type(mpas_pool_type),pointer :: configs,mesh,diag_physics
 type(field1DReal),pointer :: land
 character(len=*),intent(in) :: init_path,state_path
 integer,intent(in) :: order(:),owned
 type(mpas_pool_type),pointer :: state,diag
 type(field3DReal),pointer :: scalars
 type(field1DReal),pointer :: vector
 integer :: init,data,dimid,nmesh,nz,n,i,varid
 real(RKIND),allocatable :: full(:,:),surface(:)
 character(len=2),parameter :: names(6)=[character(len=2)::'qv','qc','qr','qi','qs','qg']
 call check(nf90_open(init_path,nf90_nowrite,init))
 call check(nf90_open(state_path,nf90_nowrite,data))
 call check(nf90_inq_dimid(init,'nCells',dimid))
 call check(nf90_inquire_dimension(init,dimid,len=nmesh))
 nz=kte; n=size(order)
 call mpas_pool_create_pool(state)
 call mpas_pool_create_pool(diag)
 call mpas_pool_add_config(configs,'config_microp_scheme','mp_kessler')
 call mpas_pool_add_config(configs,'config_pbl_scheme','off')
 do i=1,6
  call mpas_pool_add_dimension(state,'index_'//trim(names(i)),i)
 enddo
 allocate(scalars); allocate(scalars%array(6,nz,n),full(nz,nmesh))
 scalars%array=0._RKIND
 do i=1,6
  if(nf90_inq_varid(data,trim(names(i)),varid)/=nf90_noerr) then
   if(i<=3) error stop 'Kessler fixture requires qv, qc and qr'
   cycle
  endif
  call check(nf90_get_var(data,varid,full,start=[1,1,1],count=[nz,nmesh,1]))
  scalars%array(i,:,:)=full(:,order)
 enddo
 call mpas_pool_add_field(state,'scalars',scalars)
 call add2(state,data,'rho_zz',nz,nmesh,order)
 call add2(state,data,'theta_m',nz,nmesh,order)
 call add2(state,data,'w',nz+1,nmesh,order)
 call add2(diag,data,'exner',nz,nmesh,order)
 call add2(diag,data,'pressure_base',nz,nmesh,order)
 call add2(diag,data,'pressure_p',nz,nmesh,order)
 call add2(diag,data,'rtheta_base',nz,nmesh,order)
 call add2(diag,data,'rtheta_p',nz,nmesh,order)
 call add2(diag,data,'uReconstructZonal',nz,nmesh,order)
 call add2(diag,data,'uReconstructMeridional',nz,nmesh,order)
 call add2(mesh,init,'zgrid',nz+1,nmesh,order,.false.)
 call add2(mesh,init,'zz',nz,nmesh,order,.false.)
 call add1(mesh,init,'latCell',nmesh,order)
 call add1(mesh,init,'lonCell',nmesh,order)
 call add_vertical('fzm'); call add_vertical('fzp'); call add_vertical('rdzw')
 allocate(vector); allocate(vector%array(n)); vector%array=0._RKIND
 call mpas_pool_add_field(diag,'surface_pressure',vector)
 allocate(vector); allocate(vector%array(n)); vector%array=0._RKIND
 call mpas_pool_add_field(diag_physics,'plrad',vector)
 allocate(surface(nmesh))
 call check(nf90_inq_varid(init,'xland',varid))
 call check(nf90_get_var(init,varid,surface,start=[1,1],count=[nmesh,1]))
 land%array=surface(order)
 call allocate_forall_physics(configs)
 ! Unused top/dummy storage is explicit fixture state; native converts owned columns.
 t_p=0.; qv_p=0.; qc_p=0.; qi_p=0.; qs_p=0.; pres_p=0.; pres_hyd_p=0.; rho_p=0.
 call MPAS_to_physics(configs,mesh,state,1,diag,diag_physics,1,owned)
 call check(nf90_close(init)); call check(nf90_close(data))
 call mpas_pool_destroy_pool(state)
 call mpas_pool_destroy_pool(diag)
contains
 subroutine add_vertical(name)
  character(len=*),intent(in) :: name
  type(field1DReal),pointer :: field
  integer :: id
  allocate(field); allocate(field%array(nz))
  call check(nf90_inq_varid(init,name,id))
  call check(nf90_get_var(init,id,field%array))
  call mpas_pool_add_field(mesh,name,field)
 end subroutine
end subroutine
subroutine add1(pool,file,name,nmesh,order)
 type(mpas_pool_type),pointer :: pool
 integer,intent(in) :: file,nmesh,order(:)
 character(len=*),intent(in) :: name
 type(field1DReal),pointer :: field
 real(RKIND) :: full(nmesh)
 integer :: id
 call check(nf90_inq_varid(file,name,id))
 call check(nf90_get_var(file,id,full))
 allocate(field); allocate(field%array(size(order))); field%array=full(order)
 call mpas_pool_add_field(pool,name,field)
end subroutine
subroutine add2(pool,file,name,nz,nmesh,order,time)
 type(mpas_pool_type),pointer :: pool
 integer,intent(in) :: file,nz,nmesh,order(:)
 character(len=*),intent(in) :: name
 logical,optional,intent(in) :: time
 type(field2DReal),pointer :: field
 real(RKIND) :: full(nz,nmesh)
 integer :: id
 logical :: timed
 timed=.true.; if(present(time)) timed=time
 call check(nf90_inq_varid(file,name,id))
 if(timed) then
  call check(nf90_get_var(file,id,full,start=[1,1,1],count=[nz,nmesh,1]))
 else
  call check(nf90_get_var(file,id,full))
 endif
 allocate(field); allocate(field%array(nz,size(order))); field%array=full(:,order)
 call mpas_pool_add_field(pool,name,field)
end subroutine
subroutine check(status)
 integer,intent(in) :: status
 if(status/=nf90_noerr) then
  write(*,*) nf90_strerror(status)
  error stop 'native case decode failed'
 endif
end subroutine
end module
