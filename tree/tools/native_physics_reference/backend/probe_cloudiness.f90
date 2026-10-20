! Direct pristine driver comparison, independent of generated call/pool wrappers.
! MPAS source: src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:189-234.
program probe_cloudiness
use mpas_kind_types
use mpas_pool_routines
use mpas_atmphys_vars
use mpas_atmphys_driver_cloudiness
use netcdf
use native_case_columns, only: prepare_case
#ifdef MPAS_NATIVE_ORACLE_PROBE
use mpas_oracle_dump
#endif
implicit none
type(mpas_pool_type),pointer :: configs,mesh,diag_physics,sfc_input
type(field1DReal),pointer :: area,density,land
type(field2DReal),pointer :: cloud
real(RKIND),parameter :: temp(6)=[290.,260.,240.,280.,275.,255.]
real(RKIND),parameter :: vapor(6)=[.015,.001,.00001,.000001,.010,.002]
real(RKIND),parameter :: liquid(6)=[.00001,.000001,.0000005,0.,.00002,.00003]
real(RKIND),parameter :: ice(6)=[0.,.00002,.0000002,0.,.00001,.00005]
real(RKIND),parameter :: snow(6)=[0.,0.,.000001,0.,0.,.00001]
character(len=128) :: scheme,arg
integer :: i,k,n,nsolve,j,ncid,dimid,varid,nmesh,maxedges,unit,rank,source_cell,nz,case_cells
real(RKIND) :: len_disp
integer,allocatable :: global_ids(:),order(:),partition(:),neighbors(:,:),nedges(:),capture_ids(:)
real(RKIND),allocatable :: mesh_area(:),mesh_density(:)
logical,allocatable :: selected(:)
character(len=4096) :: mesh_path,partition_path,case_init,case_state
#ifdef MPAS_NATIVE_ORACLE_PROBE
character(len=4096) :: prefix
#endif
call get_command_argument(1,scheme)
call get_command_argument(2,arg)
f_qc=arg=='1'
call get_command_argument(3,arg)
f_qi=arg=='1'
n=6; nsolve=5
allocate(order(n),capture_ids(n))
order=[1,2,3,4,5,6]; capture_ids=[1,2,3,4,5,0]
! Optional real mesh/partition metadata; args 8/9 supply initialized weather.
! NetCDF decoding and partition construction occur in Fortran, not Python.
call get_command_argument(5,mesh_path)
if (len_trim(mesh_path)>0) then
 call check(nf90_open(trim(mesh_path),nf90_nowrite,ncid))
 call check(nf90_inq_dimid(ncid,'nCells',dimid))
 call check(nf90_inquire_dimension(ncid,dimid,len=nmesh))
 call check(nf90_inq_dimid(ncid,'maxEdges',dimid))
 call check(nf90_inquire_dimension(ncid,dimid,len=maxedges))
 allocate(global_ids(nmesh),partition(nmesh),mesh_area(nmesh),mesh_density(nmesh))
 allocate(neighbors(maxedges,nmesh),nedges(nmesh),selected(nmesh))
 call check(nf90_inq_varid(ncid,'indexToCellID',varid))
 call check(nf90_get_var(ncid,varid,global_ids))
 call check(nf90_inq_varid(ncid,'areaCell',varid))
 call check(nf90_get_var(ncid,varid,mesh_area))
 call check(nf90_inq_varid(ncid,'meshDensity',varid))
 call check(nf90_get_var(ncid,varid,mesh_density))
 call check(nf90_inq_varid(ncid,'cellsOnCell',varid))
 call check(nf90_get_var(ncid,varid,neighbors))
 call check(nf90_inq_varid(ncid,'nEdgesOnCell',varid))
 call check(nf90_get_var(ncid,varid,nedges))
 call check(nf90_close(ncid))
 call get_command_argument(6,partition_path)
 call get_command_argument(7,arg)
 read(arg,*) rank
 partition=0
 if (rank>=0) then
  open(newunit=unit,file=trim(partition_path),status='old',action='read')
  do i=1,nmesh
   read(unit,*) partition(i)
  enddo
  close(unit)
 else
  rank=0
 endif
 nsolve=count(partition==rank)
 if(nsolve==0) error stop 'empty partition'
 selected=partition==rank
 do i=1,nmesh
  if(partition(i)/=rank) cycle
  do k=1,nedges(i)
   j=neighbors(k,i)
   if(j>0) selected(j)=.true.
  enddo
 enddo
 n=count(selected)+1
 deallocate(order,capture_ids)
 allocate(order(n),capture_ids(n))
 order=1; capture_ids=0
 j=0
 ! Reverse owned-cell order deliberately: compare by ID, never storage index.
 do i=nmesh,1,-1
  if(partition(i)/=rank) cycle
  j=j+1; order(j)=i; capture_ids(j)=global_ids(i)
 enddo
 do i=1,nmesh
  if(.not.selected(i).or.partition(i)==rank) cycle
  j=j+1; order(j)=i
 enddo
endif
call get_command_argument(8,case_init)
call get_command_argument(9,case_state)
nz=3; len_disp=480000._RKIND
if(len_trim(case_init)>0) then
 if(len_trim(mesh_path)==0.or.len_trim(case_state)==0) error stop 'case requires mesh and native state output'
 call check(nf90_open(trim(case_init),nf90_nowrite,ncid))
 call check(nf90_inq_dimid(ncid,'nVertLevels',dimid))
 call check(nf90_inquire_dimension(ncid,dimid,len=nz))
 call check(nf90_inq_dimid(ncid,'nCells',dimid))
 call check(nf90_inquire_dimension(ncid,dimid,len=case_cells))
 if(case_cells/=nmesh) error stop 'case/mesh dimensions differ'
 call check(nf90_inq_varid(ncid,'nominalMinDc',varid))
 call check(nf90_get_var(ncid,varid,len_disp))
 call check(nf90_close(ncid))
endif
ids=1; ide=n; ims=1; ime=n
jds=1; jde=1; jms=1; jme=1; jts=1; jte=1
kds=1; kde=nz; kms=1; kme=nz; kts=1; kte=nz
if(len_trim(case_init)>0) kme=nz+1
call mpas_pool_create_pool(configs)
call mpas_pool_create_pool(mesh)
call mpas_pool_create_pool(diag_physics)
call mpas_pool_create_pool(sfc_input)
call mpas_pool_add_config(configs,'config_radt_cld_scheme',trim(scheme))
call mpas_pool_add_config(configs,'config_len_disp',len_disp)
allocate(area,density,land,cloud)
allocate(area%array(n),density%array(n),land%array(n),cloud%array(nz,n))
area%array=1.e10_RKIND; density%array=1._RKIND; cloud%array=-999._RKIND
do i=1,n
 source_cell=order(i)
 land%array(i)=real(merge(1,2,mod(source_cell,2)==0),RKIND)
 if(allocated(mesh_area)) then
  area%array(i)=mesh_area(source_cell); density%array(i)=mesh_density(source_cell)
 endif
enddo
call mpas_pool_add_field(mesh,'areaCell',area)
call mpas_pool_add_field(mesh,'meshDensity',density)
call mpas_pool_add_field(sfc_input,'xland',land)
call mpas_pool_add_field(diag_physics,'cldfrac',cloud)
if(len_trim(case_init)>0) then
 call prepare_case(configs,mesh,diag_physics,land,trim(case_init),trim(case_state),order,nsolve)
else
allocate(t_p(n,3,1),qv_p(n,3,1),qc_p(n,3,1),qi_p(n,3,1),qs_p(n,3,1))
allocate(pres_p(n,3,1),pres_hyd_p(n,3,1),rho_p(n,3,1))
do k=1,3
 do i=1,n
  j=mod(order(i)-1,6)+1
  t_p(i,k,1)=temp(j); qv_p(i,k,1)=vapor(j); qc_p(i,k,1)=liquid(j)
  qi_p(i,k,1)=ice(j); qs_p(i,k,1)=snow(j)
 enddo
enddo
pres_p=80000._RKIND; pres_hyd_p=80000._RKIND; rho_p=1._RKIND
endif
call allocate_cloudiness()
#ifdef MPAS_NATIVE_ORACLE_PROBE
call get_command_argument(4,prefix)
call oracle_set_output(trim(prefix))
call oracle_set_context(capture_ids,1,0)
call oracle_write('boundary__driver_cloudiness','pre','config_radt_cld_scheme',trim(scheme),1)
call oracle_write('boundary__driver_cloudiness','pre','config_len_disp',len_disp,1)
call oracle_write('boundary__driver_cloudiness','pre','areaCell',area%array,1)
call oracle_write('boundary__driver_cloudiness','pre','meshDensity',density%array,1)
call oracle_write('boundary__driver_cloudiness','pre','xland',land%array,1)
call oracle_write('boundary__driver_cloudiness','pre','cldfrac',cloud%array,1)
call oracle_write('boundary__driver_cloudiness','pre','ids',ids,1)
call oracle_write('boundary__driver_cloudiness','pre','ide',ide,1)
call oracle_write('boundary__driver_cloudiness','pre','ims',ims,1)
call oracle_write('boundary__driver_cloudiness','pre','ime',ime,1)
call oracle_write('boundary__driver_cloudiness','pre','jds',jds,1)
call oracle_write('boundary__driver_cloudiness','pre','jde',jde,1)
call oracle_write('boundary__driver_cloudiness','pre','jms',jms,1)
call oracle_write('boundary__driver_cloudiness','pre','jme',jme,1)
call oracle_write('boundary__driver_cloudiness','pre','jts',jts,1)
call oracle_write('boundary__driver_cloudiness','pre','jte',jte,1)
call oracle_write('boundary__driver_cloudiness','pre','kds',kds,1)
call oracle_write('boundary__driver_cloudiness','pre','kde',kde,1)
call oracle_write('boundary__driver_cloudiness','pre','kms',kms,1)
call oracle_write('boundary__driver_cloudiness','pre','kme',kme,1)
call oracle_write('boundary__driver_cloudiness','pre','kts',kts,1)
call oracle_write('boundary__driver_cloudiness','pre','kte',kte,1)
call oracle_write('boundary__driver_cloudiness','pre','f_qc',f_qc,1)
call oracle_write('boundary__driver_cloudiness','pre','f_qi',f_qi,1)
call oracle_write('boundary__driver_cloudiness','pre','t_p',t_p,1,lower=lbound(t_p))
call oracle_write('boundary__driver_cloudiness','pre','qv_p',qv_p,1,lower=lbound(qv_p))
call oracle_write('boundary__driver_cloudiness','pre','qc_p',qc_p,1,lower=lbound(qc_p))
call oracle_write('boundary__driver_cloudiness','pre','qi_p',qi_p,1,lower=lbound(qi_p))
call oracle_write('boundary__driver_cloudiness','pre','qs_p',qs_p,1,lower=lbound(qs_p))
call oracle_write('boundary__driver_cloudiness','pre','pres_p',pres_p,1,lower=lbound(pres_p))
call oracle_write('boundary__driver_cloudiness','pre','pres_hyd_p',pres_hyd_p,1,lower=lbound(pres_hyd_p))
call oracle_write('boundary__driver_cloudiness','pre','rho_p',rho_p,1,lower=lbound(rho_p))
#endif
call driver_cloudiness(configs,mesh,diag_physics,sfc_input,1,nsolve)
#ifdef MPAS_NATIVE_ORACLE_PROBE
call oracle_write('boundary__driver_cloudiness','post','cldfrac',cloud%array,1,lower=lbound(cloud%array),cell_axis=2,level_axis=1)
call oracle_clear_output()
#endif
! Print native words in Fortran order; only test evidence, never forecast data I/O.
do i=1,n
 do k=1,nz
  write(*,'(z8.8)') transfer(cloud%array(k,i),0)
 enddo
enddo
call deallocate_cloudiness()
call mpas_pool_destroy_pool(configs)
call mpas_pool_destroy_pool(mesh)
call mpas_pool_destroy_pool(diag_physics)
call mpas_pool_destroy_pool(sfc_input)
contains
subroutine check(status)
 integer,intent(in) :: status
 if(status/=nf90_noerr) then
  write(*,*) nf90_strerror(status)
  error stop 'mesh decode failed'
 endif
end subroutine
end program
