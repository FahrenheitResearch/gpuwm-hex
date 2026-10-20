! Direct native monthly surface update; independent of generated ABI calls.
! MPAS source: src/core_atmosphere/physics/mpas_atmphys_update_surface.F:54-111.
! MPAS source: src/core_atmosphere/physics/mpas_atmphys_date_time.F:126-219.
program probe_surface
use iso_c_binding, only: c_int32_t
use mpas_kind_types, only: RKIND, StrKIND
use mpas_pool_routines
use mpas_atmphys_update_surface, only: physics_update_surface
implicit none
type(mpas_pool_type), pointer :: mesh, surface
type(field1DInteger), pointer :: land
type(field2DReal), pointer :: albedo, green
type(field1DReal), pointer :: background, vegetation, minimum, maximum
character(len=StrKIND) :: date, option
integer :: cell, month
call get_command_argument(1,date)
call get_command_argument(2,option)
call mpas_pool_create_pool(mesh)
call mpas_pool_create_pool(surface)
call mpas_pool_add_dimension(mesh,'nCellsSolve',4)
allocate(land,albedo,green,background,vegetation,minimum,maximum)
allocate(land%array(5),albedo%array(12,5),green%array(12,5))
allocate(background%array(5),vegetation%array(5),minimum%array(5),maximum%array(5))
land%fieldName='landmask'; land%dimNames=''; land%dimSizes=5
land%defaultValue=0; land%missingValue=0
land%isDecomposed=.false.; land%hasTimeDimension=.false.
land%isActive=.true.; land%isVarArray=.false.; land%isPersistent=.true.
call metadata2(albedo,'albedo12m')
call metadata2(green,'greenfrac')
call metadata1(background,'sfc_albbck')
call metadata1(vegetation,'vegfra')
call metadata1(minimum,'shdmin')
call metadata1(maximum,'shdmax')
land%array=[0,1,1,0,1]
do cell=1,5
 do month=1,12
  albedo%array(month,cell)=real(month+cell,RKIND)
  green%array(month,cell)=real(13-month+cell,RKIND)
 enddo
enddo
background%array=-999._RKIND
vegetation%array=-999._RKIND
minimum%array=-999._RKIND
maximum%array=-999._RKIND
call mpas_pool_add_field(surface,'landmask',land)
call mpas_pool_add_field(surface,'albedo12m',albedo)
call mpas_pool_add_field(surface,'greenfrac',green)
call mpas_pool_add_field(surface,'sfc_albbck',background)
call mpas_pool_add_field(surface,'vegfra',vegetation)
call mpas_pool_add_field(surface,'shdmin',minimum)
call mpas_pool_add_field(surface,'shdmax',maximum)
call physics_update_surface(date,option=='1',mesh,surface)
call words(background%array)
call words(vegetation%array)
call words(minimum%array)
call words(maximum%array)
call mpas_pool_destroy_pool(surface)
call mpas_pool_destroy_pool(mesh)
contains
subroutine metadata1(field,name)
 type(field1DReal),intent(inout) :: field
 character(len=*),intent(in) :: name
 field%fieldName=name; field%dimNames=''; field%dimSizes=5
 field%defaultValue=0; field%missingValue=0
 field%isDecomposed=.false.; field%hasTimeDimension=.false.
 field%isActive=.true.; field%isVarArray=.false.; field%isPersistent=.true.
end subroutine
subroutine metadata2(field,name)
 type(field2DReal),intent(inout) :: field
 character(len=*),intent(in) :: name
 field%fieldName=name; field%dimNames=''; field%dimSizes=[12,5]
 field%defaultValue=0; field%missingValue=0
 field%isDecomposed=.false.; field%hasTimeDimension=.false.
 field%isActive=.true.; field%isVarArray=.false.; field%isPersistent=.true.
end subroutine
subroutine words(values)
 real(RKIND),intent(in) :: values(:)
 integer :: i
 do i=1,size(values)
  write(*,'(z8.8)') transfer(values(i),0_c_int32_t)
 enddo
end subroutine
end program
