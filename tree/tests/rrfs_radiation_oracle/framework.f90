! Test-only framework. Numerical routines come from pristine RRFS 1d6e370e.
module mpas_kind_types
 integer,parameter:: RKIND=4,StrKIND=512
end module
module mpas_derived_types
 integer,parameter:: MPAS_LOG_ERR=1
end module
module mpas_log
contains
 subroutine mpas_log_write(text,messagetype)
 character(*),intent(in)::text
 integer,optional::messagetype
 end subroutine
end module
module mpas_timer
contains
 subroutine mpas_timer_start(text)
 character(*)::text
 end subroutine
 subroutine mpas_timer_stop(text)
 character(*)::text
 end subroutine
end module
module mpas_atmphys_utilities
contains
 subroutine physics_error_fatal(text)
 character(*)::text
 print*,text
 stop 1
 end subroutine
 subroutine physics_message(text)
 character(*)::text
 print*,text
 end subroutine
end module
module mpas_io_units
contains
 subroutine mpas_new_unit(unit)
 integer::unit
 unit=77
 end subroutine
 subroutine mpas_release_unit(unit)
 integer::unit
 end subroutine
end module
module mpas_atmphys_vars
 integer::ims=1,ime,kms=1,kme,jms=1,jme=1,ids=1,ide,kds=1,kde,jds=1,jde=1
 integer::its=1,ite,kts=1,kte,jts=1,jte=1
 logical::f_qc=.true.,f_qi=.true.
 real,allocatable::dx_p(:,:),xland_p(:,:),cldfrac_p(:,:,:),qvrad_p(:,:,:),qcrad_p(:,:,:)
 real,allocatable::qirad_p(:,:,:),qsrad_p(:,:,:),qv_p(:,:,:),qc_p(:,:,:),qi_p(:,:,:),qs_p(:,:,:)
 real,allocatable::pres_p(:,:,:),pres_hyd_p(:,:,:),t_p(:,:,:),rho_p(:,:,:)
 real,allocatable::qcbl_p(:,:,:),qibl_p(:,:,:),qccu_p(:,:,:),qicu_p(:,:,:)
end module
module oracle_state
 integer,target::nc,nl,nz,nmon=12
 real,target::len_disp
 character(512),target::scheme
 character(512),target::convection='off'
 real,allocatable,target::density(:),land(:),area(:),lat(:),lon(:),pin(:),monthly(:,:,:),ozn(:,:),cld(:,:)
 real,allocatable,target::blfrac(:,:),blc(:,:),bli(:,:),cufrac(:,:),cuc(:,:),cui(:,:)
end module
module mpas_pool_routines
 use oracle_state
 type mpas_pool_type
 integer::unused
 end type
 interface mpas_pool_get_config
 module procedure config_real,config_char
 end interface
 interface mpas_pool_get_array
 module procedure array1,array2,array3
 end interface
contains
 subroutine config_real(pool,key,val)
 type(mpas_pool_type),intent(in)::pool
 character(*)::key
 real,pointer::val
 val=>len_disp
 end subroutine
 subroutine config_char(pool,key,val)
 type(mpas_pool_type),intent(in)::pool
 character(*)::key
 character(512),pointer::val
 if(key=='config_convection_scheme')then
 val=>convection
 else
 val=>scheme
 endif
 end subroutine
 subroutine mpas_pool_get_dimension(pool,key,val)
 type(mpas_pool_type),intent(in)::pool
 character(*)::key
 integer,pointer::val
 select case(key)
 case('nCells','nCellsSolve');val=>nc
 case('nVertLevels');val=>nz
 case('nOznLevels');val=>nl
 case('nMonths');val=>nmon
 case default;stop 'bad dimension'
 end select
 end subroutine
 subroutine array1(pool,key,val)
 type(mpas_pool_type),intent(in)::pool
 character(*)::key
 real,pointer::val(:)
 select case(key)
 case('areaCell');val=>area
 case('meshDensity');val=>density
 case('xland');val=>land
 case('latCell');val=>lat
 case('lonCell');val=>lon
 case('pin');val=>pin
 case default;stop 'bad array1'
 end select
 end subroutine
 subroutine array2(pool,key,val)
 type(mpas_pool_type),intent(in)::pool
 character(*)::key
 real,pointer::val(:,:)
 select case(key)
 case('cldfrac');val=>cld
 case('o3clim');val=>ozn
 case('cldfrac_bl');val=>blfrac
 case('qc_bl');val=>blc
 case('qi_bl');val=>bli
 case('cldfrac_cu');val=>cufrac
 case('qc_cu');val=>cuc
 case('qi_cu');val=>cui
 case default;stop 'bad array2'
 end select
 end subroutine
 subroutine array3(pool,key,val)
 type(mpas_pool_type),intent(in)::pool
 character(*)::key
 real,pointer::val(:,:,:)
 val=>monthly
 end subroutine
end module
