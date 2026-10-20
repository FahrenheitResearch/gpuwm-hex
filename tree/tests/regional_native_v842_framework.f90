! Framework-only associations and stream/clock fixtures for the pristine module.
! No replacement boundary arithmetic: compile mpas_atm_boundaries.F unchanged.
module mpas_kind_types
 integer, parameter :: RKIND=kind(1.0), StrKIND=256
end module
module mpas_derived_types
 use mpas_kind_types
 use iso_fortran_env, only: int64
 implicit none
 integer, parameter :: MPAS_NOW=0,MPAS_STREAM_LATEST_BEFORE=1,MPAS_STREAM_EARLIEST_STRICTLY_AFTER=2
 integer, parameter :: MPAS_STREAM_MGR_NOERR=0,MPAS_LOG_ERR=1,MPAS_POOL_SILENT=0
 integer, parameter :: MPAS_STREAM_PROPERTY_RECORD_INTV=1,MPAS_STREAM_INPUT=1
 type mpas_time_type
   integer(int64) :: ticks=0
 end type
 type mpas_timeinterval_type
   integer(int64) :: ticks=0
 end type
 type mpas_clock_type
   type(mpas_time_type) :: now
 end type
 type dm_info
   integer :: unused=0
 end type
 type field2
   real(RKIND), pointer :: current(:,:)=>null(),previous(:,:)=>null()
 end type
 type mpas_pool_type
   type(mpas_pool_type), pointer :: mesh=>null(),state=>null(),lbc=>null()
   integer, pointer :: nCells=>null(),nEdges=>null(),nVertLevels=>null(),num_scalars=>null(),index_qv=>null(),index_qc=>null(),nCellsSolve=>null()
   logical, pointer :: config_apply_lbcs=>null()
   integer, pointer :: bdyMaskCell(:)=>null(),bdyMaskEdge(:)=>null(),bdyMaskVertex(:)=>null()
   integer, pointer :: nearestRelaxationCell(:)=>null(),nEdgesOnCell(:)=>null(),cellsOnCell(:,:)=>null(),cellsOnEdge(:,:)=>null()
   real(RKIND), pointer :: specZoneMaskCell(:)=>null(),specZoneMaskEdge(:)=>null(),specZoneMaskVertex(:)=>null()
   real(RKIND), pointer :: xCell(:)=>null(),yCell(:)=>null(),zCell(:)=>null(),zz(:,:)=>null()
   logical :: scalar_present(64)=.true.
   type(field2) :: fields(8)
   real(RKIND), pointer :: scalars(:,:,:)=>null(),old_scalars(:,:,:)=>null()
 end type
 type block_type
   type(mpas_pool_type), pointer :: structs=>null(),configs=>null()
   type(block_type), pointer :: next=>null()
 end type
 type MPAS_streamManager_type
   type(mpas_pool_type), pointer :: pool=>null(),file=>null()
   integer :: read_error=0,property_error=0,last_whence=0
   character(StrKIND) :: read_time='0',input_interval='03:00:00'
 end type
 character(16), parameter :: names(8)=[character(16)::'lbc_u','lbc_ru','lbc_rho_edge','lbc_w', &
                                      'lbc_theta','lbc_rtheta_m','lbc_rho_zz','lbc_rho']
end module
module mpas_pool_routines
 use mpas_derived_types
 implicit none
 interface mpas_pool_get_array
   module procedure get_real1,get_real2,get_real3,get_int1,get_int2
 end interface
contains
 subroutine mpas_pool_get_subpool(pool,name,out)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   type(mpas_pool_type),pointer::out
   select case(name)
   case('mesh');out=>pool%mesh
   case('state');out=>pool%state
   case('lbc');out=>pool%lbc
   case default;error stop 'unknown subpool'
   end select
 end subroutine
 subroutine mpas_pool_get_dimension(pool,name,out)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   integer,pointer::out
   select case(name)
   case('nCells');out=>pool%nCells
   case('nEdges');out=>pool%nEdges
   case('nVertLevels');out=>pool%nVertLevels
   case('num_scalars');out=>pool%num_scalars
   case('nCellsSolve');out=>pool%nCellsSolve
   case('index_qv');out=>pool%index_qv
   case('index_qc');out=>pool%index_qc
   case default;error stop 'unknown dimension'
   end select
 end subroutine
 subroutine mpas_pool_get_config(pool,name,out)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   logical,pointer::out
   if(name/='config_apply_lbcs') error stop 'unknown config'
   out=>pool%config_apply_lbcs
 end subroutine
 subroutine get_real1(pool,name,out,timelevel)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   real(RKIND),pointer::out(:)
   integer,optional::timelevel
   select case(name)
   case('specZoneMaskCell');out=>pool%specZoneMaskCell
   case('specZoneMaskEdge');out=>pool%specZoneMaskEdge
   case('specZoneMaskVertex');out=>pool%specZoneMaskVertex
   case('xCell');out=>pool%xCell
   case('yCell');out=>pool%yCell
   case('zCell');out=>pool%zCell
   case default;error stop 'unknown real1'
   end select
 end subroutine
 subroutine get_real2(pool,name,out,timelevel)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   real(RKIND),pointer::out(:,:)
   integer,optional::timelevel
   integer::j
   nullify(out)
   if(name=='zz') then
     out=>pool%zz;return
   end if
   do j=1,8
     if(name==trim(names(j))) then
       if(present(timelevel)) then
         if(timelevel==1) then
           out=>pool%fields(j)%previous;return
         end if
       end if
       out=>pool%fields(j)%current;return
     end if
   end do
   ! A missing named scalar intentionally leaves the pointer disassociated.
 end subroutine
 subroutine get_real3(pool,name,out,timelevel)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   real(RKIND),pointer::out(:,:,:)
   integer,optional::timelevel
   if(name/='lbc_scalars') error stop 'unknown real3'
   if(present(timelevel)) then
     if(timelevel==1) then
       out=>pool%old_scalars;return
     end if
   end if
   out=>pool%scalars
 end subroutine
 subroutine get_int1(pool,name,out,timelevel)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   integer,pointer::out(:)
   integer,optional::timelevel
   select case(name)
   case('bdyMaskCell');out=>pool%bdyMaskCell
   case('bdyMaskEdge');out=>pool%bdyMaskEdge
   case('bdyMaskVertex');out=>pool%bdyMaskVertex
   case('nearestRelaxationCell');out=>pool%nearestRelaxationCell
   case('nEdgesOnCell');out=>pool%nEdgesOnCell
   case default;error stop 'unknown int1'
   end select
 end subroutine
 subroutine get_int2(pool,name,out,timelevel)
   type(mpas_pool_type),intent(in)::pool
   character(*),intent(in)::name
   integer,pointer::out(:,:)
   integer,optional::timelevel
   select case(name)
   case('cellsOnCell');out=>pool%cellsOnCell
   case('cellsOnEdge');out=>pool%cellsOnEdge
   case default;error stop 'unknown int2'
   end select
 end subroutine
 subroutine mpas_pool_shift_time_levels(pool)
   ! Native mpas_field_routines.F:3335-3347 and shift_time_levs_array.inc:
   ! recycle the old level-1 allocation, rather than copying level 2.
   type(mpas_pool_type),intent(inout)::pool
   real(RKIND),pointer::field_tmp(:,:),scalar_tmp(:,:,:)
   integer::j
   do j=1,8
     field_tmp=>pool%fields(j)%previous
     pool%fields(j)%previous=>pool%fields(j)%current
     pool%fields(j)%current=>field_tmp
   end do
   scalar_tmp=>pool%old_scalars
   pool%old_scalars=>pool%scalars
   pool%scalars=>scalar_tmp
 end subroutine
 integer function mpas_pool_get_error_level()
   mpas_pool_get_error_level=0
 end function
 subroutine mpas_pool_set_error_level(level)
   integer,intent(in)::level
 end subroutine
end module
module mpas_timekeeping
 use mpas_derived_types
 implicit none
 interface operator(-)
   module procedure sub_t_t
 end interface
contains
 function sub_t_t(a,b) result(out)
   type(mpas_time_type),intent(in)::a,b
   type(mpas_timeinterval_type)::out
   out%ticks=a%ticks-b%ticks
 end function
 function mpas_get_clock_time(clock,which,ierr) result(out)
   type(mpas_clock_type),intent(in)::clock
   integer,intent(in)::which
   integer,intent(out)::ierr
   type(mpas_time_type)::out
   out=clock%now;ierr=0
 end function
 subroutine mpas_get_timeInterval(interval,DD,S,S_n,S_d,ierr)
   type(mpas_timeinterval_type),intent(in)::interval
   integer,intent(out)::DD,S,S_n,S_d,ierr
   integer(int64)::left
   DD=int(interval%ticks/86400000000_int64)
   left=interval%ticks-int(DD,int64)*86400000000_int64
   S=int(left/1000000_int64);S_n=int(mod(left,1000000_int64));S_d=1000000;ierr=0
 end subroutine
 subroutine mpas_set_time(out,dateTimeString)
   type(mpas_time_type),intent(out)::out
   character(*),intent(in)::dateTimeString
   read(dateTimeString,*) out%ticks
 end subroutine
 subroutine mpas_get_time(time,dateTimeString)
   type(mpas_time_type),intent(in)::time
   character(*),intent(out)::dateTimeString
   write(dateTimeString,*) time%ticks
 end subroutine
end module
module mpas_constants
 use mpas_kind_types
 real(RKIND),parameter::rvord=461.6_RKIND/287.0_RKIND
end module
module mpas_log
 use mpas_kind_types
contains
 subroutine mpas_log_write(message,messageType,logicArgs,intArgs)
   character(*),intent(in)::message
   integer,optional,intent(in)::messageType,intArgs(:)
   logical,optional,intent(in)::logicArgs(:)
 end subroutine
end module
module mpas_stream_manager
 use mpas_derived_types
contains
 subroutine mpas_stream_mgr_read(manager,streamID,timeLevel,whence,actualWhen,ierr)
   type(MPAS_streamManager_type),intent(inout)::manager
   character(*),intent(in)::streamID
   integer,intent(in)::timeLevel,whence
   character(*),intent(out)::actualWhen
   integer,intent(out)::ierr
   integer::j
   manager%last_whence=whence;ierr=manager%read_error
   actualWhen=manager%read_time
   if(ierr/=0) return
   ! File admission is an association/copy fixture, with no boundary math.
   do j=1,8
     if(j==1.or.j==4.or.j==5.or.j==8) manager%pool%fields(j)%current=manager%file%fields(j)%current
   end do
   ! mpas_io.F:858-878 retains allocated storage for a missing variable.
   do j=1,size(manager%pool%scalars,1)
     if(manager%file%scalar_present(j)) manager%pool%scalars(j,:,:)=manager%file%scalars(j,:,:)
   end do
 end subroutine
 subroutine mpas_stream_mgr_get_property(manager,streamID,property,out,direction,ierr)
   type(MPAS_streamManager_type),intent(in)::manager
   character(*),intent(in)::streamID
   integer,intent(in)::property,direction
   character(*),intent(out)::out
   integer,intent(out)::ierr
   out=manager%input_interval;ierr=manager%property_error
 end subroutine
end module
module mpas_dmpar
 use mpas_derived_types
contains
 subroutine mpas_dmpar_max_int(info,local,global)
   type(dm_info),intent(in)::info
   integer,intent(in)::local
   integer,intent(out)::global
   global=local ! Single-rank identity fixture; not MPI proof.
 end subroutine
end module
