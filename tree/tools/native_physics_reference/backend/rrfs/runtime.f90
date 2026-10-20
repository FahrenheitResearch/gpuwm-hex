! Native framework lifetime for the test-only MPAS driver ABI.
module hex_native_runtime
use iso_c_binding
use mpi_f08, only: MPI_Initialized, MPI_Init, MPI_COMM_WORLD
use mpas_kind_types
use mpas_pool_routines
use mpas_dmpar, only: mpas_dmpar_init, mpas_dmpar_finalize, &
 mpas_dmpar_init_multihalo_exchange_list,mpas_dmpar_destroy_mulithalo_exchange_list,mpas_dmpar_get_exch_list, &
 mpas_dmpar_exch_group_destroy
use mpas_log, only: mpas_log_init, mpas_log_finalize
use mpas_timekeeping
use mpas_stream_manager
use mpas_io
use mpas_attlist, only: mpas_add_att,mpas_modify_att
use mpas_string_utils, only: mpas_string_replace
implicit none
interface
 subroutine gfortran_set_convert(mode) bind(C,name='_gfortran_set_convert')
  import c_int
  integer(c_int),value :: mode
 end subroutine
end interface
type native_exchange_type
 type(mpas_multihalo_exchange_list),pointer :: send=>null(),recv=>null(),copy=>null()
 logical :: ready=.false.
end type
type native_runtime_type
 type(domain_type),pointer :: domain=>null()
 type(mpas_pool_type),pointer :: fields=>null(),packages=>null(),structs=>null()
 type(native_exchange_type) :: exchange(3)
 logical :: has_clock=.false.,has_streams=.false.,has_block=.false.
end type
contains
function string(s) result(v)
 character(c_char),intent(in) :: s(*)
 character(len=StrKIND) :: v
 integer :: i
 v=''
 do i=1,StrKIND
  if(s(i)==c_null_char) exit
  v(i:i)=s(i)
 enddo
end function
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:273-344; mpas_log.F:99-273.
function native_runtime_create() result(handle) bind(C)
 type(c_ptr) :: handle
 type(native_runtime_type),pointer :: runtime
 logical :: initialized
 integer :: status
 ! GNU's -fconvert=big-endian is initialized by the Fortran executable main,
 ! which ctypes does not execute. Match the strict probe/native main exactly.
 call gfortran_set_convert(2_c_int)
 allocate(runtime)
 allocate(runtime%domain)
 allocate(runtime%domain%dminfo)
 allocate(runtime%domain%core)
 runtime%domain%core%coreName='atmosphere'
 runtime%domain%domainID=1
 call MPI_Initialized(initialized,status)
 if(.not.initialized) call MPI_Init(status)
 ! Keep process MPI alive across serial test contexts, and never finalize caller MPI.
 call mpas_dmpar_init(runtime%domain%dminfo,external_comm=MPI_COMM_WORLD)
 call mpas_log_init(runtime%domain%logInfo,runtime%domain,unitNumbers=[6,0],err=status)
 ! The strict fork has no MPAS_DEBUG: its pool default is SILENT. In particular,
 ! lsm_to_MPAS requests absent, unused raincv even with convection disabled.
 ! Preserve the original null pointer instead of aborting a valid native call.
 ! MPAS source: MPAS-Model-rrfs/src/framework/mpas_pool_routines.F:109,5839-5858;
 ! MPAS-Model-rrfs/src/framework/mpas_framework.F:101-103;
 ! MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:704.
 call mpas_pool_set_error_level(MPAS_POOL_SILENT)
 call mpas_pool_create_pool(runtime%fields)
 call mpas_pool_create_pool(runtime%packages)
 call mpas_pool_create_pool(runtime%structs)
 handle=c_loc(runtime)
end function
! The container members borrow caller pools; these aliases never own targets.
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_pool_types.inc:67-101.
subroutine destroy_borrowed_pool(pool)
 type(mpas_pool_type),pointer :: pool
 type(mpas_pool_member_type),pointer :: member,next
 integer :: i
 do i=1,pool%size
  member=>pool%table(i)%head
  do while(associated(member))
   next=>member%next
   deallocate(member%data)
   deallocate(member)
   member=>next
  enddo
  nullify(pool%table(i)%head)
 enddo
 nullify(pool%iterator,pool%iteration_head,pool%iteration_tail)
 call mpas_pool_destroy_pool(pool)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_block_types.inc:19-49;
! MPAS-Model-rrfs/src/framework/mpas_pool_routines.F:5097-5129.
subroutine native_runtime_block(handle,configs,mesh,diag_physics,sfc_input,atm_input,status) bind(C)
 type(c_ptr),value :: handle,configs,mesh,diag_physics,sfc_input,atm_input
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 type(mpas_pool_type),pointer :: config_pool,mesh_pool,diag_pool,sfc_pool,atm_pool
 call c_f_pointer(handle,runtime)
 status=-1
 if(runtime%has_block) return
 call c_f_pointer(configs,config_pool)
 call c_f_pointer(mesh,mesh_pool)
 call c_f_pointer(diag_physics,diag_pool)
 call c_f_pointer(sfc_input,sfc_pool)
 call c_f_pointer(atm_input,atm_pool)
 allocate(runtime%domain%blocklist)
 runtime%domain%blocklist%blockID=1
 runtime%domain%blocklist%localBlockID=0
 runtime%domain%blocklist%domain=>runtime%domain
 runtime%domain%configs=>config_pool
 runtime%domain%blocklist%configs=>config_pool
 runtime%domain%blocklist%dimensions=>mesh_pool
 runtime%domain%blocklist%structs=>runtime%structs
 runtime%domain%blocklist%allStructs=>runtime%structs
 runtime%domain%blocklist%allFields=>runtime%fields
 runtime%domain%blocklist%packages=>runtime%packages
 nullify(runtime%domain%blocklist%parinfo)
 call mpas_pool_add_subpool(runtime%structs,'mesh',mesh_pool)
 call mpas_pool_add_subpool(runtime%structs,'diag_physics',diag_pool)
 call mpas_pool_add_subpool(runtime%structs,'sfc_input',sfc_pool)
 call mpas_pool_add_subpool(runtime%structs,'atm_input',atm_pool)
 runtime%has_block=.true.
 status=0
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_pool_routines.F:5119-5139.
subroutine native_runtime_attach_pool(handle,key,pool,status) bind(C)
 type(c_ptr),value :: handle,pool
 character(c_char),intent(in) :: key(*)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 type(mpas_pool_type),pointer :: borrowed
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_block) return
 call c_f_pointer(pool,borrowed)
 call mpas_pool_add_subpool(runtime%structs,trim(string(key)),borrowed)
 status=0
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:1599-2159,6455-6502.
! All owned elements precede halos; native get_exch_list constructs the maps.
subroutine native_runtime_decomposition(handle,element,owned,count,ids,status) bind(C)
 type(c_ptr),value :: handle
 integer(c_int),value :: element,owned,count
 integer(c_int),intent(in),target :: ids(count)
 integer(c_int),intent(out) :: status
 call native_runtime_decomposition_layers(handle,element,owned,count,ids,1,[count],status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:1599-2159,6456-6503.
subroutine native_runtime_decomposition_layers(handle,element,owned,count,ids,nlayers,ends,status) bind(C)
 type(c_ptr),value :: handle
 integer(c_int),value :: element,owned,count,nlayers
 integer(c_int),intent(in) :: ends(nlayers)
 integer :: layer,first,last
 integer(c_int),intent(in),target :: ids(count)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 type(field1DInteger),pointer :: owned_field,needed_field
 type(field0DInteger),pointer :: offset
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_block.or.element<1.or.element>3) return
 if(owned<0.or.owned>count.or.any(ids<=0)) return
 if(runtime%exchange(element)%ready.or.nlayers<1) return
 if(ends(nlayers)/=count.or.any(ends<owned).or.any(ends>count)) return
 do layer=2,nlayers
  if(ends(layer)<ends(layer-1)) return
 enddo
 call mpas_dmpar_init_multihalo_exchange_list(runtime%exchange(element)%send,nlayers)
 call mpas_dmpar_init_multihalo_exchange_list(runtime%exchange(element)%recv,nlayers)
 call mpas_dmpar_init_multihalo_exchange_list(runtime%exchange(element)%copy,nlayers)
 allocate(owned_field,needed_field,offset)
 owned_field%array=>ids(1:owned)
 owned_field%dimSizes=[owned]
 owned_field%block=>runtime%domain%blocklist
 owned_field%sendList=>runtime%exchange(element)%send
 owned_field%copyList=>runtime%exchange(element)%copy
 needed_field%dimSizes=[count-owned]
 needed_field%block=>runtime%domain%blocklist
 needed_field%recvList=>runtime%exchange(element)%recv
 offset%block=>runtime%domain%blocklist
 first=owned+1
 do layer=1,nlayers
  last=ends(layer)
  needed_field%array=>ids(first:last)
  needed_field%dimSizes=[last-first+1]
  offset%scalar=first-1
  call mpas_dmpar_get_exch_list(layer,owned_field,needed_field,offsetListField=offset)
  first=last+1
 enddo
 deallocate(owned_field,needed_field,offset)
 runtime%exchange(element)%ready=.true.
 status=0
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:87-130,175-220,1142-1297,1346-1589.
subroutine native_runtime_clock(handle,calendar,start,stop,dt,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: calendar(*),start(*),stop(*)
 real(c_float),value :: dt
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 type(MPAS_Time_type) :: start_time,stop_time
 type(MPAS_TimeInterval_type) :: step
 call c_f_pointer(handle,runtime)
 status=0
 if(runtime%has_clock) then
  status=-1
  return
 endif
 call mpas_timekeeping_init(trim(string(calendar)))
 call mpas_set_time(start_time,dateTimeString=trim(string(start)),ierr=status)
 if(status/=0) return
 call mpas_set_time(stop_time,dateTimeString=trim(string(stop)),ierr=status)
 if(status/=0) return
 call mpas_set_timeInterval(step,dt=real(dt,RKIND),ierr=status)
 if(status/=0) return
 allocate(runtime%domain%clock)
 call mpas_create_clock(runtime%domain%clock,start_time,step,stopTime=stop_time,ierr=status)
 runtime%has_clock=status==0
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_io.F:143-234; mpas_stream_manager.F:120-195.
subroutine native_runtime_streams(handle,status) bind(C)
 type(c_ptr),value :: handle
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_clock.or.runtime%has_streams) return
 allocate(runtime%domain%ioContext)
 runtime%domain%ioContext%dminfo=>runtime%domain%dminfo
 call MPAS_io_init(runtime%domain%ioContext,1,1,ierr=status)
 if(status/=0) return
 call MPAS_stream_mgr_init(runtime%domain%streamManager,runtime%domain%ioContext, &
      runtime%domain%clock,runtime%fields,runtime%packages,runtime%structs,ierr=status)
 runtime%has_streams=status==0
end subroutine
function native_runtime_object(handle,which) result(object) bind(C)
 type(c_ptr),value :: handle
 integer(c_int),value :: which
 type(c_ptr) :: object
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 object=c_null_ptr
 select case(which)
 case(1)
  object=c_loc(runtime%domain%dminfo)
 case(2)
  if(runtime%has_clock) object=c_loc(runtime%domain%clock)
 case(3)
  if(runtime%has_streams) object=c_loc(runtime%domain%streamManager)
 case(4)
  if(runtime%has_block) object=c_loc(runtime%domain)
 case(5)
  if(runtime%has_block) object=c_loc(runtime%domain%blocklist)
 end select
end function
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:319-444,1167-1340;
! MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:489-560.
subroutine native_runtime_stream(handle,id,filename,direction,input_interval,output_interval,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: id(*),filename(*),input_interval(*),output_interval(*)
 integer(c_int),value :: direction
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 integer :: native_direction
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_streams) return
 select case(direction)
 case(1)
  native_direction=MPAS_STREAM_INPUT
 case(2)
  native_direction=MPAS_STREAM_OUTPUT
 case(3)
  native_direction=MPAS_STREAM_INPUT_OUTPUT
 case default
  return
 end select
 call MPAS_stream_mgr_create_stream(runtime%domain%streamManager,trim(string(id)), &
      native_direction,trim(string(filename)),ierr=status)
 if(status/=0) return
 if(direction==1.or.direction==3) then
  call native_stream_alarm(runtime,trim(string(id)),trim(string(input_interval)),MPAS_STREAM_INPUT,status)
  if(status/=0) return
 endif
 if(direction==2.or.direction==3) then
  call native_stream_alarm(runtime,trim(string(id)),trim(string(output_interval)),MPAS_STREAM_OUTPUT,status)
 endif
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:489-560;
! MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:1167-1340.
subroutine native_stream_alarm(runtime,id,interval,direction,status)
 type(native_runtime_type),intent(inout) :: runtime
 character(len=*),intent(in) :: id,interval
 integer,intent(in) :: direction
 integer,intent(out) :: status
 type(MPAS_TimeInterval_type) :: step
 type(MPAS_Time_type) :: start
 character(len=512) :: alarm
 status=0
 if(interval=='none') return
 start=mpas_get_clock_time(runtime%domain%clock,MPAS_START_TIME)
 if(direction==MPAS_STREAM_INPUT) then
  alarm=trim(id)//':in'
 else
  alarm=trim(id)//':out'
 endif
 if(interval=='initial_only'.or.interval=='final_only') then
  if(interval=='final_only') start=mpas_get_clock_time(runtime%domain%clock,MPAS_STOP_TIME)
  call mpas_add_clock_alarm(runtime%domain%clock,trim(alarm),start,ierr=status)
 else
  call mpas_set_timeInterval(step,timeString=interval,ierr=status)
  if(status/=0) return
  call mpas_add_clock_alarm(runtime%domain%clock,trim(alarm),start,alarmTimeInterval=step,ierr=status)
 endif
 if(status/=0) return
 call MPAS_stream_mgr_add_alarm(runtime%domain%streamManager,id,trim(alarm),direction,ierr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:1848-1935.
subroutine native_runtime_stream_filename_interval(handle,id,interval,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: id(*),interval(*)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_streams) return
 call MPAS_stream_mgr_set_property(runtime%domain%streamManager,trim(string(id)), &
      MPAS_STREAM_PROPERTY_FILENAME_INTV,trim(string(interval)),ierr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:396-425.
subroutine native_runtime_advance(handle,status) bind(C)
 type(c_ptr),value :: handle
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_clock) return
 call mpas_advance_clock(runtime%domain%clock,ierr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:813-841.
function native_runtime_alarm_ringing(handle,id,status) result(ringing) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: id(*)
 integer(c_int),intent(out) :: status
 logical(c_bool) :: ringing
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 ringing=.false.
 if(.not.runtime%has_clock) return
 ringing=mpas_is_alarm_ringing(runtime%domain%clock,trim(string(id)),ierr=status)
end function
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:924-984.
subroutine native_runtime_alarm_reset(handle,id,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: id(*)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_clock) return
 call mpas_reset_clock_alarm(runtime%domain%clock,trim(string(id)),ierr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:283-295.
subroutine native_runtime_initialize_state_time(handle,pool,status) bind(C)
 type(c_ptr),value :: handle,pool
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 type(mpas_pool_type),pointer :: state
 type(field0DReal),pointer :: time_field
 character(len=StrKIND),pointer :: initial_time1,initial_time2,xtime
 character(len=StrKIND) :: timestamp
 type(MPAS_Time_type) :: start
 integer :: level
 call c_f_pointer(handle,runtime)
 call c_f_pointer(pool,state)
 status=-1
 if(.not.runtime%has_clock) return
 start=mpas_get_clock_time(runtime%domain%clock,MPAS_START_TIME)
 call mpas_get_time(start,dateTimeString=timestamp)
 call mpas_pool_get_array(state,'initial_time',initial_time1,1)
 call mpas_pool_get_array(state,'initial_time',initial_time2,2)
 initial_time2=initial_time1
 do level=1,2
  call mpas_pool_get_array(state,'xtime',xtime,level)
  xtime=timestamp
  call mpas_pool_get_field(state,'Time',time_field,timeLevel=level)
  if(.not.associated(time_field%attLists)) allocate(time_field%attLists(1))
  call mpas_modify_att(time_field%attLists(1)%attList,'units', &
      'seconds since '//mpas_string_replace(initial_time1,'_',' '),ierr=status)
  if(status/=0) call mpas_add_att(time_field%attLists(1)%attList,'units', &
      'seconds since '//mpas_string_replace(initial_time1,'_',' '),ierr=status)
  if(status/=0) return
 enddo
end subroutine
! MPAS source: MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:783-797.
subroutine native_runtime_update_state_time(handle,pool,dt,clock_advanced,source_level,target_level,status) bind(C)
 type(c_ptr),value :: handle,pool
 real(c_float),value :: dt
 logical(c_bool),value :: clock_advanced
 integer(c_int),value :: source_level,target_level
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 type(mpas_pool_type),pointer :: state
 type(MPAS_Time_type) :: currTime
 type(MPAS_TimeInterval_type) :: dtInterval
 character(len=StrKIND),pointer :: xtime
 character(len=StrKIND) :: xtime_new
 real(kind=RKIND),pointer :: Time
 real(kind=RKIND) :: Time_new
 call c_f_pointer(handle,runtime)
 call c_f_pointer(pool,state)
 status=-1
 if(.not.runtime%has_clock) return
 currTime=mpas_get_clock_time(runtime%domain%clock,MPAS_NOW)
 if(.not.clock_advanced) then
  call mpas_set_timeInterval(dtInterval,dt=dt)
  currTime=currTime+dtInterval
 endif
 call mpas_get_time(currTime,dateTimeString=xtime_new)
 call mpas_pool_get_array(state,'xtime',xtime,target_level)
 xtime=xtime_new
 call mpas_pool_get_array(state,'Time',Time,source_level)
 Time_new=Time+dt
 call mpas_pool_get_array(state,'Time',Time,target_level)
 Time=Time_new
 status=0
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:213-277; mpas_io.F:6396-6456;
! mpas_timekeeping.F:224-250; mpas_dmpar.F:359-386; mpas_log.F:660-726.
subroutine native_runtime_destroy(handle) bind(C)
 type(c_ptr),value :: handle
 type(native_runtime_type),pointer :: runtime
 integer :: status,i
 character(len=StrKIND) :: group_name
 call c_f_pointer(handle,runtime)
 if(runtime%has_streams) then
  call MPAS_stream_mgr_finalize(runtime%domain%streamManager,ierr=status)
  call MPAS_io_finalize(runtime%domain%ioContext,finalize_iosystem=.true.,ierr=status)
  deallocate(runtime%domain%ioContext)
 endif
 do while(associated(runtime%domain%exchangeGroups))
  group_name=runtime%domain%exchangeGroups%groupName
  call mpas_dmpar_exch_group_destroy(runtime%domain,trim(group_name))
 enddo
 call destroy_borrowed_pool(runtime%fields)
 call destroy_borrowed_pool(runtime%structs)
 call mpas_pool_destroy_pool(runtime%packages)
 do i=1,3
  if(.not.runtime%exchange(i)%ready) cycle
  call mpas_dmpar_destroy_mulithalo_exchange_list(runtime%exchange(i)%send)
  call mpas_dmpar_destroy_mulithalo_exchange_list(runtime%exchange(i)%recv)
  call mpas_dmpar_destroy_mulithalo_exchange_list(runtime%exchange(i)%copy)
 enddo
 if(runtime%has_block) deallocate(runtime%domain%blocklist)
 if(runtime%has_clock) then
  call mpas_destroy_clock(runtime%domain%clock,ierr=status)
  deallocate(runtime%domain%clock)
 endif
 call mpas_log_finalize(status)
 deallocate(runtime%domain%logInfo%outputLog,runtime%domain%logInfo%errorLog)
 deallocate(runtime%domain%logInfo)
 call mpas_dmpar_finalize(runtime%domain%dminfo)
 deallocate(runtime%domain%dminfo)
 deallocate(runtime%domain%core)
 deallocate(runtime%domain)
 deallocate(runtime)
end subroutine
end module

! MPAS authority: mpas_atmphys_manager.F:204-213,263-375.
 subroutine native_manager_state(handle,flags,stamp,dates,julian,status) bind(C)
 use iso_c_binding
 use mpas_kind_types
 use mpas_timekeeping
 use mpas_atmphys_manager,only:year,month,day,julday,curr_julday
 use mpas_atmphys_vars,only:l_radtlw,l_radtsw,l_conv,doabsems,l_camlw,l_acrain,l_acradt,l_diags
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),intent(in) :: flags(8)
 character(c_char),intent(out) :: stamp(512)
 integer(c_int),intent(out) :: dates(5),status
 real(c_float),intent(out) :: julian
 type(MPAS_Clock_type),pointer :: clock
 type(MPAS_Time_type) :: currTime
 character(len=StrKIND) :: timeStamp
 integer :: yr,h,m,s,s_n,s_d,DoY,ierr,i
 real(RKIND) :: utc_h,utc_s
 logical :: LeapYear
 call c_f_pointer(handle,clock)
 currTime = mpas_get_clock_time(clock,MPAS_NOW,ierr)
 call mpas_get_time(curr_time=currTime,dateTimeString=timeStamp,YYYY=yr,MM=month,DD=day,H=h,M=m, &
                    S=s,S_n=s_n,S_d=s_d,DoY=DoY,ierr=ierr)

 utc_h = real(h) + real(m) / 60.0 + real(s + s_n / s_d) / 3600.0
 utc_s = real(s + s_n + s_d) 
 year   = yr
 julday = DoY
 curr_julday = real(julday-1) + utc_h / 24.0
 LeapYear = isLeapYear(year)

 status=ierr
 if(status/=0) return
 do i=1,len_trim(timeStamp)
  stamp(i)=timeStamp(i:i)
 enddo
 stamp(len_trim(timeStamp)+1)=c_null_char
 dates=[year,month,day,julday,merge(1,0,LeapYear)]
 julian=curr_julday
 l_radtlw=flags(1)/=0; l_radtsw=flags(2)/=0; l_conv=flags(3)/=0
 doabsems=flags(4)/=0; l_camlw=flags(5)/=0; l_acrain=flags(6)/=0
 l_acradt=flags(7)/=0; l_diags=flags(8)/=0
 end subroutine
