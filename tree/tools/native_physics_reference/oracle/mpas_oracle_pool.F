module mpas_oracle_pool
 use mpas_pool_routines
 use mpas_oracle_dump
 implicit none
 private
 public :: oracle_pool_capture, oracle_mesh_context
contains
 subroutine oracle_mesh_context(mesh,step)
  type(mpas_pool_type), intent(in) :: mesh
  integer, intent(in), optional :: step
  integer, pointer :: ids(:)
  character(len=32) :: rank_text
  integer :: rank_id,status,step_id
  call mpas_pool_get_array(mesh,'indexToCellID',ids)
  rank_id=0
  call get_environment_variable('OMPI_COMM_WORLD_RANK',rank_text,status=status)
  if(status/=0) call get_environment_variable('PMI_RANK',rank_text,status=status)
  if(status/=0) call get_environment_variable('PMIX_RANK',rank_text,status=status)
  if(status==0) then
   read(rank_text,*,iostat=status) rank_id
   if(status/=0) error stop 'oracle invalid MPI rank environment'
  endif
  step_id=oracle_current_step()
  if(present(step)) step_id=step
  call oracle_set_native_context(ids,step_id,rank_id)
 end subroutine
 recursive subroutine oracle_pool_capture(key,phase,prefix,pool,event)
  character(len=*), intent(in) :: key,phase,prefix
  type(mpas_pool_type), intent(in) :: pool
  integer, intent(in) :: event
  type(mpas_pool_member_type), pointer :: member
  type(mpas_pool_data_type), pointer :: data
  character(len=256) :: name
  character(len=24) :: lev
  integer :: t,j,ca,la,member_count,iterator_member
  character(len=256), allocatable :: member_names(:)
  logical :: field_present
  if(.not.oracle_enabled(key)) return
  call oracle_write(key,phase,trim(prefix)//'.pool_size',pool%size,event)
  member_count=0; iterator_member=0
  member=>pool%iteration_head
  do while(associated(member))
   member_count=member_count+1
   if(associated(member,pool%iterator)) iterator_member=member_count
   member=>member%iteration_next
  enddo
  call oracle_write(key,phase,trim(prefix)//'.iterator_member',iterator_member,event)
  allocate(member_names(member_count))
  member=>pool%iteration_head
  do j=1,member_count
   member_names(j)=member%key
   member=>member%iteration_next
  enddo
  call oracle_write(key,phase,trim(prefix)//'.members',member_names,event)
  member=>pool%iteration_head
  do while(associated(member))
   data=>member%data
   name=trim(prefix)//'.'//trim(member%key)
   if(len_trim(prefix)+1+len_trim(member%key)>len(name)-len('.field_descriptor.present')) &
     error stop 'oracle pool name exceeds format with descriptor suffix'
   call oracle_write(key,phase,trim(name)//'.member_kind',member%contentsType,event)
   call oracle_write(key,phase,trim(name)//'.descriptor.present',associated(data),event)
   if(.not.associated(data)) then
    member=>member%iteration_next
    cycle
   endif
   select case(member%contentsType)
   case(MPAS_POOL_SUBPOOL)
    call oracle_write(key,phase,trim(name)//'.subpool.present',associated(data%p),event)
    if(associated(data%p)) call oracle_pool_capture(key,phase,name,data%p,event)
   case(MPAS_POOL_CONFIG,MPAS_POOL_PACKAGE,MPAS_POOL_DIMENSION)
    call oracle_write(key,phase,trim(name)//'.type',data%contentsType,event)
    call oracle_write(key,phase,trim(name)//'.rank',data%contentsDims,event)
    call oracle_write(key,phase,trim(name)//'.time_levels',data%contentsTimeLevs,event)
    call oracle_write(key,phase,trim(name)//'.integer.present',associated(data%simple_int),event)
    call oracle_write(key,phase,trim(name)//'.integer_array.present',associated(data%simple_int_arr),event)
    call oracle_write(key,phase,trim(name)//'.real.present',associated(data%simple_real),event)
    call oracle_write(key,phase,trim(name)//'.logical.present',associated(data%simple_logical),event)
    call oracle_write(key,phase,trim(name)//'.character.present',associated(data%simple_char),event)
    if(associated(data%simple_int)) call oracle_write(key,phase,name,data%simple_int,event)
    if(associated(data%simple_int_arr)) call oracle_write(key,phase,name,data%simple_int_arr,event, &
         lower=lbound(data%simple_int_arr))
    if(associated(data%simple_real)) call oracle_write(key,phase,name,data%simple_real,event)
    if(associated(data%simple_logical)) call oracle_write(key,phase,name,data%simple_logical,event)
    if(associated(data%simple_char)) call oracle_write(key,phase,name,data%simple_char,event)
   case(MPAS_POOL_FIELD)
    call oracle_write(key,phase,trim(name)//'.type',data%contentsType,event)
    call oracle_write(key,phase,trim(name)//'.rank',data%contentsDims,event)
    call oracle_write(key,phase,trim(name)//'.time_levels',data%contentsTimeLevs,event)
    select case(data%contentsType)
    case(MPAS_POOL_REAL)
     select case(data%contentsDims)
     case(0)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r0a),event)
       if(associated(data%r0a)) then
        do t=1,size(data%r0a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%r0a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%r0a(t)%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%r0a(t)%scalar,event,cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r0),event)
       if(associated(data%r0)) then
          call oracle_write(key,phase,trim(name)//'.active',data%r0%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%r0%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name), &
               data%r0%scalar,event,cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(1)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r1a),event)
       if(associated(data%r1a)) then
        do t=1,size(data%r1a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%r1a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%r1a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%r1a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%r1a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r1a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,1
           if(trim(data%r1a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%r1a(t)%dimNames(j))=='nVertLevels' .or. trim(data%r1a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%r1a(t)%array,event, lower=lbound(data%r1a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r1),event)
       if(associated(data%r1)) then
          call oracle_write(key,phase,trim(name)//'.active',data%r1%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%r1%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%r1%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%r1%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r1%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,1
           if(trim(data%r1%dimNames(j))=='nCells') ca=j
           if(trim(data%r1%dimNames(j))=='nVertLevels' .or. trim(data%r1%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%r1%array,event, lower=lbound(data%r1%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(2)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r2a),event)
       if(associated(data%r2a)) then
        do t=1,size(data%r2a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%r2a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%r2a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%r2a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%r2a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r2a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,2
           if(trim(data%r2a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%r2a(t)%dimNames(j))=='nVertLevels' .or. trim(data%r2a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%r2a(t)%array,event, lower=lbound(data%r2a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r2),event)
       if(associated(data%r2)) then
          call oracle_write(key,phase,trim(name)//'.active',data%r2%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%r2%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%r2%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%r2%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r2%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,2
           if(trim(data%r2%dimNames(j))=='nCells') ca=j
           if(trim(data%r2%dimNames(j))=='nVertLevels' .or. trim(data%r2%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%r2%array,event, lower=lbound(data%r2%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(3)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r3a),event)
       if(associated(data%r3a)) then
        do t=1,size(data%r3a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%r3a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%r3a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%r3a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%r3a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r3a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,3
           if(trim(data%r3a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%r3a(t)%dimNames(j))=='nVertLevels' .or. trim(data%r3a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%r3a(t)%array,event, lower=lbound(data%r3a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r3),event)
       if(associated(data%r3)) then
          call oracle_write(key,phase,trim(name)//'.active',data%r3%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%r3%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%r3%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%r3%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r3%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,3
           if(trim(data%r3%dimNames(j))=='nCells') ca=j
           if(trim(data%r3%dimNames(j))=='nVertLevels' .or. trim(data%r3%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%r3%array,event, lower=lbound(data%r3%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(4)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r4a),event)
       if(associated(data%r4a)) then
        do t=1,size(data%r4a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%r4a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%r4a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%r4a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%r4a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r4a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,4
           if(trim(data%r4a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%r4a(t)%dimNames(j))=='nVertLevels' .or. trim(data%r4a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%r4a(t)%array,event, lower=lbound(data%r4a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r4),event)
       if(associated(data%r4)) then
          call oracle_write(key,phase,trim(name)//'.active',data%r4%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%r4%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%r4%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%r4%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r4%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,4
           if(trim(data%r4%dimNames(j))=='nCells') ca=j
           if(trim(data%r4%dimNames(j))=='nVertLevels' .or. trim(data%r4%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%r4%array,event, lower=lbound(data%r4%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(5)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r5a),event)
       if(associated(data%r5a)) then
        do t=1,size(data%r5a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%r5a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%r5a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%r5a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%r5a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r5a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,5
           if(trim(data%r5a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%r5a(t)%dimNames(j))=='nVertLevels' .or. trim(data%r5a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%r5a(t)%array,event, lower=lbound(data%r5a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%r5),event)
       if(associated(data%r5)) then
          call oracle_write(key,phase,trim(name)//'.active',data%r5%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%r5%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%r5%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%r5%dimSizes,event)
          field_present=.true.
          field_present=associated(data%r5%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,5
           if(trim(data%r5%dimNames(j))=='nCells') ca=j
           if(trim(data%r5%dimNames(j))=='nVertLevels' .or. trim(data%r5%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%r5%array,event, lower=lbound(data%r5%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case default
      error stop 'oracle pool field rank unsupported'
     end select
    case(MPAS_POOL_INTEGER)
     select case(data%contentsDims)
     case(0)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i0a),event)
       if(associated(data%i0a)) then
        do t=1,size(data%i0a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%i0a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%i0a(t)%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%i0a(t)%scalar,event,cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i0),event)
       if(associated(data%i0)) then
          call oracle_write(key,phase,trim(name)//'.active',data%i0%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%i0%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name), &
               data%i0%scalar,event,cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(1)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i1a),event)
       if(associated(data%i1a)) then
        do t=1,size(data%i1a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%i1a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%i1a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%i1a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%i1a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%i1a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,1
           if(trim(data%i1a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%i1a(t)%dimNames(j))=='nVertLevels' .or. trim(data%i1a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%i1a(t)%array,event, lower=lbound(data%i1a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i1),event)
       if(associated(data%i1)) then
          call oracle_write(key,phase,trim(name)//'.active',data%i1%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%i1%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%i1%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%i1%dimSizes,event)
          field_present=.true.
          field_present=associated(data%i1%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,1
           if(trim(data%i1%dimNames(j))=='nCells') ca=j
           if(trim(data%i1%dimNames(j))=='nVertLevels' .or. trim(data%i1%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%i1%array,event, lower=lbound(data%i1%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(2)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i2a),event)
       if(associated(data%i2a)) then
        do t=1,size(data%i2a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%i2a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%i2a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%i2a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%i2a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%i2a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,2
           if(trim(data%i2a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%i2a(t)%dimNames(j))=='nVertLevels' .or. trim(data%i2a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%i2a(t)%array,event, lower=lbound(data%i2a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i2),event)
       if(associated(data%i2)) then
          call oracle_write(key,phase,trim(name)//'.active',data%i2%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%i2%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%i2%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%i2%dimSizes,event)
          field_present=.true.
          field_present=associated(data%i2%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,2
           if(trim(data%i2%dimNames(j))=='nCells') ca=j
           if(trim(data%i2%dimNames(j))=='nVertLevels' .or. trim(data%i2%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%i2%array,event, lower=lbound(data%i2%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(3)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i3a),event)
       if(associated(data%i3a)) then
        do t=1,size(data%i3a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%i3a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%i3a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%i3a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%i3a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%i3a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,3
           if(trim(data%i3a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%i3a(t)%dimNames(j))=='nVertLevels' .or. trim(data%i3a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%i3a(t)%array,event, lower=lbound(data%i3a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%i3),event)
       if(associated(data%i3)) then
          call oracle_write(key,phase,trim(name)//'.active',data%i3%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%i3%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%i3%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%i3%dimSizes,event)
          field_present=.true.
          field_present=associated(data%i3%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,3
           if(trim(data%i3%dimNames(j))=='nCells') ca=j
           if(trim(data%i3%dimNames(j))=='nVertLevels' .or. trim(data%i3%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%i3%array,event, lower=lbound(data%i3%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case default
      error stop 'oracle pool field rank unsupported'
     end select
    case(MPAS_POOL_LOGICAL)
     select case(data%contentsDims)
     case(0)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%l0a),event)
       if(associated(data%l0a)) then
        do t=1,size(data%l0a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%l0a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%l0a(t)%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%l0a(t)%scalar,event,cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%l0),event)
       if(associated(data%l0)) then
          call oracle_write(key,phase,trim(name)//'.active',data%l0%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%l0%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name), &
               data%l0%scalar,event,cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case default
      error stop 'oracle pool field rank unsupported'
     end select
    case(MPAS_POOL_CHARACTER)
     select case(data%contentsDims)
     case(0)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%c0a),event)
       if(associated(data%c0a)) then
        do t=1,size(data%c0a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%c0a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%c0a(t)%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%c0a(t)%scalar,event,cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%c0),event)
       if(associated(data%c0)) then
          call oracle_write(key,phase,trim(name)//'.active',data%c0%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%c0%fieldName,event)
          field_present=.true.
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          call oracle_write(key,phase,trim(name), &
               data%c0%scalar,event,cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case(1)
      if(data%contentsTimeLevs>1) then
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%c1a),event)
       if(associated(data%c1a)) then
        do t=1,size(data%c1a)
          write(lev,'(A,I0)') '.t',t
          call oracle_write(key,phase,trim(name)//trim(lev)//'.active',data%c1a(t)%isActive,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.field_name',data%c1a(t)%fieldName,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_names',data%c1a(t)%dimNames,event)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.dimension_sizes',data%c1a(t)%dimSizes,event)
          field_present=.true.
          field_present=associated(data%c1a(t)%array)
          call oracle_write(key,phase,trim(name)//trim(lev)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,1
           if(trim(data%c1a(t)%dimNames(j))=='nCells') ca=j
           if(trim(data%c1a(t)%dimNames(j))=='nVertLevels' .or. trim(data%c1a(t)%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name)//trim(lev), &
               data%c1a(t)%array,event, lower=lbound(data%c1a(t)%array),cell_axis=ca,level_axis=la)
         endif
        enddo
       endif
      else
       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%c1),event)
       if(associated(data%c1)) then
          call oracle_write(key,phase,trim(name)//'.active',data%c1%isActive,event)
          call oracle_write(key,phase,trim(name)//'.field_name',data%c1%fieldName,event)
          call oracle_write(key,phase,trim(name)//'.dimension_names',data%c1%dimNames,event)
          call oracle_write(key,phase,trim(name)//'.dimension_sizes',data%c1%dimSizes,event)
          field_present=.true.
          field_present=associated(data%c1%array)
          call oracle_write(key,phase,trim(name)//'.present',field_present,event)
         if(field_present) then
          ca=0; la=0
          do j=1,1
           if(trim(data%c1%dimNames(j))=='nCells') ca=j
           if(trim(data%c1%dimNames(j))=='nVertLevels' .or. trim(data%c1%dimNames(j))=='nVertLevelsP1') la=j
          enddo
          call oracle_write(key,phase,trim(name), &
               data%c1%array,event, lower=lbound(data%c1%array),cell_axis=ca,level_axis=la)
         endif
       endif
      endif
     case default
      error stop 'oracle pool field rank unsupported'
     end select
    case default
     error stop 'oracle pool field type unsupported'
    end select
   end select
   member=>member%iteration_next
  enddo
 end subroutine
end module
