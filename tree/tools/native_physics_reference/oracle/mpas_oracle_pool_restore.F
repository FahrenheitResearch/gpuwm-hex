module mpas_oracle_pool_restore
 use mpas_pool_routines
 use mpas_oracle_dump
 implicit none
 private
 public :: oracle_pool_restore, oracle_pool_restore_in_place
 logical,save :: pool_context_set=.false.
 integer,save :: restore_depth=0
contains
 recursive subroutine oracle_pool_restore(prefix,pool)
  character(len=*),intent(in) :: prefix
  type(mpas_pool_type),pointer :: pool
  type(mpas_pool_type),pointer :: child
  type(mpas_pool_member_type),pointer :: new_member
  character(len=256),allocatable :: names(:)
  character(len=4096) :: member_path
  character(len=24) :: time_suffix
  character(len=StrKIND) :: char_value
  integer :: context_event,j,t,member_kind,field_type,field_rank,time_levels,iterator_member
  integer :: code,width,rank_n,dims(5),lb(5),unit,pool_size,int_value
  integer,pointer :: dimension_values(:)
  real(kind=RKIND) :: real_value
  logical :: has_descriptor,has_value,logical_value
  type(field0DReal),pointer :: field_r0
  type(field0DReal),pointer :: field_r0_levels(:)
  type(field1DReal),pointer :: field_r1
  type(field1DReal),pointer :: field_r1_levels(:)
  type(field2DReal),pointer :: field_r2
  type(field2DReal),pointer :: field_r2_levels(:)
  type(field3DReal),pointer :: field_r3
  type(field3DReal),pointer :: field_r3_levels(:)
  type(field4DReal),pointer :: field_r4
  type(field4DReal),pointer :: field_r4_levels(:)
  type(field5DReal),pointer :: field_r5
  type(field5DReal),pointer :: field_r5_levels(:)
  type(field0DInteger),pointer :: field_i0
  type(field0DInteger),pointer :: field_i0_levels(:)
  type(field1DInteger),pointer :: field_i1
  type(field1DInteger),pointer :: field_i1_levels(:)
  type(field2DInteger),pointer :: field_i2
  type(field2DInteger),pointer :: field_i2_levels(:)
  type(field3DInteger),pointer :: field_i3
  type(field3DInteger),pointer :: field_i3_levels(:)
  type(field0DLogical),pointer :: field_l0
  type(field0DLogical),pointer :: field_l0_levels(:)
  type(field0DChar),pointer :: field_c0
  type(field0DChar),pointer :: field_c0_levels(:)
  type(field1DChar),pointer :: field_c1
  type(field1DChar),pointer :: field_c1_levels(:)
  if(restore_depth==0) pool_context_set=.false.
  restore_depth=restore_depth+1
  call oracle_read(trim(prefix)//'.pool_size.bin',pool_size)
  call mpas_pool_create_pool(pool,pool_size)
  call oracle_read(trim(prefix)//'.iterator_member.bin',iterator_member)
  call oracle_header(trim(prefix)//'.members.bin',code,width,rank_n,dims,lb,unit)
  close(unit)
  allocate(names(dims(1)))
  call oracle_read(trim(prefix)//'.members.bin',names)
  do j=1,size(names)
   member_path=trim(prefix)//'.'//trim(names(j))
   call oracle_read(trim(member_path)//'.member_kind.bin',member_kind)
   call oracle_read(trim(member_path)//'.descriptor.present.bin',has_descriptor)
   if(.not.has_descriptor) then
    call restore_member(pool,trim(names(j)),member_kind,new_member)
    cycle
   endif
   select case(member_kind)
   case(MPAS_POOL_SUBPOOL)
    call oracle_read(trim(member_path)//'.subpool.present.bin',has_value)
    nullify(child)
    if(has_value) call oracle_pool_restore(trim(member_path),child)
    call mpas_pool_add_subpool(pool,trim(names(j)),child)
   case(MPAS_POOL_CONFIG,MPAS_POOL_PACKAGE,MPAS_POOL_DIMENSION)
    call restore_member(pool,trim(names(j)),member_kind,new_member)
    allocate(new_member%data)
    call oracle_read(trim(member_path)//'.type.bin',new_member%data%contentsType)
    call oracle_read(trim(member_path)//'.rank.bin',new_member%data%contentsDims)
    call oracle_read(trim(member_path)//'.time_levels.bin',new_member%data%contentsTimeLevs)
    call oracle_read(trim(member_path)//'.integer.present.bin',has_value)
    if(has_value) then
     allocate(new_member%data%simple_int)
     call oracle_read(trim(member_path)//'.bin',new_member%data%simple_int)
    endif
    call oracle_read(trim(member_path)//'.integer_array.present.bin',has_value)
    if(has_value) then
     call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
     close(unit)
     allocate(new_member%data%simple_int_arr(lb(1):lb(1)+dims(1)-1))
     call oracle_read(trim(member_path)//'.bin',new_member%data%simple_int_arr)
    endif
    call oracle_read(trim(member_path)//'.real.present.bin',has_value)
    if(has_value) then
     allocate(new_member%data%simple_real)
     call oracle_read(trim(member_path)//'.bin',new_member%data%simple_real)
    endif
    call oracle_read(trim(member_path)//'.logical.present.bin',has_value)
    if(has_value) then
     allocate(new_member%data%simple_logical)
     call oracle_read(trim(member_path)//'.bin',new_member%data%simple_logical)
    endif
    call oracle_read(trim(member_path)//'.character.present.bin',has_value)
    if(has_value) then
     allocate(new_member%data%simple_char)
     call oracle_read(trim(member_path)//'.bin',new_member%data%simple_char)
    endif
   case(MPAS_POOL_FIELD)
    call oracle_read(trim(member_path)//'.type.bin',field_type)
    call oracle_read(trim(member_path)//'.rank.bin',field_rank)
    call oracle_read(trim(member_path)//'.time_levels.bin',time_levels)
    call oracle_read(trim(member_path)//'.field_descriptor.present.bin',has_descriptor)
    call restore_member(pool,trim(names(j)),member_kind,new_member)
    allocate(new_member%data)
    new_member%data%contentsType=field_type
    new_member%data%contentsDims=field_rank
    new_member%data%contentsTimeLevs=time_levels
    if(.not.has_descriptor) cycle
    select case(field_type)
    case(MPAS_POOL_REAL)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       allocate(field_r0_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_r0_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_r0_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_r0_levels(t)%scalar)
        endif
       enddo
       new_member%data%r0a=>field_r0_levels
      else
       allocate(field_r0)
        call oracle_read(trim(member_path)//'.active.bin',field_r0%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_r0%fieldName)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//'.bin',field_r0%scalar)
        endif
       new_member%data%r0=>field_r0
      endif
     case(1)
      if(time_levels>1) then
       allocate(field_r1_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_r1_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_r1_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_r1_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_r1_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r1_levels(t)%array(lb(1):lb(1)+dims(1)-1))
         if(.not.pool_context_set.and.any(field_r1_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_r1_levels(t)%array)
        endif
       enddo
       new_member%data%r1a=>field_r1_levels
      else
       allocate(field_r1)
        call oracle_read(trim(member_path)//'.active.bin',field_r1%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_r1%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_r1%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_r1%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r1%array(lb(1):lb(1)+dims(1)-1))
         if(.not.pool_context_set.and.any(field_r1%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_r1%array)
        endif
       new_member%data%r1=>field_r1
      endif
     case(2)
      if(time_levels>1) then
       allocate(field_r2_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_r2_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_r2_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_r2_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_r2_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r2_levels(t)%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1))
         if(.not.pool_context_set.and.any(field_r2_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_r2_levels(t)%array)
        endif
       enddo
       new_member%data%r2a=>field_r2_levels
      else
       allocate(field_r2)
        call oracle_read(trim(member_path)//'.active.bin',field_r2%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_r2%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_r2%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_r2%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r2%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1))
         if(.not.pool_context_set.and.any(field_r2%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_r2%array)
        endif
       new_member%data%r2=>field_r2
      endif
     case(3)
      if(time_levels>1) then
       allocate(field_r3_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_r3_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_r3_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_r3_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_r3_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r3_levels(t)%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1))
         if(.not.pool_context_set.and.any(field_r3_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_r3_levels(t)%array)
        endif
       enddo
       new_member%data%r3a=>field_r3_levels
      else
       allocate(field_r3)
        call oracle_read(trim(member_path)//'.active.bin',field_r3%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_r3%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_r3%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_r3%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r3%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1))
         if(.not.pool_context_set.and.any(field_r3%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_r3%array)
        endif
       new_member%data%r3=>field_r3
      endif
     case(4)
      if(time_levels>1) then
       allocate(field_r4_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_r4_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_r4_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_r4_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_r4_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r4_levels(t)%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1,lb(4):lb(4)+dims(4)-1))
         if(.not.pool_context_set.and.any(field_r4_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_r4_levels(t)%array)
        endif
       enddo
       new_member%data%r4a=>field_r4_levels
      else
       allocate(field_r4)
        call oracle_read(trim(member_path)//'.active.bin',field_r4%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_r4%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_r4%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_r4%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r4%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1,lb(4):lb(4)+dims(4)-1))
         if(.not.pool_context_set.and.any(field_r4%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_r4%array)
        endif
       new_member%data%r4=>field_r4
      endif
     case(5)
      if(time_levels>1) then
       allocate(field_r5_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_r5_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_r5_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_r5_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_r5_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r5_levels(t)%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1,lb(4):lb(4)+dims(4)-1,lb(5):lb(5)+dims(5)-1))
         if(.not.pool_context_set.and.any(field_r5_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_r5_levels(t)%array)
        endif
       enddo
       new_member%data%r5a=>field_r5_levels
      else
       allocate(field_r5)
        call oracle_read(trim(member_path)//'.active.bin',field_r5%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_r5%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_r5%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_r5%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_r5%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1,lb(4):lb(4)+dims(4)-1,lb(5):lb(5)+dims(5)-1))
         if(.not.pool_context_set.and.any(field_r5%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_r5%array)
        endif
       new_member%data%r5=>field_r5
      endif
     case default
      error stop 'oracle pool restoration unsupported field rank'
     end select
    case(MPAS_POOL_INTEGER)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       allocate(field_i0_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_i0_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_i0_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_i0_levels(t)%scalar)
        endif
       enddo
       new_member%data%i0a=>field_i0_levels
      else
       allocate(field_i0)
        call oracle_read(trim(member_path)//'.active.bin',field_i0%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_i0%fieldName)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//'.bin',field_i0%scalar)
        endif
       new_member%data%i0=>field_i0
      endif
     case(1)
      if(time_levels>1) then
       allocate(field_i1_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_i1_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_i1_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_i1_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_i1_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_i1_levels(t)%array(lb(1):lb(1)+dims(1)-1))
         if(.not.pool_context_set.and.any(field_i1_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_i1_levels(t)%array)
        endif
       enddo
       new_member%data%i1a=>field_i1_levels
      else
       allocate(field_i1)
        call oracle_read(trim(member_path)//'.active.bin',field_i1%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_i1%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_i1%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_i1%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_i1%array(lb(1):lb(1)+dims(1)-1))
         if(.not.pool_context_set.and.any(field_i1%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_i1%array)
        endif
       new_member%data%i1=>field_i1
      endif
     case(2)
      if(time_levels>1) then
       allocate(field_i2_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_i2_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_i2_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_i2_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_i2_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_i2_levels(t)%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1))
         if(.not.pool_context_set.and.any(field_i2_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_i2_levels(t)%array)
        endif
       enddo
       new_member%data%i2a=>field_i2_levels
      else
       allocate(field_i2)
        call oracle_read(trim(member_path)//'.active.bin',field_i2%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_i2%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_i2%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_i2%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_i2%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1))
         if(.not.pool_context_set.and.any(field_i2%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_i2%array)
        endif
       new_member%data%i2=>field_i2
      endif
     case(3)
      if(time_levels>1) then
       allocate(field_i3_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_i3_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_i3_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_i3_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_i3_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_i3_levels(t)%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1))
         if(.not.pool_context_set.and.any(field_i3_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_i3_levels(t)%array)
        endif
       enddo
       new_member%data%i3a=>field_i3_levels
      else
       allocate(field_i3)
        call oracle_read(trim(member_path)//'.active.bin',field_i3%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_i3%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_i3%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_i3%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_i3%array(lb(1):lb(1)+dims(1)-1,lb(2):lb(2)+dims(2)-1,lb(3):lb(3)+dims(3)-1))
         if(.not.pool_context_set.and.any(field_i3%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_i3%array)
        endif
       new_member%data%i3=>field_i3
      endif
     case default
      error stop 'oracle pool restoration unsupported field rank'
     end select
    case(MPAS_POOL_LOGICAL)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       allocate(field_l0_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_l0_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_l0_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_l0_levels(t)%scalar)
        endif
       enddo
       new_member%data%l0a=>field_l0_levels
      else
       allocate(field_l0)
        call oracle_read(trim(member_path)//'.active.bin',field_l0%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_l0%fieldName)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//'.bin',field_l0%scalar)
        endif
       new_member%data%l0=>field_l0
      endif
     case default
      error stop 'oracle pool restoration unsupported field rank'
     end select
    case(MPAS_POOL_CHARACTER)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       allocate(field_c0_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_c0_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_c0_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_c0_levels(t)%scalar)
        endif
       enddo
       new_member%data%c0a=>field_c0_levels
      else
       allocate(field_c0)
        call oracle_read(trim(member_path)//'.active.bin',field_c0%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_c0%fieldName)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_read(trim(member_path)//'.bin',field_c0%scalar)
        endif
       new_member%data%c0=>field_c0
      endif
     case(1)
      if(time_levels>1) then
       allocate(field_c1_levels(time_levels))
       do t=1,time_levels
        write(time_suffix,'(A,I0)') '.t',t
        call oracle_read(trim(member_path)//trim(time_suffix)//'.active.bin',field_c1_levels(t)%isActive)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.field_name.bin',field_c1_levels(t)%fieldName)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_names.bin',field_c1_levels(t)%dimNames)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.dimension_sizes.bin',field_c1_levels(t)%dimSizes)
        call oracle_read(trim(member_path)//trim(time_suffix)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//trim(time_suffix)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_c1_levels(t)%array(lb(1):lb(1)+dims(1)-1))
         if(.not.pool_context_set.and.any(field_c1_levels(t)%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//trim(time_suffix)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//trim(time_suffix)//'.bin',field_c1_levels(t)%array)
        endif
       enddo
       new_member%data%c1a=>field_c1_levels
      else
       allocate(field_c1)
        call oracle_read(trim(member_path)//'.active.bin',field_c1%isActive)
        call oracle_read(trim(member_path)//'.field_name.bin',field_c1%fieldName)
        call oracle_read(trim(member_path)//'.dimension_names.bin',field_c1%dimNames)
        call oracle_read(trim(member_path)//'.dimension_sizes.bin',field_c1%dimSizes)
        call oracle_read(trim(member_path)//'.present.bin',has_value)
        if(has_value) then
         call oracle_header(trim(member_path)//'.bin',code,width,rank_n,dims,lb,unit)
         close(unit)
         allocate(field_c1%array(lb(1):lb(1)+dims(1)-1))
         if(.not.pool_context_set.and.any(field_c1%dimNames=='nCells')) then
          call oracle_restore_context(trim(member_path)//'.bin',context_event)
          pool_context_set=.true.
         endif
         call oracle_read(trim(member_path)//'.bin',field_c1%array)
        endif
       new_member%data%c1=>field_c1
      endif
     case default
      error stop 'oracle pool restoration unsupported field rank'
     end select
    case default
     error stop 'oracle pool restoration unsupported field type'
    end select
   case default
    error stop 'oracle pool restoration unsupported member kind'
   end select
  enddo
  nullify(pool%iterator)
  new_member=>pool%iteration_head
  do j=1,iterator_member
   if(.not.associated(new_member)) error stop 'oracle invalid pool iterator index'
   if(j==iterator_member) pool%iterator=>new_member
   new_member=>new_member%iteration_next
  enddo
  restore_depth=restore_depth-1
 end subroutine
 subroutine restore_member(pool,key,kind,new_member)
  type(mpas_pool_type),intent(inout) :: pool
  character(len=*),intent(in) :: key
  integer,intent(in) :: kind
  type(mpas_pool_member_type),pointer :: new_member
  allocate(new_member)
  new_member%key=key
  new_member%keyLen=len_trim(key)
  new_member%contentsType=kind
  if(.not.pool_add_member(pool,key,new_member)) error stop 'oracle duplicate restored pool member'
 end subroutine

 ! RRFS fork src/framework/mpas_pool_types.inc:23-98 and
 ! mpas_pool_routines.F:169-191,5423-5430,5694-5709 at 1d6e370e.
 recursive subroutine oracle_pool_restore_in_place(prefix,pool)
  character(len=*),intent(in) :: prefix
  type(mpas_pool_type),intent(inout) :: pool
  type(mpas_pool_member_type),pointer :: member,saved_member
  character(len=256),allocatable :: names(:)
  character(len=4096) :: member_path,value_path
  character(len=24) :: time_suffix
  integer :: pool_size,iterator_member,j,t,kind,field_type,field_rank,time_levels
  integer :: code,width,rank_n,dims(5),lb(5),unit,context_event
  logical :: has_descriptor,has_value
  if(restore_depth==0) pool_context_set=.false.
  restore_depth=restore_depth+1
  call oracle_read(trim(prefix)//'.pool_size.bin',pool_size)
  if(pool%size/=pool_size) error stop 'oracle initialized pool size mismatch'
  call oracle_read(trim(prefix)//'.iterator_member.bin',iterator_member)
  call oracle_header(trim(prefix)//'.members.bin',code,width,rank_n,dims,lb,unit)
  close(unit)
  if(rank_n/=1.or.dims(1)<0) error stop 'oracle invalid pool member extent'
  allocate(names(dims(1)))
  call oracle_read(trim(prefix)//'.members.bin',names)
  if(iterator_member<0.or.iterator_member>size(names)) error stop 'oracle invalid pool iterator'
  member=>pool%iteration_head
  nullify(saved_member)
  do j=1,size(names)
   if(.not.associated(member)) error stop 'oracle initialized pool missing member'
   if(member%key/=names(j)) error stop 'oracle initialized pool member order mismatch'
   if(j==iterator_member) saved_member=>member
   member_path=trim(prefix)//'.'//trim(names(j))
   call oracle_read(trim(member_path)//'.member_kind.bin',kind)
   if(kind/=member%contentsType) error stop 'oracle initialized pool member kind mismatch'
   call oracle_read(trim(member_path)//'.descriptor.present.bin',has_descriptor)
   call require_association(has_descriptor,associated(member%data))
   if(.not.has_descriptor) then
    member=>member%iteration_next
    cycle
   endif
   select case(kind)
   case(MPAS_POOL_SUBPOOL)
    call oracle_read(trim(member_path)//'.subpool.present.bin',has_value)
    call require_association(has_value,associated(member%data%p))
    if(has_value) call oracle_pool_restore_in_place(trim(member_path),member%data%p)
   case(MPAS_POOL_CONFIG,MPAS_POOL_PACKAGE,MPAS_POOL_DIMENSION)
    call oracle_read(trim(member_path)//'.type.bin',field_type)
    call oracle_read(trim(member_path)//'.rank.bin',field_rank)
    call oracle_read(trim(member_path)//'.time_levels.bin',time_levels)
    if(field_type/=member%data%contentsType.or.field_rank/=member%data%contentsDims.or. &
       time_levels/=member%data%contentsTimeLevs) error stop 'oracle initialized simple descriptor mismatch'
    call oracle_read(trim(member_path)//'.integer.present.bin',has_value)
    call require_association(has_value,associated(member%data%simple_int))
    if(has_value) then
     call oracle_read(trim(member_path)//'.bin',member%data%simple_int)
    endif
    call oracle_read(trim(member_path)//'.integer_array.present.bin',has_value)
    call require_association(has_value,associated(member%data%simple_int_arr))
    if(has_value) then
     call require_bounds(trim(member_path)//'.bin',shape(member%data%simple_int_arr),lbound(member%data%simple_int_arr))
     call oracle_read(trim(member_path)//'.bin',member%data%simple_int_arr)
    endif
    call oracle_read(trim(member_path)//'.real.present.bin',has_value)
    call require_association(has_value,associated(member%data%simple_real))
    if(has_value) then
     call oracle_read(trim(member_path)//'.bin',member%data%simple_real)
    endif
    call oracle_read(trim(member_path)//'.logical.present.bin',has_value)
    call require_association(has_value,associated(member%data%simple_logical))
    if(has_value) then
     call oracle_read(trim(member_path)//'.bin',member%data%simple_logical)
    endif
    call oracle_read(trim(member_path)//'.character.present.bin',has_value)
    call require_association(has_value,associated(member%data%simple_char))
    if(has_value) then
     call oracle_read(trim(member_path)//'.bin',member%data%simple_char)
    endif
   case(MPAS_POOL_FIELD)
    call oracle_read(trim(member_path)//'.type.bin',field_type)
    call oracle_read(trim(member_path)//'.rank.bin',field_rank)
    call oracle_read(trim(member_path)//'.time_levels.bin',time_levels)
    if(field_type/=member%data%contentsType.or.field_rank/=member%data%contentsDims.or. &
       time_levels/=member%data%contentsTimeLevs) error stop 'oracle initialized field descriptor mismatch'
    call oracle_read(trim(member_path)//'.field_descriptor.present.bin',has_descriptor)
    select case(field_type)
    case(MPAS_POOL_REAL)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%r0a))
       if(has_descriptor) then
        if(size(member%data%r0a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%r0a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r0a(t)%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%r0a(t)%scalar)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%r0))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%r0%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r0%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%r0%scalar)
         endif
       endif
      endif
     case(1)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%r1a))
       if(has_descriptor) then
        if(size(member%data%r1a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%r1a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r1a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r1a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r1a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r1a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r1a(t)%array),lbound(member%data%r1a(t)%array))
          if(.not.pool_context_set.and.any(member%data%r1a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r1a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%r1))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%r1%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r1%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r1%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r1%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r1%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r1%array),lbound(member%data%r1%array))
          if(.not.pool_context_set.and.any(member%data%r1%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r1%array)
         endif
       endif
      endif
     case(2)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%r2a))
       if(has_descriptor) then
        if(size(member%data%r2a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%r2a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r2a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r2a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r2a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r2a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r2a(t)%array),lbound(member%data%r2a(t)%array))
          if(.not.pool_context_set.and.any(member%data%r2a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r2a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%r2))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%r2%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r2%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r2%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r2%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r2%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r2%array),lbound(member%data%r2%array))
          if(.not.pool_context_set.and.any(member%data%r2%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r2%array)
         endif
       endif
      endif
     case(3)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%r3a))
       if(has_descriptor) then
        if(size(member%data%r3a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%r3a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r3a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r3a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r3a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r3a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r3a(t)%array),lbound(member%data%r3a(t)%array))
          if(.not.pool_context_set.and.any(member%data%r3a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r3a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%r3))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%r3%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r3%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r3%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r3%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r3%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r3%array),lbound(member%data%r3%array))
          if(.not.pool_context_set.and.any(member%data%r3%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r3%array)
         endif
       endif
      endif
     case(4)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%r4a))
       if(has_descriptor) then
        if(size(member%data%r4a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%r4a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r4a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r4a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r4a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r4a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r4a(t)%array),lbound(member%data%r4a(t)%array))
          if(.not.pool_context_set.and.any(member%data%r4a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r4a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%r4))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%r4%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r4%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r4%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r4%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r4%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r4%array),lbound(member%data%r4%array))
          if(.not.pool_context_set.and.any(member%data%r4%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r4%array)
         endif
       endif
      endif
     case(5)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%r5a))
       if(has_descriptor) then
        if(size(member%data%r5a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%r5a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r5a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r5a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r5a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r5a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r5a(t)%array),lbound(member%data%r5a(t)%array))
          if(.not.pool_context_set.and.any(member%data%r5a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r5a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%r5))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%r5%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%r5%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%r5%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%r5%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%r5%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%r5%array),lbound(member%data%r5%array))
          if(.not.pool_context_set.and.any(member%data%r5%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%r5%array)
         endif
       endif
      endif
     case default
      error stop 'oracle context unsupported field rank'
     end select
    case(MPAS_POOL_INTEGER)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%i0a))
       if(has_descriptor) then
        if(size(member%data%i0a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%i0a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i0a(t)%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%i0a(t)%scalar)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%i0))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%i0%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i0%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%i0%scalar)
         endif
       endif
      endif
     case(1)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%i1a))
       if(has_descriptor) then
        if(size(member%data%i1a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%i1a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i1a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%i1a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%i1a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%i1a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%i1a(t)%array),lbound(member%data%i1a(t)%array))
          if(.not.pool_context_set.and.any(member%data%i1a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%i1a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%i1))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%i1%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i1%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%i1%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%i1%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%i1%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%i1%array),lbound(member%data%i1%array))
          if(.not.pool_context_set.and.any(member%data%i1%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%i1%array)
         endif
       endif
      endif
     case(2)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%i2a))
       if(has_descriptor) then
        if(size(member%data%i2a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%i2a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i2a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%i2a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%i2a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%i2a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%i2a(t)%array),lbound(member%data%i2a(t)%array))
          if(.not.pool_context_set.and.any(member%data%i2a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%i2a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%i2))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%i2%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i2%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%i2%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%i2%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%i2%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%i2%array),lbound(member%data%i2%array))
          if(.not.pool_context_set.and.any(member%data%i2%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%i2%array)
         endif
       endif
      endif
     case(3)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%i3a))
       if(has_descriptor) then
        if(size(member%data%i3a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%i3a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i3a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%i3a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%i3a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%i3a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%i3a(t)%array),lbound(member%data%i3a(t)%array))
          if(.not.pool_context_set.and.any(member%data%i3a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%i3a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%i3))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%i3%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%i3%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%i3%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%i3%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%i3%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%i3%array),lbound(member%data%i3%array))
          if(.not.pool_context_set.and.any(member%data%i3%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%i3%array)
         endif
       endif
      endif
     case default
      error stop 'oracle context unsupported field rank'
     end select
    case(MPAS_POOL_LOGICAL)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%l0a))
       if(has_descriptor) then
        if(size(member%data%l0a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%l0a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%l0a(t)%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%l0a(t)%scalar)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%l0))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%l0%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%l0%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%l0%scalar)
         endif
       endif
      endif
     case default
      error stop 'oracle context unsupported field rank'
     end select
    case(MPAS_POOL_CHARACTER)
     select case(field_rank)
     case(0)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%c0a))
       if(has_descriptor) then
        if(size(member%data%c0a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%c0a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%c0a(t)%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%c0a(t)%scalar)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%c0))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%c0%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%c0%fieldName)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,.true.)
         if(has_value) then
          call oracle_read(trim(value_path)//'.bin',member%data%c0%scalar)
         endif
       endif
      endif
     case(1)
      if(time_levels>1) then
       call require_association(has_descriptor,associated(member%data%c1a))
       if(has_descriptor) then
        if(size(member%data%c1a)/=time_levels) error stop 'oracle context time-level extent mismatch'
        do t=1,time_levels
         write(time_suffix,'(A,I0)') '.t',t
         value_path=trim(member_path)//trim(time_suffix)
         call oracle_read(trim(value_path)//'.active.bin',member%data%c1a(t)%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%c1a(t)%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%c1a(t)%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%c1a(t)%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%c1a(t)%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%c1a(t)%array),lbound(member%data%c1a(t)%array))
          if(.not.pool_context_set.and.any(member%data%c1a(t)%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%c1a(t)%array)
         endif
        enddo
       endif
      else
       call require_association(has_descriptor,associated(member%data%c1))
       if(has_descriptor) then
        value_path=member_path
         call oracle_read(trim(value_path)//'.active.bin',member%data%c1%isActive)
         call oracle_read(trim(value_path)//'.field_name.bin',member%data%c1%fieldName)
         call oracle_read(trim(value_path)//'.dimension_names.bin',member%data%c1%dimNames)
         call oracle_read(trim(value_path)//'.dimension_sizes.bin',member%data%c1%dimSizes)
         call oracle_read(trim(value_path)//'.present.bin',has_value)
         call require_association(has_value,associated(member%data%c1%array))
         if(has_value) then
          call require_bounds(trim(value_path)//'.bin',shape(member%data%c1%array),lbound(member%data%c1%array))
          if(.not.pool_context_set.and.any(member%data%c1%dimNames=='nCells')) then
           call oracle_restore_context(trim(value_path)//'.bin',context_event)
           pool_context_set=.true.
          endif
          call oracle_read(trim(value_path)//'.bin',member%data%c1%array)
         endif
       endif
      endif
     case default
      error stop 'oracle context unsupported field rank'
     end select
    case default
     error stop 'oracle context unsupported field type'
    end select
   case default
    error stop 'oracle context unsupported member kind'
   end select
   member=>member%iteration_next
  enddo
  if(associated(member)) error stop 'oracle initialized pool has extra members'
  pool%iterator=>saved_member
  restore_depth=restore_depth-1
 end subroutine
 subroutine require_association(captured,native)
  logical,intent(in) :: captured,native
  if(captured.neqv.native) error stop 'oracle initialized context association mismatch'
 end subroutine
 subroutine require_bounds(path,extents,lower)
  character(len=*),intent(in) :: path
  integer,intent(in) :: extents(:),lower(:)
  integer :: code,width,rank_n,dims(5),lb(5),unit
  call oracle_header(path,code,width,rank_n,dims,lb,unit)
  close(unit)
  if(rank_n/=size(extents)) error stop 'oracle initialized context rank mismatch'
  if(any(dims(:rank_n)/=extents).or.any(lb(:rank_n)/=lower)) &
   error stop 'oracle initialized context bounds mismatch'
 end subroutine
end module mpas_oracle_pool_restore
