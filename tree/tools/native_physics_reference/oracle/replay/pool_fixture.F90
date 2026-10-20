! Native framework/mpas_pool_routines.F:169-191,2108-3195,4540-5216.
program pool_fixture
 use mpas_oracle_pool
 use mpas_oracle_pool_restore
 use mpas_pool_routines
 use mpas_oracle_dump
 implicit none
 type(mpas_pool_type),pointer :: source,restored,child
 type(mpas_pool_member_type),pointer :: node,saved_iterator
 type(field2DReal),pointer :: levels(:)
 type(field2DReal),pointer :: alias
 type(field1DReal),pointer :: inactive
 type(field1DInteger),pointer :: absent
 type(field1DChar),pointer :: labels
 type(field0DInteger),pointer :: counter
 type(field0DLogical),pointer :: condition
 character(len=4096) :: original,target,in_place,mode
 integer :: t
 integer,pointer :: integer_option
 call get_command_argument(1,original)
 call get_command_argument(2,target)
 call get_command_argument(3,in_place)
 call get_command_argument(4,mode)
 call mpas_pool_create_pool(source,37)
 call mpas_pool_add_config(source,'integer_option',17)
 call mpas_pool_add_config(source,'real_option',1.5_RKIND)
 call mpas_pool_add_config(source,'logical_option',.true.)
 call mpas_pool_add_config(source,'character_option','cold')
 call mpas_pool_add_dimension(source,'nCells',2)
 call mpas_pool_add_dimension(source,'ranges',[3,4])
 call mpas_pool_add_package(source,'enabled',.true.)
 call mpas_pool_create_pool(child,19)
 call mpas_pool_add_config(child,'child_option',23)
 call mpas_pool_add_subpool(source,'child',child)
 allocate(levels(2))
 do t=1,2
  levels(t)%fieldName='profiles'
  levels(t)%dimNames=['nVertLevels','nCells     ']
  levels(t)%dimSizes=[2,2]
  levels(t)%isActive=.true.
  allocate(levels(t)%array(-1:0,3:4))
  levels(t)%array=real(t,RKIND)
  levels(t)%array(-1,3)=-0.0_RKIND
 enddo
 call mpas_pool_add_field(source,'profiles',levels)
 ! Real native storage shared by two distinct pools; replay in place must
 ! retain both the descriptor identity and the array pointer association.
 call mpas_pool_add_field(child,'profiles_alias',levels)
 allocate(inactive)
 inactive%fieldName='inactive'
 inactive%dimNames='nCells'
 inactive%dimSizes=2
 inactive%isActive=.false.
 allocate(inactive%array(3:4))
 inactive%array=[-0.0_RKIND,2.5_RKIND]
 call mpas_pool_add_field(source,'inactive',inactive)
 nullify(absent)
 call mpas_pool_add_field(source,'absent',absent)
 allocate(labels)
 labels%fieldName='labels'
 labels%dimNames='nCells'
 labels%dimSizes=2
 labels%isActive=.true.
 allocate(labels%array(-1:0))
 labels%array=['first ','second']
 call mpas_pool_add_field(source,'labels',labels)
 allocate(counter)
 counter%fieldName='counter'
 counter%isActive=.true.
 counter%scalar=127
 call mpas_pool_add_field(source,'counter',counter)
 ! The native framework declares logical fields but has no typed add interface.
 ! Its actual generic member insertion supports the native descriptor directly.
 allocate(condition,node)
 allocate(node%data)
 condition%fieldName='condition'
 condition%isActive=.true.
 condition%scalar=.false.
 node%key='condition';node%keyLen=9;node%contentsType=MPAS_POOL_FIELD
 node%data%contentsType=MPAS_POOL_LOGICAL
 node%data%contentsDims=0;node%data%contentsTimeLevs=1
 node%data%l0=>condition
 if(.not.pool_add_member(source,'condition',node)) stop 1
 allocate(node)
 node%key='missing_descriptor';node%keyLen=18;node%contentsType=MPAS_POOL_FIELD
 if(.not.pool_add_member(source,'missing_descriptor',node)) stop 2
 allocate(node)
 allocate(node%data)
 node%key='missing_config_value';node%keyLen=20;node%contentsType=MPAS_POOL_CONFIG
 node%data%contentsType=MPAS_POOL_REAL
 node%data%contentsDims=0;node%data%contentsTimeLevs=0
 if(.not.pool_add_member(source,trim(node%key),node)) stop 4
 allocate(node)
 allocate(node%data)
 node%key='offset_dimension';node%keyLen=16;node%contentsType=MPAS_POOL_DIMENSION
 node%data%contentsType=MPAS_POOL_INTEGER
 node%data%contentsDims=1;node%data%contentsTimeLevs=0
 allocate(node%data%simple_int_arr(-2:-1))
 node%data%simple_int_arr=[9,11]
 if(.not.pool_add_member(source,trim(node%key),node)) stop 5
 source%iterator=>source%iteration_head
 saved_iterator=>source%iterator
 call oracle_set_context([101,205,309,413],1,0)
 call oracle_set_output(trim(original))
 call oracle_pool_capture('pool_test','pre','p',source,1)
 call oracle_pool_capture('pool_test','post','p',source,1)
 if(.not.associated(source%iterator,saved_iterator)) stop 3
 call oracle_pool_restore(trim(original)//'.pre.p',restored)
 call oracle_set_output(trim(target))
 call oracle_pool_capture('pool_test','post','p',restored,1)
 if(len_trim(in_place)>0) then
  levels(1)%array=93.0_RKIND
  levels(2)%array=-17.0_RKIND
  inactive%array=11.0_RKIND
  condition%scalar=.true.
  counter%scalar=-21
  labels%array='wrong'
  call mpas_pool_get_config(source,'integer_option',integer_option)
  integer_option=-999
  nullify(source%iterator)
  if(trim(mode)=='bad-bounds') then
   deallocate(inactive%array)
   allocate(inactive%array(2:3))
   inactive%array=11.0_RKIND
  endif
  call oracle_pool_restore_in_place(trim(original)//'.pre.p',source)
  if(.not.associated(source%iterator,saved_iterator)) error stop 'native iterator alias lost'
  call mpas_pool_get_field(child,'profiles_alias',alias,1)
  if(.not.associated(alias,levels(1))) error stop 'native cross-pool descriptor alias lost'
  if(.not.associated(alias%array,levels(1)%array)) error stop 'native cross-pool array alias lost'
  call oracle_set_output(trim(in_place))
  call oracle_pool_capture('pool_test','post','p',source,1)
  alias%array(-1,4)=19.0_RKIND
  if(levels(1)%array(-1,4)/=19.0_RKIND) error stop 'restored alias writes lost'
 endif
end program
