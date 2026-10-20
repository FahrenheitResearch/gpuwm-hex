"""Generate native pool reconstruction for captured driver helper arguments.

MPAS framework/mpas_pool_routines.F:169-191,2108-3195,4540-5216.
Uses MPAS's own insertion APIs, preserving captured iteration order and bounds.
Fresh reconstruction needs separate alias/I/O closure proof. In-place replay
uses an initialized native context and retains its storage and ancillary graph.
"""
from pathlib import Path


def generate_in_place():
    """Restore all supported native members without allocating target storage.

    Descriptor/extent mismatches mean this is the wrong initialized context;
    they must fail before writing through a pointer with incompatible bounds.
    This is a replay validation, never an added native forecast refusal.
    """
    field_cases=[]
    for native,tag,maximum in [('Real','r',5),('Integer','i',3),('Logical','l',0),('Char','c',1)]:
        enum={'Real':'REAL','Integer':'INTEGER','Logical':'LOGICAL','Char':'CHARACTER'}[native]
        ranks=[]
        for rank in range(maximum+1):
            bodies=[]
            for levels in (True,False):
                obj=f'member%data%{tag}{rank}'+('a(t)' if levels else '')
                descriptor=f'member%data%{tag}{rank}'+('a' if levels else '')
                if levels:
                    bodies += ['      if(time_levels>1) then',
                               f'       call require_association(has_descriptor,associated({descriptor}))',
                               '       if(has_descriptor) then',
                               f"        if(size({descriptor})/=time_levels) error stop 'oracle context time-level extent mismatch'",
                               '        do t=1,time_levels',
                               "         write(time_suffix,'(A,I0)') '.t',t",
                               '         value_path=trim(member_path)//trim(time_suffix)']
                else:
                    bodies += ['        enddo','       endif','      else',
                               f'       call require_association(has_descriptor,associated({descriptor}))',
                               '       if(has_descriptor) then','        value_path=member_path']
                bodies += [f"         call oracle_read(trim(value_path)//'.active.bin',{obj}%isActive)",
                           f"         call oracle_read(trim(value_path)//'.field_name.bin',{obj}%fieldName)"]
                if rank:
                    bodies += [f"         call oracle_read(trim(value_path)//'.dimension_names.bin',{obj}%dimNames)",
                               f"         call oracle_read(trim(value_path)//'.dimension_sizes.bin',{obj}%dimSizes)"]
                bodies += ["         call oracle_read(trim(value_path)//'.present.bin',has_value)"]
                value=obj+('%array' if rank else '%scalar')
                bodies += [f'         call require_association(has_value,associated({value}))' if rank else
                           '         call require_association(has_value,.true.)',
                           '         if(has_value) then']
                if rank:
                    bodies += [f"          call require_bounds(trim(value_path)//'.bin',shape({value}),lbound({value}))",
                               f"          if(.not.pool_context_set.and.any({obj}%dimNames=='nCells')) then",
                               "           call oracle_restore_context(trim(value_path)//'.bin',context_event)",
                               '           pool_context_set=.true.','          endif']
                bodies += [f"          call oracle_read(trim(value_path)//'.bin',{value})",'         endif']
            bodies += ['       endif','      endif']
            ranks.append(f'     case({rank})\n'+'\n'.join(bodies))
        field_cases.append(f'    case(MPAS_POOL_{enum})\n     select case(field_rank)\n'+'\n'.join(ranks)+
                           "\n     case default\n      error stop 'oracle context unsupported field rank'\n     end select")
    simple=[]
    for name,component in [('integer','simple_int'),('integer_array','simple_int_arr'),('real','simple_real'),
                           ('logical','simple_logical'),('character','simple_char')]:
        value='member%data%'+component
        simple += [f"    call oracle_read(trim(member_path)//'.{name}.present.bin',has_value)",
                   f'    call require_association(has_value,associated({value}))','    if(has_value) then']
        if name=='integer_array':
            simple += [f"     call require_bounds(trim(member_path)//'.bin',shape({value}),lbound({value}))"]
        simple += [f"     call oracle_read(trim(member_path)//'.bin',{value})",'    endif']
    return '''
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
'''+ '\n'.join(simple)+'''
   case(MPAS_POOL_FIELD)
    call oracle_read(trim(member_path)//'.type.bin',field_type)
    call oracle_read(trim(member_path)//'.rank.bin',field_rank)
    call oracle_read(trim(member_path)//'.time_levels.bin',time_levels)
    if(field_type/=member%data%contentsType.or.field_rank/=member%data%contentsDims.or. &
       time_levels/=member%data%contentsTimeLevs) error stop 'oracle initialized field descriptor mismatch'
    call oracle_read(trim(member_path)//'.field_descriptor.present.bin',has_descriptor)
    select case(field_type)
'''+ '\n'.join(field_cases)+'''
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
'''


def generate():
    declarations=[];cases=[]
    for native,tag,maximum in [('Real','r',5),('Integer','i',3),('Logical','l',0),('Char','c',1)]:
        enum={'Real':'REAL','Integer':'INTEGER','Logical':'LOGICAL','Char':'CHARACTER'}[native]
        ranks=[]
        for rank in range(maximum+1):
            typ=f'field{rank}D{native}';name=f'field_{tag}{rank}'
            declarations.extend([f'  type({typ}),pointer :: {name}',f'  type({typ}),pointer :: {name}_levels(:)'])
            restore=[]
            for levels in [True,False]:
                obj=name+'_levels(t)' if levels else name
                label="trim(member_path)//trim(time_suffix)" if levels else 'trim(member_path)'
                if levels:
                    restore.append(f'      if(time_levels>1) then\n       allocate({name}_levels(time_levels))\n       do t=1,time_levels\n        write(time_suffix,\'(A,I0)\') \'.t\',t')
                else:restore.append(f'       enddo\n       new_member%data%{tag}{rank}a=>{name}_levels\n      else\n       allocate({name})')
                restore.extend([f'        call oracle_read({label}//\'.active.bin\',{obj}%isActive)',
                                f'        call oracle_read({label}//\'.field_name.bin\',{obj}%fieldName)'])
                if rank:
                    restore.extend([f'        call oracle_read({label}//\'.dimension_names.bin\',{obj}%dimNames)',
                                    f'        call oracle_read({label}//\'.dimension_sizes.bin\',{obj}%dimSizes)'])
                restore.append(f'        call oracle_read({label}//\'.present.bin\',has_value)\n        if(has_value) then')
                value=obj+('%array' if rank else '%scalar')
                if rank:
                    restore.extend([f'         call oracle_header({label}//\'.bin\',code,width,rank_n,dims,lb,unit)\n         close(unit)',
                                    f"         allocate({value}({','.join(f'lb({i}):lb({i})+dims({i})-1' for i in range(1,rank+1))}))"])
                if rank:
                    restore.append(f"         if(.not.pool_context_set.and.any({obj}%dimNames=='nCells')) then\n          call oracle_restore_context({label}//'.bin',context_event)\n          pool_context_set=.true.\n         endif")
                restore.append(f'         call oracle_read({label}//\'.bin\',{value})\n        endif')
                if not levels:restore.append(f'       new_member%data%{tag}{rank}=>{name}\n      endif')
            ranks.append(f'     case({rank})\n'+ '\n'.join(restore))
        cases.append(f'    case(MPAS_POOL_{enum})\n     select case(field_rank)\n'+ '\n'.join(ranks)+"\n     case default\n      error stop 'oracle pool restoration unsupported field rank'\n     end select")
    return '''module mpas_oracle_pool_restore
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
'''+ '\n'.join(declarations)+'''
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
'''+ '\n'.join(cases)+'''
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
'''+generate_in_place()+'''end module mpas_oracle_pool_restore
'''


if __name__=='__main__':
    Path(__file__).with_name('mpas_oracle_pool_restore.F').write_text(generate())
