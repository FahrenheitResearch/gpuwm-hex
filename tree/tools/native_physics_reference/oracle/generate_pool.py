#!/usr/bin/env python3
"""Generate non-mutating MPAS pool traversal for boundary snapshots.

MPAS src/framework/mpas_pool_types.inc:23-112 and mpas_field_types.inc.
Uses the linked member list directly so dumping cannot move the pool iterator.
"""
from pathlib import Path


def generate():
    out = '''module mpas_oracle_pool
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
'''
    for type_name, tag, maximum in [('REAL','r',5),('INTEGER','i',3),('LOGICAL','l',0),('CHARACTER','c',1)]:
        out += f'    case(MPAS_POOL_{type_name})\n     select case(data%contentsDims)\n'
        for rank in range(maximum+1):
            out += f'     case({rank})\n      if(data%contentsTimeLevs>1) then\n'
            for array in [True,False]:
                obj=f'data%{tag}{rank}'+ ('a(t)' if array else '')
                if array:
                    out += f"       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated(data%{tag}{rank}a),event)\n"
                    out += f'       if(associated(data%{tag}{rank}a)) then\n        do t=1,size(data%{tag}{rank}a)\n'
                    out += "          write(lev,'(A,I0)') '.t',t\n"
                else:
                    out += f"      else\n       call oracle_write(key,phase,trim(name)//'.field_descriptor.present',associated({obj}),event)\n       if(associated({obj})) then\n"
                name='trim(name)//trim(lev)' if array else 'trim(name)'
                out += f"          call oracle_write(key,phase,{name}//'.active',{obj}%isActive,event)\n"
                out += f"          call oracle_write(key,phase,{name}//'.field_name',{obj}%fieldName,event)\n"
                if rank:
                    out += f"          call oracle_write(key,phase,{name}//'.dimension_names',{obj}%dimNames,event)\n"
                    out += f"          call oracle_write(key,phase,{name}//'.dimension_sizes',{obj}%dimSizes,event)\n"
                out += '          field_present=.true.\n'
                if rank:out += f'          field_present=associated({obj}%array)\n'
                out += f"          call oracle_write(key,phase,{name}//'.present',field_present,event)\n"
                out += '         if(field_present) then\n'
                out += "          ca=0; la=0\n"
                if rank:
                    out += f'''          do j=1,{rank}
           if(trim({obj}%dimNames(j))=='nCells') ca=j
           if(trim({obj}%dimNames(j))=='nVertLevels' .or. trim({obj}%dimNames(j))=='nVertLevelsP1') la=j
          enddo
'''
                var=obj+('%array' if rank else '%scalar')
                lower=f', lower=lbound({var})' if rank else ''
                out += f'          call oracle_write(key,phase,{name}, &\n               {var},event{lower},cell_axis=ca,level_axis=la)\n'
                out += '         endif\n'
                if array:
                    out += '        enddo\n       endif\n'
                else:
                    out += '       endif\n      endif\n'
        out += "     case default\n      error stop 'oracle pool field rank unsupported'\n     end select\n"
    out += "    case default\n     error stop 'oracle pool field type unsupported'\n    end select\n   end select\n   member=>member%iteration_next\n  enddo\n end subroutine\nend module\n"
    return out


if __name__=='__main__':
    Path(__file__).with_name('mpas_oracle_pool.F').write_text(generate())
