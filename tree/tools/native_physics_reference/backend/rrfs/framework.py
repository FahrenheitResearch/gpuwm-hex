"""Generate native field metadata/stream registration, without field arithmetic."""
from generate import write_changed


def generate_framework(output):
    f = ['! Native field registration; MPAS source: MPAS-Model-rrfs/src/framework/mpas_pool_routines.F:2110-3143.',
         'module hex_native_framework', 'use iso_c_binding', 'use hex_native_runtime',
         'use mpas_pool_routines', 'use mpas_stream_manager',
         'use mpas_dmpar, only: mpas_dmpar_exch_halo_field,mpas_dmpar_exch_group_create,mpas_dmpar_exch_group_add_field,mpas_dmpar_exch_group_full_halo_exch',
         'implicit none', 'contains', '''
function member(pool,key) result(found)
 type(mpas_pool_type),intent(in) :: pool
 character(len=*),intent(in) :: key
 type(mpas_pool_member_type),pointer :: found
 found=>pool%iteration_head
 do while(associated(found))
  if(found%contentsType==MPAS_POOL_FIELD.and.trim(found%key)==trim(key)) return
  found=>found%iteration_next
 enddo
end function
''']
    for kind, native, ranks in [('real','Real',range(6)), ('integer','Integer',range(4)), ('character','Char',range(1))]:
        for rank in ranks:
            f += [f'subroutine native_runtime_field_{kind}_{rank}(handle,pool,key,registry,dimensions,decomposed,ncomponents,components,has_time,status) bind(C)',
                  'type(c_ptr),value :: handle,pool', 'character(c_char),intent(in) :: key(*)',
                  'character(c_char),intent(in) :: registry(*)',
                  f'character(c_char),intent(in) :: dimensions(512,{rank})',
                  'integer(c_int),value :: ncomponents',
                  'character(c_char),intent(in) :: components(512,ncomponents)',
                  'logical(c_bool),value :: has_time',
                  'logical(c_bool),value :: decomposed', 'integer(c_int),intent(out) :: status',
                  'type(native_runtime_type),pointer :: runtime', 'type(mpas_pool_type),pointer :: source',
                  f'type(field{rank}D{native}),pointer :: field',
                  'type(mpas_pool_member_type),pointer :: original,alias',
                  'integer :: level,i,element', 'call c_f_pointer(handle,runtime)',
                  'call c_f_pointer(pool,source)', 'status=-1', 'if(.not.runtime%has_block) return',
                  'original=>member(source,trim(string(key)))', 'if(.not.associated(original)) return',
                  'alias=>member(runtime%fields,trim(string(registry)))',
                  'if(associated(alias)) then', ' status=-2', ' return', 'endif', 'element=0']
            if rank:
                f += [f"select case(trim(string(dimensions(:,{rank}))))", "case('nCells')", ' element=1',
                      "case('nEdges')", ' element=2', "case('nVertices')", ' element=3', 'end select',
                      'if(decomposed) then', ' if(element==0) return',
                      ' if(.not.runtime%exchange(element)%ready) return', 'endif']
            else:
                f += ['if(decomposed) return']
            f += ['do level=1,original%data%contentsTimeLevs',
                  ' call mpas_pool_get_field(source,trim(string(key)),field,timeLevel=level)',
                  ' field%block=>runtime%domain%blocklist', ' field%isDecomposed=logical(decomposed)',
                  ' field%fieldName=string(registry)',
                  ' field%hasTimeDimension=logical(has_time)',
                  ' field%isVarArray=ncomponents>0',
                  ' if(ncomponents>0) then',
                  '  allocate(field%constituentNames(ncomponents))',
                  '  do i=1,ncomponents',
                  '   field%constituentNames(i)=string(components(:,i))',
                  '  enddo', ' endif',
                  '! Native stream registration dereferences attLists(1), even with no attributes.',
                  '! MPAS source: MPAS-Model-rrfs/src/framework/mpas_io_streams.F:972.',
                  ' if(.not.associated(field%attLists)) allocate(field%attLists(max(1,ncomponents)))']
            if rank:
                f += [' field%isPersistent=.true.', f' do i=1,{rank}', '  field%dimNames(i)=string(dimensions(:,i))', ' enddo',
                      ' if(decomposed) then', '  field%sendList=>runtime%exchange(element)%send',
                      '  field%recvList=>runtime%exchange(element)%recv',
                      '  field%copyList=>runtime%exchange(element)%copy', ' endif']
            f += ['enddo',
                  'call mpas_pool_get_field(source,trim(string(key)),field,timeLevel=1)',
                  'call mpas_pool_add_field(runtime%fields,trim(string(registry)),field)',
                  'alias=>member(runtime%fields,trim(string(registry)))',
                  '! Copy native metadata pointers, retaining all separately owned time levels.',
                  'alias%data=original%data', 'status=0', 'end subroutine']
            if rank and kind != 'character':
                f += ['! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:4787-6256.',
                      f'subroutine native_runtime_halo_{kind}_{rank}(handle,key,level,status) bind(C)',
                      'type(c_ptr),value :: handle', 'character(c_char),intent(in) :: key(*)',
                      'integer(c_int),value :: level', 'integer(c_int),intent(out) :: status',
                      'type(native_runtime_type),pointer :: runtime',
                      f'type(field{rank}D{native}),pointer :: field',
                      'call c_f_pointer(handle,runtime)', 'status=-1',
                      'if(.not.runtime%has_block) return',
                      'call mpas_pool_get_field(runtime%fields,trim(string(key)),field,timeLevel=level)',
                      'if(.not.field%isDecomposed) return',
                      'call mpas_dmpar_exch_halo_field(field)', 'status=0', 'end subroutine']
    f += ['''
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:6852-6902,7008-7146.
subroutine native_runtime_group(handle,key,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: key(*)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_block) return
 call mpas_dmpar_exch_group_create(runtime%domain,trim(string(key)),iErr=status)
end subroutine
subroutine native_runtime_group_field(handle,group,key,level,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: group(*),key(*)
 integer(c_int),value :: level
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_block) return
 call mpas_dmpar_exch_group_add_field(runtime%domain,trim(string(group)),trim(string(key)),timeLevel=level,iErr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:7010-7192.
subroutine native_runtime_group_field_layers(handle,group,key,level,nlayers,layers,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: group(*),key(*)
 integer(c_int),value :: level,nlayers
 integer(c_int),intent(in) :: layers(nlayers)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_block.or.nlayers<1.or.any(layers<1)) return
 call mpas_dmpar_exch_group_add_field(runtime%domain,trim(string(group)),trim(string(key)), &
     timeLevel=level,haloLayers=layers,iErr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_dmpar.F:7452-7502.
subroutine native_runtime_group_exchange(handle,group,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: group(*)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_block) return
 call mpas_dmpar_exch_group_full_halo_exch(runtime%domain,trim(string(group)),iErr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:843-939.
subroutine native_runtime_stream_field(handle,stream,key,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: stream(*),key(*)
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_streams) return
 call MPAS_stream_mgr_add_field(runtime%domain%streamManager,trim(string(stream)),trim(string(key)),ierr=status)
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:3561-3680,2800-2896.
subroutine native_runtime_stream_transfer(handle,stream,timelevel,read,status) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: stream(*)
 integer(c_int),value :: timelevel
 logical(c_bool),value :: read
 integer(c_int),intent(out) :: status
 type(native_runtime_type),pointer :: runtime
 call c_f_pointer(handle,runtime)
 status=-1
 if(.not.runtime%has_streams) return
 if(read) then
  call MPAS_stream_mgr_read(runtime%domain%streamManager,streamID=trim(string(stream)), &
       timeLevel=timelevel,rightNow=.true.,whence=MPAS_STREAM_NEAREST,ierr=status)
 else
  call MPAS_stream_mgr_write(runtime%domain%streamManager,streamID=trim(string(stream)), &
       timeLevel=timelevel,forceWriteNow=.true.,ierr=status)
 endif
end subroutine
end module
''']
    write_changed(output / 'native_framework.f90', '\n'.join(f) + '\n')
