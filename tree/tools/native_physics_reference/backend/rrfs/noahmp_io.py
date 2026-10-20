"""Generate read-only access to the complete pinned Noah-MP derived IO type.

RRFS-MPAS 1d6e370e source: MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_vars.F:907.
RRFS-MPAS 1d6e370e source: MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarType.F90:18-938.
Native storage, allocation and lower bounds are copied without initialization.
"""
import hashlib
import re
import subprocess


def generate_noahmp_io(source, output):
    from generate import statements, split_list, write_changed
    path = source / 'src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarType.F90'
    # Match the MPAS build, so WRF-only members cannot shift the indexed ABI.
    text = subprocess.run(['cpp', '-P', '-traditional', '-Dmpas', '-DSINGLE_PRECISION',
                           str(path)], check=True, capture_output=True, text=True).stdout
    source_lines = {}
    for line, declaration in statements(path.read_text()):
        if '::' in declaration:
            for member in split_list(declaration.split('::', 1)[1]):
                source_lines[member.split('=', 1)[0].strip().lower()] = line
    fields = {}
    active = False
    for line, declaration in statements(text):
        if re.match(r'type\s*,\s*public\s*::\s*noahmpio_type\b', declaration, re.I):
            active = True
            continue
        if active and re.match(r'end\s*type', declaration, re.I):
            break
        if not active or '::' not in declaration:
            continue
        head, tail = declaration.lower().split('::', 1)
        kind = re.match(r'(real|integer|logical|character)\b', head)
        if not kind:
            raise ValueError(f'unhandled Noah-MP IO declaration at {line}: {declaration}')
        dimension = re.search(r'dimension\((.*?)\)', head)
        rank = len(split_list(dimension[1])) if dimension else 0
        if rank and 'allocatable' not in head:
            raise ValueError(f'Noah-MP IO nonallocatable array at {line}')
        length = re.search(r'len\s*=\s*(\d+)', head)
        for name in split_list(tail):
            name = name.split('=', 1)[0].strip()
            if not re.fullmatch(r'\w+', name) or name in fields:
                raise ValueError(f'unhandled Noah-MP IO component at {line}: {name}')
            fields[name] = dict(name=name, index=len(fields) + 1, kind=kind[1], rank=rank,
                                character_length=(int(length[1]) if length else 1)
                                if kind[1] == 'character' else None,
                                logical_encoding='native storage, not C bool'
                                if kind[1] == 'logical' else None,
                                source=str(path.relative_to(source)), line=source_lines[name],
                                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not fields:
        raise ValueError('native Noah-MP IO type not found')
    code = ['! Generated read-only native IO transport; no scheme arithmetic.',
            'module hex_native_noahmp_io', 'use iso_c_binding',
            'use mpas_atmphys_vars, only: mpas_noahmp', 'implicit none', 'contains',
            'subroutine native_noahmp_io_copy(index,dims,lower,width,buffer,status) bind(C)',
            'integer(c_int), value :: index',
            'integer(c_int), intent(out) :: dims(5),lower(5),width,status',
            'type(c_ptr), value :: buffer', 'integer(c_int32_t), pointer :: words(:)',
            'character(kind=c_char), pointer :: characters(:)', 'integer :: count,j',
            'dims=1; lower=1; width=0; status=0', 'select case(index)']
    for name, field in fields.items():
        item = 'mpas_noahmp%' + name
        rank = field['rank']
        code += [f'! MPAS source: MPAS-Model-rrfs/{field["source"]}:{field["line"]}', f'case({field["index"]})']
        if rank:
            code += [f'if (.not.allocated({item})) then', 'status=1; return', 'endif',
                     f'dims(1:{rank})=shape({item}); lower(1:{rank})=lbound({item})',
                     f'count=size({item})']
        else:
            code += ['count=1']
        code += [f'width=storage_size({item})/8', 'if(c_associated(buffer)) then']
        if field['kind'] == 'character':
            if rank:
                raise ValueError('native Noah-MP type has an unsupported character array')
            code += ['call c_f_pointer(buffer,characters,[width])', 'do j=1,width',
                     f'characters(j)={item}(j:j)', 'enddo']
        else:
            code += ['call c_f_pointer(buffer,words,[count*width/4])',
                     f'words=transfer({item},words)']
        code += ['endif']
    code += ['case default', 'status=2', 'end select', 'end subroutine', 'end module']
    write_changed(output / 'native_noahmp_io.f90', '\n'.join(code) + '\n')
    return fields
