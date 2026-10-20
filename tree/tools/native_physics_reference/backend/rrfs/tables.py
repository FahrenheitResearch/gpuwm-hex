"""Generate read-only native table access; never evaluate table arithmetic.

Pristine module declarations remain unchanged. Public module variables are
accessed through their actual Fortran kinds and bounds, including RRTMG real64.
"""
import hashlib
import re
import subprocess

from generate import split_list, statements, write_changed


def base_inventory(source):
    physics = source / 'src/core_atmosphere/physics'
    paths = [physics / 'physics_wrf/module_ra_rrtmg_lw.F',
             physics / 'physics_wrf/module_ra_rrtmg_sw.F',
             physics / 'mpas_atmphys_manager.F']
    entries = {}
    for path in paths:
        module = None
        spec = False
        for line, statement in statements(path.read_text()):
            s = statement.lower()
            match = re.match(r'module\s+(\w+)\s*$', s)
            if match:
                module = match[1]
                spec = module.startswith(('rrlw_', 'rrsw_')) or module == 'mpas_atmphys_manager'
                continue
            if re.match(r'contains\b|end\s*module\b', s):
                spec = False
            if not spec or '::' not in s:
                continue
            head, tail = s.split('::', 1)
            if 'parameter' in head or not head.startswith(('real', 'integer')):
                continue
            if module == 'mpas_atmphys_manager' and 'public' not in head:
                continue
            if head.startswith('integer'):
                kind = 'integer'
            elif re.match(r'real\s*\(\s*kind\s*=\s*rb\s*\)', head):
                kind = 'real'
            elif re.match(r'real\s*\(\s*kind\s*=\s*rkind\s*\)', head):
                kind = 'real'
            else:
                raise ValueError(f'unresolved public table precision: {path}:{line}: {head}')
            attr = re.search(r'dimension\s*\(([^)]*)\)', head)
            for variable in split_list(tail):
                var = re.match(r'\s*(\w+)\s*(?:\(([^)]*)\))?', variable)
                name = var[1]
                dims = var[2] if var[2] is not None else attr[1] if attr else ''
                rank = len(split_list(dims)) if dims else 0
                key = module + '.' + name
                entries[key] = {'module': module, 'variable': name, 'rank': rank, 'kind': kind,
                                'allocatable': 'allocatable' in head, 'pointer': 'pointer' in head,
                                'source': str(path.relative_to(source)), 'line': line,
                                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return entries


def active_statements(path):
    """Inventory exactly the GNU-preprocessed MPAS variant with original lines.

    Even Fortran comment backslashes affect traditional cpp continuation; using
    raw declarations could expose a symbol absent from the compiled module.
    """
    macros = ['-DSINGLE_PRECISION', '-Dmpas', '-D_MPI', '-DMPAS_USE_MPI_F08',
              '-DCORE_ATMOSPHERE', '-DMPAS_NAMELIST_SUFFIX=atmosphere']
    if 'TEMPO' in path.parts:
        macros.append('-Dtempo_mpas')
    text = subprocess.run(['cpp', '-traditional', *macros, str(path)], check=True,
                          text=True, capture_output=True).stdout
    native_line, native_file, joined, first = 1, '', '', 0
    for raw in text.splitlines():
        marker = re.match(r'#\s+(\d+)\s+"([^"]+)"', raw)
        if marker:
            native_line, native_file = int(marker[1]), marker[2]
            continue
        line_number = native_line
        native_line += 1
        if native_file != str(path):
            continue
        line = raw.split('!')[0].strip()
        if not line or line.startswith('#'):
            continue
        if not joined:
            first = line_number
        joined += ' ' + line.lstrip('&').rstrip('&').strip()
        if not line.endswith('&'):
            yield first, joined.strip()
            joined = ''


def inventory(source):
    """Expose public NOAA initialization state and constants at compiled precision.

    Derived/private state stays inside the pristine modules and survives ordered
    initialization/calls. This inventory does not claim a private-state checkpoint.
    """
    entries = base_inventory(source)
    physics = source / 'src/core_atmosphere/physics'
    from sources import physics_sources
    for path in physics_sources(source):
        if 'physics_noaa' not in path.parts:
            continue
        if path.suffix.lower() not in {'.f', '.f90'}:
            continue
        module, declarations, access = None, [], {}
        default_public, spec, depth = True, False, 0

        def finish():
            for line, head, tail in declarations:
                for variable in split_list(tail):
                    var = re.fullmatch(r'\s*(\w+)\s*(?:\((.*)\))?\s*', variable.split('=', 1)[0].strip())
                    if var is None:
                        raise ValueError(f'unparsed NOAA declaration: {path}:{line}: {variable}')
                    name = var[1]
                    public = access.get(name, 'public' in head or (default_public and 'private' not in head))
                    if not public:
                        continue
                    attr = re.search(r'dimension\s*\((.*?)\)', head)
                    dims = var[2] if var[2] is not None else attr[1] if attr else ''
                    rank = len(split_list(dims)) if dims else 0
                    if rank > 5:
                        raise ValueError(f'NOAA table rank exceeds transport: {path}:{line}')
                    key = module + '.' + name
                    record = {'module': module, 'variable': name, 'rank': rank,
                              'kind': 'real' if head.startswith('real') else 'integer',
                              'allocatable': 'allocatable' in head, 'pointer': 'pointer' in head,
                              'parameter': 'parameter' in head,
                              'source': str(path.relative_to(source)), 'line': line,
                              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    if key in entries and entries[key]['source_sha256'] != record['source_sha256']:
                        raise ValueError(f'duplicate NOAA table: {key}')
                    entries[key] = record

        for line, statement in active_statements(path):
            text = statement.lower()
            match = re.match(r'module\s+(?!procedure)(\w+)\s*$', text)
            if match:
                module, spec, depth = match[1], True, 0
                declarations, access, default_public = [], {}, True
                continue
            if not spec:
                continue
            if re.match(r'contains\b|end\s*module\b', text) and not depth:
                finish()
                spec = False
                continue
            if re.match(r'type\s*(?:,.*)?::\s*\w+|type\s+\w+\s*$', text):
                depth += 1
                continue
            if re.match(r'end\s*type\b', text):
                depth -= 1
                continue
            if depth:
                continue
            if text in {'private', 'public'}:
                default_public = text == 'public'
                continue
            match = re.match(r'(private|public)\s*::\s*(.*)', text)
            if match:
                access.update({n.strip(): match[1] == 'public' for n in split_list(match[2])})
            if '::' in text:
                head, tail = text.split('::', 1)
                if head.startswith(('real', 'integer')):
                    declarations.append((line, head, tail))
    return entries


def generate_tables(source, output):
    entries = inventory(source)
    constants = {}
    physics_constants = source / 'src/core_atmosphere/physics/mpas_atmphys_constants.F'
    framework_constants = source / 'src/framework/mpas_constants.F'
    for path in (physics_constants, framework_constants):
        module = 'mpas_atmphys_constants' if path == physics_constants else 'mpas_constants'
        for line, statement in statements(path.read_text()):
            if '::' not in statement or 'parameter' not in statement.lower():
                continue
            head, tail = statement.split('::', 1)
            if not re.match(r'real\s*\(\s*kind\s*=\s*rkind\s*\)', head, re.I):
                continue
            for declaration in split_list(tail):
                name = declaration.split('=', 1)[0].strip().lower()
                if not re.fullmatch(r'\w+', name) or name in constants:
                    continue
                constants[name] = dict(module=module, variable=name, kind='real', rank=0,
                                       source=str(path.relative_to(source)), line=line,
                                       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    # Public renamed imports in the physics constants module retain native words.
    constants['r_d'] = dict(constants['rgas'], variable='rgas')
    for i, entry in enumerate(constants.values(), 1):
        entry['index'] = i
    f = ['! Generated public native initialization-table access.',
         'module hex_native_tables', 'use iso_c_binding']
    for i, (key, entry) in enumerate(entries.items(), 1):
        entry['index'] = i
        f += [f"use {entry['module']}, only: table_{i} => {entry['variable']}"]
    for entry in constants.values():
        f += [f"use {entry['module']}, only: constant_{entry['index']} => {entry['variable']}"]
    f += ['implicit none', 'contains', '''
subroutine native_table_shape(index,dims,lower,width,status) bind(C)
 integer(c_int),value :: index
 integer(c_int),intent(out) :: dims(5),lower(5),width,status
 dims=1
 lower=1
 width=0
 status=0
 select case(index)
''']
    for key, e in entries.items():
        i, rank = e['index'], e['rank']
        f += [f"! MPAS source: MPAS-Model-rrfs/{e['source']}:{e['line']}", f'case({i})',
              f' width=storage_size(table_{i})/8']
        guard = f'allocated(table_{i})' if e['allocatable'] else f'associated(table_{i})' if e['pointer'] else None
        if guard:
            f += [f' if(.not.{guard}) then', '  status=1', '  return', ' endif']
        if rank:
            f += [f' dims(1:{rank})=shape(table_{i})', f' lower(1:{rank})=lbound(table_{i})']
    f += ['case default', ' status=-1', 'end select', 'end subroutine']
    for key, e in entries.items():
        i, rank = e['index'], e['rank']
        # Match the compiler's actual declaration kind, including rb. The pinned
        # parkind defines rb=kind(1.0); its commented real64 alternative is inactive.
        decl = f"{e['kind']}(kind=kind(table_{i}))"
        shape = '(' + ','.join(':' for _ in range(rank)) + ')' if rank else ''
        f += [f"! MPAS source: MPAS-Model-rrfs/{e['source']}:{e['line']}",
              f'subroutine native_table_copy_{i}(data) bind(C)', 'type(c_ptr),value :: data',
              decl + ',pointer :: buffer' + shape]
        f += [f'call c_f_pointer(data,buffer,shape(table_{i}))' if rank else 'call c_f_pointer(data,buffer)',
              f'buffer=table_{i}', 'end subroutine']
    f += ['subroutine native_constant_copy(index,data,width,status) bind(C)',
          'integer(c_int),value :: index', 'type(c_ptr),value :: data',
          'integer(c_int),intent(out) :: width,status', 'status=0', 'width=0',
          'select case(index)']
    for entry in constants.values():
        i = entry['index']
        f += [f"! MPAS source: MPAS-Model-rrfs/{entry['source']}:{entry['line']}", f'case({i})',
              ' block', f'  real(kind=kind(constant_{i})),pointer :: buffer',
              f'  width=storage_size(constant_{i})/8',
              '  if(c_associated(data)) then', '   call c_f_pointer(data,buffer)',
              f'   buffer=constant_{i}', '  endif', ' end block']
    f += ['case default', ' status=-1', 'end select', 'end subroutine', 'end module']
    write_changed(output / 'native_tables.f90', '\n'.join(f) + '\n')
    return entries, constants
