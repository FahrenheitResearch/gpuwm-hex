"""Generate a typed native MPAS/ESMF clock graph transport, never raw addresses.

mpas_timekeeping_types.inc:19-38, ESMF_ClockMod.F90:59-82,
ESMF_AlarmMod.F90:59-80. Native declarations define every serialized leaf.
"""
from dataclasses import asdict
import hashlib
import re
from fortran_source import active_source, statements, declarations


def schema(root):
    paths=['src/framework/mpas_timekeeping_types.inc']
    paths += [str(p.relative_to(root)) for p in sorted((root/'src/external/esmf_time_f90').glob('*.F90'))]
    types={}
    for path in paths:
        current=None
        for st in statements(active_source((root/path).read_text())):
            match=re.fullmatch(r'type\s+(\w+)',st.text,re.I)
            if match:
                current=match[1].lower();types[current]=dict(path=path,line=st.start+1,fields=[])
            elif re.match(r'end\s*type\b',st.text,re.I):current=None
            elif current:
                types[current]['fields'] += [dict(asdict(v),supported=v.supported,line=st.start+1)
                                             for v in declarations(st.text)]
    needed=set();pointers={('mpas_clock_type',0)}
    def visit(name):
        if name in needed:return
        if name not in types:raise ValueError(f'unknown clock graph type: {name}')
        needed.add(name)
        for field in types[name]['fields']:
            if 'allocatable' in field['attrs']:raise ValueError(f'changed clock allocation: {name}.{field["name"]}')
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
            if nested:
                target=nested[1].lower()
                if 'pointer' in field['attrs']:
                    if len(field['dims'])>1:raise ValueError('clock pointer rank changed')
                    pointers.add((target,len(field['dims'])))
                elif field['dims']:raise ValueError('clock structure array storage changed')
                visit(target)
            elif not field['supported'] or 'pointer' in field['attrs']:
                raise ValueError(f'unsupported clock leaf: {name}.{field["name"]}')
    visit('mpas_clock_type')
    selected={n:types[n] for n in sorted(needed)}
    return dict(types=selected,pointers=sorted(pointers),source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
                for p in sorted({v['path'] for v in selected.values()})},
                external_root_aliases_complete=False,
                scope='One native MPAS clock graph; all internal typed pointer identities and represented global calendars. External domain-root aliases remain separate.')


def generate(root):
    info=schema(root);types=info['types'];pointers=[tuple(p)for p in info['pointers']]
    tag=lambda typ,rank:typ+('_array' if rank else '')
    text='''! GENERATED from pinned native MPAS/ESMF declarations; no native arithmetic changes.
module mpas_oracle_clock
 use mpas_timekeeping,only:MPAS_Clock_type,MPAS_Alarm_type
 use ESMF_ClockMod,only:ESMF_ClockInt
 use ESMF_AlarmMod,only:ESMF_Alarm,ESMF_AlarmInt
 use ESMF_CalendarMod,only:ESMF_Calendar,defaultcal,gregoriancal,noleapcal
 use mpas_oracle_dump,only:oracle_write,oracle_read,oracle_enabled
 implicit none
 private
 public::oracle_clock_capture,oracle_clock_restore
'''
    for typ,rank in pointers:
        name=tag(typ,rank);shape='(:)'if rank else ''
        text+=f' type ref_{name}\n  type({typ}),pointer::p{shape}=>null()\n end type\n type(ref_{name}),allocatable::refs_{name}(:)\n'
    text+=' contains\n subroutine clear_refs\n'
    for typ,rank in pointers:
        name=tag(typ,rank);text+=f'  if(allocated(refs_{name})) deallocate(refs_{name})\n'
    text+=' end subroutine\n'
    text+=''' function number(i) result(s)
 integer,intent(in)::i
 character(len=32)::s
 write(s,'(i0)')i
 end function
 subroutine oracle_clock_capture(key,phase,name,value,event)
 character(len=*),intent(in)::key,phase,name
 integer,intent(in)::event
 type(MPAS_Clock_type),pointer,intent(in)::value
 if(.not.oracle_enabled(key))return
 call clear_refs
 call oracle_write(key,phase,name//'.present',associated(value),event)
 call write_ptr_mpas_clock_type(value,key,phase,name,name,event)
 call clear_refs
 end subroutine
 subroutine oracle_clock_restore(prefix,name,value)
 character(len=*),intent(in)::prefix,name
 type(MPAS_Clock_type),pointer,intent(inout)::value
 logical::present
 call clear_refs
 call oracle_read(prefix//'.pre.'//name//'.present.bin',present)
 call read_ptr_mpas_clock_type(value,prefix,name,name)
 if(present.neqv.associated(value))error stop 'clock root presence disagrees with graph target'
 call clear_refs
 end subroutine
'''
    def leaves(typ,expr,path,mode):
        result=''
        fields=types[typ]['fields']
        if typ=='esmf_timeinterval':
            fields=sorted(fields,key=lambda f:f['name']!='starttime_set')
        for field in fields:
            value=expr+'%'+field['name'];label=path+"//'."+field['name']+"'"
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
            guard=typ=='esmf_timeinterval' and field['name']=='starttime'
            if guard:result+=f' if({expr}%starttime_set)then\n'
            if nested:
                target=nested[1].lower()
                if 'pointer'in field['attrs']:
                    proc=('write'if mode=='write'else'read')+'_ptr_'+tag(target,len(field['dims']))
                    args=f'{value},key,phase,base,{label},event'if mode=='write'else f'{value},prefix,base,{label}'
                    result+=f' call {proc}({args})\n'
                else:result+=leaves(target,value,label,mode)
            elif mode=='write':
                lower=f',lower=lbound({value})'if field['dims']else''
                result+=f' call oracle_write(key,phase,{label},{value},event{lower})\n'
            else:result+=f" call oracle_read(prefix//'.pre.'//{label}//'.bin',{value})\n"
            if guard:result+=' endif\n'
        return result
    for typ,rank in pointers:
        name=tag(typ,rank);shape='(:)'if rank else '';refs='refs_'+name
        for mode in ['write','read']:
            writer=mode=='write';args='p,key,phase,base,path,event'if writer else'p,prefix,base,path'
            text+=f' recursive subroutine {mode}_ptr_{name}({args})\n type({typ}),pointer,intent({"in"if writer else"inout"})::p{shape}\n'
            text+=(' character(len=*),intent(in)::key,phase,base,path\n integer,intent(in)::event\n'if writer else' character(len=*),intent(in)::prefix,base,path\n')
            text+=f' type(ref_{name}),allocatable::grown(:)\n integer::id,n,j,extent(2)\n character(len=256)::node,element\n n=0\n if(allocated({refs}))n=size({refs})\n'
            if writer:
                text+=' id=0\n if(associated(p))then\n'
                if typ=='esmf_calendar':
                    for idx,slot in enumerate(['defaultcal','gregoriancal','noleapcal'],1):
                        text+=f' if(id==0.and.associated({slot}))then\n if(associated(p,{slot}))id=-{idx}\n endif\n'
                text+=f' if(id==0)then\n do j=1,n\n if(associated(p,{refs}(j)%p))then\n id=j\n exit\n endif\n enddo\n endif\n if(id==0)id=n+1\n endif\n call oracle_write(key,phase,path//\'.target\',id,event)\n if(id<=n)return\n'
            else:
                text+=" call oracle_read(prefix//'.pre.'//path//'.target.bin',id)\n if(id==0)then\n nullify(p)\n return\n endif\n"
                if typ=='esmf_calendar':
                    text+=' if(id<0)then\n select case(id)\n'
                    for idx,slot in enumerate(['defaultcal','gregoriancal','noleapcal'],1):text+=f' case(-{idx})\n if(.not.associated({slot}))error stop \'clock global calendar target not restored\'\n p=>{slot}\n'
                    text+=" case default\n error stop 'clock invalid global calendar class'\n end select\n return\n endif\n"
                text+=f" if(id<0.or.id>n+1)error stop 'clock target ordinal missing or forward reference'\n if(id<=n)then\n p=>{refs}(id)%p\n return\n endif\n"
                # Reuse compatible native objects. In particular, replacing
                # atm_core.clock would detach domain%clock from the restored
                # target (mpas_atm_core.F:97; mpas_domain_types.inc:6).
                # A captured distinct node must never overwrite a target that
                # has already been assigned to another captured ordinal.
                text+=f' if(associated(p))then\n do j=1,n\n if(associated(p,{refs}(j)%p))then\n nullify(p)\n exit\n endif\n enddo\n endif\n'
                if typ=='esmf_calendar':
                    # Positive ordinals represent local calendars, even if a
                    # mutated pointer now aliases a global provider target.
                    for slot in ['defaultcal','gregoriancal','noleapcal']:
                        text+=f' if(associated(p).and.associated({slot}))then\n if(associated(p,{slot}))nullify(p)\n endif\n'
            text+=f" node=base//'.node.{name}.'//trim(number(id))\n"
            if rank:
                if writer:text+=" extent=[lbound(p,1),size(p)]\n call oracle_write(key,phase,trim(node)//'.bounds',extent,event)\n"
                else:text+=" call oracle_read(prefix//'.pre.'//trim(node)//'.bounds.bin',extent)\n if(extent(2)<0)error stop 'clock graph array extent is negative'\n if(associated(p))then\n if(lbound(p,1)/=extent(1).or.size(p)/=extent(2))nullify(p)\n endif\n if(.not.associated(p))allocate(p(extent(1):extent(1)+extent(2)-1))\n"
            elif not writer:text+=' if(.not.associated(p))allocate(p)\n'
            text+=f' allocate(grown(n+1))\n if(n>0)grown(:n)={refs}\n grown(n+1)%p=>p\n call move_alloc(grown,{refs})\n'
            if rank:text+=" do j=lbound(p,1),ubound(p,1)\n element=trim(node)//'.element.'//trim(number(j))\n"+leaves(typ,'p(j)','trim(element)',mode)+' enddo\n'
            else:text+=leaves(typ,'p','trim(node)',mode)
            text+=' end subroutine\n'
    text+='end module mpas_oracle_clock\n'
    return text,info


def audit_membership(info, prefix, phase, name, filenames, read):
    """Audit the complete represented graph; Rust decodes all record controls.

    Walk in the same source order as the generated native reader. Registries
    are typed: repeated ordinals preserve aliases/cycles, new nodes must be the
    next ordinal, and only native calendar provider classes accept negatives.
    This does not certify aliases held outside this clock root.
    """
    expected, missing, issues = set(), set(), []
    counts = {tuple(pointer): 0 for pointer in info['pointers']}
    global_calendars = set()
    types = info['types']
    stem = f'{prefix}.{phase}.'

    def member(path, code=None, width=None, shape=None):
        filename = stem + path + '.bin'
        expected.add(filename)
        if filename not in filenames:
            missing.add(filename)
            return None
        header = read('header', filename, '-')
        if (header['name'] != path or header['phase'] != phase or
                code is not None and header['code'] != code or
                width is not None and header['width'] != width or
                shape is not None and header['shape'] != shape):
            issues.append(dict(file=filename, reason='clock record layout or context disagrees with native declaration'))
            return None
        return header

    def control(path, code, indices='-'):
        header = member(path, code, 4, [] if indices == '-' else [2])
        if header is None:
            return None
        sample = read('sample', stem + path + '.bin', indices)
        value = sample['value']
        if ((code == 4 and type(value) is not bool) or
                (code == 3 and type(value) is not int)):
            issues.append(dict(file=stem + path + '.bin', reason='clock control is not a native logical/integer'))
            return None
        return value

    def leaves(typ, path):
        fields = types[typ]['fields']
        if typ == 'esmf_timeinterval':
            fields = sorted(fields, key=lambda field: field['name'] != 'starttime_set')
        starttime = None
        for field in fields:
            label = path + '.' + field['name']
            if typ == 'esmf_timeinterval' and field['name'] == 'starttime':
                if starttime is not True:
                    continue
            nested = re.fullmatch(r'type\s*\(\s*(\w+)\s*\)', field['type'], re.I)
            if nested:
                target = nested[1].lower()
                if 'pointer' in field['attrs']:
                    pointer(target, len(field['dims']), label)
                else:
                    leaves(target, label)
            elif typ == 'esmf_timeinterval' and field['name'] == 'starttime_set':
                starttime = control(label, 4)
            else:
                declaration = field['type'].lower()
                if declaration.startswith('character'):
                    member(label, 5, 1, [] if not field['dims'] else None)
                else:
                    code = 4 if declaration.startswith('logical') else 6 if 'esmf_kind_i8' in declaration else 3
                    width = 8 if code == 6 else 4
                    header = member(label, code, width, [] if not field['dims'] else None)
                    if header is not None and len(header['shape']) != len(field['dims']):
                        issues.append(dict(file=stem + label + '.bin', reason='clock primitive storage rank changed'))

    def pointer(typ, rank, path):
        identity = (typ, rank)
        if identity not in counts:
            issues.append(dict(path=path, reason='clock pointer class absent from pinned schema'))
            return None
        ordinal = control(path + '.target', 3)
        if ordinal is None or ordinal == 0:
            return ordinal
        if ordinal < 0:
            if typ != 'esmf_calendar' or rank or ordinal not in (-1, -2, -3):
                issues.append(dict(path=path, ordinal=ordinal, reason='invalid native calendar provider class'))
            else:
                global_calendars.add(['defaultcal', 'gregoriancal', 'noleapcal'][-ordinal - 1])
            return ordinal
        current = counts[identity]
        if ordinal <= current:
            return ordinal
        if ordinal != current + 1:
            issues.append(dict(path=path, ordinal=ordinal, reason='clock target ordinal missing or forward reference'))
            return ordinal
        counts[identity] = ordinal
        tag = typ + ('_array' if rank else '')
        node = name + f'.node.{tag}.{ordinal}'
        if rank:
            lower, extent = control(node + '.bounds', 3, '1'), control(node + '.bounds', 3, '2')
            if lower is None or extent is None:
                return ordinal
            if extent < 0 or extent > len(filenames):
                issues.append(dict(path=node, extent=extent, reason='clock array extent cannot be represented by captured records'))
                return ordinal
            for index in range(lower, lower + extent):
                leaves(typ, node + f'.element.{index}')
        else:
            leaves(typ, node)
        return ordinal

    present = control(name + '.present', 4)
    root = pointer('mpas_clock_type', 0, name)
    if present is not None and root is not None and present != (root != 0):
        issues.append(dict(path=name, reason='clock root presence disagrees with graph target'))
    observed = {filename for filename in filenames if filename.startswith(stem + name + '.')}
    unexpected = sorted(observed - expected)
    return dict(expected_records=len(expected), missing=sorted(missing),
                unexpected=unexpected, topology_issues=issues,
                typed_node_counts={typ + ('_array' if rank else ''): count
                                   for (typ, rank), count in sorted(counts.items())},
                calendar_provider_references=sorted(global_calendars),
                external_aliases_complete=False, global_hidden_state_complete=False,
                complete=not missing and not unexpected and not issues,
                scope='Represented native clock graph membership and internal typed topology only')
