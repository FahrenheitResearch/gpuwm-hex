"""Capture framework module globals without a framework/physics module cycle.

MPAS src/framework/mpas_timekeeping.F:22-23 has mutable TheCalendar/yearWidth.
Framework compiles before physics. Explicit external interfaces reach the same
oracle transport object; no native arithmetic, initialization or ordering changes.
"""
from generate_transport import TYPES

KINDS = ' use iso_fortran_env, only: int32,int64,real32,real64\n'


def procedure(tag, typ, rank, mode, implementation):
    name = f'oracle_framework_{mode}_{tag}_{rank}'
    args = {'write':'key,phase,name,value,event,lower,cell_axis,level_axis',
            'read':'path,value','hidden':'name,value,invocation,phase,lower',
            'hidden_alloc':'name,value,invocation,phase','allocation':'name,value,lower'}[mode]
    text = f' subroutine {name}({args})\n'
    if implementation:
        text += f' use mpas_oracle_dump, only: oracle_{mode}\n'
    text += KINDS
    if mode == 'write':
        text += (' character(len=*), intent(in) :: key,phase,name\n'
                 ' integer, intent(in) :: event\n'
                 ' integer, optional, intent(in) :: lower(:),cell_axis,level_axis\n')
    elif mode == 'read':
        text += ' character(len=*), intent(in) :: path\n'
    elif mode=='allocation':
        text += ' character(len=*), intent(in) :: name\n integer, optional, intent(in) :: lower(:)\n'
    else:
        text += (' character(len=*), intent(in) :: name,phase\n'
                 ' integer, intent(in) :: invocation\n'+
                 (' integer, optional, intent(in) :: lower(:)\n' if mode=='hidden' else ''))
    shape = '(' + ','.join(':' for _ in range(rank)) + ')' if rank else ''
    allocation=', allocatable' if mode=='hidden_alloc' else ''
    text += f' {typ}{allocation}, intent({"in" if mode == "write" else "inout"}) :: value{shape}\n'
    if implementation:
        if mode == 'write':
            text += (' call oracle_write(key,phase,name,value,event,lower=lower, &\n'
                     '      cell_axis=cell_axis,level_axis=level_axis)\n')
        elif mode == 'read':
            text += ' call oracle_read(path,value)\n'
        elif mode=='hidden_alloc':
            text += ' call oracle_hidden_alloc(name,value,invocation,phase)\n'
        elif mode=='allocation':
            text += ' call oracle_allocation(name,value,lower=lower)\n'
        else:
            text += ' call oracle_hidden(name,value,invocation,phase,lower=lower)\n'
    return text + f' end subroutine {name}\n'


def special(mode, implementation):
    if mode == 'local_begin':
        text=' subroutine oracle_framework_local_begin(name,invocation)\n'
        if implementation:text+=' use mpas_oracle_dump, only: oracle_local_begin\n'
        text+=(' character(len=*), intent(in) :: name\n'
               ' integer, intent(out) :: invocation\n')
        if implementation:text+=' call oracle_local_begin(name,invocation)\n'
        return text+' end subroutine oracle_framework_local_begin\n'
    if mode == 'enabled':
        name = 'oracle_framework_enabled'
        text = f' logical function {name}(key)\n'
        if implementation:
            text += ' use mpas_oracle_dump, only: oracle_enabled\n'
        text += ' character(len=*), intent(in) :: key\n'
        if implementation:
            text += f' {name}=oracle_enabled(key)\n'
        return text + f' end function {name}\n'
    text = ' subroutine oracle_framework_header(path,code,width,rank_n,dims,lower,unit)\n'
    if implementation:
        text += ' use mpas_oracle_dump, only: oracle_header\n'
    text += (' character(len=*), intent(in) :: path\n'
             ' integer, intent(out) :: code,width,rank_n,dims(5),lower(5),unit\n')
    if implementation:
        text += ' call oracle_header(path,code,width,rank_n,dims,lower,unit)\n'
    return text + ' end subroutine oracle_framework_header\n'


def interface_source():
    text = ''
    for mode in ['write', 'read','hidden','hidden_alloc','allocation']:
        text += f' interface oracle_{mode}\n'
        for tag, typ, _, _ in TYPES:
            if mode=='hidden_alloc' and tag=='c':continue
            for rank in range(6):
                text += procedure(tag, typ, rank, mode, False)
        text += ' end interface\n'
    # Keep the helper's original names; procedure aliases are explicit generic
    # interfaces, so keyword/optional arguments retain the transport contract.
    for mode in ['enabled', 'header','local_begin']:
        text += f' interface oracle_{mode}\n' + special(mode, False) + ' end interface\n'
    text += ' private :: oracle_write,oracle_read,oracle_enabled,oracle_header,oracle_hidden,oracle_hidden_alloc,oracle_allocation,oracle_local_begin\n'
    for mode in ['write', 'read','hidden','hidden_alloc','allocation']:
        for tag, _, _, _ in TYPES:
            if mode=='hidden_alloc' and tag=='c':continue
            for rank in range(6):
                text += f' private :: oracle_framework_{mode}_{tag}_{rank}\n'
    text += ' private :: oracle_framework_enabled,oracle_framework_header,oracle_framework_local_begin\n'
    return text


def transport_bridge():
    text = '\n! Framework bridge: external interfaces preserve native module ordering.\n'
    for mode in ['write', 'read','hidden','hidden_alloc','allocation']:
        for tag, typ, _, _ in TYPES:
            if mode=='hidden_alloc' and tag=='c':continue
            for rank in range(6):
                text += procedure(tag, typ, rank, mode, True)
    for mode in ['enabled', 'header','local_begin']:
        text += special(mode, True)
    return text
