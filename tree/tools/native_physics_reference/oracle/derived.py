"""Source-only derived argument inventory for MPAS NoahMP's whole driver object.

MPAS physics_noahmp/drivers/mpas/NoahmpIOVarType.F90 and
NoahmpIOVarInitMod.F90. Allocation expressions establish physical cell axes;
table/soil-only arrays retain unmapped axes rather than guessed global IDs.
"""
from dataclasses import asdict
from fortran_source import active_source,statements,declarations,split_top
import re


def noahmp_components(root,dummy,actual,intent):
    directory=root/'src/core_atmosphere/physics/physics_noahmp/drivers/mpas'
    fields=[];inside=False
    for st in statements(active_source((directory/'NoahmpIOVarType.F90').read_text())):
        if re.match(r'type\s*,\s*public\s*::\s*NoahmpIO_type\b',st.text,re.I):inside=True;continue
        if inside and re.match(r'end\s*type\b',st.text,re.I):break
        if inside:
            for v in declarations(st.text):
                if not v.supported:raise ValueError(f'unsupported NoahMP component {v.name}')
                fields.append(dict(asdict(v),name=dummy.replace('(','.').replace(')','').replace(',','.')+'.'+v.name,replay_expr=dummy+'%'+v.name,
                                   actual=actual+'%'+v.name,intent=intent,
                                   optional='allocatable' in v.attrs,
                                   presence_guard=f'allocated({actual}%{v.name})' if 'allocatable' in v.attrs else '',
                                   cell_axis=0,level_axis=0))
    allocations={}
    for st in statements(active_source((directory/'NoahmpIOVarInitMod.F90').read_text())):
        for match in re.finditer(r'\ballocate\s*\(\s*NoahmpIO%(\w+)\s*\(([^)]*)\)',st.text,re.I):
            dims=split_top(match[2]);axis=next((i+1 for i,d in enumerate(dims) if re.search(r'\bits\b|\bite\b',d,re.I)),0)
            name=match[1].lower()
            if name in allocations and allocations[name]!=axis:raise ValueError(f'ambiguous NoahMP cell axis: {name}')
            allocations[name]=axis
    for f in fields:f['cell_axis']=allocations.get(f['name'].split('.')[-1],0)
    if not fields:raise ValueError('NoahMP component inventory is empty')
    return fields


def has_presence(argument):
    """Separate the callee declaration from an absent forwarded actual."""
    return argument['optional'] or argument.get('actual_optional', False)


def array_arguments(site):
    return [f for arg in site['arguments'] if not (arg.get('native_pool') or arg.get('pool_argument')) for f in arg.get('components',[arg])]


def time_interval_components(root,actual):
    """Flatten MPAS/ESMF clock values from their pinned type declarations.

    mpas_timekeeping_types.inc:18-20 wraps ESMF_TimeIntervalMod.F90:63-73.
    Calendar pointers are represented by association plus semantic fields;
    pointer addresses are never serialized. Unset reference time is absent.
    """
    files=['src/framework/mpas_timekeeping_types.inc',
           *('src/external/esmf_time_f90/'+n for n in (
               'ESMF_TimeIntervalMod.F90','ESMF_ShrTimeMod.F90',
               'ESMF_BaseTimeMod.F90','ESMF_CalendarMod.F90'))]
    types={}
    for relative in files:
        current=None
        for st in statements(active_source((root/relative).read_text())):
            start=re.fullmatch(r'type\s+(\w+)',st.text,re.I)
            if start:current=start[1].lower();types[current]=[];continue
            if re.match(r'end\s*type\b',st.text,re.I):current=None;continue
            if current:types[current].extend(dict(asdict(v),line=st.start+1,path=relative) for v in declarations(st.text))
    fields=[]
    def visit(typ,prefix,conditions):
        if typ not in types:raise ValueError(f'unknown clock component type: {typ}')
        for v in types[typ]:
            name=prefix+'%'+v['name'];guards=list(conditions)
            if typ=='esmf_timeinterval' and v['name']=='starttime':guards.append(prefix+'%starttime_set')
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',v['type'],re.I)
            if nested:
                if 'pointer' in v['attrs']:
                    fields.append(dict(v,name=name,container_pointer=True,conditions=list(guards)))
                    guards.append(f'associated({name})')
                visit(nested[1].lower(),name,guards)
            else:
                if not v['type'].startswith(('integer','logical','real','character')):raise ValueError(f'unsupported clock value: {name}')
                fields.append(dict(v,name=name,conditions=guards))
    visit('mpas_timeinterval_type',actual,[])
    return fields


def primitive_components(root, config, typename, dummy, actual, intent, dims=None):
    """Capture every field of a pinned primitive-only derived configuration.

    RRFS 1d6e370e: physics_noaa/TEMPO/src/module_mp_tempo_cfgs.F90:14-46.
    Descriptor/pointer/nested types need a separate audited restoration path.
    """
    if dims:
        if any(not re.fullmatch(r'[1-9][0-9]*',d.strip()) for d in dims):
            raise ValueError(f'derived array requires fixed explicit extents: {typename}: {dims}')
        from itertools import product
        result=[]
        for index in product(*(range(1,int(d)+1) for d in dims)):
            element=','.join(str(i) for i in index)
            result.extend(primitive_components(root,config,typename,dummy+'('+element+')',actual+'('+element+')',intent))
        return result
    fields=[];inside=False
    for st in statements(active_source((root/config['path']).read_text())):
        if re.fullmatch(r'type\s*(?:::)?\s*'+re.escape(typename),st.text,re.I):
            inside=True;continue
        if inside and re.match(r'end\s*type\b',st.text,re.I):break
        if inside:
            for v in declarations(st.text):
                if not v.supported or 'pointer' in v.attrs:
                    raise ValueError(f'unsupported config component: {typename}%{v.name}')
                fields.append(dict(asdict(v),name=dummy.replace('(','.').replace(')','').replace(',','.')+'.'+v.name,replay_expr=dummy+'%'+v.name,
                                   actual=actual+'%'+v.name,intent=intent,optional='allocatable' in v.attrs,
                                   presence_guard=f'allocated({actual}%{v.name})' if 'allocatable' in v.attrs else '',
                                   entry_storage=intent=='out',cell_axis=0,level_axis=0,
                                   source_path=config['path'],source_line=st.start+1))
    if not fields:raise ValueError(f'empty config type: {typename}')
    return fields
