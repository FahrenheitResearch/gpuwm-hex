"""Primitive leaves of pinned ESMF calendar module targets and pointer objects.

ESMF_CalendarMod.F90:56-63,122-124 and ESMF_BaseTimeMod.F90:52-55.
No pointer address or assumed target layout is serialized. Components are read
from the pinned declarations, including integer precision and array bounds.
"""
from dataclasses import asdict
import re
from fortran_source import active_source,statements,declarations


def components(root,var):
    paths=['src/external/esmf_time_f90/ESMF_CalendarMod.F90',
           'src/external/esmf_time_f90/ESMF_BaseTimeMod.F90']
    types={}
    for path in paths:
        current=None
        for st in statements(active_source((root/path).read_text())):
            start=re.fullmatch(r'type\s+(\w+)',st.text,re.I)
            if start:
                current=start[1].lower();types[current]=[];continue
            if re.match(r'end\s*type\b',st.text,re.I):current=None;continue
            if current:
                types[current].extend(dict(asdict(v),supported=v.supported,line=st.start+1,path=path)
                                      for v in declarations(st.text))
    result=[]
    name=var.name
    pointer='pointer' in var.attrs
    guards=[f'associated({name})'] if pointer else []
    if pointer:
        result.append(dict(asdict(var),name=name,container_pointer=True,conditions=[]))
    def visit(typ,prefix,dims,conditions):
        if typ not in types:raise ValueError(f'unknown native calendar component type: {typ}')
        for field in types[typ]:
            value=prefix+'%'+field['name']
            if 'pointer' in field['attrs'] or 'allocatable' in field['attrs']:
                raise ValueError(f'calendar component storage changed: {value}')
            # Fortran permits one nonzero-rank part in a component expression.
            if dims and field['dims']:raise ValueError(f'two array parts in calendar state: {value}')
            shape=dims or field['dims']
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
            if nested:visit(nested[1].lower(),value,shape,conditions)
            elif field['supported']:
                record=dict(field,name=value,dims=shape,conditions=list(conditions))
                if var.dims:record['lower_expr']=f'lbound({name})'
                result.append(record)
            else:raise ValueError(f'unsupported native calendar component: {value}')
    typ=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',var.type,re.I)
    if not typ:raise ValueError('calendar root must be a native derived object')
    visit(typ[1].lower(),name,var.dims,guards)
    return result
