"""Whole native logger target state, from mpas_log_types.inc:11-43.

mpas_log.F:54 exposes the active domain's logger through a module pointer.
Capture its complete native leaves and output/error target alias classes.
Restoration keeps an existing root so the domain's logger alias remains live.
Other external aliases and actual file-descriptor positions remain unproven.
"""
from dataclasses import asdict
import hashlib
import re
from fortran_source import active_source, statements, declarations


def components(root, var):
    path = 'src/framework/mpas_log_types.inc'
    source = (root / path).read_text()
    types = {}
    current = None
    for st in statements(active_source(source)):
        match = re.fullmatch(r'type\s+(\w+)', st.text, re.I)
        if match:
            current = match[1].lower()
            types[current] = []
        elif re.match(r'end\s*type\b', st.text, re.I):
            current = None
        elif current:
            types[current].extend(dict(asdict(field), supported=field.supported,
                                       line=st.start + 1, path=path)
                                  for field in declarations(st.text))
    if set(types) != {'mpas_log_type', 'mpas_log_file_type'}:
        raise ValueError('native logger type set changed')
    if var.dims or 'pointer' not in var.attrs:
        raise ValueError('native logger root storage changed')
    result = []
    source_hash = hashlib.sha256(source.encode()).hexdigest()

    def visit(typ, value, conditions, ancestors, declaration=None):
        if typ in ancestors or typ not in types:
            raise ValueError('native logger graph topology changed')
        descendants = []
        for field in types[typ]:
            if field['dims'] or 'allocatable' in field['attrs']:
                raise ValueError('native logger component storage changed')
            nested = re.fullmatch(r'type\s*\(\s*(\w+)\s*\)', field['type'], re.I)
            if nested and 'pointer' in field['attrs']:
                descendants.append(field['name'])
        root_record = dict(name=value, type='type(' + typ + ')', attrs=', pointer ',
                           dims=[], intent='inout', optional=False, initialized=False,
                           container_pointer=True, preserve_live_target=True,
                           nullify_new_components=descendants, conditions=list(conditions),
                           path=declaration['path'] if declaration else 'src/framework/mpas_log.F',
                           declaration_source_sha256=source_hash,
                           external_aliases_complete=False, file_descriptors_restored=False)
        if declaration:
            root_record['line'] = declaration['line']
        result.append(root_record)
        guard = conditions + [f'associated({value})']
        for field in types[typ]:
            child = value + '%' + field['name']
            nested = re.fullmatch(r'type\s*\(\s*(\w+)\s*\)', field['type'], re.I)
            if nested:
                if 'pointer' not in field['attrs']:
                    raise ValueError('native logger child storage changed')
                visit(nested[1].lower(), child, guard, ancestors + [typ], field)
            elif field['supported'] and 'pointer' not in field['attrs']:
                result.append(dict(field, name=child, conditions=list(guard),
                                   declaration_source_sha256=source_hash))
            else:
                raise ValueError('unsupported native logger leaf: ' + child)
    visit('mpas_log_type', var.name, [], [])
    return result
