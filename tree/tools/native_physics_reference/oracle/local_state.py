"""Capture local SAVE, DATA and COMMON through native routine entry/exit hooks.

First entry supplies the hidden input; the final exit supplies hidden output.
Source declarations and storage association are unchanged. MPAS source citations
and unsupported state remain in the generated inventory.
"""
from dataclasses import asdict
import re
from fortran_source import active_source,statements,routines,split_top,first_executable


def local_variables(routine,root=None,pin_set_name="mpas-v842"):
    parameters={v.name for v in routine.variables.values() if 'parameter' in v.attrs}
    saved={v.name for v in routine.variables.values() if 'save' in v.attrs or v.initialized}
    issues=[]
    for st in routine.body:
        text=st.text.lower()
        if text.startswith('parameter'):
            match=re.match(r'parameter\s*\((.*)\)',text)
            if match:parameters.update(x.split('=',1)[0].strip() for x in split_top(match[1]))
        elif re.match(r'save\b',text):
            tail=re.sub(r'^save\s*(?:::)?','',text).strip()
            if not tail:saved.update(set(routine.variables)-set(routine.args))
            else:saved.update(x.strip() for x in split_top(tail) if '/' not in x)
        elif re.match(r'common\b',text):
            tail=re.sub(r'/[^/]*/',',',re.sub(r'^common\s*','',text))
            saved.update(re.match(r'\w+',x)[0] for x in split_top(tail) if re.match(r'\w+',x))
        elif re.match(r'data\b',text):
            parts=split_top(re.sub(r'^data\s*','',text),'/')
            if len(parts)%2!=1:issues.append('unparsed DATA statement');continue
            for target in parts[0::2]:
                for entity in split_top(target):
                    if not entity:continue
                    if entity.startswith('('):
                        # Implied DO controls are not DATA objects.
                        saved.update(re.findall(r'(\w+)\s*\(',entity))
                    else:
                        match=re.match(r'(\w+)',entity)
                        if match:saved.add(match[1])
    saved-=parameters
    variables=[]
    for name in sorted(saved):
        var=routine.variables.get(name)
        if var is None:issues.append(f'{name}: hidden declaration missing');continue
        from pin_sets import pin_set
        derived=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',var.type,re.I)
        config=pin_set(pin_set_name)
        if root is not None and derived and derived[1].lower() in config['derived_types']:
            from derived import primitive_components
            try:
                components=primitive_components(root,config['derived_types'][derived[1].lower()],derived[1],name,name,'inout',var.dims)
            except ValueError as error:issues.append(str(error));continue
            variables.extend(dict(v,name=v['actual']) for v in components)
            continue
        if not var.supported or 'pointer' in var.attrs or ('allocatable' in var.attrs and var.type.startswith('character')):
            issues.append(f'{name}: hidden storage requires descriptor restoration');continue
        variables.append(dict(asdict(var),name=name))
    return variables,issues


def patch_routines(original,text,module_names,root=None,pin_set_name="mpas-v842",framework=False):
    initial={(r.module,r.name):r for r in routines(original,original.parent) if r.module in module_names}
    # Coordinates use the already generated owned text, not stale upstream spans.
    lines=text.splitlines(keepends=True);inserts={};replacements={};records=[]
    # Source parser accepts a Path. Avoid a scratch file by a tiny read-only facade.
    class Source:
        def read_text(self,**kw):return text
        def relative_to(self,root):return original.name
    for routine in routines(Source(),original.parent):
        source=initial.get((routine.module,routine.name))
        if source is None:continue
        variables,issues=local_variables(source,root,pin_set_name)
        if not variables and not issues:continue
        signature=next(s for s in statements(text) if s.start==routine.start)
        if re.search(r'\b(?:pure|elemental)\b',signature.text,re.I):
            issues.append('PURE/ELEMENTAL routine cannot perform instrumentation I/O');variables=[]
        record={'module':routine.module,'routine':routine.name,'line':source.start+1,
                'end_line':source.end,'variables':variables,'issues':issues}
        records.append(record)
        if not variables:continue
        first=first_executable(routine)
        if first is None:raise ValueError(f'local state executable entry missing: {routine.name}')
        imports='' if framework else '  use mpas_oracle_dump, only: oracle_local_begin,oracle_hidden,oracle_hidden_alloc\n'
        inserts.setdefault(signature.end,[]).append(imports)
        tag=routine.module+'.'+routine.name
        before=f"  call oracle_local_begin('{tag}',oracle_local_event)\n"
        post=''
        for var in variables:
            value=var['name'];lower=f',lower=lbound({value})' if var['dims'] else ''
            name='local.'+tag+'.'+value.replace('%','.').replace('(','.').replace(')','').replace(',','.')
            if 'allocatable' in var['attrs']:
                before+=f"  call oracle_hidden_alloc('{name}',{value},oracle_local_event,'pre')\n"
                post+=f"  call oracle_hidden_alloc('{name}',{value},oracle_local_event,'post')\n"
                continue
            before+=f"  call oracle_hidden('{name}',{value},oracle_local_event,'pre'{lower})\n"
            post+=f"  call oracle_hidden('{name}',{value},oracle_local_event,'post'{lower})\n"
        inserts.setdefault(first.start,[]).append('  integer :: oracle_local_event\n')
        # A boundary implemented inside the routine must open its own scope
        # before observing its SAVE inputs, and flush after observing outputs.
        own_begin=next((s for s in routine.body if re.match(
            r"call\s+oracle_scope_begin\('(?:boundary__"+re.escape(routine.name)+r"|state__physics_init)'\s*,",s.text,re.I)),None)
        own_ends=[s for s in routine.body if re.fullmatch(r'call\s+oracle_scope_end\(\)',s.text,re.I)] if own_begin else []
        own_end=own_ends[-1] if own_ends else None
        if own_begin and not own_end:raise ValueError('local boundary scope end missing')
        inserts.setdefault(own_begin.end if own_begin else first.start,[]).append(before)
        for number,st in enumerate(routine.body):
            tail=re.sub(r'^\d+\s+','',st.text)
            if re.fullmatch(r'return',tail,re.I):
                previous=routine.body[number-1] if number else None
                at=previous.start if own_begin and previous in own_ends else st.start
                inserts.setdefault(at,[]).append(post)
            elif re.match(r'if\s*\(.*\)\s*return\s*$',tail,re.I):
                if st.start in replacements:raise ValueError('multiple returns share a source line')
                condition=re.match(r'if\s*(\(.*\))\s*return',tail,re.I)[1]
                label=re.match(r'\s*(\d+)\s+',lines[st.start])
                lead=(label[1]+' ' if label else '  ')
                replacements[st.start]=(st.end,f'{lead}if {condition} then\n'+post+'    return\n  endif\n')
        inserts.setdefault(own_end.start if own_end else routine.end-1,[]).append(post)
    result=[];i=0
    while i<len(lines):
        result.extend(inserts.get(i,[]))
        if i in replacements:
            i,text=replacements[i];result.append(text)
        else:result.append(lines[i]);i+=1
    return ''.join(result),records


def generate_locals(root,out,manifest):
    paths={m['path'] for m in manifest['module_state']}
    names={m['name'] for m in manifest['module_state']}
    records=[]
    for relative in sorted(paths):
        target=out/relative
        text,found=patch_routines(root/relative,target.read_text(),names,root,manifest.get("pin_set","mpas-v842"),relative.startswith(("src/framework/","src/external/")))
        if found:target.write_text(text)
        records.extend(dict(r,path=relative) for r in found)
    manifest['local_state']=records
    init=manifest.get('initialization_scope')
    if init:
        init['local_state_inventory']=[r for r in records if r['module'] in init['state_modules']]
        init['local_state_complete']=False
    captured={(r['module'],r['routine'],v['name']) for r in records for v in r['variables']}
    captured.update((r['module'],r['routine'],re.match(r'\w+',v['name'])[0]) for r in records for v in r['variables'])
    # A broad old COMMON/DATA issue is retired only when all objects in that
    # routine are parsed and supported; otherwise it remains explicit.
    complete={(r['module'],r['routine']) for r in records if not r['issues']}
    for module in manifest['module_state']:
        module['issues']=[i for i in module['issues'] if not (
            (module['name'],i.get('routine'),i.get('name')) in captured or
            i.get('reason')=='COMMON/DATA local state requires capture' and (module['name'],i.get('routine')) in complete)]
    by_module={m['name']:m for m in manifest['module_state']}
    for boundary in manifest.get('boundaries',[]):
        if not boundary.get('callee'):continue
        names=boundary.get('state_modules',[])
        # Module issues were copied before local hooks existed. Refresh that
        # copy so a supported SAVE/COMMON/DATA defect is actually retired.
        boundary['state_issues']=[{'module':name,**issue}
                                 for name in names for issue in by_module[name]['issues']]
        boundary['local_state_inventory']=[r for r in records if r['module'] in names]
        boundary['local_state_complete']=False
    manifest['hidden_state_complete']=False
    from static_accessors import generate as generate_accessors
    generate_accessors(root,out,manifest)
