"""Capture allocator-return bytes before native initialization or first use.

Fresh uninitialized storage is a hidden input. Replay executes the original
ALLOCATE then restores that entry input, never a recorded post-call result.
Every supported site and unresolved allocation is source-cited in inventory.
"""
import json
from pathlib import Path
import re
from fortran_source import routines,statements,split_top,first_executable

from fortran_source import routines,statements,split_top,declarations
from state import modules


def patch_allocations(original,text,infos,names,framework=False):
    lines=text.splitlines(keepends=True);replacements={};imports={};records=[]
    class Source:
        def read_text(self,**kw):return text
        def relative_to(self,root):return original.name
    originals={(r.module,r.name):r for r in routines(original,original.parent)}
    for routine in routines(Source(),original.parent):
        if routine.module not in names:continue
        origin=originals.get((routine.module,routine.name))
        if origin is None:continue
        signature=next(s for s in statements(text) if s.start==routine.start)
        candidates=[s for s in routine.body if re.search(r'\ballocate\s*\(',s.text,re.I)]
        original_sites=[s for s in origin.body if re.search(r'\ballocate\s*\(',s.text,re.I)]
        # Generated state restorers are separate routines, never native sites.
        if len(candidates)!=len(original_sites):
            raise ValueError(f'allocation source spans differ: {routine.module}.{routine.name}')
        available={v['name']:v for v in infos[routine.module]['roots']}
        # Resolve unrenamed imports conservatively. Ambiguous storage is explicit.
        imported={}
        for dep in infos[routine.module]['uses']:
            for v in infos.get(dep,{}).get('roots',[]):imported.setdefault(v['name'],[]).append(v)
        for name,values in imported.items():
            if name not in available and len(values)==1:available[name]=values[0]
        available.update({name:vars(v) for name,v in origin.variables.items()})
        for st,source in zip(candidates,original_sites):
            row={'module':routine.module,'routine':routine.name,'line':source.start+1,
                 'end_line':source.end,'statement':source.text,'variables':[],'issues':[]}
            records.append(row)
            match=re.fullmatch(r'(?:(if\s*\(.*\))\s*)?allocate\s*\((.*)\)',st.text,re.I)
            if not match or re.search(r'\b(?:pure|elemental)\b',signature.text,re.I):
                row['issues'].append('allocation control form requires explicit instrumentation');continue
            entities=split_top(match[2])
            if any('=' in entity or '::' in entity for entity in entities):
                row['issues'].append('allocation options/type specification require explicit instrumentation');continue
            hooks=[]
            for entity in entities:
                value=re.match(r'(\w+)\s*(?:\(.*\))?$',entity)
                var=available.get(value[1].lower()) if value else None
                if not var or not var['type'].startswith(('real','complex','integer','logical','character','double precision','double complex')) or len(var['dims'])>5:
                    row['issues'].append(f'{entity}: allocation declaration/type unresolved');continue
                name=value[1];tag=f'{routine.module}.{routine.name}.{name.lower()}.L{source.start+1}'
                row['variables'].append({'name':name,'record':tag,'type':var['type'],'dims':var['dims']})
                lower=f', lower=lbound({name})' if var['dims'] else ''
                hooks.append(f"  call oracle_allocation('{tag}', &\n       {name}{lower})\n")
            if not hooks:continue
            if st.start in replacements:raise ValueError('multiple allocation statements on one source line')
            # Preserve the original conditional allocation control flow.
            # Native modules may define a procedure named LBOUND (SMOKE does).
            # Restrict intrinsic resolution to instrumentation, preserving
            # every native call to that procedure outside this block.
            body='  allocate('+match[2]+')\n  block\n    intrinsic :: lbound\n'+''.join(hooks)+'  end block\n'
            if match[1]:body='  '+match[1]+' then\n'+body+'  endif\n'
            replacements[st.start]=(st.end,body)
            if not framework:
                imports[signature.end]='  use mpas_oracle_dump, only: oracle_allocation\n'
    output=[];i=0
    while i<len(lines):
        if i in imports:output.append(imports[i])
        if i in replacements:i,body=replacements[i];output.append(body)
        else:output.append(lines[i]);i+=1
    return ''.join(output),records


def generate_allocations(root,out,manifest):
    infos=modules(root,manifest.get('pin_set','mpas-v842'),(m['path'] for m in manifest['module_state']))
    names={m['name'] for m in manifest['module_state']};records=[]
    for relative in sorted({m['path'] for m in manifest['module_state']}):
        target=out/relative
        text,found=patch_allocations(root/relative,target.read_text(),infos,names,
                                     framework=relative.startswith(('src/framework/','src/external/')))
        if found:target.write_text(text)
        records.extend(dict(row,path=relative) for row in found)
    manifest['allocation_state']=records
    manifest['hidden_state_complete']=False


def generate_automatic_entries(root,out,manifest):
    """Preserve native temporary and primitive OUT entry storage.

    Automatic local arrays can leave observable output words unset. Each
    invocation restores its captured entry bytes, never its output bytes.
    Scalar OUT dummies can also be left unwritten on a native branch. Their
    caller's entry words are inputs to that execution, even though Fortran
    marks the dummy undefined. Restore observed PRE words, never a substitute.
    Saved and explicitly allocated storage already have their own hooks.
    """
    from local_state import local_variables
    names={m['name'] for m in manifest['module_state']};records=[]
    for relative in sorted({m['path'] for m in manifest['module_state']}):
        original=root/relative;target=out/relative;text=target.read_text()
        originals={(r.module,r.name):r for r in routines(original,original.parent)}
        # Callback interface declarations have OUT dummies but no execution.
        # Their source bodies must remain declarations, without I/O hooks.
        interface_spans=[];interface_start=None
        for statement in statements(original.read_text()):
            if re.match(r'(?:abstract\s+)?interface\b',statement.text,re.I):
                interface_start=statement.start
            elif re.match(r'end\s*interface\b',statement.text,re.I) and interface_start is not None:
                interface_spans.append((interface_start,statement.end));interface_start=None
        class Source:
            def read_text(self,**kw):return text
            def relative_to(self,root):return original.name
        inserts={}
        for routine in routines(Source(),original.parent):
            origin=originals.get((routine.module,routine.name))
            if origin is None or routine.module not in names:continue
            if any(start<=origin.start<end for start,end in interface_spans):continue
            saved,_=local_variables(origin,root,manifest.get('pin_set','mpas-v842'))
            saved={v['name'] for v in saved}
            variables=[v for v in origin.variables.values() if
                       ((v.dims and v.name not in origin.args) or
                        (v.name in origin.args and v.intent=='out')) and v.name not in saved and
                       not re.search(r'\b(?:parameter|save|allocatable|pointer)\b',v.attrs,re.I)]
            if not variables:continue
            signature=next(s for s in statements(text) if s.start==routine.start)
            row={'module':routine.module,'routine':routine.name,'path':relative,
                 'line':origin.start+1,'end_line':origin.end,'variables':[],'issues':[]}
            records.append(row)
            if re.search(r'\b(?:pure|elemental)\b',signature.text,re.I):
                row['issues'].append('PURE/ELEMENTAL temporary entry cannot perform instrumentation I/O');continue
            hooks=[]
            for var in variables:
                if any(dimension.rsplit(':',1)[-1].strip()=='*' for dimension in var.dims):
                    row['issues'].append(f'{var.name}: assumed-size entry extent requires caller descriptor')
                    continue
                if not var.supported or len(var.dims)>5:
                    row['issues'].append(f'{var.name}: temporary entry type/rank unresolved');continue
                tag=f'{origin.module}.{origin.name}.{var.name}.entry.L{origin.start+1}'
                row['variables'].append({'name':var.name,'record':tag,'type':var.type,'dims':var.dims,
                    'storage':'dummy-out' if var.name in origin.args else 'automatic-local'})
                lower=f', lower=lbound({var.name})' if var.dims else ''
                hook=f"  call oracle_allocation('{tag}', &\n       {var.name}{lower})\n"
                if var.optional:hook=f'  if(present({var.name})) then\n'+hook+'  endif\n'
                hooks.append(hook)
            if not hooks:continue
            first=first_executable(routine)
            if first is None:
                # The pinned ESMF attribute routines are empty implementations
                # with optional OUT return codes. Capture at their actual END,
                # while callback interface declarations remain excluded above.
                first=next((statement for statement in statements(text)
                    if statement.end==routine.end and
                    re.match(r'end\s*(?:subroutine|function)\b',statement.text,re.I)),None)
            if first is None:raise ValueError(f'native storage entry missing: {routine.name}')
            if not relative.startswith(('src/framework/','src/external/')):
                inserts.setdefault(signature.end,[]).append('  use mpas_oracle_dump, only: oracle_allocation\n')
            inserts.setdefault(first.start,[]).append('  block\n    intrinsic :: lbound\n'+''.join(hooks)+'  end block\n')
        if inserts:
            lines=text.splitlines(keepends=True)
            target.write_text(''.join(''.join(inserts.get(i,[]))+line for i,line in enumerate(lines)))
    manifest['automatic_entry_state']=records
    manifest['hidden_state_complete']=False

def patch_entry_storage(original,text,requests):
    """Capture requested native local arrays at compiler-allocated entry.

    Like ALLOCATE inputs, replay restores PRE bytes before numerical execution.
    The request table names source storage, never values or forecast cases.
    Declaration/control changes are explicit failures, not silent omissions.
    """
    from local_state import local_variables
    from fortran_source import first_executable
    class Source:
        def read_text(self,**kw):return text
        def relative_to(self,root):return original.name
    native={(r.module,r.name):r for r in routines(original,original.parent)}
    generated={(r.module,r.name):r for r in routines(Source(),original.parent)}
    lines=text.splitlines(keepends=True);inserts={};records=[]
    for request in requests:
        key=(request['module'],request['routine'])
        origin=native.get(key);current=generated.get(key)
        if origin is None or current is None:
            raise ValueError(f'entry storage native routine missing: {key}')
        signature=next(s for s in statements(text) if s.start==current.start)
        if re.search(r'\b(?:pure|elemental)\b',signature.text,re.I):
            raise ValueError(f'entry storage cannot perform I/O in PURE/ELEMENTAL routine: {key}')
        first=first_executable(current)
        if first is None:raise ValueError(f'entry storage executable entry missing: {key}')
        hooks=[]
        saved,issues=local_variables(origin)
        for name in request['variables']:
            var=origin.variables.get(name)
            if (var is None or name in origin.args or not var.supported or not var.dims
                    or any(a in var.attrs for a in ('allocatable','pointer','parameter','save'))
                    or var.initialized or len(var.dims)>5):
                raise ValueError(f'entry storage declaration changed: {key}.{name}')
            if issues or any(v['name']==name for v in saved):
                raise ValueError(f'entry storage is no longer automatic/local: {key}.{name}')
            declaration=next(s for s in origin.body if any(v.name==name for v in declarations(s.text)))
            tag=f'{origin.module}.{origin.name}.{name}.L{declaration.start+1}'
            row=dict(request,name=name,record=tag,type=var.type,dims=var.dims,
                     line=declaration.start+1,end_line=declaration.end,
                     capture='native-entry-before-first-executable',issues=[])
            records.append(row)
            hooks.append(f"  call oracle_allocation('{tag}', &\n       {name},lower=lbound({name}))\n")
        if not any(re.match(r'use\s+mpas_oracle_dump\b',s.text,re.I)
                   and re.search(r'\boracle_allocation\b',s.text,re.I) for s in current.body):
            if not request.get('path',original.as_posix()).startswith(('src/framework/','src/external/')):
                inserts.setdefault(signature.end,[]).append('  use mpas_oracle_dump, only: oracle_allocation\n')
        inserts.setdefault(first.start,[]).extend(hooks)
    output=[]
    for index,line in enumerate(lines):
        output.extend(inserts.get(index,[]));output.append(line)
    return ''.join(output),records


def generate_entry_storage(root,out,manifest):
    table=Path(__file__).with_name('entry-storage.v1.json')
    requests=json.loads(table.read_text())['requests'];records=[]
    for relative in sorted({r['path'] for r in requests}):
        matching=[r for r in requests if r['path']==relative]
        target=out/relative
        if not target.is_file():
            records.extend(dict(r,issues=['requested native entry storage source is absent']) for r in matching)
            continue
        text,found=patch_entry_storage(root/relative,target.read_text(),matching)
        target.write_text(text);records.extend(found)
    manifest['entry_storage']=records
