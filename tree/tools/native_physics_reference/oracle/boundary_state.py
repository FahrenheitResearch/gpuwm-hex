"""Capture private primitive module inputs/outputs at native driver boundaries.

MPAS: physics/mpas_atmphys_driver.F:202-372, dynamics/
mpas_atm_time_integration.F:2146,2170,2811 and mpas_atm_core.F output diagnostics.
This extends the existing pool hooks. Nested scopes retain supported local SAVE
inputs and outputs; unsupported descriptors and external state remain explicit.
"""
from collections import defaultdict
from fortran_source import routines,statements


def inventory_boundaries(root,manifest,infos):
    by_name=defaultdict(list)
    for relative in sorted({info['path'] for info in infos.values()}):
        for routine in routines(root/relative,root):
            by_name[routine.name].append(routine)
    for boundary in manifest.get('boundaries',[]):
        name=boundary['key'].removeprefix('boundary__')
        candidates=by_name.get(name,[])
        boundary['state_issues']=[]
        if len(candidates)!=1:
            boundary['state_issues'].append('boundary callee is missing or ambiguous')
            continue
        callee=candidates[0]
        boundary['callee']={'module':callee.module,'path':callee.path,
                            'line':callee.start+1,'end_line':callee.end}


def instrument_boundaries(root,out,manifest,infos,needed):
    imports=defaultdict(set)
    for boundary in manifest.get('boundaries',[]):
        boundary['module_state_complete']=False
        boundary['local_state_complete']=False
        if not boundary.get('callee'):continue
        closure={boundary['callee']['module']};pending=list(closure)
        external=set()
        while pending:
            module=pending.pop()
            for dep in infos.get(module,{}).get('uses',[]):
                if dep not in infos:external.add(dep)
                elif dep in needed and dep not in closure:
                    closure.add(dep);pending.append(dep)
        names=sorted(closure & needed)
        boundary['state_modules']=names
        boundary['uninventoried_state_modules']=sorted(external)
        boundary['state_issues'].extend({'module':name,**issue}
                                       for name in names for issue in infos[name]['issues'])
        relative=boundary['path'];target=out/relative;text=target.read_text()
        raw=(root/relative).read_text().splitlines(keepends=True)
        routine=next(r for r in routines(root/relative,root) if r.name==boundary['driver'])
        imports[(relative,routine.name)].update(n for n in names if n!=routine.module)
        event='oracle_output_event' if boundary.get('mode')=='routine' else 'oracle_boundary_event'
        calls=lambda phase:''.join(f"      call oracle_state_{n}('{boundary['key']}','{phase}',{event})\n" for n in names)
        if boundary.get('mode')=='routine':
            entry=f"      call oracle_begin('{boundary['key']}',{event})\n"
            exit=f"      call oracle_pool_capture('{boundary['key']}','post','state',state,{event})\n"
            if text.count(entry)!=1 or text.count(exit)!=1:
                raise ValueError('output boundary state anchors are missing or ambiguous')
            begin=f"      call oracle_scope_begin('{boundary['key']}',{event})\n"
            text=text.replace(entry,entry+begin+calls('pre'),1).replace(exit,'      call oracle_scope_end()\n'+calls('post')+exit,1)
        else:
            begin=f"! oracle_boundary_call_begin {boundary['key']} L{boundary['line']}\n"
            end=f"! oracle_boundary_call_end {boundary['key']} L{boundary['line']}\n"
            if begin in text or end in text:
                if text.count(begin)!=1 or text.count(end)!=1:
                    raise ValueError(f"ambiguous boundary state markers: {boundary['key']}")
                text=text.replace(begin,calls('pre')+begin,1).replace(end,end+calls('post'),1)
            else:
                # Read-only interface fixtures can supply an unmodified call.
                anchor=''.join(raw[boundary['line']-1:boundary['end_line']])
                if text.count(anchor)!=1:
                    raise ValueError(f"ambiguous boundary state call anchor: {boundary['key']}")
                text=text.replace(anchor,
                    f"      call oracle_scope_begin('{boundary['key']}',{event})\n"+
                    calls('pre')+anchor+calls('post')+'      call oracle_scope_end()\n',1)
        target.write_text(text)
    for (relative,name),names in imports.items():
        target=out/relative;text=target.read_text();raw=(root/relative).read_text()
        routine=next(r for r in routines(root/relative,root) if r.name==name)
        signature=next(st for st in statements(raw) if st.start==routine.start)
        anchor=''.join(raw.splitlines(keepends=True)[signature.start:signature.end])
        if text.count(anchor)!=1:raise ValueError(f'ambiguous boundary caller signature: {name}')
        uses='      use mpas_oracle_dump,only:oracle_scope_begin,oracle_scope_end\n'
        uses+=''.join(f'      use {n},only:oracle_state_{n}\n' for n in sorted(names))
        target.write_text(text.replace(anchor,anchor+uses,1))
