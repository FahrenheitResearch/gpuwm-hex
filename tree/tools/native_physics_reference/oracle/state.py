"""Generate private module-state snapshots and list unresolved hidden inputs.

Module variables have implicit SAVE in Fortran. Local explicit/initialized SAVE
and imported modules are separately enumerated so a module snapshot cannot be
mistaken for a complete replay. MPAS physics_init.F:83-481 is the init hook.
"""
from dataclasses import asdict
from pathlib import Path
import re
from fortran_source import active_source,statements,declarations,routines,split_top


def modules(root, pin_set_name="mpas-v842", extra_paths=()):
    from pin_sets import source_files, pin_set
    config=pin_set(pin_set_name)
    result={}
    selected=source_files(root,pin_set_name,'src/core_atmosphere/physics')
    selected.extend(root/p for p in extra_paths if (root/p).is_file())
    selected.extend(root/p for p in ('src/framework/mpas_timekeeping.F',) if (root/p).is_file())
    selected.extend((root/'src/external/esmf_time_f90').glob('*.F90'))
    # Keep fork source-family selection and the landed framework SAVE inventory.
    selected.extend((root/'src/framework').glob('*'))
    for path in sorted(set(selected)):
        if path.suffix.lower() not in {'.f','.f90'}:continue
        current=None;in_spec=False;type_depth=0;interface_depth=0
        for st in statements(active_source(path.read_text(errors='replace'))):
            match=re.fullmatch(r'module\s+(?!procedure\b|function\b|subroutine\b)(\w+)',st.text,re.I)
            if match:
                name=match[1].lower();in_spec=True;type_depth=0;interface_depth=0
                current={'name':name,'path':str(path.relative_to(root)),'line':st.start+1,
                         'contains':None,'end':None,'variables':[],'roots':[], 'uses':[], 'use_statements':[], 'issues':[],'parameters':[]}
                result[name]=current
                continue
            if not current:continue
            if re.match(r'end\s*module\b',st.text,re.I):current['end']=st.start;current=None;continue
            if re.match(r'(?:abstract\s+)?interface\b',st.text,re.I):
                interface_depth+=1;continue
            if re.match(r'end\s*interface\b',st.text,re.I):
                if interface_depth<1:raise ValueError(f'unmatched END INTERFACE at {path}:{st.start+1}')
                interface_depth-=1;continue
            if interface_depth:continue
            if re.match(r'contains\b',st.text,re.I) and in_spec and not type_depth:
                current['contains']=st.start;in_spec=False;continue
            use=re.match(r'use\s+(\w+)',st.text,re.I)
            if use:
                current['uses'].append(use[1].lower())
                if in_spec:current['use_statements'].append(st.text)
            if not in_spec:continue
            if re.match(r'type\s*(?:,.*)?::\s*\w+|type\s+\w+\s*$',st.text,re.I):type_depth+=1;continue
            if re.match(r'end\s*type\b',st.text,re.I):
                if type_depth<1:raise ValueError(f'unmatched END TYPE at {path}:{st.start+1}')
                type_depth-=1;continue
            if type_depth:continue
            parameter=re.match(r'parameter\s*\((.*)\)',st.text,re.I)
            if parameter:
                current['parameters'].extend(e.split('=',1)[0].strip().lower() for e in split_top(parameter[1]))
                continue
            for var in declarations(st.text):
                if 'parameter' in var.attrs:continue
                record=dict(asdict(var),line=st.start+1)
                current['roots'].append(record)
                if current['name']=='mpas_stream_manager' and re.fullmatch(r'type\s*\(\s*field2dinteger\s*\)',var.type,re.I):
                    if var.dims or 'pointer'not in var.attrs:
                        raise ValueError('native stream saved root storage changed')
                    if not any(v.get('graph_stream')for v in current['variables']):
                        from stream_graph import schema
                        record['graph_stream']=schema(root)
                        current['variables'].append(record)
                        current['issues'].append(dict(name=var.name,line=st.start+1,
                            reason='stream external aliases, non-null external pointees and uninitialized unused metadata remain unproven'))
                    continue
                if re.fullmatch(r'type\s*\(\s*mpas_log_type\s*\)',var.type,re.I):
                    from framework_log import components
                    leaves=components(root,var)
                    for leaf in leaves:
                        leaf.setdefault('line',st.start+1)
                    current['variables'].extend(leaves)
                    current['issues'].append(dict(name=var.name,line=st.start+1,
                        reason='logger external aliases and live file-descriptor positions remain unproven'))
                    continue
                if re.fullmatch(r'type\s*\(\s*mpas_clock_type\s*\)',var.type,re.I):
                    if var.dims or 'pointer' not in var.attrs:
                        raise ValueError('native clock root storage changed')
                    from clock_graph import schema
                    record['graph_clock']=schema(root)
                    current['variables'].append(record)
                    continue
                if re.fullmatch(r'type\s*\(\s*mpas_timer_root\s*\)',var.type,re.I):
                    if var.dims or 'pointer' not in var.attrs:
                        raise ValueError('native timer root storage changed')
                    from timer_graph import schema
                    record['graph_timer']=schema(root)
                    current['variables'].append(record)
                    current['issues'].append(dict(name=var.name,line=st.start+1,
                        reason='timer external aliases, MPI resources and future wall-clock observations remain unproven'))
                    continue
                if re.fullmatch(r'type\s*\(\s*esmf_(?:basetime|calendar)\s*\)',var.type,re.I):
                    from framework_calendar import components
                    current['variables'].extend(components(root,var))
                    continue
                if re.fullmatch(r'type\s*\(\s*mpas_timeinterval_type\s*\)',var.type,re.I):
                    from derived import time_interval_components
                    current['variables'].extend(time_interval_components(root,var.name))
                    continue
                if re.fullmatch(r'type\s*\(\s*noahmpio_type\s*\)',var.type,re.I):
                    from derived import noahmp_components
                    for component in noahmp_components(root,var.name,var.name,'inout'):
                        current['variables'].append(dict(component,name=component['actual'],line=st.start+1))
                    continue
                derived=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',var.type,re.I)
                if derived and derived[1].lower() in config['derived_types']:
                    from derived import primitive_components
                    pointer='pointer' in var.attrs
                    if pointer:current['variables'].append(dict(record,container_pointer=True,conditions=[]))
                    for component in primitive_components(root,config['derived_types'][derived[1].lower()],derived[1],var.name,var.name,'inout',var.dims):
                        current['variables'].append(dict(component,name=component['actual'],line=st.start+1,conditions=[f'associated({var.name})'] if pointer else []))
                    continue
                supported=var.supported
                # Helpers run inside the native module and retain its REAL kind
                # aliases. Only INTEGER needs validation against transport kinds;
                # replay-local type spelling must not discard native REAL tables.
                if var.type.lower().startswith(('integer','complex','double complex')):
                    from replay.generate_replay import replay_type
                    try: replay_type(var.type)
                    except ValueError: supported=False
                if not supported:
                    current['issues'].append({'name':var.name,'line':st.start+1,'reason':'unsupported hidden-state declaration'})
                else:current['variables'].append(record)
        # Mutable routine SAVE state cannot be reached from module helpers.
        for routine in routines(path,root):
            if routine.module not in result:continue
            saved={v.name for v in routine.variables.values() if 'save' in v.attrs or v.initialized and 'parameter' not in v.attrs}
            for st in routine.body:
                if re.match(r'save\b',st.text,re.I):
                    names=re.sub(r'^save\s*(?:::)?','',st.text,flags=re.I).strip()
                    saved.update(x.strip().lower() for x in names.split(',') if x.strip())
                if re.match(r'common\b|data\b',st.text,re.I):
                    result[routine.module]['issues'].append({'routine':routine.name,'line':st.start+1,'reason':'COMMON/DATA local state requires capture'})
            for name in sorted(saved):
                result[routine.module]['issues'].append({'routine':routine.name,'name':name,'line':routine.start+1,'reason':'local SAVE state requires capture'})
    for info in result.values():
        info['variables']=[v for v in info['variables'] if v['name'] not in info['parameters']]
        info['roots']=[v for v in info['roots'] if v['name'] not in info['parameters']]
    return result


def state_call(module,phase,event):
    return f"      call oracle_state_{module}('state__{module}','{phase}',{event})\n"


def module_anchors(text,name):
    """Locate module CONTAINS, skipping derived-type bound procedures."""
    lines=text.splitlines(keepends=True)
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    inside=False;depth=0;contains=None
    for st in statements(text):
        if not inside:
            if re.fullmatch(r'module\s+'+re.escape(name),st.text,re.I):inside=True
            continue
        if re.match(r'end\s*module\b',st.text,re.I):
            return contains,offsets[st.start]
        if re.match(r'type\s*(?:,.*)?::\s*\w+|type\s+\w+\s*$',st.text,re.I):
            depth+=1
        elif re.match(r'end\s*type\b',st.text,re.I):depth-=1
        elif re.fullmatch(r'contains',st.text,re.I) and depth==0 and contains is None:
            contains=offsets[st.start]
    raise ValueError(f'module end anchor missing: {name}')


def generate_state(root,out,manifest):
    from boundary_state import inventory_boundaries,instrument_boundaries
    infos=modules(root,manifest.get("pin_set","mpas-v842"),(
        "src/core_atmosphere/dynamics/mpas_atm_iau.F","src/core_atmosphere/mpas_atm_core.F"))
    inventory_boundaries(root,manifest,infos)
    needed={s['callee']['module'] for s in manifest['sites']}
    needed.update(b['callee']['module'] for b in manifest.get('boundaries',[]) if b.get('callee'))
    needed.update({'mpas_atmphys_init','mpas_atmphys_vars','mpas_atmphys_rrtmg_lwinit','mpas_atmphys_rrtmg_swinit',
                   'mpas_atmphys_lsm_noahinit','mpas_atmphys_o3climatology'})
    pending=list(needed)
    while pending:
        name=pending.pop()
        for dep in infos.get(name,{}).get('uses',[]):
            if dep in infos and dep not in needed:needed.add(dep);pending.append(dep)
    needed={n for n in needed if n in infos}
    by_path={}
    for name in needed:by_path.setdefault(infos[name]['path'],[]).append(infos[name])
    for relative,mods in by_path.items():
        target=out/relative
        text=target.read_text() if target.exists() else (root/relative).read_text()
        # If argument overlay already changed a driver module, use textual anchors
        # from the original specification and end module, never stale line offsets.
        for info in mods:
            name=info['name']
            start=re.search(r'^\s*module\s+'+re.escape(name)+r'\s*$',text,re.I|re.M)
            if not start:raise ValueError(f'module anchor missing: {name}')
            if info['path'].startswith(('src/framework/','src/external/')):
                from framework_state import interface_source
                # Interfaces are specification statements: native USE/IMPLICIT
                # statements must precede them. Hide transport names from users
                # of this public framework module.
                contains_at,end_at=module_anchors(text,name)
                at=contains_at if contains_at is not None else end_at
                interface=interface_source()
                if any(v.get('graph_timer') for v in info['variables']):
                    from timer_graph import interface_source as timer_interface
                    interface+=timer_interface()
                for v in info['variables']:
                    if v.get('graph_stream'):
                        from stream_graph import interface_source as stream_interface
                        interface+=stream_interface(v['graph_stream']['root_names'])
                text=text[:at]+'\n'+interface+text[at:]
            else:
                imports="\n use mpas_oracle_dump, only: oracle_write,oracle_read,oracle_header,oracle_enabled\n"
                if any(v.get('graph_clock') for v in info['variables']):
                    imports+=' use mpas_oracle_clock, only: oracle_clock_capture,oracle_clock_restore\n'
                if any(v.get('graph_timer') for v in info['variables']):
                    imports+=' use mpas_oracle_timer, only: oracle_timer_capture,oracle_timer_restore\n'
                text=text[:start.end()]+imports+text[start.end():]
            offset,end_offset=module_anchors(text,name)
            if offset is None:
                text=text[:end_offset]+'\n contains\n'+text[end_offset:]
                offset,_=module_anchors(text,name)
            exports=[s['callee'].get('name',s['scheme']) for s in manifest['sites'] if s['callee']['module']==name and s['callee']['path']==s['path']]
            public=f'\n public :: oracle_state_{name}, oracle_restore_{name}\n'
            # Driver-local scheme routines can otherwise be private to replay.
            # Only add access declarations where the original module defaults private.
            spec=text[start.end():offset]
            if re.search(r'^\s*private\s*$',spec,re.I|re.M):
                public+=''.join(f' public :: {s}\n' for s in sorted(set(exports)) if not any(re.match(r'public\s*(?:::)?',st.text,re.I) and re.search(r'\b'+s+r'\b',st.text,re.I) for st in statements(spec)))
            text=text[:offset]+public+text[offset:]
            pointer_containers=[v for v in info['variables'] if v.get('container_pointer')]
            global_calendars=[v for v in pointer_containers
                              if re.fullmatch(r'type\s*\(\s*esmf_calendar\s*\)',v['type'],re.I)
                              and name!='esmf_calendarmod']
            calendar_slots=('defaultcal','gregoriancal','noleapcal')
            if global_calendars:
                imports='\n use ESMF_CalendarMod, only: '+', &\n     '.join(
                    f'oracle_global_{slot}=>{slot}' for slot in calendar_slots)+'\n'
                text=text[:start.end()]+imports+text[start.end():]
                contains=re.search(r'^\s*contains\s*$',text[start.end():],re.I|re.M)
                at=start.end()+contains.start()
                private='\n private :: '+','.join('oracle_global_'+slot for slot in calendar_slots)+'\n'
                text=text[:at]+private+text[at:]
            helper=f'''\n subroutine oracle_state_{name}(key,phase,event)
  character(len=*), intent(in) :: key,phase
  integer, intent(in) :: event
  integer :: state_alias
  character(len=256) :: state_prefix
  if(.not.oracle_enabled(key)) return
  state_prefix='state.{name}.'
'''
            for var in info['variables']:
                value=var['name'];guard=''
                if var.get('graph_stream'):
                    roots=','.join(var['graph_stream']['root_names'])
                    helper+=f"  call oracle_framework_stream_capture(key,phase,trim(state_prefix)//'{value}',{roots},event)\n"
                    continue
                if var.get('graph_clock'):
                    helper+=f"  call oracle_clock_capture(key,phase,trim(state_prefix)//'{value}',{value},event)\n"
                    continue
                if var.get('graph_timer'):
                    capture='oracle_framework_timer_capture' if info['path'].startswith(('src/framework/','src/external/')) else 'oracle_timer_capture'
                    helper+=f"  call {capture}(key,phase,trim(state_prefix)//'{value}',{value},event)\n"
                    continue
                conditions=var.get('conditions',[])
                for condition in conditions:helper+=f'  if({condition}) then\n'
                if var.get('container_pointer'):
                    helper+=f"  call oracle_write(key,phase,trim(state_prefix)//'{value}.present',associated({value}),event)\n"
                    alias_id=pointer_containers.index(var)+1
                    helper+=f'  state_alias=0\n  if(associated({value})) then\n   state_alias={alias_id}\n'
                    for earlier in pointer_containers[:alias_id-1]:
                        if earlier['type']!=var['type']:continue
                        for condition in earlier.get('conditions',[]):helper+=f'  if({condition}) then\n'
                        previous=earlier['name'];previous_id=pointer_containers.index(earlier)+1
                        helper+=f'  if(state_alias=={alias_id}) then\n   if(associated({previous})) then\n    if(associated({value},{previous})) state_alias={previous_id}\n   endif\n  endif\n'
                        helper+='  endif\n'*len(earlier.get('conditions',[]))
                    if var in global_calendars:
                        for slot_id,slot in enumerate(calendar_slots,1):
                            alias='oracle_global_'+slot
                            helper+=f'  if(state_alias>0.and.associated({alias})) then\n'
                            helper+=f'   if(associated({value},{alias})) state_alias=-{slot_id}\n  endif\n'
                    helper+='  endif\n'
                    helper+=f"  call oracle_write(key,phase,trim(state_prefix)//'{value}.alias',state_alias,event)\n"
                    helper+='  endif\n'*len(conditions)
                    continue
                if 'allocatable' in var['attrs']:guard=f'allocated({value})'
                elif 'pointer' in var['attrs']:guard=f'associated({value})'
                if guard:
                    helper+=f"  call oracle_write(key,phase,trim(state_prefix)//'{value}.present',{guard},event)\n  if({guard}) then\n"
                lower=',lower='+var.get('lower_expr',f'lbound({value})') if var['dims'] else ''
                helper+=f"  call oracle_write(key,phase,trim(state_prefix)//'{value}',{value},event{lower})\n"
                if guard:helper+='  endif\n'
                helper+='  endif\n'*len(conditions)
            helper+=' end subroutine\n'
            helper+=f''' subroutine oracle_restore_{name}(prefix)
  character(len=*), intent(in) :: prefix
  character(len=4096) :: path
  integer :: code,width,rank_n,dims(5),lb(5),unit,state_alias
  logical :: state_present
'''
            for var in info['variables']:
                value=var['name'];rank=len(var['dims'])
                if var.get('graph_stream'):
                    roots=','.join(var['graph_stream']['root_names'])
                    helper+=f"  call oracle_framework_stream_restore(prefix,'state.{name}.{value}',{roots})\n"
                    continue
                if var.get('graph_clock'):
                    helper+=f"  call oracle_clock_restore(prefix,'state.{name}.{value}',{value})\n"
                    continue
                if var.get('graph_timer'):
                    restore='oracle_framework_timer_restore' if info['path'].startswith(('src/framework/','src/external/')) else 'oracle_timer_restore'
                    helper+=f"  call {restore}(prefix,'state.{name}.{value}',{value})\n"
                    continue
                conditions=var.get('conditions',[])
                for condition in conditions:helper+=f'  if({condition}) then\n'
                if var.get('container_pointer'):
                    helper+=f"  call oracle_read(trim(prefix)//'.pre.state.{name}.{value}.present.bin',state_present)\n"
                    helper+=f"  call oracle_read(trim(prefix)//'.pre.state.{name}.{value}.alias.bin',state_alias)\n"
                    alias_id=pointer_containers.index(var)+1
                    preserve=var.get('preserve_live_target',False)
                    if not preserve:helper+=f'  nullify({value})\n'
                    helper+=f'  if(state_present) then\n   select case(state_alias)\n   case({alias_id})\n'
                    if preserve:
                        # A live target can have been rebound to an earlier
                        # class since capture. Split that alias before restoring
                        # the distinct captured object; never deallocate a
                        # target still owned by the earlier class.
                        for earlier in pointer_containers[:alias_id-1]:
                            if earlier['type']!=var['type']:continue
                            previous=earlier['name']
                            for condition in earlier.get('conditions',[]):helper+=f'    if({condition}) then\n'
                            helper+=f'    if(associated({value})) then\n     if(associated({previous})) then\n      if(associated({value},{previous})) nullify({value})\n     endif\n    endif\n'
                            helper+='    endif\n'*len(earlier.get('conditions',[]))
                        helper+=f'    if(.not.associated({value})) then\n'
                    helper+=f'    allocate({value})\n'
                    for component in var.get('nullify_new_components',[]):
                        helper+=f'    nullify({value}%{component})\n'
                    if preserve:helper+='    endif\n'
                    for earlier in pointer_containers[:alias_id-1]:
                        if earlier['type']!=var['type']:continue
                        previous=earlier['name'];previous_id=pointer_containers.index(earlier)+1
                        helper+=f"   case({previous_id})\n    if(.not.associated({previous})) error stop 'oracle absent pointer alias target'\n    {value}=>{previous}\n"
                    if var in global_calendars:
                        for slot_id,slot in enumerate(calendar_slots,1):
                            alias='oracle_global_'+slot
                            helper+=f"   case(-{slot_id})\n    if(.not.associated({alias})) error stop 'oracle global calendar target not restored'\n    {value}=>{alias}\n"
                    helper+="   case default\n    error stop 'oracle invalid pointer alias class'\n   end select\n  else\n   if(state_alias/=0) error stop 'oracle absent pointer has alias'\n"
                    if preserve:helper+=f'   nullify({value})\n'
                    helper+='  endif\n'
                    helper+='  endif\n'*len(conditions)
                    continue
                alloc='allocatable' in var['attrs'];pointer='pointer' in var['attrs']
                if pointer:
                    info['issues'].append({'name':value,'line':var['line'],'reason':'pointer target alias identity requires restoration'})
                helper+=f"  path=trim(prefix)//'.pre.state.{name}.{value}.bin'\n"
                if alloc or pointer:
                    helper+=f"  call oracle_read(trim(prefix)//'.pre.state.{name}.{value}.present.bin',state_present)\n"
                    guard=f'allocated({value})' if alloc else f'associated({value})'
                    if alloc:helper+=f'  if({guard}) deallocate({value})\n'
                    if pointer:helper+=f'  nullify({value})\n'
                    helper+='  if(state_present) then\n'
                    if rank:
                        helper+='  call oracle_header(path,code,width,rank_n,dims,lb,unit)\n  close(unit)\n'
                    bounds=','.join(f'lb({i}):lb({i})+dims({i})-1' for i in range(1,rank+1))
                    helper+=f'  allocate({value}'+(f'({bounds})' if rank else '')+')\n'
                helper+=f'  call oracle_read(path,{value})\n'
                if alloc or pointer:helper+='  endif\n'
                helper+='  endif\n'*len(conditions)
            helper+=' end subroutine\n'
            end=re.search(r'^\s*end\s*module\s+'+re.escape(name)+r'\b',text,re.I|re.M)
            if not end:raise ValueError(f'end module anchor missing: {name}')
            text=text[:end.start()]+helper+text[end.start():]
            if any(v.get('graph_stream')for v in info['variables']):
                from stream_graph import instrument_stream_writes
                text=instrument_stream_writes(text)
            if any(v.get('graph_clock') for v in info['variables']):
                # A scalar external bridge avoids physics_init -> atm_core ->
                # physics_init module dependence. Both units use native ABI.
                text+=f'''\nsubroutine oracle_init_clock_{name}(key,phase,event)
 use {name},only:oracle_state_{name}
 implicit none
 character(len=*),intent(in)::key,phase
 integer,intent(in)::event
 call oracle_state_{name}(key,phase,event)
end subroutine
'''
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    # Each actual call captures the transitive module graph as input/output.
    # Local SAVE and pointer alias identity remain explicit completeness gaps.
    driver_imports={}
    for site in manifest['sites']:
        if site['issues']:continue
        closure={site['callee']['module']};pending=list(closure)
        while pending:
            for dep in infos.get(pending.pop(),{}).get('uses',[]):
                if dep in needed and dep not in closure:closure.add(dep);pending.append(dep)
        site['state_modules']=sorted(closure & needed)
        site['runtime_modules']=sorted({'mpas_log'} & {
            dep for name in closure for dep in infos.get(name,{}).get('uses',[])})
        candidates=[(n,v) for n in site['state_modules'] for v in infos[n]['variables'] if not v.get('conditions')]
        if candidates:
            module,var=candidates[0]
            suffix='.present' if var.get('container_pointer') or any(a in var['attrs'] for a in ('pointer','allocatable')) else ''
            site['context_record']='state.'+module+'.'+var['name']+suffix

        target=out/site['path'];text=target.read_text()
        raw=(root/site['path']).read_text().splitlines(keepends=True)
        begin=f"! oracle_native_call_begin {site['site']}\n"
        end=f"! oracle_native_call_end {site['site']}\n"
        if text.count(begin)!=1 or text.count(end)!=1:raise ValueError(f'ambiguous site marker: {site["site"]}')
        start=text.index(begin);finish=text.index(end,start)+len(end)
        anchor=text[start:finish]
        if text.count(anchor)!=1:raise ValueError(f'ambiguous state call anchor: {site["site"]}')
        before='      if(.not.oracle_replaying()) then\n'+''.join(f"      call oracle_state_{n}('{site['site']}','pre',oracle_event)\n" for n in site['state_modules'])
        after='      if(.not.oracle_replaying()) then\n'+''.join(f"      call oracle_state_{n}('{site['site']}','post',oracle_event)\n" for n in site['state_modules'])
        text=text.replace(anchor,before+'      endif\n'+anchor+after+'      endif\n',1);target.write_text(text)
        driver_imports.setdefault((site['path'],site['driver']),set()).update(site['state_modules'])
    for (relative,driver),names in driver_imports.items():
        target=out/relative;text=target.read_text()
        routine=next(r for r in routines(root/relative,root) if r.name==driver)
        raw=(root/relative).read_text().splitlines(keepends=True)
        sig=next(st for st in statements(''.join(raw)) if st.start==routine.start)
        anchor=''.join(raw[sig.start:sig.end])
        imports=''.join(f'      use {n}, only: oracle_state_{n}\n' for n in sorted(names) if n!=routine.module)
        text=text.replace(anchor,anchor+imports,1);target.write_text(text)
    instrument_boundaries(root,out,manifest,infos,needed)
    # Physics initialization cannot import its callers in the core/dynamics:
    # that would create a Fortran module cycle. Those boundary modules retain
    # their own entry/exit checkpoints; expose the init inventory separately.
    init_needed={n for n in needed if infos[n]['path'].startswith(('src/core_atmosphere/physics/', 'src/framework/','src/external/'))}
    manifest['init_state_modules']=sorted(init_needed)
    clock_modules={n for n in needed if any(v.get('graph_clock') for v in infos[n]['variables'])}
    manifest['init_state_modules']=sorted(init_needed|clock_modules)
    # Transitive physics module state is dumped once per physics_init call,
    # at step zero with that mesh's global IDs and rank.
    relative='src/core_atmosphere/physics/mpas_atmphys_init.F'
    target=out/relative
    text=target.read_text() if target.exists() else (root/relative).read_text()
    signature=next(r for r in routines(root/relative,root) if r.name=='physics_init')
    original=(root/relative).read_text().splitlines(keepends=True)
    from fortran_source import statements as lex
    sig=next(st for st in lex(''.join(original)) if st.start==signature.start)
    # This module may already contain generated USE statements and state
    # helpers. Original byte offsets no longer identify its routine signature.
    anchor=''.join(original[sig.start:sig.end])
    if text.count(anchor)!=1:raise ValueError('ambiguous physics_init signature anchor')
    imports='      use mpas_oracle_dump, only: oracle_begin,oracle_scope_begin,oracle_scope_end\n      use mpas_oracle_pool, only: oracle_mesh_context\n'
    for name in sorted(init_needed):
        if name!=signature.module:imports+=f'      use {name}, only: oracle_state_{name}\n'
    text=text.replace(anchor,anchor+imports,1)
    from fortran_source import first_executable
    first=first_executable(signature)
    anchor=''.join(original[first.start:first.end])
    entry="      integer :: oracle_init_event\n      call oracle_mesh_context(mesh,0)\n      call oracle_begin('state__physics_init',oracle_init_event)\n      call oracle_scope_begin('state__physics_init',oracle_init_event)\n"
    text=text.replace(anchor,entry+anchor,1)
    end=re.search(r'^\s*end\s*subroutine\s+physics_init\b',text,re.I|re.M)
    calls=""
    calls+=''.join(f"      call oracle_state_{n}('state__physics_init','init',oracle_init_event)\n" for n in sorted(init_needed))
    calls+=''.join(f"      call oracle_init_clock_{n}('state__physics_init','init',oracle_init_event)\n" for n in sorted(clock_modules))
    calls+='      call oracle_scope_end()\n'
    text=text[:end.start()]+calls+text[end.start():]
    # Close the whole init scope on each native return; later local hooks must
    # observe their POST before this close rather than leak into the caller.
    class InitSource:
        def read_text(self,**kw):return text
        def relative_to(self,root):return relative
    patched=next(r for r in routines(InitSource(),root) if r.name=='physics_init')
    lines=text.splitlines(keepends=True);changes={}
    for st in patched.body:
        tail=re.sub(r'^\d+\s+','',st.text)
        if re.fullmatch(r'return',tail,re.I):
            changes[st.start]=(st.end,calls+''.join(lines[st.start:st.end]))
        elif re.match(r'if\s*\(.*\)\s*return\s*$',tail,re.I):
            condition=re.match(r'if\s*(\(.*\))\s*return',tail,re.I)[1]
            label=re.match(r'\s*(\d+)\s+',lines[st.start])
            lead=(label[1]+' ' if label else '      ')
            changes[st.start]=(st.end,f'{lead}if {condition} then\n'+calls+'      return\n      endif\n')
    result=[];i=0
    while i<len(lines):
        if i in changes:i,part=changes[i];result.append(part)
        else:result.append(lines[i]);i+=1
    text=''.join(result)
    manifest['initialization_scope']=dict(key='state__physics_init',path=relative,line=signature.start+1,end_line=signature.end,state_modules=sorted(init_needed),local_state_complete=False,external_aliases_complete=False)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    manifest['module_state']=[infos[n] for n in sorted(needed)]
    manifest['hidden_state_complete']=False
