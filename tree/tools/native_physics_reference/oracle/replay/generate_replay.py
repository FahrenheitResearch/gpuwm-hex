"""Generate standalone argument replay from pinned primitive/derived interfaces.

Inputs are decoded in Fortran. Module state restores precede the original call;
unresolved hidden state is still a separate completeness requirement.
"""
import re
from derived import array_arguments, has_presence


def replay_type(typ):
    if typ.startswith('real'):
        m = re.search(r'(?:kind\s*=\s*)?(\w+)\s*\)', typ)
        kind = m[1] if m else ('8' if '*8' in typ else '4')
        if kind in {'8', 'r8kind', 'kind_rb', 'rb', 'dp', 'real64'}:
            return 'real(real64)'
        if kind in {'4', 'rkind', 'kind_phys', 'kind_noahmp', 'real32', 'wp', 'sp'}:
            return 'real(real32)'
        raise ValueError(f'unresolved real kind: {typ}')
    if typ.startswith('double precision'):
        return 'real(real64)'
    if typ.startswith('double complex'):
        return 'complex(real64)'
    if typ.startswith('complex'):
        m=re.fullmatch(r'complex(?:\s*\(\s*(?:kind\s*=\s*)?(\w+)\s*\)|\*(\d+))?',typ)
        kind=(m[1] or ('4' if m[2]=='8' else '8' if m[2]=='16' else m[2]) or '4') if m else ''
        if kind in {'4','rkind','kind_phys','kind_noahmp','real32','wp','sp'}:return 'complex(real32)'
        if kind in {'8','r8kind','kind_rb','rb','real64','dp'}:return 'complex(real64)'
        raise ValueError(f'unresolved complex kind: {typ}')
    if typ.startswith('integer'):
        m=re.fullmatch(r'integer(?:\s*\(\s*(?:kind\s*=\s*)?(\w+)\s*\)|\*(\d+))?',typ)
        kind=(m[1] or m[2] or '4') if m else ''
        if kind in {'4','int32','im','kind_im','i4kind'}:return 'integer(int32)'
        if kind in {'8','int64','i8kind','esmf_kind_i8'}:return 'integer(int64)'
        raise ValueError(f'unresolved integer kind: {typ}')
    if typ.startswith('logical'):
        if '(' in typ or '*' in typ:
            raise ValueError(f'unresolved logical kind: {typ}')
        return 'logical'
    if typ.startswith('character'):
        return 'character(len=:), allocatable'
    raise ValueError(f'unresolved type: {typ}')


def axes(a):
    return (a.get('cell_axis',next((i+1 for i,d in enumerate(a['dims']) if re.search(r'\bims\b|\bits\b',d,re.I)),0)),
            a.get('level_axis',next((i+1 for i,d in enumerate(a['dims']) if re.search(r'\bkms\b|\bkts\b',d,re.I)),0)))


def native_runtime(site):
    """Initialize native logging for both captured replay and source fixtures."""
    if 'mpas_log' not in site.get('runtime_modules',[]):return '', '', ''
    imports='  use mpas_log, only: mpas_log_init\n  use mpas_derived_types, only: domain_type'
    declarations='  type(domain_type), pointer :: replay_domain'
    # MPAS framework/mpas_log.F:99-294 requires native domain/core and DM
    # metadata even for informational messages. Keep its original initializer.
    setup="""  allocate(replay_domain)
  allocate(replay_domain%core,replay_domain%dminfo)
  replay_domain%core%coreName='atmosphere'
  replay_domain%domainID=0
  replay_domain%dminfo%my_proc_id=0
  replay_domain%dminfo%nprocs=1
  call mpas_log_init(replay_domain%logInfo,replay_domain,unitNumbers=[6,0])"""
    return imports, declarations, setup


def generate(site):
    if site['issues']:
        raise ValueError('; '.join(site['issues']))
    args=site['arguments'];leaves=array_arguments(site)
    declarations,reads,writes,imports,contexts=[],[],[],[],[]
    aliases={}
    buffer_aliases=site.get('argument_aliases',{})
    buffer_targets=set(buffer_aliases.values())
    associations=[]
    by_name={a['name']:a for a in args}
    for name,target in buffer_aliases.items():
        if name not in by_name or target not in by_name or target in buffer_aliases:
            raise ValueError('captured argument alias must name a distinct canonical target')
        a,b=by_name[name],by_name[target]
        if not a['dims'] or a['dims']!=b['dims'] or replay_type(a['type'])!=replay_type(b['type']):
            raise ValueError('captured argument alias type or bounds differ')
    distributed=[a for a in args if a['type']=='type(dm_info)']
    startup=[];shutdown=[]
    if distributed:
        imports.append('  use mpas_dmpar,only:mpas_dmpar_init,mpas_dmpar_finalize')
        declarations.append('  type(dm_info) :: oracle_live_dm')
        startup.append('  call mpas_dmpar_init(oracle_live_dm)')
        for a in distributed:
            name=a['name']
            startup.append(f"  if({name}%nprocs/=oracle_live_dm%nprocs.or.{name}%my_proc_id/=oracle_live_dm%my_proc_id) error stop 'oracle captured MPI topology differs from replay'")
            startup.append(f'  {name}%comm=oracle_live_dm%comm\n  {name}%info=oracle_live_dm%info')
        shutdown.append('  call mpas_dmpar_finalize(oracle_live_dm)')
    pools=[a for a in args if a.get('native_pool') or a.get('pool_argument')]
    if pools:imports+=['  use mpas_pool_routines,only:mpas_pool_type','  use mpas_oracle_pool,only:oracle_pool_capture','  use mpas_oracle_pool_restore,only:oracle_pool_restore']
    for pool in pools:
        name=pool['name'];label=('argument.' if pool.get('native_pool') else '')+name;declarations.append(f'  type(mpas_pool_type),pointer :: {name}')
        reads.append(f"  call oracle_pool_restore(trim(prefix)//'.pre.{label}',{name})")
        writes.append(f"  call oracle_pool_capture('{site['site']}','post','{label}',{name},event)")
    for parent in args:
        actual=parent.get('module_actual')
        # An owner outside the callee's reachable module graph is represented
        # entirely by its argument records. Owners in that graph must be the
        # same storage used by the native state restoration/readers.
        if actual and actual['module'] in site.get('state_modules',[]):
            value='oracle_actual_'+parent['name']
            aliases[parent['name']]=value
            imports.append(f"  use {actual['module']}, only: {value} => {actual['name']}")
    for parent in args:
        if 'components' in parent:

            # dmpar keeps imported types private; the public native type is
            # declared by mpas_derived_types (mpas_dmpar_types.inc:9-27).
            # Preserve compatibility with earlier inventories that named
            # the routine provider rather than the type provider.
            provider='mpas_derived_types' if parent in distributed else parent['derived_module']
            imports.append(f"  use {provider}, only: {parent['type'][5:-1]}")
            name=parent['name'];parent_dims=parent.get('dims',[]);shape='('+','.join(parent_dims)+')' if parent_dims else ''
            if has_presence(parent):
                shape='('+','.join(':' for _ in parent_dims)+')' if parent_dims else ''
                declarations.append(f"  {parent['type']},allocatable :: {name}{shape}")
                declarations.append(f"  logical :: {name}_present")
                reads.append(f"  call oracle_read(trim(prefix)//'.pre.{name}.present.bin',{name}_present)")
                extent='('+','.join(parent_dims)+')' if parent_dims else ''
                reads.append(f"  if({name}_present) allocate({name}{extent})")
            elif name not in aliases:declarations.append(f"  {parent['type']} :: {name}{shape}")
    for a in leaves:
        name=a['name'];value=a.get('replay_expr',name);rank=a.get('capture_rank',len(a['dims']))
        root=re.match(r'\w+',value)[0];alias=aliases.get(root)
        parent=next((p for p in args if p['name']==root),None)
        if alias:value=alias+value[len(root):]
        native_attrs=parent['module_actual']['attrs'] if alias and 'replay_expr' not in a else a['attrs']
        identifier=name.replace('.','_');flag=identifier+'_present'
        typ=replay_type(a['type']);component='replay_expr' in a
        if name in buffer_aliases:
            if component or alias or typ.startswith('character'):
                raise ValueError('captured argument alias needs a primitive array')
            target=buffer_aliases[name]
            declarations.append(f"  {typ}, pointer :: {name}({','.join(':' for _ in range(rank))}) => null()")
            declarations.append(f'  logical :: {flag}')
            reads.append(f"  call oracle_read(trim(prefix)//'.pre.{name}.present.bin',{flag})")
            associations.append(f"  if({flag}) then\n    if(.not.allocated({target})) error stop 'captured alias target absent'\n    {name}=>{target}\n    call oracle_header(trim(prefix)//'.pre.{name}.bin',code,width,rank_n,dims,lb,unit)\n    close(unit)\n    if(rank_n/={rank}.or.any(dims(1:{rank})/=shape({name})).or.any(lb(1:{rank})/=lbound({name}))) error stop 'captured alias shape differs'\n  endif")
            ca,la=axes(a)
            if a['intent']!='in':
                if has_presence(a):
                    writes.append(f"  call oracle_write('{site['site']}','post','{name}.present',{flag},event)")
                writes.append(f"  if({flag}) call oracle_write('{site['site']}','post','{name}',{name},event,lower=lbound({name}),cell_axis={ca},level_axis={la})")
            continue
        if not component and not alias:
            if rank:
                alloc='' if typ.startswith('character') else ', allocatable'
                if name in buffer_targets:alloc+=', target'
                declarations.append(f"  {typ}{alloc} :: {value}({','.join(':' for _ in range(rank))})")
            elif has_presence(a) and not typ.startswith('character'):
                declarations.append(f'  {typ}, allocatable :: {value}')
            else:declarations.append(f'  {typ} :: {value}')
        parent_optional=parent and 'components' in parent and has_presence(parent)
        if parent_optional:reads.append(f"  if({root}_present) then")
        if has_presence(a):declarations.append(f'  logical :: {flag}')
        phase='post' if a['intent']=='out' and not a.get('entry_storage') else 'pre'
        path=f"trim(prefix)//'.{phase}.{name}.bin'"
        if has_presence(a):
            reads.append(f"  call oracle_read(trim(prefix)//'.{phase}.{name}.present.bin',{flag})\n  if({flag}) then")
        reads.append(f'  call oracle_header({path},code,width,rank_n,dims,lb,unit)\n  close(unit)')
        # Components are already declared in the native derived type. Only its
        # allocatable components can be allocated here; fixed components retain shape.
        allocate=(not component and not alias) or 'allocatable' in native_attrs or 'pointer' in native_attrs
        if allocate and (alias or component):
            inquiry='associated' if 'pointer' in native_attrs else 'allocated'
            reads.append(f'  if({inquiry}({value})) deallocate({value})')
        if rank and allocate:
            char_type='character(len=width) :: ' if typ.startswith('character') else ''
            reads.append(f"  allocate({char_type}{value}({','.join(f'lb({i}):lb({i})+dims({i})-1' for i in range(1,rank+1))}))")
        elif not rank and typ.startswith('character') and allocate:
            reads.append(f'  allocate(character(len=dims(1)) :: {value})')
        elif not rank and has_presence(a) and allocate:reads.append(f'  allocate({value})')
        if a['intent']!='out' or a.get('entry_storage'):reads.append(f'  call oracle_read({path},{value})')
        if has_presence(a):reads.append('  endif')
        if parent_optional:reads.append('  endif')
        ca,la=axes(a)
        if ca:
            condition=f'.not.context_set.and.{flag}' if has_presence(a) else '.not.context_set'
            contexts.append(f'  if({condition}) then\n    call oracle_restore_context({path},event)\n    context_set=.true.\n  endif')
        if a['intent']!='in':
            if parent_optional:writes.append(f"  if({root}_present) then")
            lower=f', lower=lbound({value})' if rank else ''
            if has_presence(a):
                # Derived component allocation can change in an INOUT call.
                if component:writes.append(f'  {flag}=allocated({value})')
                writes.append(f"  call oracle_write('{site['site']}','post','{name}.present',{flag},event)\n  if({flag}) then")
            writes.append(f"  call oracle_write('{site['site']}','post','{name}', &\n       {value},event{lower},cell_axis={ca},level_axis={la})")
            if has_presence(a):writes.append('  endif')
            if parent_optional:writes.append('  endif')
    if leaves:
        fallback=leaves[0];parent=next(p for p in args if p['name']==re.match(r'\w+',fallback.get('replay_expr',fallback['name']))[0]);phase='post' if fallback['intent']=='out' and not fallback.get('entry_storage') else 'pre'
        suffix='.present' if has_presence(fallback) else ''
        context=f".{phase}.{fallback['name']}{suffix}.bin"
        if parent.get('components') and has_presence(parent):context=f".pre.{parent['name']}.present.bin"
    elif pools:
        context='.pre.'+('argument.' if pools[0].get('native_pool') else '')+pools[0]['name']+'.pool_size.bin'
    elif site.get('context_record'):
        context='.pre.'+site['context_record']+'.bin'
    else:raise ValueError('argument-free call needs a captured module-state context')
    contexts.append(f"  if(.not.context_set) call oracle_restore_context(trim(prefix)//'{context}',event)")
    actuals=(', &'+chr(10)).join('       '+a['name']+'='+aliases.get(a['name'],a['name']) for a in args)
    callee_name=site['callee'].get('name',site['scheme'])
    invocation=f"  call {callee_name}( &\n{actuals} &\n       )" if args else f"  call {callee_name}()"
    # A WRF cell-memory slice retains its original lower bound. Update both
    # memory and active loop bounds explicitly when replaying that extracted
    # column; retaining whole-mesh scalar bounds would read outside the slice.
    scalar_names={a['name'] for a in args if a['type'].startswith('integer') and not a['dims']}
    column_anchor=next((a for a in leaves if axes(a)[0] and
                        re.fullmatch(r'ims\s*:\s*ime',a['dims'][axes(a)[0]-1],re.I)),None)
    column_setup="  if(column_mode) error stop 'this interface lacks source-bound WRF column bounds'"
    if column_anchor and {'ims','ime','its','ite'}.issubset(scalar_names) and not pools:
        anchor=column_anchor['name'];axis=axes(column_anchor)[0]
        column_setup=f"""  if(column_mode) then
    call oracle_header(trim(prefix)//'.pre.{anchor}.bin',code,width,rank_n,dims,lb,unit)
    close(unit)
    if(dims({axis})/=1) error stop 'column replay requires one extracted cell'
    ims=lb({axis}); ime=ims; its=ims; ite=ims
  endif"""
    runtime_imports,runtime_declarations,runtime_setup=native_runtime(site)
    return f"""! GENERATED: MPAS {site['callee']['path']}:{site['callee']['line']}-{site['callee']['end_line']}
! Call site: {site['path']}:{site['line']}-{site['end_line']}
! Argument replay is debugging evidence; hidden state and forecasts need receipts.
program native_argument_replay
  use iso_fortran_env, only: int32, int64, real32, real64
  use {site['callee']['module']}, only: {site['callee'].get('name',site['scheme'])}
{chr(10).join(imports)}
{chr(10).join('  use '+n+', only: oracle_restore_'+n+',oracle_state_'+n for n in site.get('state_modules',[]))}
  use mpas_oracle_dump
{runtime_imports}
  implicit none
  character(len=4096) :: prefix, output, mode
  integer :: code,width,rank_n,dims(5),lb(5),unit,event
  logical :: context_set, column_mode
{runtime_declarations}
{chr(10).join(declarations)}
  if(command_argument_count()<2.or.command_argument_count()>3) &
       error stop 'usage: replay CAPTURE_PREFIX OUTPUT_PREFIX [column]'
  call get_command_argument(1,prefix)
  call get_command_argument(2,output)
  column_mode=command_argument_count()==3
  if(column_mode) then
    call get_command_argument(3,mode)
    if(trim(mode)/='column') error stop 'unknown replay mode'
  endif
{chr(10).join(reads)}
{chr(10).join(associations)}
{column_setup}
{runtime_setup}
{chr(10).join(startup)}
  context_set=.false.
{chr(10).join(contexts)}
{chr(10).join('  call oracle_restore_'+n+'(trim(prefix))' for n in site.get('state_modules',[]))}
  call oracle_replay_begin(trim(prefix),'{site['site']}',event)
  call oracle_set_output(trim(output))
{invocation}
  ! oracle native call completed
{chr(10).join(writes)}
{chr(10).join("  call oracle_state_"+n+"('"+site['site']+"','post',event)" for n in site.get('state_modules',[]))}
  call oracle_scope_end()
  call oracle_clear_output()
{chr(10).join(shutdown)}
end program
"""


def write_replays(manifest,out):
    out.mkdir(parents=True,exist_ok=True);failures=[]
    for site in manifest['sites']:
        try:source=generate(site)
        except (ValueError,IndexError) as error:
            failures.append({'site':site['site'],'reason':str(error)})
            continue
        (out/f"replay_{site['site']}.F90").write_text(source)
    manifest['replay_generation_failures']=failures
