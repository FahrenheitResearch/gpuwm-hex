"""Insert pool snapshots at MPAS's complete subsystem call boundaries.

MPAS: mpas_atmphys_driver.F:202-372; mpas_atm_time_integration.F:2146,
2168,2811; diagnostics/mpas_atm_diagnostics.F (output diagnostics call).
"""
import re
from fortran_source import routines, statements, call, declarations,active_source,first_executable

BOUNDARIES={'physics_get_tend','atm_add_tend_anal_incr','atm_compute_output_diagnostics','driver_microphysics'}


def generate_boundaries(root,out,manifest):
    paths=[root/'src/core_atmosphere/physics/mpas_atmphys_driver.F',
           root/'src/core_atmosphere/dynamics/mpas_atm_time_integration.F']
    from pin_sets import pin_set
    from derived import primitive_components
    from generate import dump_call
    config=pin_set(manifest.get('pin_set','mpas-v842'))
    result=[]
    for path in paths:
        relative=str(path.relative_to(root))
        target=out/relative
        text=target.read_text() if target.exists() else path.read_text()
        # Boundary pass runs only on files not touched by the scheme overlay.
        if target.exists():
            raise ValueError('boundary file already changed; regenerate in a fresh output directory')
        lines=text.splitlines(keepends=True)
        inserts={}
        replacements={}
        module_vars={}
        for declaration in statements(active_source(path.read_text())):
            if re.fullmatch(r'contains',declaration.text,re.I):break
            module_vars.update({v.name:v for v in declarations(declaration.text)})
        import xml.etree.ElementTree as ET
        field_dims={f.get('name'):f.get('dimensions','').split() for f in ET.parse(root/'src/core_atmosphere/Registry.xml').iter() if f.tag in {'var','var_array'}}
        for routine in routines(path,root):
            sites=[]
            for st in routine.body:
                c=call(st)
                if not c or not(c[0].startswith('driver_') or c[0] in BOUNDARIES):
                    continue
                name,actuals=c
                pools=[];values=[];objects=[];uncaptured=[]
                for argument_index,actual in enumerate(actuals):
                    actual=re.sub(r'^\w+\s*=','',actual).strip()
                    v=routine.variables.get(actual.lower(),module_vars.get(actual.lower()))
                    expression_base=re.match(r'(\w+)\s*\((.*)\)$',actual)
                    base_var=routine.variables.get(expression_base[1].lower(),module_vars.get(expression_base[1].lower())) if expression_base else None
                    if v and re.match(r'type\s*\(mpas_pool_type\)',v.type):
                        pools.append(actual)
                    elif v and v.type.startswith('type(') and v.type[5:-1] in config['derived_types']:
                        typename=v.type[5:-1];objects.extend(primitive_components(root,config['derived_types'][typename],typename,'argument.'+actual,actual,'inout',v.dims))
                    elif v and v.supported and re.fullmatch(r'\w+',actual):
                        values.append((actual,v))
                    elif base_var and base_var.supported:
                        from dataclasses import replace
                        values.append((actual,replace(base_var,dims=[],attrs='')))
                    elif re.fullmatch(r'[+-]?[0-9]+',actual):
                        from fortran_source import Variable
                        values.append((actual,Variable('literal','integer','',[],'in',False,False)))
                    else:
                        member=re.fullmatch(r'(\w+)\s*%\s*(\w+)',actual)
                        base=(routine.variables.get(member[1].lower(),module_vars.get(member[1].lower())) if member else None)
                        if base and re.fullmatch(r'type\s*\(\s*block_type\s*\)',base.type) and member[2].lower() in {'configs','structs','dimensions','packages','allfields','allstructs'}:
                            pools.append(actual)
                        elif base and base.type=='type(domain_type)' and member[2].lower()=='dminfo' and 'dm_info' in config['derived_types']:
                            objects.extend(primitive_components(root,config['derived_types']['dm_info'],'dm_info','argument.'+actual.replace('%','.'),actual,'inout'))
                        elif v and re.fullmatch(r'type\s*\(\s*block_type\s*\)',v.type):
                            pools.extend(actual+'%'+n for n in ('configs','dimensions','allFields','allStructs','structs','packages'))
                        else:uncaptured.append({'expression':actual,'type':v.type if v else 'unresolved expression or procedure'})
                sites.append((st,name,pools,values,objects,uncaptured))
            if not sites:
                continue
            first=next(st for st in statements(text) if st.start==routine.start)
            inserts.setdefault(first.end,[]).append('      use mpas_oracle_dump, only: oracle_begin,oracle_write,oracle_scope_begin,oracle_scope_end\n      use mpas_oracle_pool\n')
            exec_start=first_executable(routine).start
            inserts.setdefault(exec_start,[]).append('      integer :: oracle_boundary_event\n')
            for st,name,pools,values,objects,uncaptured in sites:
                key='boundary__'+name
                # Driver arguments omit the dimensions pool, although its
                # thread/owned ranges are inputs to every MPAS driver slice.
                if 'block' in routine.variables:
                    pools.extend(['block%configs','block%dimensions'])
                pre=f"      call oracle_begin('{key}',oracle_boundary_event)\n      call oracle_scope_begin('{key}',oracle_boundary_event)\n"
                if 'mesh' in routine.variables:
                    if 'itimestep' not in routine.variables:
                        raise ValueError(f'no timestep context: {routine.name}')
                    pre+='      call oracle_mesh_context(mesh,itimestep)\n'
                post=''
                for pool in dict.fromkeys(pools):
                    for phase in ['pre','post']:
                        label=re.sub(r'\s*%\s*','.',pool)
                        line=f"      call oracle_pool_capture('{key}','{phase}','{label}',{pool},oracle_boundary_event)\n"
                        if phase=='pre':pre+=line
                        else:post+=line
                for value,var in values:
                    dims=field_dims.get(value,[])
                    cell=next((i+1 for i,d in enumerate(dims) if d=='nCells'),0)
                    level=next((i+1 for i,d in enumerate(dims) if d in {'nVertLevels','nVertLevelsP1'}),0)
                    lower=f',lower=lbound({value})' if var.dims else ''
                    guard=(f'associated({value})' if 'pointer' in var.attrs else f'allocated({value})' if 'allocatable' in var.attrs else '')
                    for phase in ['pre','post']:
                        label=re.sub(r'[^a-z0-9_.]','_',value.lower())
                        line=f"      call oracle_write('{key}','{phase}','argument.{label}',{value},oracle_boundary_event{lower},cell_axis={cell},level_axis={level})\n"
                        if guard:
                            line=f"      call oracle_write('{key}','{phase}','argument.{label}.present',{guard},oracle_boundary_event)\n      if({guard}) then\n"+line+'      endif\n'
                        if phase=='pre':pre+=line
                        else:post+=line
                for obj in objects:
                    for phase in ('pre','post'):
                        # The boundary event has a distinct counter from scheme taps.
                        line=dump_call({'site':key},obj,phase).replace('oracle_event','oracle_boundary_event')
                        if phase=='pre':pre+=line
                        else:post+=line
                post+='      call oracle_scope_end()\n'
                # A conditional native call must emit records only when the
                # call actually executes, and its condition is evaluated once.
                conditional=re.match(r'\s*if\s*(\(.*\))\s*call\b',lines[st.start],re.I)
                if conditional:
                    pre='      if'+conditional[1]+' then\n'+pre
                    post+='      endif\n'
                    replacements[st.start]=re.sub(r'^\s*if\s*\(.*\)\s*(?=call\b)','      ',lines[st.start],count=1,flags=re.I)
                pre+=f'! oracle_boundary_call_begin {key} L{st.start+1}\n'
                post=f'! oracle_boundary_call_end {key} L{st.start+1}\n'+post
                inserts.setdefault(st.start,[]).append(pre)
                inserts.setdefault(st.end,[]).append(post)
                result.append({'key':key,'path':relative,'line':st.start+1,'end_line':st.end,'driver':routine.name,'pools':pools,'values':[v for v,_ in values],'object_components':[v['name'] for v in objects],'uncaptured_arguments':uncaptured})
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(''.join(inserts.get(i,[]))+replacements.get(i,line) for i,line in enumerate(lines)))
    manifest['boundaries']=result


def generate_output_diagnostics(root,out,manifest):
    relative='src/core_atmosphere/mpas_atm_core.F'
    path=root/relative;target=out/relative
    if target.exists():raise ValueError('output diagnostics patch needs fresh source')
    lines=path.read_text().splitlines(keepends=True)
    routine=next(r for r in routines(path,root) if r.name=='atm_compute_output_diagnostics')
    sig=next(st for st in statements(''.join(lines)) if st.start==routine.start)
    first=first_executable(routine).start
    imports='      use mpas_oracle_dump, only: oracle_begin,oracle_scope_begin,oracle_scope_end\n      use mpas_oracle_pool\n'
    begin="      integer :: oracle_output_event\n      call oracle_mesh_context(mesh)\n      call oracle_begin('boundary__atm_compute_output_diagnostics',oracle_output_event)\n      call oracle_scope_begin('boundary__atm_compute_output_diagnostics',oracle_output_event)\n"
    finish=''
    for pool in ['state','diag','mesh']:
        begin+=f"      call oracle_pool_capture('boundary__atm_compute_output_diagnostics','pre','{pool}',{pool},oracle_output_event)\n"
        finish+=f"      call oracle_pool_capture('boundary__atm_compute_output_diagnostics','post','{pool}',{pool},oracle_output_event)\n"
    finish+='      call oracle_scope_end()\n'
    inserts={sig.end:imports,first:begin,routine.end-1:finish}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(inserts.get(i,'')+line for i,line in enumerate(lines)))
    manifest['boundaries'].append({'key':'boundary__atm_compute_output_diagnostics','path':relative,
                                   'line':routine.start+1,'end_line':routine.end,'driver':routine.name,'mode':'routine','pools':['state','diag','mesh']})
