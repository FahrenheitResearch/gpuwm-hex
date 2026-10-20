"""Observe native local SAVE storage even when a column skips its owning routine.

MPAS physics_wrf/module_ra_rrtmg_lw.F:12334-12444 (whole O3DATA), called
from rrtmg_lwrad (module_ra_rrtmg_lw.F:11574-12305) and
physics_wrf/module_ra_rrtmg_sw.F:10026-10664 (whole rrtmg_swrad).
An alternate native ENTRY shares the original declarations and DATA storage;
its bounds dummies keep automatic-array entry allocation well defined.
"""
import re
from fortran_source import routines,statements,first_executable
from local_state import local_variables


REQUESTS = ({'module':'module_ra_rrtmg_lw','routine':'o3data',
             'variables':{'o3sum','o3win','ppsum','ppwin'},
             'bounds':('kts','kte'),
             'callers':(('module_ra_rrtmg_lw','rrtmg_lwrad'),
                        ('module_ra_rrtmg_sw','rrtmg_swrad'))},)


def patch(text,path,root,requests=REQUESTS):
    """Return one complete file patch; unknown native layouts refuse generation."""
    class Source:
        def read_text(self,**kw):return text
        def relative_to(self,unused):return path
    parsed=list(routines(Source(),root));lines=text.splitlines(keepends=True)
    inserts={};records=[]
    for request in requests:
        name='oracle_prime_'+request['routine']
        owner=next((r for r in parsed if (r.module,r.name)==
                    (request['module'],request['routine'])),None)
        callers=[r for r in parsed if (r.module,r.name) in request['callers']]
        if owner:
            variables,issues=local_variables(owner)
            if issues or {v['name'] for v in variables}!=request['variables']:
                raise ValueError('native static accessor SAVE inventory changed')
            for argument in owner.args:
                var=owner.variables[argument]
                if var.type.startswith('character'):
                    raise ValueError('native static accessor needs character length dummies')
            bound_names=set()
            for variable in owner.variables.values():
                for dim in variable.dims:
                    bound_names.update(set(re.findall(r'\b\w+\b',dim.lower()))&set(owner.args))
            if bound_names!=set(request['bounds']):
                raise ValueError('native static accessor automatic bounds changed')
            for bound in request['bounds']:
                var=owner.variables[bound]
                if var.dims or var.type!='integer' or var.intent!='in':
                    raise ValueError('native static accessor bound declaration changed')
            code='  return\n  entry '+name+'('+','.join(request['bounds'])+')\n'
            code+=f"  call oracle_local_begin('{owner.module}.{owner.name}',oracle_local_event)\n"
            for phase in ('pre','post'):
                for variable in variables:
                    value=variable['name'];lower=f',lower=lbound({value})' if variable['dims'] else ''
                    code+=f"  call oracle_hidden('local.{owner.module}.{owner.name}.{value}',{value},oracle_local_event,'{phase}'{lower})\n"
            inserts.setdefault(owner.end-1,[]).append(code)
            records.append(dict(module=owner.module,routine=owner.name,entry=name,
                path=path,bounds=list(request['bounds']),variables=variables,
                storage='original native SAVE/DATA, alternate ENTRY',
                numerical_body_executed=False))
        for caller in callers:
            for bound in request['bounds']:
                variable=caller.variables.get(bound)
                if not variable or variable.type!='integer' or variable.dims:
                    raise ValueError('native static accessor caller bounds missing')
            signature=next(s for s in statements(text) if s.start==caller.start)
            imports='  use mpas_oracle_dump, only: oracle_scope_active\n'
            if caller.module!=request['module']:
                imports+=f"  use {request['module']}, only: {name}\n"
            inserts.setdefault(signature.end,[]).append(imports)
            first=first_executable(caller)
            if first is None:raise ValueError('native static accessor caller has no executable entry')
            inserts.setdefault(first.start,[]).append(
                '  if(oracle_scope_active()) call '+name+'('+','.join(request['bounds'])+')\n')
    return ''.join(''.join(inserts.get(i,[]))+line for i,line in enumerate(lines)),records


def generate(root,out,manifest):
    records=[]
    for relative in sorted({m['path'] for m in manifest['module_state']}):
        target=out/relative
        text,found=patch(target.read_text(),relative,root)
        if text!=target.read_text():target.write_text(text)
        records.extend(found)
    manifest['local_state_accessors']=records
    for site in manifest['sites']:
        site['local_state_accessors']=[r for r in records if r['module'] in site.get('state_modules',[])]
