#!/usr/bin/env python3
"""Emit native Fortran captures from explicit synthetic source configurations.

This constructs columns inside the native program; it never decodes field data.
Profile dimensions and expressions are debugging inputs, not forecast evidence.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from generate import dump_call
from replay.generate_replay import replay_type, native_runtime


def generate(site,profile):
    if site['issues']:raise ValueError('unresolved native arguments')
    if any(a.get('native_pool') for a in site['arguments']):
        raise ValueError('pool source fixtures need native framework allocation and alias initialization')
    dimensions={name.lower():value for name,value in profile['dimensions'].items()}
    if len(dimensions)!=len(profile['dimensions']):
        raise ValueError('profile dimensions differ only by Fortran letter case')
    values=profile['values']
    names={a['name'] for a in site['arguments']}
    if set(values)-names:raise ValueError('profile names do not match native interface')
    omitted=set(profile.get('omitted',[]))
    if not omitted<=names or any(a['name'] in omitted and not a['optional'] for a in site['arguments']):
        raise ValueError('only native optional arguments can be omitted')
    derived_initial=profile.get('derived_initialize',{})
    derived_names={a['name'] for a in site['arguments'] if 'components' in a}
    if set(derived_initial)!=derived_names-omitted:
        raise ValueError('each present derived source argument needs explicit native initialization')
    declarations=[];initial=[];before=[];after=[]
    site=copy.deepcopy(site)
    derived_imports=[]
    for a in site['arguments']:
        name=a['name'];typ=a['type'] if 'components' in a else replay_type(a['type'])
        # Fixtures own dummy storage; the model caller's pointers are absent.
        a['actual']=name
        if typ.startswith('character'):typ=f"character(len={profile['character_lengths'][name]})"
        dims=[]
        for d in a['dims']:
            # Dimensions retain native order and explicit lower bounds.
            expr=d
            for symbol in re.findall(r'\b[a-z]\w*\b',d,re.I):
                if symbol.lower() not in dimensions:raise ValueError(f'profile dimension missing: {symbol}')
                expr=re.sub(r'\b'+re.escape(symbol)+r'\b',str(dimensions[symbol.lower()]),expr,flags=re.I)
            if not re.fullmatch(r'[0-9:+*/() -]+',expr):raise ValueError(f'invalid dimension: {expr}')
            dims.append(expr)
        declarations.append(f"  {typ} :: {name}"+('('+','.join(dims)+')' if dims else ''))
        if 'components' in a:
            derived_imports.append(f"  use {a['derived_module']},only:{a['type'][5:-1]}")
            for field in a['components']:
                field['actual']=field['replay_expr']
                if field['optional']:
                    field['presence_guard']=f"allocated({field['actual']})"
            if name not in omitted:initial.extend(derived_initial[name])
        elif name not in omitted:
            if name not in values:raise ValueError(f'explicit source value missing: {name}')
            initial.append(f'  {name}={values[name]}')
        a['presence_guard']=('.false.' if name in omitted else '.true.') if a['optional'] else ''
        before.append(dump_call(site,a,'pre'))
        if a['intent']!='in':after.append(dump_call(site,a,'post'))
    args=(', &\n').join(f"    {a['name']}={a['name']}" for a in site['arguments'] if a['name'] not in omitted)
    states=site.get('state_modules',[])
    imports=[f"  use {site['callee']['module']},only:{site['scheme']}"]
    imports+=sorted(set(derived_imports))
    imports+=profile.get('imports',[])
    imports += ['  use '+n+',only:oracle_state_'+n for n in states]
    state_pre='\n'.join(f"  call oracle_state_{n}('{site['site']}','pre',oracle_event)" for n in states)
    state_post='\n'.join(f"  call oracle_state_{n}('{site['site']}','post',oracle_event)" for n in states)
    invocation=f"  call {site['scheme']}( &\n{args})" if args else f"  call {site['scheme']}()"
    runtime_imports,runtime_declarations,runtime_setup=native_runtime(site)
    return f'''! GENERATED synthetic source profile, native {site['callee']['path']}:{site['callee']['line']}-{site['callee']['end_line']}
program synthetic_capture
  use iso_fortran_env,only:int32,int64,real32,real64
  use mpas_oracle_dump
{chr(10).join(imports)}
{runtime_imports}
  implicit none
  integer :: oracle_event
{chr(10).join(declarations)}
{runtime_declarations}
{chr(10).join(profile.get('declarations',[]))}
{runtime_setup}
{chr(10).join(initial)}
{chr(10).join(profile.get('initialize',[]))}
  call oracle_begin('{site['site']}',oracle_event)
  call oracle_scope_begin('{site['site']}',oracle_event)
{''.join(before)}
{state_pre}
{invocation}
{''.join(after)}
{state_post}
{chr(10).join(profile.get('assertions',[]))}
  call oracle_scope_end()
end program
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--site',required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.inventory.read_text());profile=json.loads(a.profile.read_text())
    site=next(s for s in m['sites'] if s['site']==a.site)
    if a.out.exists():raise ValueError('source fixture must be fresh')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(generate(site,profile))
    receipt={'schema':'mpas-native-synthetic-source-v1','site':site['site'],'profile':profile,
             'inventory_sha256':hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
             'generated_source_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'forecast_proven':False}
    a.out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
