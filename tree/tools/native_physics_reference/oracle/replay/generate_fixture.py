#!/usr/bin/env python3
"""Generate native perturbed-column captures using the complete scheme replay.

The profile is source configuration, never decoded weather data. Fortran reads
and perturbs the captured inputs, then the unchanged native module produces
expected outputs. Local SAVE pre-records must be copied unchanged from the
authenticated parent capture; module-state pre-records are written here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from derived import array_arguments, has_presence
from replay.generate_replay import generate,axes,replay_type


def fixture(site,profile):
    args=site['arguments'];leaves=[a for a in array_arguments(site) if not a.get('pool_argument')]
    pools=[a for a in args if a.get('pool_argument')]
    aliases={a['name']:'oracle_actual_'+a['name'] for a in args
             if a.get('module_actual') and a['module_actual']['module'] in site.get('state_modules',[])}
    def value(a):
        expression=a.get('replay_expr',a['name']);root=expression.split('%')[0]
        return aliases.get(root,root)+expression[len(root):]
    by_name={a['name']:a for a in leaves};updates=[]
    required=profile.get('required_present',[])
    if (not isinstance(required,list) or any(not isinstance(n,str) for n in required)
            or len(set(required))!=len(required)
            or any(n not in by_name or not by_name[n]['optional'] for n in required)):
        raise ValueError('required_present must name distinct optional native arguments')
    required=set(required)
    presence_checks=[f"  if(.not.{n.replace('.','_')}_present) error stop 'native fixture required optional input absent'"
                     for n in sorted(required)]
    kind=profile['precision']
    if kind not in {'real32','real64'}:raise ValueError('profile requires an explicit native real precision')
    for name,expression in profile['assignments'].items():
        if name not in by_name:raise ValueError(f'profile argument not present: {name}')
        a=by_name[name]
        if a['intent']=='out':raise ValueError(f'cannot perturb undefined OUT input: {name}')
        if a.get('actual_optional') and not a['optional']:
            raise ValueError(f'profile requires a present mandatory input: {name}')
        if a['optional'] and name not in required:
            raise ValueError(f'profile optional input needs required_present: {name}')
        native_type=replay_type(a['type'])
        if native_type.startswith('real') and native_type!=f'real({kind})':
            raise ValueError(f'profile/native precision mismatch: {name}')
        if native_type.startswith('character'):raise ValueError(f'character fixture assignment unsupported: {name}')
        if not re.fullmatch(r'[\w\s%.+*/(),\-]+',expression) or '\n' in expression:
            raise ValueError(f'profile expression must be one arithmetic assignment: {name}')
        # Map dummy names to their native shared module storage where needed.
        names=re.findall(r'\b[a-z]\w*\b',expression,re.I)
        allowed={*by_name,'min','max','abs',kind,'int32','int64','true','false','and','or','not','eqv','neqv'}
        if any(n.lower() not in allowed for n in names):raise ValueError(f'unknown profile expression symbol: {expression}')
        used={n.lower() for n in names}&set(by_name)
        if any(by_name[n]['optional'] and n not in required for n in used):
            raise ValueError('profile expression optional input needs required_present')
        if any(by_name[n]['intent']=='out' for n in used):
            raise ValueError('profile expression cannot read undefined OUT storage')
        for argument in sorted(leaves,key=lambda a:len(a['name']),reverse=True):
            expression=re.sub(r'(?<![\w.])'+re.escape(argument['name'])+r'(?![\w.])',value(argument),expression)
        updates.append(f"  {value(a)}={expression}")
    if not updates:raise ValueError('profile has no perturbations')
    capture=[]
    for a in leaves:
        if a['intent']=='out' and not a.get('entry_storage'):
            raise ValueError('synthetic capture requires original OUT entry storage')
        name=a['name'];v=value(a);ca,la=axes(a);rank=len(a['dims'])
        if has_presence(a):
            flag=name.replace('.','_')+'_present'
            capture.append(f"  call oracle_write('{site['site']}','pre','{name}.present',{flag},event)\n  if({flag}) then")
        lower=f',lower=lbound({v})' if rank else ''
        capture.append(f"  call oracle_write('{site['site']}','pre','{name}',{v},event{lower},cell_axis={ca},level_axis={la})")
        if has_presence(a):capture.append('  endif')
    capture.extend(f"  call oracle_state_{n}('{site['site']}','pre',event)" for n in site.get('state_modules',[]))
    capture.extend(f"  call oracle_pool_capture('{site['site']}','pre','{a['name']}',{a['name']},event)" for a in pools)
    def checks(phase):
        statements=[]
        for expression in profile.get(phase+'_assertions',[]):
            if not isinstance(expression,str) or not re.fullmatch(r'[\w\s%.+*/(),:<>=\-]+',expression) or '\n' in expression:
                raise ValueError('profile assertion must be one logical expression')
            names=re.findall(r'\b[a-z]\w*\b',expression,re.I)
            allowed={*by_name,'min','max','abs','any','all','count',kind,'int32','int64',
                     'true','false','and','or','not','eqv','neqv','eq','ne','lt','le','gt','ge'}
            if any(n.lower() not in allowed for n in names):
                raise ValueError(f'unknown profile assertion symbol: {expression}')
            used={n.lower() for n in names}&set(by_name)
            if any(by_name[n].get('actual_optional') and not by_name[n]['optional'] for n in used):
                raise ValueError('profile assertion requires mandatory arguments')
            if any(by_name[n]['optional'] and n not in required for n in used):
                raise ValueError('profile assertion optional input needs required_present')
            if phase=='pre' and any(by_name[n]['intent']=='out' for n in used):
                raise ValueError('precondition cannot read undefined OUT storage')
            for argument in sorted(leaves,key=lambda a:len(a['name']),reverse=True):
                expression=re.sub(r'(?<![\w.])'+re.escape(argument['name'])+r'(?![\w.])',value(argument),expression)
            statements.append(f"  if(.not.({expression})) error stop 'native fixture {phase}condition failed'")
        return statements

    source=generate(site)
    anchor='  call oracle_set_output(trim(output))\n'
    if source.count(anchor)!=1:raise ValueError('native replay output anchor is not unique')
    source=source.replace(anchor,anchor+'\n'.join(presence_checks+updates+checks('pre')+capture)+'\n',1)
    post=checks('post')
    if post:
        boundary='  ! oracle native call completed\n'
        if source.count(boundary)!=1:raise ValueError('native fixture lacks a unique post-call boundary')
        source=source.replace(boundary,boundary+'\n'.join(post)+'\n',1)
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--site',required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.inventory.read_text())
    site=next(s for s in manifest['sites'] if s['site']==a.site)
    profile=json.loads(a.profile.read_text());source=fixture(site,profile)
    if a.out.exists():raise ValueError('fixture output must be fresh')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(source)
    meta={'schema':'mpas-native-perturbed-fixture-v1','site':a.site,'profile':profile,
          'inventory_sha256':hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
          'profile_sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest(),
          'generated_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'source_sha256':manifest['source_sha256'],'forecast_proven':False}
    a.out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
