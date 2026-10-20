"""Source-derived GNU explicit-shape ABI for the shared oracle's external tap.

The observer never recompiles or allocates inside the native child. Generated
captures remain inadmissible until full forecast observer passivity is measured.
GNU assumed-shape descriptors are compiler-bound; derived/value interfaces remain
unsupported rather than being interpreted as primitive data buffers.
"""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
from fortran_source import routines, call, bind, split_top, statements
from replay.generate_replay import replay_type, axes


def rust_expression(expression, routine):
    """Compile integer bounds; source dummy names retain their original ABI index."""
    node=ast.parse(expression.lower(),mode='eval').body
    def emit(n):
        if isinstance(n,ast.Constant) and type(n.value) is int:return str(n.value)+'i64'
        if isinstance(n,ast.Name) and n.id in routine.args:
            variable=routine.variables[n.id]
            if variable.dims or not variable.type.startswith('integer'):
                raise ValueError('array bound is not an integer scalar dummy: '+n.id)
            width=8 if replay_type(variable.type)=='integer(int64)' else 4
            return f'scalar(pid,p,{routine.args.index(n.id)},{width})?'
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return emit(n.operand) if isinstance(n.op,ast.UAdd) else f'({emit(n.operand)}).checked_neg().ok_or_else(||bad("bound overflow"))?'
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div'}[type(n.op)]
            return f'({emit(n.left)}).checked_{op}({emit(n.right)}).ok_or_else(||bad("bound overflow or division by zero"))?'
        raise ValueError('unsupported native bound expression: '+expression)
    return emit(node)


def kind(variable):
    typ=replay_type(variable.type)
    pairs={'real(real32)':(1,4),'real(real64)':(2,8),'integer(int32)':(3,4),
           'integer(int64)':(6,8),'logical':(4,4),'complex(real32)':(7,8),
           'complex(real64)':(8,16),'character(len=:), allocatable':(5,1)}
    return pairs[typ]


def variable_layout(variable,routine,allow_descriptor=False):
    if not variable.supported or any(a in variable.attrs for a in ('pointer','allocatable','value')):
        raise ValueError('unsupported native dummy storage: '+variable.name)
    bounds=[]
    for dimension in variable.dims:
        parts=split_top(dimension,':')
        lower,upper=('1',parts[0]) if len(parts)==1 else parts if len(parts)==2 else ('','')
        if allow_descriptor and len(parts)==2 and not upper:
            bounds.append(rust_expression(lower or '1',routine))
            continue
        if not lower or not upper or '*' in upper:
            raise ValueError('external observer needs explicit-shape dummy: '+variable.name)
        bounds.append('('+rust_expression(lower,routine)+','+rust_expression(upper,routine)+')')
    code,width=kind(variable)
    ca,la=axes(asdict(variable))
    return code,width,ca,la,'&['+','.join(bounds)+']'


def hidden_probe(root,routine,index,selection):
    """Derive local argument labels and an integer call discriminator from source."""
    name,argument=selection.lower().split(':')
    targets=index.get(name,[])
    if len(targets)!=1:raise ValueError('external hidden callee must resolve uniquely: '+name)
    target=targets[0]
    if argument not in target.args:raise ValueError('hidden dummy absent: '+selection)
    calls=[]
    for st in routine.body:
        parsed=call(st)
        if parsed and parsed[0]==name:
            binding=bind(target,parsed[1]);actual=binding[argument].strip().lower()
            if actual not in routine.variables:raise ValueError('hidden actual is not simple native local: '+actual)
            calls.append((st,binding,actual))
    if not calls:raise ValueError('hidden callee is not called by selected wrapper: '+name)
    if len(calls)>1:
        def constant(actual, statement):
            if re.fullmatch(r'[+-]?\d+', actual):return int(actual)
            previous=routine.body[routine.body.index(statement)-1].text
            guard=re.fullmatch(r'if\s*\(\s*'+re.escape(actual)+r'\s*==\s*([+-]?\d+)\s*\)\s*then',previous,re.I)
            return int(guard[1]) if guard else None
        discriminators=[n for n in target.args if n in target.variables and not target.variables[n].dims
            and target.variables[n].type.startswith('integer')
            and all(constant(binding.get(n,''),st) is not None for st,binding,_ in calls)
            and len({constant(binding[n],st) for st,binding,_ in calls})==len(calls)]
        if len(discriminators)!=1:raise ValueError('hidden call discriminator not unique: '+repr(discriminators))
        discriminator=target.args.index(discriminators[0])
        labels=[(constant(binding[discriminators[0]],st),'local.'+routine.module+'.'+actual) for st,binding,actual in calls]
    else:discriminator=None;labels=[(0,'local.'+routine.module+'.'+calls[0][2])]
    variable=target.variables[argument]
    code,width,ca,la,layout=variable_layout(variable,target)
    return dict(name=name,module=target.module,path=target.path,line=target.start+1,end_line=target.end,
                arguments=len(target.args),argument=target.args.index(argument),selector=discriminator,
                labels=labels,code=code,width=width,cell_axis=ca,level_axis=la,layout=layout,
                insertions=[dict(line=st.start,label='local.'+routine.module+'.'+actual,actual=actual) for st,_,actual in calls])


def emit(root,out,manifest,selections,hidden=()):
    """An additional backend of generate.py; one native caller ABI per selection."""
    out.mkdir(parents=True,exist_ok=True)
    templates=Path(__file__).parent/'replay'
    candidates={}
    # Hidden callees are in the complete module inventory already parsed by the
    # shared pin-selected inventory. Source hashes stay in the generated spec.
    from pin_sets import source_files
    for path in source_files(root,manifest.get('reference','mpas-v842')):
        if path.suffix.lower() not in {'.f','.f90'}:continue
        for r in routines(path,root):candidates.setdefault(r.name,[]).append(r)
    outcomes=[]
    for selection in selections:
        matching=[s for s in manifest['sites'] if s['scheme']==selection or s['callee']['name']==selection]
        unique={(s['callee']['module'],s['callee']['name'],s['callee']['path']):s for s in matching}
        if len(unique)!=1:raise ValueError('external selection must resolve to one native ABI: '+selection)
        site=next(iter(unique.values()))
        routine=next(r for r in routines(root/site['callee']['path'],root) if r.name==site['callee']['name'])
        directory=out/routine.name;directory.mkdir(exist_ok=False)
        argument_specs=[];layouts=[];characters=[];char_index=0
        descriptors=[]
        for i,name in enumerate(routine.args):
            var=routine.variables[name]
            descriptor=bool(var.dims) and all(':' in d and not d.split(':',1)[1].strip() for d in var.dims)
            code,width,ca,la,bounds=variable_layout(var,routine,allow_descriptor=descriptor)
            if code==5:
                characters.append((i,len(routine.args)+char_index));char_index+=1
            argument_specs.append(f'Arg{{name:{json.dumps(name)},code:{code},width:{width},cell:{ca},level:{la},optional:{str(var.optional).lower()},descriptor:{str(descriptor).lower()}}}')
            layouts.append(f'{i}=>descriptor_layout(pid,p[{i}],{bounds},{code},width),' if descriptor else f'{i}=>layout({bounds}),')
            if descriptor:descriptors.append(name)
        probes=[hidden_probe(root,routine,candidates,h) for h in hidden]
        code=f'const SYMBOL:&str="__{routine.module}_MOD_{routine.name}";\nconst NARGS:usize={len(routine.args)};\n'
        code+='const ARGS:&[Arg]=&[\n'+',\n'.join(argument_specs)+'\n];\n'
        code+='const CHARACTERS:&[(usize,usize)]=&'+repr(characters).replace('[','[').replace(']',']')+';\n'
        code+='fn argument_layout(pid:i32,p:&[u64],i:usize,width:usize)->io::Result<(Vec<i32>,Vec<i32>)>{match i{\n'+''.join(layouts)+'_=>Err(bad("argument ordinal"))}}\n'
        code+='const PROBES:&[Probe]=&[\n'+',\n'.join(f'Probe{{symbol:"__{p["module"]}_MOD_{p["name"]}",nargs:{p["arguments"]},argument:{p["argument"]},selector:'+('None' if p['selector'] is None else 'Some('+str(p['selector'])+')')+',labels:&['+','.join(f'({n},{json.dumps(label)})' for n,label in p['labels'])+f'],code:{p["code"]},width:{p["width"]},cell:{p["cell_axis"]},level:{p["level_axis"]}'+'}' for p in probes)+'\n];\n'
        code+='fn probe_layout(pid:i32,p:&[u64],i:usize)->io::Result<(Vec<i32>,Vec<i32>)>{match i{\n'+''.join(f'{i}=>layout({p["layout"]}),\n' for i,p in enumerate(probes))+'_=>Err(bad("probe ordinal"))}}\n'
        (directory/'argument_spec.rs').write_text(code)
        shutil.copy2(templates/'native_argument_watch.rs',directory/'native_argument_watch.rs')
        shutil.copy2(templates/'native_entry_watch.rs',directory/'native_entry_watch.rs')
        (directory/'Makefile').write_text('all: observer\nobserver: native_argument_watch.rs native_entry_watch.rs argument_spec.rs\n\trustc --edition=2021 -O native_argument_watch.rs -o $@\n')
        data=dict(schema='mpas-external-native-abi-v1',selection=selection,callee=site['callee'],site=site,
                  arguments=[asdict(routine.variables[n]) for n in routine.args],probes=probes,
                  descriptor_arguments=descriptors,
                  source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in {routine.path,*[p['path'] for p in probes]}},
                  pin_set=manifest.get('reference'),source_set_sha256=manifest.get('source_set_sha256'),
                  hidden_state_complete=False,global_cell_ids_known=False,forecast_proven=False,
                  passivity_requirement='Every complete native forecast output compared with an ordinary pristine run at a fixed process count before capture admission.')
        data['generator_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__),Path(__file__).parent/'generate.py',
                                  Path(__file__).parent/'fortran_source.py',templates/'generate_replay.py']}
        data['generated_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
        (directory/'spec.json').write_text(json.dumps(data,indent=2)+'\n');outcomes.append(dict(selection=selection,spec=str(directory/'spec.json')))
    manifest['external_observers']=outcomes


def prepare_native_replay(root,spec,manifest,out,recipes=('gnu-strict','gnu-o3','coverage'),preserve_calls=False):
    """Compile a complete pristine numerical closure with recorded local inputs.

    This is standalone replay, never an instrumented production native model.
    The only added numerical-module statements restore captured native locals at
    their measured internal entries. Caller aliases must be checked against each
    external observation by the replay runner before executing the program.
    """
    import copy
    from replay.generate_replay import generate as replay_source
    from replay.prepare_standalone import RECIPES,module_dependencies
    from generate_transport import generate as transport
    site=copy.deepcopy(spec['site'])
    names={a['name']:a for a in spec['arguments']}
    actuals={a['name']:a['actual'] for a in site['arguments']}
    groups={}
    for name,actual in actuals.items():
        if names[name]['dims'] and re.fullmatch(r'\w+',actual):groups.setdefault(actual,[]).append(name)
    aliases={}
    for members in groups.values():
        if len(members)<2:continue
        target=next((n for n in members if not names[n]['optional']),members[0])
        aliases.update({n:target for n in members if n!=target})
    site['arguments']=[dict(a,actual=a['name'],entry_storage=True) for a in spec['arguments']]
    site['argument_aliases']=aliases
    site['state_modules']=[];site['runtime_modules']=[];site['issues']=[]
    source=replay_source(site)
    # Resolve the complete source closure even when this backend is invoked
    # directly by generate.py without a pre-existing instrumented build.
    from pin_sets import source_files
    providers={};dependencies={}
    for path in source_files(root,manifest.get('reference','mpas-v842')):
        if path.suffix.lower() not in {'.f','.f90'}:continue
        relative=str(path.relative_to(root))
        defined,used=module_dependencies(path.read_text())
        dependencies[relative]=used
        for module in defined:providers.setdefault(module,set()).add(relative)
    paths=set();pending=[spec['callee']['path']]
    while pending:
        path=pending.pop()
        if path in paths:continue
        paths.add(path)
        for module in dependencies[path]:
            if module=='mpas_log':continue
            candidates=providers.get(module,set())
            if len(candidates)!=1:raise ValueError('native replay module provider is not unique: '+module)
            pending.append(next(iter(candidates)))
    contents={p:(root/p).read_text() for p in paths}
    # A USE of mpas_log without any referenced log API requires no numerical
    # framework stand-in. Refuse this standalone closure if the native module
    # actually calls a framework procedure; full-framework replay remains the
    # appropriate backend for that interface.
    if any(re.search(r'\bcall\s+mpas_\w+',t,re.I) for t in contents.values()):
        raise ValueError('external standalone closure needs real framework calls')
    contents['framework_log_interface.F90']='! No native logging API is referenced by this selected numerical closure.\nmodule mpas_log\nend module mpas_log\n'
    contents['mpas_oracle_dump.F']=transport()
    wrapper=spec['callee']['path'];lines=contents[wrapper].splitlines(keepends=True)
    routine=next(r for r in routines(root/wrapper,root) if r.name==spec['callee']['name'])
    signature=next(s for s in __import__('fortran_source').statements(contents[wrapper]) if s.start==routine.start)
    inserts={signature.end:[' use mpas_oracle_dump,only:oracle_hidden\n']}
    # Native wrappers also call private module procedures. Export that procedure
    # only in this standalone source copy so the replay has the real compiler
    # interface. Its executable body and production source remain unchanged.
    visibility_export=False
    module_spec=[];inside=False
    for statement in statements(contents[wrapper]):
        if re.fullmatch(r'module\s+'+re.escape(routine.module),statement.text,re.I):inside=True
        elif inside and statement.text.strip().lower()=='contains':
            if any(s.text.strip().lower()=='private' for s in module_spec):
                public=[n.strip().lower() for s in module_spec
                        if re.match(r'public\s*::',s.text,re.I)
                        for n in s.text.split('::',1)[1].split(',')]
                if routine.name not in public:
                    inserts.setdefault(statement.start,[]).append(' public :: '+routine.name+'\n')
                    visibility_export=True
            break
        elif inside:module_spec.append(statement)
    for probe in spec['probes']:
        for entry in probe['insertions']:
            inserts.setdefault(entry['line'],[]).append(f" call oracle_hidden('{entry['label']}',{entry['actual']},1,'pre',lower=lbound({entry['actual']}))\n")
    contents[wrapper]=''.join(''.join(inserts.get(i,[]))+line for i,line in enumerate(lines))
    dependencies={p:module_dependencies(t) for p,t in contents.items()}
    owners={n:p for p,(modules,_) in dependencies.items() for n in modules}
    unknown={p:sorted(used-set(owners)) for p,(_,used) in dependencies.items() if used-set(owners)}
    if unknown:raise ValueError('complete native replay closure unresolved: '+repr(unknown))
    graph={p:{owners[n] for n in used if n in owners}-{p} for p,(_,used) in dependencies.items()}
    remaining=set(contents);ordered=[]
    while remaining:
        ready=sorted(p for p in remaining if not graph[p]&remaining)
        if not ready:raise ValueError('native replay dependency cycle')
        ordered.extend(ready);remaining-=set(ready)
    out.mkdir(parents=True,exist_ok=False);compiled={}
    for recipe in recipes:
        compiler,flags=RECIPES[recipe]
        if preserve_calls:
            flags=flags+['-fno-inline','-fno-inline-functions','-fno-inline-small-functions',
                         '-fno-inline-functions-called-once','-fno-ipa-cp','-fno-ipa-sra']
        flags=flags+['-cpp','-ffree-form','-ffree-line-length-none','-fconvert=big-endian','-DSINGLE_PRECISION','-Dmpas']
        directory=out/recipe;directory.mkdir()
        objects={p:'native_'+hashlib.sha256(p.encode()).hexdigest()[:16]+'.o' for p in ordered}
        text=f'FC={compiler}\nFLAGS={" ".join(flags)}\nall: replay\n'
        for p in ordered:
            target=directory/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(contents[p])
            text+=f'{objects[p]}: {p} Makefile '+ ' '.join(objects[d] for d in sorted(graph[p]))+'\n\t$(FC) $(FLAGS) -c $< -o $@\n'
        (directory/'replay.F90').write_text(source)
        text+='replay: replay.F90 Makefile '+' '.join(objects[p] for p in ordered)+'\n\t$(FC) $(FLAGS) replay.F90 '+' '.join(objects[p] for p in ordered)+' -o $@\n'
        (directory/'Makefile').write_text(text)
        compiled[recipe]=dict(compiler=compiler,flags=flags,objects=objects)
    (out/'Makefile').write_text('all: '+' '.join(recipes)+'\n.PHONY: all '+' '.join(recipes)+'\n'+''.join(f'{r}:\n\t$(MAKE) -C {r}\n' for r in recipes))
    receipt=dict(schema='pristine-full-module-external-replay-v1',callee=spec['callee'],argument_aliases=aliases,
                 recipes=compiled,source_order=ordered,
                 pristine_source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                 generated_source_sha256={p:hashlib.sha256(t.encode()).hexdigest() for p,t in contents.items()},
                 replay_source_sha256=hashlib.sha256(source.encode()).hexdigest(),site=site,
                 hidden_local_restore_points=spec['probes'],native_numerical_rewrite=False,forecast_proven=False,
                 standalone_visibility_export=visibility_export,
                 standalone_preserve_calls=preserve_calls,
                 scope='Complete native module replay with recorded hidden entry inputs. Must match every captured output word. Pristine observer passivity measured separately; callback/module SAVE completeness stays open.')
    (out/'replay-spec.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
