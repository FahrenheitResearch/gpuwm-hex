#!/usr/bin/env python3
"""Build whole native scheme modules under four recipes and GNU coverage.

The owned oracle supplies source-generated state hooks and replay programs.
MPAS framework kind/constants sources are unchanged. Logging-only framework
stubs implement mpas_log.F:471-637's message interface and fatal termination;
they do not implement a numerical operation. Unresolved dependencies fail.
Compilation is a separate make -j4 invocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from fortran_source import active_source,statements,routines
from state import module_anchors

RECIPES={
    'gnu-strict':('gfortran',['-O3','-ffp-contract=off','-fno-tree-vectorize','-fno-unsafe-math-optimizations']),
    'gnu-o3':('gfortran',['-O3']),
    'ifx-o3':('/opt/intel/oneapi/compiler/2026.0/bin/ifx',
              ['-O3','-align','array64byte','-Qoption,fpp,-macro_expand=vc']),
    'ifx-valuesafe':('/opt/intel/oneapi/compiler/2026.0/bin/ifx',
                    ['-O3','-align','array64byte','-Qoption,fpp,-macro_expand=vc',
                     '-fp-model=source','-no-fma','-fimf-arch-consistency=true']),
    # GNU 15 emits an invalid forwarding-function source span for Fortran
    # ENTRY. Keep the diagnostic visible without rejecting profiling code;
    # native line/branch mapping and numerical replay remain separate checks.
    'coverage':('gfortran',['-O3','-ffp-contract=off','-fno-tree-vectorize','-fno-unsafe-math-optimizations',
                          '--coverage','-Wno-error=coverage-invalid-line-number']),
}


def module_dependencies(text):
    provides=set();uses=set()
    for st in statements(active_source(text)):
        module=re.match(r'module\s+(?!procedure\b)(\w+)',st.text,re.I)
        if module:provides.add(module[1].lower())
        use=re.match(r'use\s+(\w+)',st.text,re.I)
        if use:uses.add(use[1].lower())
    return provides,uses-{'iso_fortran_env','iso_c_binding','ieee_arithmetic','ieee_exceptions'}


def expose_native_routines(contents,build,selections):
    """Expose explicitly selected native initializers in owned copies only."""
    changes=[]
    for selection in selections:
        if not re.fullmatch(r'[a-zA-Z]\w*:[a-zA-Z]\w*',selection):
            raise ValueError('native routine access requires module:routine')
        module,name=selection.lower().split(':')
        matches=[(p,r) for p in contents if module in module_dependencies(contents[p])[0]
                 for r in routines(build/p,build)
                 if r.module==module and r.name==name]
        if len(matches)!=1:raise ValueError('native access routine is not in the selected source closure: '+selection)
        path,routine=matches[0]
        anchor,_=module_anchors(contents[path],module)
        if anchor is None:raise ValueError('native module has no CONTAINS: '+module)
        insertion='  public :: '+name+' ! oracle standalone native access\n'
        contents[path]=contents[path][:anchor]+insertion+contents[path][anchor:]
        changes.append({'module':module,'routine':name,'path':path,
                        'instrumented_lines':[routine.start+1,routine.end],
                        'instrumented_source_sha256':hashlib.sha256((build/path).read_bytes()).hexdigest(),
                        'insertion':insertion,'numerical_source_changed':False})
    return changes


def prepare(build,scheme,out,recipes=None,expose_routines=()):
    recipes=recipes or list(RECIPES)
    if any(r not in RECIPES for r in recipes):raise ValueError("unknown standalone recipe")
    build=build.resolve();out=out.resolve()
    if out.exists():raise ValueError('standalone output must be fresh')
    manifest=json.loads((build/'oracle-inventory.json').read_text())
    selected=[s for s in manifest['sites'] if s['site']==scheme or s['scheme']==scheme]
    if len(selected)!=1:raise ValueError('select one unambiguous scheme call site')
    site=selected[0]
    if site['issues']:raise ValueError('scheme arguments are unresolved')
    paths={m['path'] for m in manifest['module_state'] if m['name'] in site['state_modules']}
    paths.update(['src/framework/mpas_kind_types.F','src/framework/mpas_constants.F',
                  'src/core_atmosphere/physics/mpas_oracle_dump.F'])
    contents={p:(build/p).read_text() for p in paths}
    # NSSL calls the external ESMF-time message routine. Copy the native
    # implementation verbatim: it prints and returns (it does not abort).
    if any(re.search(r'\bcall\s+wrf_error_fatal\s*\(',t,re.I) for t in contents.values()):
        fatal_path='src/external/esmf_time_f90/wrf_error_fatal.F90'
        contents[fatal_path]=(build/fatal_path).read_text()
    # Copy the actual public log-level parameters, rather than duplicating values.
    logtypes=(build/'src/framework/mpas_log_types.inc').read_text()
    constants='\n'.join(st.text for st in statements(logtypes)
                        if st.text.lower().startswith('integer, parameter'))
    if not constants:raise ValueError('native logging parameters missing')
    logger=next(r for r in routines(build/'src/framework/mpas_log.F',build) if r.name=='mpas_log_write')
    logger_declarations=[]
    for name in logger.args:
        var=logger.variables[name]
        logger_declarations.append(var.type+var.attrs+' :: '+name+('('+','.join(var.dims)+')' if var.dims else ''))
    contents['framework_log_stub.F90']=f'''! Interface: MPAS src/framework/mpas_log.F:471-637
module mpas_derived_types
implicit none
{constants}
end module
module mpas_log
use mpas_derived_types
use mpas_kind_types
implicit none
contains
subroutine mpas_log_write({','.join(logger.args)})
{chr(10).join(logger_declarations)}
write(*,'(a)') trim(message)
if(present(err)) err=0
if(present(messageType)) then
 if(messageType==MPAS_LOG_CRIT) error stop 'native physics fatal'
endif
end subroutine
end module
'''
    # Native I/O-unit allocation is an ordinary framework module, needed by
    # Thompson/CAM even when a captured replay does not open a table file.
    if any('mpas_io_units' in module_dependencies(t)[1] for t in contents.values()):
        io_path='src/framework/mpas_io_units.F'
        contents[io_path]=(build/io_path).read_text()
    dependencies={p:module_dependencies(t) for p,t in contents.items()}
    owners={}
    for p,(provided,_) in dependencies.items():
        for name in provided:
            if name in owners:raise ValueError(f'duplicate native module {name}')
            owners[name]=p
    graph={p:{owners[n] for n in used if n in owners}-{p} for p,(_,used) in dependencies.items()}
    unknown={p:sorted(used-set(owners)) for p,(_,used) in dependencies.items() if used-set(owners)}
    if unknown:raise ValueError(f'framework dependencies need native sources/stubs: {unknown}')
    remaining=set(contents);ordered=[]
    while remaining:
        ready=sorted(p for p in remaining if not graph[p]&remaining)
        if not ready:raise ValueError(f'cyclic native module source dependencies: {sorted(remaining)}')
        ordered.extend(ready);remaining-=set(ready)
    replay=build/'oracle_replay'/('replay_'+site['site']+'.F90')
    if not replay.is_file():raise ValueError('generated replay source missing')
    access=expose_native_routines(contents,build,expose_routines)
    out.mkdir(parents=True)
    receipts={}
    for recipe in recipes:
        compiler,optimization=RECIPES[recipe]
        directory=out/recipe;directory.mkdir()
        for p,t in contents.items():
            target=directory/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(t)
        shutil.copy2(replay,directory/'replay.F90')
        flags=(['-ffree-form','-ffree-line-length-none','-fconvert=big-endian','-fallow-argument-mismatch']
               if compiler=='gfortran' else ['-free','-extend-source','-convert','big_endian','-fpp'])
        flags+=optimization+['-Dmpas','-DSINGLE_PRECISION','-DCORE_ATMOSPHERE']
        if manifest.get('reference')=='rrfs':flags+=['-Dtempo_mpas']
        # Keep object identity tied to its source when a dependency is added.
        # Ordinal names can reuse a different source's older object silently.
        names={p:'native_'+hashlib.sha256(p.encode()).hexdigest()[:16]+'.o' for p in ordered}
        text=f'FC={compiler}\nFLAGS={" ".join(flags)}\nall: replay\n'
        for p in ordered:
            text+=f'{names[p]}: {p} Makefile '+ ' '.join(names[d] for d in sorted(graph[p]))+'\n'
            text+='\t$(FC) $(FLAGS) -c $< -o $@\n'
        objects=' '.join(names[p] for p in ordered)
        text+='replay: replay.F90 Makefile '+objects+'\n'
        text+='\t$(FC) $(FLAGS) replay.F90 '+objects+' -o $@\n'
        (directory/'Makefile').write_text(text)
        receipts[recipe]={'compiler':compiler,'flags':flags,'source_order':ordered,'objects':names}
    (out/'Makefile').write_text('all: '+' '.join(recipes)+'\n.PHONY: all '+' '.join(recipes)+'\n'+
                              ''.join(f'{r}:\n\t$(MAKE) -C {r}\n' for r in recipes))
    info={'schema':'mpas-native-standalone-recipes-v1','scheme':site['site'],
          'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'inventory_sha256':hashlib.sha256((build/'oracle-inventory.json').read_bytes()).hexdigest(),
          'source_sha256':{p:hashlib.sha256(t.encode()).hexdigest() for p,t in contents.items()},
          'logging_interface_source_sha256':hashlib.sha256((build/'src/framework/mpas_log.F').read_bytes()).hexdigest(),
          'replay_source_sha256':hashlib.sha256(replay.read_bytes()).hexdigest(),
          'framework_stubs':['logging interface and log-level parameters only'],
          'native_routine_access':access,
          'recipes':receipts,'forecast_proven':False}
    (out/'build-manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--scheme',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recipe',choices=list(RECIPES),action='append')
    parser.add_argument('--expose-routine',action='append',default=[],metavar='MODULE:ROUTINE')
    a=parser.parse_args();info=prepare(a.build,a.scheme,a.out,a.recipe,a.expose_routine)
    print(json.dumps({'scheme':info['scheme'],'recipes':list(info['recipes'])}))


if __name__=='__main__':main()
