"""Prepare unchanged-machine debugger and host boundary diagnostics.

Only source syntax is inspected here. Rust reads native memory and compares
words. Every original bank profile remains executed and fully graded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
from translate_module import parse_modules


def main(a):
    a.output.mkdir(exist_ok=False)
    paths = [a.native / n for n in ('mynn_shared.F90', 'bl_mynn_subroutines.F90', 'bl_mynn.F90')]
    routines = parse_modules(paths)[1]
    selected = tuple(a.routines)
    calls = [2*(deck-a.first_deck)+step for deck in a.decks for step in (1, 2)]
    plan = dict(scope=__doc__, driver_symbol='__bl_mynn_MOD_bl_mynn_run',
        selected_driver_calls=calls, process_count=1, routines=[])
    for name in selected:
        r = routines[name]
        args = []
        for name2 in r.arguments:
            v = r.variables[name2]
            assert not v.optional and v.kind in ('float', 'double', 'int', 'bool')
            assert len(v.dimensions) <= 1 and all(re.fullmatch(r'kts:kte(?:[+]1)?', d) for d in v.dimensions)
            args.append(dict(name=v.name, kind=v.kind, dimensions=v.dimensions, intent=v.intent,
                source=f'{r.source}:{v.line}'))
        plan['routines'].append(dict(name=name, symbol='__bl_mynn_subroutines_MOD_'+name,
            source=f'{r.source}:{r.start}-{r.end}', arguments=args))
    (a.output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if a.plan_only:
        (a.output / 'admission.json').write_text(json.dumps(dict(
            scope=__doc__, source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in paths}, selected_driver_calls=calls,
            complete_bank_profiles=a.profiles, acceptance_proven=False,
            excluded_words=[], plan_sha256=hashlib.sha256((a.output / 'plan.json').read_bytes()).hexdigest()),
            indent=2) + '\n')
        return
    target = a.output / 'host'; target.mkdir()
    for p in a.host.iterdir():
        if p.suffix in ('.F90', '.F', '.cpp', '.cuh') or p.name == 'Makefile':
            shutil.copy2(p, target / p.name)
    cpp = (target / 'translation.cpp').read_text()
    marker = '// Generated from pinned MPAS MMM; see translation receipt.'
    helper = r'''
#include <cstdio>
#include <functional>
#include <map>
#include <string>
static FILE* p12_boundary_file=nullptr;
static int p12_boundary_driver=0;
static std::map<std::string,int> p12_boundary_counts;
template<class T> static void p12_boundary_word(const char* name,int driver,int invocation,int k,T x) {
 if(!p12_boundary_file)p12_boundary_file=std::fopen("boundary.words","w");
 uint32_t bits[2]={0,0};std::memcpy(bits,&x,sizeof(T));
 if(sizeof(T)<=4)std::fprintf(p12_boundary_file,"R out:%s %d %d %d 0 %08X\n",name,driver,invocation,k,bits[0]);
 else {std::fprintf(p12_boundary_file,"R out:%s_lo %d %d %d 0 %08X\n",name,driver,invocation,k,bits[0]);std::fprintf(p12_boundary_file,"R out:%s_hi %d %d %d 0 %08X\n",name,driver,invocation,k,bits[1]);}
}
struct P12BoundaryGuard {
 std::function<void()> exit;
 ~P12BoundaryGuard(){exit();}
};
'''
    assert cpp.count(marker) == 1
    cpp = cpp.replace(marker, helper+marker)
    condition = ' || '.join(f'p12_boundary_driver=={n}' for n in calls)
    lines = []
    for line in cpp.splitlines():
        lines.append(line)
        m = re.match(r'__device__ __noinline__ void f_(\w+)\(', line)
        if not m: continue
        name = m[1]
        if name == 'bl_mynn_run':
            lines += ['++p12_boundary_driver;p12_boundary_counts.clear();']
        if name not in selected: continue
        r = routines[name]
        def snapshot(phase):
            body = []
            for arg in r.arguments:
                v = r.variables[arg]
                label = f'{name}_{phase}_{arg}'
                if v.dimensions:
                    lo, hi = v.dimensions[0].split(':')
                    body.append(f'for(int p12_k={lo};p12_k<={hi};++p12_k)p12_boundary_word("{label}",p12_driver,p12_invocation,p12_k,{arg}(p12_k));')
                else:
                    body.append(f'p12_boundary_word("{label}",p12_driver,p12_invocation,0,{"int("+arg+")" if v.kind=="bool" else arg});')
            return '\n'.join(body)
        lines += [f'const bool p12_selected=({condition});',
            'const int p12_driver=p12_boundary_driver;',
            f'const int p12_invocation=++p12_boundary_counts["{name}"];',
            'if(p12_selected){'+snapshot('entry')+'}',
            'P12BoundaryGuard p12_guard{[&,p12_driver,p12_invocation,p12_selected]{if(p12_selected){'+snapshot('exit')+'}}};']
    (target / 'translation.cpp').write_text('\n'.join(lines)+'\n')
    files = {str(p.relative_to(a.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in a.output.rglob('*') if p.is_file()}
    (a.output / 'admission.json').write_text(json.dumps(dict(files=files,
        original_native=str(a.native), original_host=str(a.host),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        selected_driver_calls=calls, complete_bank_profiles=a.profiles,
        acceptance_proven=False, excluded_words=[]), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'host', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--profiles',type=int,required=True)
    p.add_argument('--first-deck',type=int,default=0)
    p.add_argument('--decks',type=int,nargs='+',required=True)
    p.add_argument('--routines',nargs='+',default=['mym_level2', 'mym_condensation', 'mym_turbulence', 'mym_predict', 'mynn_tendencies'])
    p.add_argument('--plan-only', action='store_true')
    main(p.parse_args())
