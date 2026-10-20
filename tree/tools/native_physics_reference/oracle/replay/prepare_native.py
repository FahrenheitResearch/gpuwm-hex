#!/usr/bin/env python3
"""Build generated strict replays against the owned instrumented native archive.

The native model must already be compiled with its audited pinned recipe. This
utility only writes a Makefile; invoke it as make -j4.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

FLAGS=['-O3','-ffp-contract=off','-fno-tree-vectorize','-fno-unsafe-math-optimizations',
       '-ffree-line-length-none','-ffree-form','-fconvert=big-endian']


def prepare(build,log,out,replay_root=None,recipe='gnu-strict'):
    build=build.resolve();out=out.resolve()
    if out.exists():raise ValueError('replay build directory must be fresh')
    if not (build/'src/libdycore.a').exists():raise ValueError('native model archive is not built')
    links=[]
    for line in log.read_text().splitlines():
        try:args=shlex.split(line)
        except ValueError:continue
        if '-o' in args and args[args.index('-o')+1]=='atmosphere_model' and '-ldycore' in args:links.append(args)
    if not links:raise ValueError('native final link command missing')
    args=links[-1];libraries=args[args.index('-L.'):]
    libraries=[f'-L{build}/src' if a=='-L.' else a for a in libraries]
    includes=sorted({p.parent for p in build.rglob('*.mod')})
    sources=sorted((replay_root or build/'oracle_replay').resolve().glob('replay_*.F90'))
    if not sources:raise ValueError('no generated replay sources')
    for path in [build,out,*includes,*sources]:
        if any(c in str(path) for c in ' \t\n#$'):raise ValueError(f'unsupported Makefile path: {path}')
    if recipe=='gnu-strict':flags=FLAGS
    elif recipe=='gnu-o3':flags=['-O3','-ffree-line-length-none','-ffree-form','-fconvert=big-endian']
    elif recipe=='coverage':flags=FLAGS+['--coverage']
    elif recipe in {'ifx-o3','ifx-valuesafe'}:
        flags=['-O3','-free','-extend-source','-convert','big_endian','-align','array64byte','-Qoption,fpp,-macro_expand=vc']
        if recipe=='ifx-valuesafe':flags+=['-fp-model=source','-no-fma','-fimf-arch-consistency=true']
    else:raise ValueError(f'unsupported replay recipe: {recipe}')
    out.mkdir(parents=True)
    compiler='mpiifx' if recipe.startswith('ifx-') else 'mpif90'
    text='FC='+compiler+'\nFLAGS='+shlex.join(flags+[f'-I{p}' for p in includes])+'\n'
    # Group native archives because module restoration imports can create cycles.
    text+='LIBS=-Wl,--start-group '+shlex.join(libraries)+' -Wl,--end-group\n'
    text+='all: '+' '.join(s.stem for s in sources)+'\n'
    for source in sources:
        text+=f'{source.stem}: {source}\n\t$(FC) $(FLAGS) $< $(LIBS) -o $@\n'
    (out/'Makefile').write_text(text)
    meta={'schema':'mpas-oracle-native-replay-build-v1','recipe':recipe,
          'native_binary_sha256':hashlib.sha256((build/'atmosphere_model').read_bytes()).hexdigest(),
          'inventory_sha256':hashlib.sha256((build/'oracle-inventory.json').read_bytes()).hexdigest(),
          'sources':{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
          'flags':flags,'link_libraries':libraries,'native_build_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
    (out/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','log','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replay-root',type=Path,help='fresh generated replay sources against the same audited native archives')
    p.add_argument('--recipe',choices=['gnu-strict','gnu-o3','ifx-o3','ifx-valuesafe','coverage'],default='gnu-strict')
    a=p.parse_args();m=prepare(a.build,a.log,a.out,a.replay_root,a.recipe)
    print(json.dumps({'recipe':m['recipe'],'replays':len(m['sources'])}))


if __name__=='__main__':main()
