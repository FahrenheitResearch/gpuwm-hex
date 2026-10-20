#!/usr/bin/env python3
"""Prepare owned full-framework oracle builds for native replay recipes.

Driver-local numerical routines need MPAS framework interfaces. Compile the
entire native tree rather than replacing numerical framework code with stubs.
No build runs here: invoke each emitted command as make -j4.
Recipe flags follow native_dry_reference/build_native_reference.sh:78-99.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from prepare_build import prepare


def configure(tree,recipe):
    path=tree/'Makefile';text=path.read_text()
    target='intel-mpi' if recipe.startswith('ifx-') else 'gfortran'
    start=re.search(r'^'+re.escape(target)+r':',text,re.M)
    if not start:raise ValueError(f'native build target missing: {target}')
    end=re.search(r'^\w[\w-]*:',text[start.end():],re.M)
    stop=start.end()+end.start() if end else len(text)
    section=text[start.start():stop]
    if recipe.startswith('ifx-'):
        flags='-O3 -convert big_endian -free -align array64byte -Qoption,fpp,-macro_expand=vc'
        if recipe=='ifx-valuesafe':flags+=' -fp-model=source -no-fma -fimf-arch-consistency=true'
        replacements={'FC_PARALLEL':'mpiifx','CC_PARALLEL':'mpigcc','CXX_PARALLEL':'mpigxx',
                      'FC_SERIAL':'ifx','CC_SERIAL':'icx','CXX_SERIAL':'icpx'}
    else:
        flags='-O3 '
        if recipe in {'gnu-strict','coverage'}:
            flags+='-ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations '
        flags+='-ffree-line-length-none -fconvert=big-endian -ffree-form'
        if recipe=='coverage':flags+=' --coverage -Wno-error=coverage-invalid-line-number'
        replacements={}
    replacements['FFLAGS_OPT']=flags
    if recipe=='coverage':replacements['LDFLAGS_OPT']='-O3 --coverage'
    for name,value in replacements.items():
        section,count=re.subn(r'"'+name+r' = [^"]*"',lambda _:f'"{name} = {value}"',section)
        if count!=1:raise ValueError(f'expected one {target}/{name} assignment, found {count}')
    path.write_text(text[:start.start()]+section+text[stop:])
    return target,flags


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pin-set',choices=['mpas-v842','rrfs'],default='mpas-v842')
    parser.add_argument('--recipe',choices=['gnu-strict','gnu-o3','ifx-o3','ifx-valuesafe','coverage'],action='append',required=True)
    a=parser.parse_args()
    if a.out.exists():parser.error('full-recipe root must be fresh')
    a.out.mkdir(parents=True);rows=[]
    for recipe in dict.fromkeys(a.recipe):
        tree=a.out/recipe
        manifest=prepare(a.source,tree,a.reference,pin_set_name=a.pin_set)
        target,flags=configure(tree,recipe)
        rows.append({'recipe':recipe,'directory':str(tree.resolve()),'make_target':target,
                     'fortran_flags':flags,'mpi_fortran_compiler':'mpiifx' if recipe.startswith('ifx-') else 'mpif90',
                     'mpi_environment_requirements':['I_MPI_ROOT must name the Intel MPI installation'] if recipe.startswith('ifx-') else [],
                     'inventory_sha256':hashlib.sha256((tree/'oracle-inventory.json').read_bytes()).hexdigest(),
                     'makefile_sha256':hashlib.sha256((tree/'Makefile').read_bytes()).hexdigest(),
                     'instrumentation_sha256':manifest['instrumentation_sha256']})
    (a.out/'build-manifest.json').write_text(json.dumps({'schema':'mpas-full-framework-replay-recipes-v1',
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'recipes':rows,
        'forecast_proven':False},indent=2)+'\n')
    print(json.dumps({'prepared_recipes':[r['recipe'] for r in rows]}))


if __name__=='__main__':main()
