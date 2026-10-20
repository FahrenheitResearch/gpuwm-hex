#!/usr/bin/env python3
"""Audit every owned oracle object against recorded compilation commands.

Extends f0 audit_flags.py to framework, dynamics, tools, external objects and
instrumentation. Fortran objects require the selected recipe and single
precision; C/C++ objects reject unsafe math. Unresolved audits fail closed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

STRICT={'-ffp-contract=off','-fno-tree-vectorize','-fno-unsafe-math-optimizations'}


def audit(root,log,recipe='gnu-strict'):
    if recipe not in {'gnu-strict','gnu-o3','coverage'}:
        raise ValueError(f'unsupported GNU object audit recipe: {recipe}')
    root=root.resolve()
    object_paths=sorted(root.rglob('*.o'))
    sources={}
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.f','.f90','.c','.cpp'}:
            sources.setdefault(path.name,[]).append(path)
    commands={}
    for line in log.read_text().splitlines():
        try:args=shlex.split(line)
        except ValueError:continue
        if '-c' not in args:continue
        source=next((x for x in args if re.search(r'\.(?:f|f90|c|cpp)$',x,re.I)),None)
        if source:
            objstem=Path(args[args.index('-o')+1]).stem if '-o' in args else Path(source).stem
            # Parallel make output interleaves directories. Resolve the source
            # against actual source/object pairs instead of a global cwd stack.
            # Native TEMPO compiles src/*.F90 from its package root; table
            # tools compile ../physics/... sources from utils. Resolve the
            # source relative to each candidate object directory, rather
            # than assuming object and source share a directory.
            matches=[obj for obj in object_paths if obj.stem==objstem
                     and (obj.parent/source).resolve() in sources.get(Path(source).name,[])]
            if Path(source).is_absolute():
                matches=[obj for obj in matches if obj.parent==Path(source).parent]
            if len(matches)!=1:
                commands.setdefault(objstem,[]).append((None,source,args))
            else:commands.setdefault(objstem,[]).append((matches[0],source,args))
    objects=[];errors=[]
    for obj in object_paths:
        candidates=commands.get(obj.stem,[])
        matched=[c for c in candidates if c[0]==obj]
        alias=None
        if not matched:
            # MPAS copies physics objects into libphys/libchem. Only accept a
            # copy whose bytes match its uniquely resolved compiled original.
            origins={c[0] for c in candidates if c[0] is not None}
            identical=[p for p in origins if p.read_bytes()==obj.read_bytes()]
            if len(identical)==1:
                alias=identical[0];matched=[c for c in candidates if c[0]==alias]
        okay=False
        if matched and not any(c[0] is None for c in candidates):
            origin,source,args=matched[-1]
            fortran=Path(source).suffix.lower() in {'.f','.f90'}
            okay=not any(f in args for f in ['-Ofast','-ffast-math','-funsafe-math-optimizations',
                                           '-ffp-contract=fast','-fdefault-real-8','-fdefault-double-8'])
            if fortran:
                okay=okay and '-O3' in args
                if recipe in {'gnu-strict','coverage'}:
                    okay=okay and STRICT.issubset(args)
                if recipe=='coverage':okay=okay and '--coverage' in args
        row={'object':str(obj.relative_to(root)),'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),
             'matched_commands':len(matched),'flags_ok':okay}
        if matched:
            row['command']=shlex.join(matched[-1][2])
            row['compiled_object']=str(matched[-1][0].relative_to(root))
        if alias:row['identical_copy_of']=str(alias.relative_to(root))
        if not okay:errors.append(row['object'])
        objects.append(row)
    if not objects:errors.append('no objects found')
    return {'schema':'mpas-oracle-object-flags-v2','recipe':recipe,'all_ok':not errors,
            'errors':errors,'objects':objects,'build_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
            'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--recipe',choices=['gnu-strict','gnu-o3','coverage'],default='gnu-strict')
    a=p.parse_args();result=audit(a.build,a.log,a.recipe)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'objects':len(result['objects']),'all_ok':result['all_ok'],'errors':result['errors']}))
    return 0 if result['all_ok'] else 1


if __name__=='__main__':raise SystemExit(main())
