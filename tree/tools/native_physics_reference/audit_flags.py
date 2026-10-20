#!/usr/bin/env python3
"""Audit every physics object against its actual compiler command, including externals.

MPAS src/core_atmosphere/physics/Makefile:61-86 passes inherited FFLAGS to
MMM, UGWP, WRF and Noah-MP; Noah-MP src/Makefile:9 strips only -std=f2008.
An empty audit or an object without a matching command fails closed.
"""
import argparse
import json
from pathlib import Path
import re
import shlex

STRICT = ('-ffp-contract=off', '-fno-tree-vectorize', '-fno-unsafe-math-optimizations')

def audit(root, recipe):
    commands = {}
    for line in (root / 'build.log').read_text().splitlines():
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        if '-c' not in args or not any('mpif' in a or 'gfortran' in a for a in args[:2]):
            continue
        source = next((a for a in args if re.search(r'\.(F90|F|f90)$', a)), None)
        if source:
            commands.setdefault(Path(source).stem, []).append(args)
    records, errors = [], []
    physics = root / 'src/core_atmosphere/physics'
    for obj in sorted(physics.rglob('*.o')):
        candidates = commands.get(obj.stem, [])
        required = ('-O3', '-DSINGLE_PRECISION') + (STRICT if recipe == 'gnu-strict' else ())
        forbidden = ('-fdefault-real-8', '-fdefault-double-8', '-ffast-math', '-Ofast')
        if recipe == 'gnu-strict':
            forbidden += ('-ffp-contract=fast', '-ffp-contract=on', '-ftree-vectorize', '-funsafe-math-optimizations')
        good = [c for c in candidates if all(f in c for f in required)
                and not any(f in c for f in forbidden)]
        flags_ok = bool(candidates) and len(good) == len(candidates)
        record = {'object': str(obj.relative_to(root)), 'source_stem': obj.stem,
                  'matched_commands': len(candidates), 'flags_ok': flags_ok}
        if flags_ok:
            record['command'] = shlex.join(good[0])
        else:
            errors.append(record['object'])
        records.append(record)
    groups = {g: sum(g in r['object'] for r in records) for g in
              ('physics_mmm/', 'physics_noaa/UGWP/', 'physics_noahmp/', 'physics_wrf/')}
    errors += [f'missing group {g}' for g, n in groups.items() if not n]
    if not records:
        errors.append('no physics objects')
    return {'schema': 'mpas.native-physics-flags/v1', 'recipe': recipe,
            'all_ok': not errors, 'groups': groups, 'errors': errors, 'objects': records}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--recipe', choices=('gnu-strict', 'gnu-o3'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); r = audit(a.root, a.recipe)
    a.out.write_text(json.dumps(r, indent=2)+'\n')
    print(json.dumps({k: r[k] for k in ('all_ok', 'groups', 'errors')}))
    return 0 if r['all_ok'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
