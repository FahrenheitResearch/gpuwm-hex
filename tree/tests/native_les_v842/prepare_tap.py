"""Record complete real LES calls using f1's native typed transport.

MPAS mpas_atm_dissipation_models.F:208-1606. Only an owned source copy
is changed; no native arithmetic expression or call argument is rewritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys

HERE = Path(__file__).resolve()
TREE = HERE.parents[2]
sys.path.insert(0, str(TREE / 'tools/native_physics_reference/oracle'))
from fortran_source import routines, statements

NAMES = ('les_models', 'calculate_n2', 'u_dissipation_3d',
         'w_dissipation_3d', 'scalar_dissipation_3d_les')
REL = Path('src/core_atmosphere/dynamics/mpas_atm_dissipation_models.F')


def instrument(source, destination):
    original = source.read_text()
    lines = original.splitlines(keepends=True)
    insertions = {}
    inventory = []
    for routine in routines(source, source.parent):
        if routine.name not in NAMES:
            continue
        first = next(st.start for st in routine.body
                     if re.match(r'(call\b|if\s*\(|do\b|select\b|\w+\s*=)', st.text, re.I))
        signature = next(st for st in statements(original) if st.start == routine.start)
        insertions.setdefault(signature.end, []).append(
            '      use mpas_oracle_dump, only: oracle_begin, oracle_write, oracle_set_context\n')
        names = [name for name in routine.args if name != 'clock']
        names += [name for name in ('nvertlevels', 'maxedges', 'num_scalars') if name not in names]
        insertions.setdefault(first, []).append(
            '      integer, save :: d2_tap_calls=0\n      integer :: d2_tap_event\n'
            '      d2_tap_calls=d2_tap_calls+1\n'
            '      call oracle_set_context([integer ::], (d2_tap_calls+2)/3, 0)\n'
            f"      call oracle_begin('{routine.name}', d2_tap_event)\n")
        for phase, index in (('pre', first), ('post', routine.end - 1)):
            calls = []
            for name in names:
                variable = routine.variables.get(name)
                lower = f', lower=lbound({name})' if variable and variable.dims else ''
                calls.append(f"      call oracle_write('{routine.name}', '{phase}', '{name}', &\n"
                             f"                        {name}, d2_tap_event{lower})\n")
            insertions.setdefault(index, []).append(''.join(calls))
        if any(re.match(r'return\b', st.text, re.I) for st in routine.body):
            raise ValueError('early return requires explicit exit instrumentation')
        inventory.append(dict(routine=routine.name, first_line=routine.start + 1,
                              last_line=routine.end, recorded=names,
                              omitted=['clock: only used in commented flux_les_sas code']
                              if 'clock' in routine.args else []))
    if {row['routine'] for row in inventory} != set(NAMES):
        raise ValueError('incomplete native LES inventory')
    destination.write_text(''.join(''.join(insertions.get(i, [])) + line
                                   for i, line in enumerate(lines)))
    return dict(original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                instrumented_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                routines=inventory, context='MPI1, three native RK calls per time step; full arrays, no cell extraction')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('refusing to overwrite a native source/build copy')
    shutil.copytree(args.source, args.out, symlinks=False)
    args.out.chmod(args.out.stat().st_mode | 0o700)
    for path in args.out.rglob('*'):
        path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o200))
    inventory = instrument(args.source / REL, args.out / REL)
    dynamics = args.out / REL.parent
    shutil.copyfile(TREE / 'tools/native_physics_reference/oracle/mpas_oracle_dump.F',
                    dynamics / 'mpas_oracle_dump.F')
    makefile = dynamics / 'Makefile'
    text = makefile.read_text().replace('OBJS = ', 'OBJS = mpas_oracle_dump.o ', 1)
    makefile.write_text(text + '\nmpas_atm_dissipation_models.o: mpas_oracle_dump.o\n')
    config = args.out / 'src/core_atmosphere/Externals.cfg'
    config.write_text(re.sub(r'(?m)^repo_url = .*$', 'repo_url = .', config.read_text()))
    subprocess_args = [sys.executable, str(TREE / 'tools/native_dry_reference/edit_makefile_flags.py'),
                       str(args.out / 'Makefile'), 'gfortran', '-O3 ',
                       '-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations ']
    import subprocess
    subprocess.run(subprocess_args, check=True)
    (args.out / 'd2-tap-source.json').write_text(json.dumps(inventory, indent=2) + '\n')


if __name__ == '__main__':
    main()
