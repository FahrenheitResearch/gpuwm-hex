"""Insertion-only, complete real-pool capture of fork cold-core initialization.

MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:375-614,1268-1309,
1481-1584 at 1d6e370e. Every retrieved numeric pool field is observed at
entry and exit, including full unvisited allocation storage.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

if __package__:
    from .prepare_dycore_tap import active_source, statements, routines, snapshot
    from .native_source import read_pinned_source
else:
    from prepare_dycore_tap import active_source, statements, routines, snapshot
    from native_source import read_pinned_source

SOURCE = 'src/core_atmosphere/mpas_atm_core.F'
ROUTINES = ('atm_compute_damping_coefs', 'atm_compute_cellsoncellsoncell')


def generate(root, out):
    path = root / SOURCE
    source = read_pinned_source(root, path)
    active = active_source(source)
    selected = {r.name: r for r in routines(path, root)}
    inserts, boundaries = {}, []
    for name in ROUTINES:
        routine = selected[name]
        getters = [st for st in routine.body if re.match(
            r'call\s+mpas_pool_get_(array|dimension|config)\s*\(', st.text, re.I)]
        fields = []
        for getter in getters:
            variable = re.search(r',\s*(\w+)\s*\)\s*$', getter.text)[1].lower()
            declaration = routine.variables[variable]
            if not declaration.supported:
                raise ValueError('unsupported native pool field: ' + variable)
            fields.append(('pool.' + variable, variable, declaration))
        key = 'core__' + name
        signature = next(st for st in statements(active) if st.start == routine.start)
        inserts.setdefault(signature.end, []).append(
            '      use mpas_oracle_dump, only: oracle_begin,oracle_enabled,oracle_write\n')
        # Declare observers before the original executable part. Entry snapshot
        # follows every original getter and precedes the first pool mutation.
        first = next(st for st in routine.body if re.match(r'(call\b|\w+\s*=)', st.text, re.I))
        inserts.setdefault(first.start, []).append('      integer :: r3_tap_event\n')
        inserts.setdefault(getters[-1].end, []).append(
            f"      call oracle_begin('{key}',r3_tap_event)\n"
            f"      if (oracle_enabled('{key}')) then\n" + snapshot(key, 'pre', fields) + '      endif\n')
        inserts.setdefault(routine.end - 1, []).append(
            f"      if (oracle_enabled('{key}')) then\n" + snapshot(key, 'post', fields) + '      endif\n')
        boundaries.append(dict(key=key, source=SOURCE, first_line=routine.start+1,
            last_line=routine.end, fields=[dict(label=label, expression=expr,
            declaration=asdict(var)) for label, expr, var in fields]))
    # Real block rank and cell IDs before either cold initializer is called.
    driver = selected['atm_mpas_init_block']
    signature = next(st for st in statements(active) if st.start == driver.start)
    inserts.setdefault(signature.end, []).append('      use mpas_oracle_dump, only: oracle_set_context\n')
    first = next(st for st in driver.body if re.match(r'call\b', st.text, re.I))
    inserts.setdefault(first.start, []).append(
        '      integer, pointer :: r3_tap_ids(:)\n'
        "      call mpas_pool_get_array(mesh,'indexToCellID',r3_tap_ids)\n"
        '      call oracle_set_context(r3_tap_ids,0,dminfo%my_proc_id)\n')
    result = ''.join(''.join(inserts.get(i, [])) + line
                     for i, line in enumerate(source.splitlines(keepends=True)))
    target = out / SOURCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result)
    contract = dict(schema='rrfs-core-native-tap-overlay-v1',
        fork_pin='1d6e370e7b43a332074809f170532a216c2212fb',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        overlay_sha256=hashlib.sha256(result.encode()).hexdigest(),
        native_arithmetic_edits=[], boundaries=boundaries,
        acceptance='diagnostic pending native observer passivity')
    (out / 'core-tap-contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    generate(args.source.resolve(), args.out.resolve())
