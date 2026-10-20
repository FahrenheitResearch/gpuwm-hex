"""Generate a complete dycore PRE/POST overlay for the native RRFS tap.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:
2225-2366,2646-2984,3613-4023,5150-6409,7399-7446 at 1d6e370e.
This is source orchestration. Numeric records use the existing native transport.
The overlay is diagnostic until native observer passivity has been measured.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parents[2] / 'tools/native_physics_reference/oracle'
sys.path.insert(0, str(TOOLS))
from fortran_source import Variable, active_source, declarations, routines, statements
if __package__:
    from .native_source import read_pinned_source
else:
    from native_source import read_pinned_source

SOURCE = 'src/core_atmosphere/dynamics/mpas_atm_time_integration.F'
ROUTINES = (
    'atm_compute_vert_imp_coefs_work', 'atm_advance_acoustic_step_work',
    'atm_advance_scalars_work', 'atm_compute_dyn_tend_work',
    'atm_zero_gradient_w_bdy_work',
    'atm_advance_scalars_mono_work',
)
NON_NUMERIC_ARGUMENTS = {'atm_advance_scalars_mono_work': {'block', 'state', 'exchange_halo_group'}}
# The fork's atm_advance_scalars declares this pointer at 3496 and passes it
# at 3574 without a pool retrieval. Only the filter-on branch reads it
# (3783-3834). Observing it while that branch is off would dereference a
# descriptor the native model itself never reads. Record the read condition;
# filter-on still attempts the original input and is not silently excluded.
CONDITIONAL_READS = {
    ('dycore__atm_advance_scalars_work', 'argument.meshscalingdel4'):
        'rk_step == 1 .and. h_scalar_filter4 > 0.0_RKIND',
}


def is_executable(statement, variables):
    if declarations(statement.text):
        return False
    # Fortran statement functions belong to the specification part. Inserting
    # a CALL before flux3/flux4 would turn their original definitions into an
    # invalid executable assignment. Array assignments remain executable.
    function = re.match(r'(\w+)\s*\([^=]*\)\s*=', statement.text)
    if function and function[1].lower() in variables and not variables[function[1].lower()].dims:
        return False
    return bool(re.match(r'(call\b|if\s*\(|do\b|select\b|return\b|\w+(?:\s*\([^=]*\))?\s*=)', statement.text, re.I))


def axes(variable):
    """Derive metadata only from declared native dimensions, never shape guesses."""
    cell = level = 0
    for index, dimension in enumerate(variable.dims, 1):
        words = set(re.findall(r'\w+', dimension.lower()))
        if 'ncells' in words:
            cell = index
        if 'nvertlevels' in words:
            level = index
    return cell, level


def snapshot(key, phase, fields):
    lines = []
    for label, expression, variable in fields:
        guards = []
        condition = CONDITIONAL_READS.get((key, label))
        if condition:
            lines.append(f"      call oracle_write('{key}','{phase}','{label}.read',({condition}),r3_tap_event)\n")
            guards.append(condition)
        if variable.optional:
            guards.append(f'present({expression})')
        if 'pointer' in variable.attrs:
            guards.append(f'associated({expression})')
        elif 'allocatable' in variable.attrs:
            guards.append(f'allocated({expression})')
        for index, guard in enumerate(guards):
            lines.append(f"      call oracle_write('{key}','{phase}','{label}.present{index}',{guard},r3_tap_event)\n")
            lines.append(f'      if ({guard}) then\n')
        metadata = f',lower=lbound({expression})' if variable.dims else ''
        cell, level = axes(variable)
        if cell:
            metadata += f',cell_axis={cell}'
        if level:
            metadata += f',level_axis={level}'
        lines.append(f"      call oracle_write('{key}','{phase}','{label}',{expression},r3_tap_event{metadata})\n")
        lines.extend('      endif\n' for _ in guards)
    return ''.join(lines)


def generate(root, out):
    path = root / SOURCE
    source = read_pinned_source(root, path)
    active = active_source(source)
    module_variables = {}
    for statement in statements(active):
        if re.fullmatch('contains', statement.text, re.I):
            break
        module_variables.update({v.name: v for v in declarations(statement.text)})
    inserts = {}
    manifest = []
    selected = {r.name: r for r in routines(path, root)}
    for name in ROUTINES:
        routine = selected[name]
        opaque = NON_NUMERIC_ARGUMENTS.get(name, set())
        unsupported = [a for a in routine.args if a not in opaque and (a not in routine.variables or not routine.variables[a].supported)]
        if unsupported:
            raise ValueError(f'{name}: unsupported complete arguments {unsupported}')
        executable = next(st for st in routine.body if is_executable(st, routine.variables))
        if any(re.match(r'return\b', st.text, re.I) for st in routine.body):
            raise ValueError(f'{name}: early return requires an exit snapshot')
        signature = next(st for st in statements(active) if st.start == routine.start)
        inserts.setdefault(signature.end, []).append('      use mpas_oracle_dump, only: oracle_begin,oracle_enabled,oracle_write\n')
        body_words = set(re.findall(r'\b\w+\b', '\n'.join(st.text for st in routine.body).lower()))
        hidden = {n: v for n, v in module_variables.items()
                  if n in body_words and n not in routine.variables
                  and v.supported and 'parameter' not in v.attrs}
        fields = [('argument.' + a, a, routine.variables[a]) for a in routine.args if a not in opaque]
        fields += [('module.' + n, n, v) for n, v in sorted(hidden.items())]
        # Array declarations use imported dimensions, which can differ from the
        # *_dummy arguments. Preserve the actual native module dimensions too.
        dimensions = []
        if any(re.match(r'use\s+mpas_atm_dimensions\b', st.text, re.I) for st in routine.body):
            dimensions = [n for n in ('nvertlevels', 'maxedges', 'maxedges2', 'num_scalars')
                          if n in body_words and n not in routine.variables]
            fields += [('dimension.' + n, n, Variable(n, 'integer', '', [], 'in', False, False))
                       for n in dimensions]
        key = 'dycore__' + name
        begin = ('      integer :: r3_tap_event\n'
                 f"      call oracle_begin('{key}',r3_tap_event)\n"
                 f"      if (oracle_enabled('{key}')) then\n" + snapshot(key, 'pre', fields) + '      endif\n')
        finish = f"      if (oracle_enabled('{key}')) then\n" + snapshot(key, 'post', fields) + '      endif\n'
        inserts.setdefault(executable.start, []).append(begin)
        inserts.setdefault(routine.end - 1, []).append(finish)
        manifest.append(dict(key=key, source=SOURCE, first_line=routine.start + 1,
                             last_line=routine.end, arguments=routine.args,
                             private_module_fields=sorted(hidden),
                             dimensions=dimensions,
                             non_numeric_arguments=sorted(opaque),
                             fields=[dict(label=label, expression=expr, declaration=asdict(var),
                                          cell_axis=axes(var)[0], level_axis=axes(var)[1],
                                          read_condition=CONDITIONAL_READS.get((key, label)))
                                     for label, expr, var in fields]))
    # Establish original native step/rank/global cell IDs at the driver boundary.
    timestep = selected['atm_timestep']
    signature = next(st for st in statements(active) if st.start == timestep.start)
    inserts.setdefault(signature.end, []).append('      use mpas_oracle_dump, only: oracle_set_context\n')
    executable = next(st for st in timestep.body if st.text.lower().startswith('clock =>'))
    inserts.setdefault(executable.start, []).append(
        "      type(mpas_pool_type), pointer :: r3_tap_mesh\n"
        "      integer, pointer :: r3_tap_ids(:)\n"
        "      call mpas_pool_get_subpool(domain%blocklist%structs,'mesh',r3_tap_mesh)\n"
        "      call mpas_pool_get_array(r3_tap_mesh,'indexToCellID',r3_tap_ids)\n"
        "      call oracle_set_context(r3_tap_ids,itimestep,domain%dminfo%my_proc_id)\n")
    result = ''.join(''.join(inserts.get(i, [])) + line
                     for i, line in enumerate(source.splitlines(keepends=True)))
    target = out / SOURCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result)
    return dict(schema='rrfs-dycore-native-tap-overlay-v1',
                fork_pin='1d6e370e7b43a332074809f170532a216c2212fb',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                overlay_sha256=hashlib.sha256(result.encode()).hexdigest(),
                native_arithmetic_edits=[], boundaries=manifest,
                acceptance='diagnostic pending native observer passivity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.source.resolve(), args.out.resolve())
    (args.out / 'dycore-tap-contract.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
