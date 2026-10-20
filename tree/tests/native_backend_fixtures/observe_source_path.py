"""Compare unchanged real-door driver calls and stage sources with strict MPAS.

MPAS source: src/core_atmosphere/physics/mpas_atmphys_driver.F:209-384;
src/core_atmosphere/physics/mpas_atmphys_todynamics.F:98-498;
src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2125-2176.
Python transfers ABI storage only; Rust decodes and compares captured words.
This diagnostic does not change any model input, output, or arithmetic.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--receipt-directory', type=Path, required=True)
    parser.add_argument('--checkout-commit', required=True)
    parser.add_argument('model_command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.model_command
    if command[:1] == ['--']:
        command = command[1:]
    if command[:2] != ['gpuwm-hex', 'forecast']:
        parser.error('requires actual gpuwm-hex forecast')
    import numpy as np
    from hexcore import cli
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.driver_v842 import NativeDriverHooks
    from hexcore.mpasphys.runtime_v842 import ResidentNativeCoupling
    lib = ct.CDLL(str(args.artifact.resolve()))
    compare = lib.compare_abi_record_v2
    compare.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                        ct.c_size_t, ct.c_char_p, ct.c_void_p, ct.c_size_t,
                        ct.c_size_t, ct.c_void_p, ct.c_size_t]
    compare.restype = ct.c_ssize_t
    prefixes = {}
    for record in args.capture_root.iterdir():
        if not record.name.startswith('boundary__') or '.post.' not in record.name:
            continue
        key = record.name.split('.s')[0].removeprefix('boundary__')
        step = int(record.name.split('.s')[1].split('.')[0])
        prefixes[key, step] = record.name.split('.post.')[0]
    args.receipt_directory.mkdir(parents=True, exist_ok=False)
    summaries = []
    context = []
    old_invoke, old_call, old_stage = NativeDriverHooks.invoke, NativePhysics.call, ResidentNativeCoupling.stage

    def compare_buffer(native, path, buffer, dimensions, shape, axis):
        buffer = np.asfortranarray(buffer)
        if axis is not None:
            entity = dimensions[axis]
            idname = {'nCells': 'indexToCellID', 'nEdges': 'indexToEdgeID', 'nVertices': 'indexToVertexID'}[entity]
            idfile = path.parent / (path.name.split('.pre.')[0].split('.post.')[0] + '.post.mesh.' + idname + '.bin')
            if not idfile.is_file():
                idfile = next(args.capture_root.glob('boundary__driver_cloudiness.s0000000001.*.post.mesh.' + idname + '.bin'))
            ids = native.get_field('mesh', idname)
            count = ids.size - 1
        else:
            idfile, ids, count = None, None, 0
        dims = (ct.c_int32 * len(shape))(*shape)
        output = ct.create_string_buffer(8192)
        length = compare(str(path).encode(), buffer.ctypes.data, buffer.nbytes, dims, len(shape),
                         str(idfile).encode() if idfile else None, ids.ctypes.data if ids is not None else None,
                         count, axis or 0, output, len(output))
        if not 0 < length <= len(output):
            raise ValueError(f'comparison returned {length}: {path.name}')
        return json.loads(output.raw[:length])

    def observe(native, hooks, entry, step, phase, arguments):
        prefix = prefixes.get((entry, step))
        if prefix is None:
            return
        rows, absent = [], []
        # The layout lives on the bound session, not in generic ABI arguments.
        session = hooks.after_call.__self__
        layout = {(x['pool'], x['pool_name']): x for x in session.layout.fields}
        for (pool, name), (dtype, shape, levels) in native._fields.items():
            if dtype.kind != 'f' or dtype.itemsize != 4:
                continue
            dimensions = layout[pool, name]['dimensions']
            axis = next((i for i, d in enumerate(dimensions) if d in {'nCells', 'nEdges', 'nVertices'}), None)
            for level in range(1, levels + 1):
                path = args.capture_root / (prefix + '.' + phase + '.' + pool + '.' + name + (f'.t{level}' if levels > 1 else '') + '.bin')
                if not path.is_file():
                    absent.append({'pool':pool, 'field':name, 'level':level})
                    continue
                result = compare_buffer(native, path, native.get_field(pool, name, level=level), dimensions, shape, axis)
                result.update(pool=pool, field=name, level=level, reference=str(path))
                rows.append(result)
        # Explicit local physics source output arguments are not pool aliases.
        for name in ('tend_ru_physics', 'tend_rtheta_physics', 'tend_rho_physics'):
            if name not in arguments:
                continue
            path = args.capture_root / (prefix + '.' + phase + '.argument.' + name + '.bin')
            if not path.is_file():
                absent.append({'argument': name})
                continue
            entity = 'nEdges' if name == 'tend_ru_physics' else 'nCells'
            buffer = arguments[name]
            result = compare_buffer(native, path, buffer, ('nVertLevels', entity), buffer.shape, 1)
            result.update(argument=name, reference=str(path))
            rows.append(result)
        receipt = args.receipt_directory / f'{len(summaries):03d}-{step}-{entry}-{phase}.json'
        data = dict(schema='native-source-path-observation-v1', process_count=1, forecast_proof=False,
                    checkout_commit=args.checkout_commit, entry=entry, step=step, phase=phase,
                    library_sha256=native.build['library_sha256'],
                    comparator_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                    fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), fields=rows, unavailable=absent)
        receipt.write_text(json.dumps(data, indent=2) + '\n')
        summaries.append(dict(entry=entry, step=step, phase=phase, receipt=str(receipt), fields=len(rows),
                              words=sum(r['words'] for r in rows), different_words=sum(r['different_words'] for r in rows),
                              unavailable=len(absent)))
        (args.receipt_directory / 'summary.json').write_text(json.dumps(dict(process_count=1, forecast_proof=False, rows=summaries), indent=2) + '\n')
        print('SOURCE_PATH', step, entry, phase, len(rows), summaries[-1]['different_words'], flush=True)

    def invoke(hooks, operation, call):
        context.append((hooks, call))
        try:
            return old_invoke(hooks, operation, call)
        finally:
            context.pop()

    def native_call(native, entry, **arguments):
        if context:
            hooks, call = context[-1]
            observe(native, hooks, entry, call.itimestep, 'pre', arguments)
        result = old_call(native, entry, **arguments)
        if context:
            observe(native, hooks, entry, call.itimestep, 'post', arguments)
        return result

    def stage(coupling, **kwargs):
        first = coupling.sources is None
        result = old_stage(coupling, **kwargs)
        if first:
            step = coupling.session.driver._pending[0]
            prefix = prefixes.get(('physics_get_tend', step))
            rows = []
            for label, source, pool, name, entity in (
                ('rho',result.rho,None,'tend_rho_physics','nCells'),
                ('rho_u',result.rho_u,None,'tend_ru_physics','nEdges'),
                ('rho_theta',result.rho_theta,None,'tend_rtheta_physics','nCells'),
                ('scalars',result.scalars,'tend','scalars_tend','nCells')):
                path = args.capture_root / (prefix + '.post.' + (pool or 'argument') + '.' + name + '.bin')
                if pool:
                    buffer = coupling.session.native.get_field(pool, name)
                else:
                    buffer = coupling.session.sources[name].copy(order='F')
                buffer[..., :-1] = source.get()
                dimensions = ('num_scalars','nVertLevels',entity) if pool else ('nVertLevels',entity)
                row = compare_buffer(coupling.session.native, path, buffer, dimensions, buffer.shape, len(buffer.shape)-1)
                row.update(carrier=label, reference=str(path), resident_dummy='absent; comparison uses native dummy copy')
                rows.append(row)
            (args.receipt_directory / f'stage-sources-{step}.json').write_text(json.dumps(dict(process_count=1,step=step,forecast_proof=False,fields=rows),indent=2)+'\n')
            print('DEVICE_HELD_SOURCES',step,[(r['carrier'],r['different_words']) for r in rows],flush=True)
        return result

    NativeDriverHooks.invoke, NativePhysics.call, ResidentNativeCoupling.stage = invoke, native_call, stage
    try:
        return cli.main(command[1:])
    finally:
        NativeDriverHooks.invoke, NativePhysics.call, ResidentNativeCoupling.stage = old_invoke, old_call, old_stage


if __name__ == '__main__':
    raise SystemExit(main())
