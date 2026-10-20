"""Observe a real forecast's driver PRE/POST words without replacing any inputs.

MPAS authority: mpas_atmphys_driver.F:202-373. Rust decodes and compares
f1 capture payloads. The wrappers below only copy fields for diagnostics;
every original native call executes once with unchanged arguments and return.
Hidden state completeness and whole-forecast acceptance are not inferred.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-manifest', type=Path, action='append', required=True)
    parser.add_argument('--comparisons-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('model_command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.model_command
    if command[:1] == ['--']:
        command = command[1:]
    if command[:2] != ['gpuwm-hex', 'forecast']:
        parser.error('the command must execute gpuwm-hex forecast')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                           'tools/native_physics_reference/backend'))
    from oracle_replay import Capture, transport
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.driver_v842 import NativeDriverHooks
    from hexcore import cli
    observed = {}
    args.comparisons_dir.mkdir(parents=True, exist_ok=False)
    for path in args.replay_manifest:
        manifest = json.loads(path.read_text())
        entry = manifest['capture_selection']['entry']
        inputs = [mapping for call in manifest['calls'] for mapping in call.get('inputs', [])]
        outputs = next(call['outputs'] for call in manifest['calls'] if call['entry'] == entry)
        root = Path(manifest['capture_selection']['capture_root'])
        observed[entry] = {'inputs': inputs, 'outputs': outputs,
                           'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'capture_seal_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    original_invoke, original_call = NativeDriverHooks.invoke, NativePhysics.call

    def invoke(hooks, operation, call):
        # MPAS source: mpas_atmphys_driver.F:202-373.
        previous = getattr(hooks.native, '_f3_observed_step', None)
        hooks.native._f3_observed_step = call.itimestep
        try:
            return original_invoke(hooks, operation, call)
        finally:
            hooks.native._f3_observed_step = previous

    def compare(native, entry, phase):
        # MPAS source: mpas_atmphys_driver.F:202-373; mpas_pool_routines.F:3436-4240.
        rows = []
        mappings = observed[entry]['inputs' if phase == 'pre' else 'outputs']
        rust = transport(native.path)
        for mapping in mappings:
            role, name = mapping['role'], mapping['name']
            row = {'role': role, 'pool': mapping.get('pool'), 'name': name,
                   'level': mapping.get('level', 1), 'capture': mapping['capture']}
            if role not in {'field', 'text_field', 'variable', 'table'}:
                row['status'] = 'metadata-only; not a scientific array comparison'
                rows.append(row)
                continue
            try:
                lower = None
                if role == 'field':
                    value = native.get_field(mapping['pool'], name, level=row['level'])
                    lower = (1,) * value.ndim
                elif role == 'text_field':
                    value = native.get_text_field_storage(mapping['pool'], name, level=row['level'])
                else:
                    value_lower = (native.get_variable(name) if role == 'variable'
                                   else native.get_table(name))
                    if value_lower is None:
                        raise ValueError('captured storage is unallocated in the actual forecast')
                    value, lower = value_lower
                with Capture(rust, Path(mapping['capture'])) as capture:
                    row.update(capture.compare(value, lower))
                row['status'] = 'compared'
            except (ValueError, KeyError, TypeError) as error:
                row['status'] = 'uncompared'
                row['error'] = str(error)
            rows.append(row)
        result = {'schema': 'native-real-door-boundary-observation-v1',
                  'driver': entry, 'phase': phase, 'step': args.step, 'process_count': 1,
                  'forecast_proof': False, 'boundary_state_complete': False,
                  'library_sha256': native.build['library_sha256'],
                  'fixture_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  **{k: observed[entry][k] for k in ('manifest_sha256', 'capture_seal_sha256')},
                  'comparisons': rows,
                  'compared_words': sum(row.get('words', 0) for row in rows),
                  'different_words': sum(row.get('different_words', 0) for row in rows),
                  'uncompared_arrays': sum(row['status'] == 'uncompared' for row in rows)}
        path = args.comparisons_dir / (entry + '-' + phase + '.json')
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('NATIVE_BOUNDARY_OBSERVATION', entry, phase, result['different_words'],
              'uncompared', result['uncompared_arrays'], flush=True)

    def call(native, name, **arguments):
        # MPAS source: mpas_atmphys_driver.F:202-373.
        selected = name in observed and getattr(native, '_f3_observed_step', None) == args.step
        if selected:
            compare(native, name, 'pre')
        result = original_call(native, name, **arguments)
        if selected:
            compare(native, name, 'post')
        return result

    NativeDriverHooks.invoke, NativePhysics.call = invoke, call
    try:
        return cli.main(command[1:])
    finally:
        NativeDriverHooks.invoke, NativePhysics.call = original_invoke, original_call


if __name__ == '__main__':
    raise SystemExit(main())
