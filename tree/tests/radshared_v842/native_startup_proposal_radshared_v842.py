"""Exercise the proposed integrator startup repair against real native pools.

The production seam is unchanged. Apply the exact proposal in memory, then
load the complete native initialization through its existing CPU loader.
This diagnostic is neither the real forecast door nor forecast acceptance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from hexcore import driver
from hexcore.mesh import Mesh
from hexcore.mpasphys.driver_v842 import resolve_namelist
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import NativePoolHydration, RegistryLayout


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def proposed_source(original, patch):
    """Apply the exact first file's unified hunks without editing the seam."""
    lines = original.splitlines(keepends=True)
    result, cursor = [], 0
    first = patch.split('--- a/tree/src/hexcore/mpasphys/runtime_v842.py')[0]
    for hunk in re.split(r'^@@ ', first, flags=re.M)[1:]:
        header, body = hunk.split('\n', 1)
        start = int(re.match(r'-(\d+)', header)[1]) - 1
        result.extend(lines[cursor:start])
        cursor = start
        for line in body.splitlines(keepends=True):
            if line.startswith((' ', '-')):
                assert lines[cursor] == line[1:], 'proposal no longer matches the checkout'
                cursor += 1
            if line.startswith((' ', '+')):
                result.append(line[1:])
    result.extend(lines[cursor:])
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'namelist', 'init', 'grid', 'reader', 'payload',
                 'library', 'patch', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = dict(purpose=__doc__, process_count=1, forecast_acceptance=False,
                  proposed_seam_only=True, passed=False)
    try:
        dimensions = {}
        for path in (args.grid, args.init):
            dimensions.update(json.loads(subprocess.check_output(
                [str(args.reader), '--inspect', str(path)], text=True)))
        text = args.namelist.read_text()
        layout = RegistryLayout(args.source, resolve_namelist(text), text, dimensions)
        with NativePhysics(args.library, test_only=True) as native:
            NativePoolHydration(native, layout, args.payload).populate()
            initialized = native.get_field('state', 'scalars')[..., :layout.dimensions['nCells']].copy()
        names = next(row['components'] for row in layout.fields
                     if row['pool'] == 'state' and row['name'] == 'scalars')
        scalar_initializations = dict(zip(names, initialized, strict=True))
        source_path = Path(driver.__file__)
        source = proposed_source(source_path.read_text(), args.patch.read_text())
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'load_mpas_initial_state')
        namespace = dict(vars(driver))
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     '<exact-integrator-startup-proposal>', 'exec'), namespace)
        mesh = Mesh.from_netcdf(args.grid)
        vertical = driver.load_mpas_vertical_grid(args.init, mesh,
            config_coef_3rd_order=layout.configs['config_coef_3rd_order'])
        state, reference, saved = namespace['load_mpas_initial_state'](
            args.init, mesh, vertical.vertical_grid, scalar_names=names,
            native_scalar_initializations=scalar_initializations,
            reference_path=args.init, terrain_metrics=vertical.terrain_metrics,
            return_saved_diagnostics=True, cold_start=True)
        request = json.loads((args.payload / 'request.json').read_text())
        row = next(r for r in request['fields'] if r['pool'] == 'state' and r['name'] == 'scalars')
        payload = json.loads((args.payload / 'manifest.json').read_text())
        origins = next(r['origins'] for r in payload['fields']
                       if r['pool'] == 'state' and r['name'] == 'scalars')
        np.testing.assert_array_equal(state.scalars.view('u4'), initialized.view('u4'))
        result.update(passed=True, scalar_components=names, scalar_origins=origins,
                      scalar_initial=row['initial'], differing_scalar_words=0,
                      compared_scalar_words=int(initialized.size),
                      rho_dtype=str(state.rho.dtype),
                      input_sha256=sha(args.init), proposal_sha256=sha(args.patch),
                      unchanged_driver_sha256=sha(source_path),
                      library_sha256=sha(args.library),
                      rust_payload_manifest_sha256=sha(args.payload / 'manifest.json'))
    except Exception as exc:
        result.update(failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
