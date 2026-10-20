#!/usr/bin/env python3
"""Authenticate the exact native provider before running a forecast.

Receipt orchestration only; this neither loads nor modifies scientific state.
The ABI, strict build receipt, generator inputs and forecast command must all
refer to the same sealed library. Forecast identity is checked separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(indexpath, checkout):
    indexpath, checkout = Path(indexpath), Path(checkout)
    index = json.loads(indexpath.read_text())
    provider = index['native_provider']
    records = provider['artifacts']
    for name, row in records.items():
        if digest(row['path']) != row['sha256']:
            raise ValueError(f'native provider {name} digest differs')
    for relative, expected in provider['generator_inputs'].items():
        if digest(checkout / relative) != expected:
            raise ValueError(f'provider generator input differs: {relative}')
    build = json.loads(Path(records['build_receipt']['path']).read_text())
    recipe = json.loads(Path(records['compile_recipe']['path']).read_text())
    library = records['library']
    if build['status'] != 'built' or build['library_sha256'] != library['sha256']:
        raise ValueError('build receipt does not authenticate this library')
    if build['abi_sha256'] != records['abi']['sha256']:
        raise ValueError('build receipt does not authenticate this ABI')
    if build['compile_recipe'] != recipe or build['flags'] != recipe['flags']:
        raise ValueError('compile recipe differs from build receipt')
    if recipe['flags'] != provider['strict_flags']:
        raise ValueError('strict flags differ from sealed provider')
    for name, key in [('wrappers', 'wrappers_sha256'),
                      ('framework', 'framework_wrapper_sha256'),
                      ('tables', 'tables_wrapper_sha256'),
                      ('noahmp_io', 'noahmp_io_wrapper_sha256')]:
        if name in records and build[key] != records[name]['sha256']:
            raise ValueError(f'build receipt does not authenticate {name}')
    runtime = 'tree/tools/native_physics_reference/backend/runtime.f90'
    if runtime in provider['generator_inputs'] and (
            build['runtime_sha256'] != provider['generator_inputs'][runtime]):
        raise ValueError('build receipt does not authenticate current runtime')
    for entry in index['entries']:
        deckpath = indexpath.parent / entry['deck']
        if digest(deckpath) != entry['deck_sha256']:
            raise ValueError('reproduction deck digest differs')
        deck = json.loads(deckpath.read_text())
        support = deck['hex_support_inputs']['native-library']
        if support['sha256'] != library['sha256'] or (
                Path(support['path']).resolve() != Path(library['path']).resolve()):
            raise ValueError('forecast deck selects a different native provider')
        command = deck['hex_command']
        if command.count('--native-physics-library') != 1 or (
                Path(command[command.index('--native-physics-library') + 1]).resolve()
                != Path(library['path']).resolve()):
            raise ValueError('forecast command selects a different native provider')
    return dict(schema='mpas-native-provider-preflight-v1', passed=True,
                library_sha256=library['sha256'], artifacts=records,
                generator_inputs=provider['generator_inputs'],
                checkout=str(checkout.resolve()), forecast_proof=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.index, args.checkout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(schema='mpas-native-provider-preflight-v1', passed=False,
                      failure=str(error), forecast_proof=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
