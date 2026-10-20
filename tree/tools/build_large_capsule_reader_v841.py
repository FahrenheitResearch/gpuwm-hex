#!/usr/bin/env python3
"""Curate and build the bounded capsule reader in an isolated CPU workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

PRIVATE_CRATES = ('rw-mpas', 'rw-store', 'rustwx-core', 'static-fields')


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(destination: Path, patch: Path, cargo: str, *, apply_patch: bool) -> int:
    receipts = destination / 'receipts'
    environment = os.environ.copy()
    for name in ('GPUWM_BRIDGE_SOURCE_REV', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS'):
        environment.pop(name, None)
    cargo_path = Path(shutil.which(cargo) or cargo).absolute()
    if not cargo_path.is_file():
        raise FileNotFoundError(cargo_path)
    environment.update(CUDA_VISIBLE_DEVICES='', GPUWM_NO_LOCAL_GPU='1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       RAYON_NUM_THREADS='2', CARGO_BUILD_JOBS='2',
                       CARGO_HOME=str(destination / 'cargo-home'),
                       CARGO_TARGET_DIR=str(destination / 'target'), TMPDIR=str(destination / 'tmp'))
    environment['PATH'] = str(cargo_path.parent) + os.pathsep + environment.get('PATH', '')
    Path(environment['TMPDIR']).mkdir(exist_ok=True)
    commands = []
    if apply_patch:
        commands.append(['patch', '--forward', '--batch', '-p1', '--input', str(patch)])
    commands.extend([
        [cargo, 'test', '--offline', '--release', '--jobs', '2', '--package', 'rw-mpas',
         '--lib', 'bounded_capsule_read', '--', '--nocapture'],
        [cargo, 'build', '--offline', '--release', '--jobs', '2', '--package', 'rw-mpas', '--bin', 'rw_mpas_init'],
    ])
    (receipts / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (receipts / 'build.log').open('a') as log:
        for command in commands:
            subprocess.run(command, cwd=destination, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    engine = destination / 'target/release/rw_mpas_init'
    receipt = {'engine': str(engine), 'engine_sha256': sha256(engine),
               'patch_sha256': sha256(patch), 'built_lock_sha256': sha256(destination / 'Cargo.lock'),
               'patched_capsule_sha256': sha256(destination / 'crates/rw-mpas/src/init/capsule.rs')}
    (receipts / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust-workspace', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--patch', required=True, type=Path)
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('--resume', action='store_true', help='resume commands in this tool\'s existing isolated copy')
    arguments = parser.parse_args()
    source = arguments.rust_workspace.resolve(strict=True)
    destination = arguments.output_dir.resolve()
    patch = arguments.patch.resolve(strict=True)
    if destination == source or source in destination.parents or destination in source.parents:
        parser.error('the build destination must be separate from the source workspace')
    if destination.exists():
        provenance = destination / 'receipts/source-manifest.json'
        if not arguments.resume or not provenance.is_file():
            parser.error('the build destination must be new, or a prior owned build must be resumed explicitly')
        if json.loads(provenance.read_text()).get('patch_sha256') != sha256(patch):
            parser.error('the resumed build has a different patch digest')
        return build(destination, patch, arguments.cargo, apply_patch=False)

    lock = tomllib.loads((source / 'Cargo.lock').read_text())
    packages = lock['package']
    by_name: dict[str, list[dict]] = {}
    for package in packages:
        by_name.setdefault(package['name'], []).append(package)
    pending = list(PRIVATE_CRATES) + ['netcrust', 'hdf5-reader']
    selected: dict[tuple[str, str], dict] = {}
    while pending:
        dependency = pending.pop()
        words = dependency.split()
        candidates = by_name.get(words[0], [])
        if len(words) > 1:
            candidates = [package for package in candidates if package['version'] == words[1]]
        if not candidates:
            parser.error(f'lockfile dependency is absent: {dependency}')
        for package in candidates:
            key = (package['name'], package['version'])
            if key in selected:
                continue
            selected[key] = package
            pending.extend(package.get('dependencies', []))

    allowed_paths = set(PRIVATE_CRATES) | {'netcrust', 'hdf5-reader'}
    for package in selected.values():
        origin = package.get('source')
        if origin is None and package['name'] not in allowed_paths:
            parser.error(f'uncurated private path dependency: {package["name"]}')
        if origin is not None and origin != 'registry+https://github.com/rust-lang/crates.io-index':
            parser.error(f'dependency is not the public crates.io registry: {package["name"]}')

    registry_index: dict[tuple[str, str], Path] = {}
    for folder in (source / 'vendor/crates-io').iterdir():
        manifest = folder / 'Cargo.toml'
        if folder.is_dir() and manifest.is_file():
            info = tomllib.loads(manifest.read_text())['package']
            registry_index[(info['name'], info['version'])] = folder
    for key, package in selected.items():
        if package.get('source') is not None and key not in registry_index:
            parser.error(f'public vendored dependency is absent: {key}')

    # Validate the entire allowlisted copy plan before creating the workspace.
    plan: dict[Path, Path] = {}
    for name in PRIVATE_CRATES:
        folder = source / 'crates' / name
        for item in (folder / 'src').rglob('*.rs'):
            plan[item] = item.relative_to(source)
        for filename in ('Cargo.toml', 'build.rs'):
            item = folder / filename
            if item.is_file():
                plan[item] = item.relative_to(source)
    registry = source / 'crates/rw-mpas/registry/driving-sources.json'
    plan[registry] = registry.relative_to(source)
    public_roots = [source / 'vendor/netcrust']
    public_roots.extend(registry_index[key] for key, package in selected.items() if package.get('source') is not None)
    for folder in public_roots:
        for current, dirs, filenames in os.walk(folder, followlinks=False):
            dirs[:] = [name for name in dirs if name != '.git' and
                       (name != 'target' or not (Path(current) / 'Cargo.toml').is_file())]
            for name in filenames:
                if name == '.git' or name == '.env' or name.startswith('.env.') or name in ('credentials.toml', 'auth.json'):
                    continue
                item = Path(current) / name
                plan[item] = item.relative_to(source)
    for item in plan:
        if item.is_symlink() or not item.is_file():
            parser.error(f'copy plan contains a link or non-file: {item.relative_to(source)}')
        try:
            item.resolve(strict=True).relative_to(source)
        except ValueError:
            parser.error('copy plan leaves the source workspace')

    destination.mkdir(parents=True)
    receipts = destination / 'receipts'
    receipts.mkdir()
    manifest = []
    for item, relative in sorted(plan.items(), key=lambda pair: str(pair[1])):
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, output)
        manifest.append({'path': relative.as_posix(), 'bytes': item.stat().st_size, 'sha256': sha256(item)})
    workspace_text = (source / 'Cargo.toml').read_text()
    members = '\n'.join(f'    "crates/{name}",' for name in PRIVATE_CRATES)
    workspace_text, substitutions = re.subn(r'(?ms)^members\s*=\s*\[.*?^\]', 'members = [\n' + members + '\n]', workspace_text, count=1)
    if substitutions != 1:
        raise RuntimeError('source workspace member table was not found')
    (destination / 'Cargo.toml').write_text(workspace_text)
    shutil.copyfile(source / 'Cargo.lock', destination / 'Cargo.lock')
    config = destination / '.cargo/config.toml'
    config.parent.mkdir()
    config.write_text('[source.crates-io]\nreplace-with = "vendored-sources"\n[source.vendored-sources]\ndirectory = ' + json.dumps(str(destination / 'vendor/crates-io')) + '\n')
    (receipts / 'source-manifest.json').write_text(json.dumps({
        'source_workspace': str(source), 'private_crates': list(PRIVATE_CRATES),
        'source_workspace_manifest_sha256': sha256(source / 'Cargo.toml'),
        'source_lock_sha256': sha256(source / 'Cargo.lock'),
        'patch_sha256': sha256(patch), 'files': manifest,
    }, indent=2) + '\n')
    return build(destination, patch, arguments.cargo, apply_patch=True)


if __name__ == '__main__':
    raise SystemExit(main())
