"""Authenticate preserved native fixtures and the qualified host source.

This prepares the CPU battery's oracle directory for a supplied clean checkout.
It orchestrates immutable artifacts and performs no scientific data conversion.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numerical_identity(source):
    """Authenticate numerical source independently of native workspace glue."""
    tree = ast.parse(source)
    boundaries = {'bind_native_pools', '_bind_native_views', '_finish_native_workspace'}
    wrapped = any(isinstance(n, ast.FunctionDef) and n.name == '_run_bound_slot' for n in tree.body)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in boundaries or (wrapped and node.name == 'run_slot'):
                continue
            if node.name == '_run_bound_slot':
                node.name = 'run_slot'
        nodes.append(node)
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]),
                                   include_attributes=False).encode()).hexdigest()


def prepare(checkout, assets, destination):
    qualified = json.loads((assets / 'manifest.json').read_text())
    # A checkout contains tree/; an unpacked source distribution is that root.
    root = checkout / 'tree' if (checkout / 'tree').is_dir() else checkout
    source = root / 'src/hexcore/mpasphys/thompson_v842.py'
    assert digest(source) == qualified['production_sha256'], 'Unqualified Thompson host witness for this checkout'
    if 'numerical_source_sha256' in qualified:
        assert numerical_identity(source.read_text()) == qualified['numerical_source_sha256'], 'Altered numerical source'
    native = json.loads((assets / 'native-fixtures-manifest.json').read_text())
    inputs = []
    for root, rows in ((Path(native['root']), native['files']),
                       (assets, qualified['files'])):
        for row in rows:
            path = root / row['path']
            assert path.stat().st_size == row['bytes'], str(path)
            assert digest(path) == row['sha256'], str(path)
            inputs.append(path)
    destination.mkdir()
    for path in inputs:
        (destination / path.name).symlink_to(path.resolve())
    (destination / 'qualified-preparation.json').write_text(json.dumps(dict(
        schema='p3-qualified-CPU-oracle-preparation/v1',
        production_sha256=qualified['production_sha256'],
        qualified_manifest_sha256=digest(assets / 'manifest.json'),
        native_manifest_sha256=digest(assets / 'native-fixtures-manifest.json'),
        host_sha256=digest(assets / 'host_port.so'),
        scope='CPU battery fixtures; no CUDA or whole-forecast acceptance.'), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('assets', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare(args.checkout.resolve(), args.assets.resolve(), args.destination.resolve())


if __name__ == '__main__':
    main()
