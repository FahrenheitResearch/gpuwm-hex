"""Reject a wheel input that would resolve an older ensemble renderer."""
import ast
import hashlib
import json
from pathlib import Path

import pytest


def test_prebuilt_requires_matching_ensemble_sibling(monkeypatch, tmp_path):
    root = Path(__file__).parents[1]
    module = ast.parse((root/'setup.py').read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == 's2s_renderer_bridge')
    namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, '_EXE': ''}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'setup.py', 'exec'), namespace)
    binary = tmp_path/'rw_wrfbatch'
    binary.write_bytes(b'test renderer')
    sibling = tmp_path/'rw_ensbatch'
    expected = b'matching ensemble renderer'
    sha = lambda data: hashlib.sha256(data).hexdigest()
    record = {'bridges': {'rw_wrfbatch': {
        'patches': {p.name: sha(p.read_bytes()) for p in (root/'docs/s2s/engine-patches').glob('*.patch')},
            'recipe_sha256': sha((root/'tools/build_s2s_diagnostics.sh').read_bytes()),
        'sha256': sha(binary.read_bytes()), 'ensemble_renderer': {'sha256': sha(expected)}}}}
    (tmp_path/'provenance.json').write_text(json.dumps(record))
    monkeypatch.setenv('GPUWM_HEX_ENGINE_BRIDGES', str(tmp_path))
    for payload in (None, b'stale engine sibling'):
        if payload is not None:
            sibling.write_bytes(payload)
        with pytest.raises(SystemExit, match='matching S2S renderer'):
            namespace['s2s_renderer_bridge'](root)
    sibling.write_bytes(expected)
    with pytest.raises(SystemExit, match='matching S2S diagnostics'):
        namespace['s2s_renderer_bridge'](root)
    reducer = tmp_path/'rw_s2s'
    reducer.write_bytes(b'test native reducer')
    record['bridges']['rw_wrfbatch']['diagnostics'] = {
        'rw_s2s_sha256':sha(reducer.read_bytes()),
        'reducer_lock_sha256':sha((root/'tools/rustwx/Cargo.lock').read_bytes()),
        'reducer_sources':{str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(
            (root/'tools/rustwx/crates/rw-s2s').rglob('*.rs'))},
        'renderer_source_sha256':sha((root/'tools/s2s_renderer/s2s_diagnostics.rs').read_bytes())}
    (tmp_path/'provenance.json').write_text(json.dumps(record))
    assert namespace['s2s_renderer_bridge'](root) == binary
    reducer.write_bytes(b'stale native reducer')
    with pytest.raises(SystemExit, match='matching S2S diagnostics'):
        namespace['s2s_renderer_bridge'](root)
