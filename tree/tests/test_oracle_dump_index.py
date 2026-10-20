"""Published index retains retirement and source interface provenance."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def test_index_counts_calls_and_refuses_inconsistent_publication(tmp_path):
    source = Path(__file__).parents[1] / 'tools/native_physics_reference/oracle/index_dumps.py'
    spec = importlib.util.spec_from_file_location('index_dumps', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'pins' / 'suite' / 'root'
    root.mkdir(parents=True)
    pub = dict(owner='f1-native-tap', suite='suite', pin_set_sha256='pins', dump_root_sha256='root')
    manifest = dict(pin_set_sha256='pins', dump_root_sha256='root',
                    scheme_aliases={'public': 'native'}, process_count=1,
                    hidden_state_complete=False, files={
                        'native__driver__L1.s0000000002.e0000000001.r000000.pre.a.bin': {'bytes': 8},
                        'native__driver__L1.s0000000002.e0000000001.r000000.post.a.bin': {'bytes': 8},
                        'native__driver__L1.s0000000002.e0000000002.r000000.post.a.bin': {'bytes': 8},
                        'state__physics_init.s0000000000.e0000000001.r000000.init.table.bin': {'bytes': 8},
                    })
    manifest.pop('dump_root_sha256')
    manifest['dump_root_sha256'] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pub['dump_root_sha256'] = manifest['dump_root_sha256']
    (root / 'PUBLICATION.json').write_text(json.dumps(pub))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'RETIREMENT.json').write_text('{"reason":"superseded"}')
    result = module.index(tmp_path, 'f1-native-tap')
    product = result['products'][0]
    assert product['retired'] and product['retirement_sha256']
    assert product['capture_process_count'] == 1
    assert product['scheme_aliases'] == {'public': 'native'}
    assert product['sites']['native__driver__L1'] == dict(
        records=3, bytes=24, steps=[2], ranks=[0], capture_count=2,
        phases={'pre': 1, 'post': 2})
    assert product['sites']['state__physics_init']['phases'] == {'init': 1}
    assert not result['forecast_proven'] and not product['hidden_state_complete']
    provenance=root/'INPUT-PROVENANCE.json'
    provenance.write_text(json.dumps(dict(dump_root_sha256=manifest['dump_root_sha256'],process_count=24)))
    with pytest.raises(ValueError,match='input provenance process count mismatch'):
        module.index(tmp_path,'f1-native-tap')
    provenance.write_text(json.dumps(dict(dump_root_sha256='wrong',process_count=1)))
    with pytest.raises(ValueError,match='input provenance root hash mismatch'):
        module.index(tmp_path,'f1-native-tap')
    provenance.unlink()
    pub['dump_root_sha256'] = 'another-root'
    (root / 'PUBLICATION.json').write_text(json.dumps(pub))
    with pytest.raises(ValueError, match='publication/manifest dump_root_sha256 mismatch'):
        module.index(tmp_path, 'f1-native-tap')
    pub['dump_root_sha256'] = manifest['dump_root_sha256']
    (root / 'PUBLICATION.json').write_text(json.dumps(pub))
    manifest['process_count'] = 24
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dump manifest root hash mismatch'):
        module.index(tmp_path, 'f1-native-tap')
