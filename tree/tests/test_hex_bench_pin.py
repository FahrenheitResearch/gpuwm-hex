import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'tools/hex_timeline/seal_forecast.py'
    spec = importlib.util.spec_from_file_location('hex_seal', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pin_binds_consumed_inputs_config_and_common_launcher(tmp_path):
    (tmp_path / 'input.nc').write_bytes(b'test input inventory, not a model')
    (tmp_path / 'unused.txt').write_text('not consumed by this forecast')
    metadata = {'case': 'fixture', 'input_paths': ['input.nc'],
                'configuration': {'forecast_argv': ['forecast'], 'model_seconds': 10}}
    receipt = module().seal(tmp_path, metadata)
    pin = json.loads((tmp_path / 'pin.json').read_text())
    assert {entry['path'] for entry in pin['files']} == {'input.nc', 'case.json', 'launch.py'}
    assert receipt['pin_sha256'] == module().sha(tmp_path / 'pin.json')
    assert receipt['prepared_content_sha256'] == pin['prepared_content_sha256']
    with pytest.raises(ValueError, match='immutable pin'):
        module().seal(tmp_path, metadata)


def test_pin_refuses_inputs_outside_the_pack(tmp_path):
    with pytest.raises(ValueError, match='contained regular'):
        module().inventory(tmp_path, ['../outside.nc'])
