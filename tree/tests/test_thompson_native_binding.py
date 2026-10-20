"""Actual native ABI binding witness; no forecast identity is inferred."""
import importlib.util
import copy
import os
from pathlib import Path

import pytest


def test_native_thompson_component_and_halo_ownership(tmp_path):
    root = Path(__file__).parents[1]
    library = os.environ.get('MPAS_NATIVE_LIBRARY')
    reference = os.environ.get('MPAS_V842_SOURCE')
    if not library or not reference:
        pytest.skip('requires the strict native library and pinned MPAS source')
    native = Path(os.environ.get('MPAS_NATIVE_BINDING_SOURCE',
                                root / 'src/hexcore/mpasphys/native_backend.py'))
    provider = Path(os.environ.get('MPAS_DEVICE_POOL_SOURCE',
                                  root / 'src/hexcore/mpasphys/device_pools_v842.py'))
    layout = Path(os.environ.get('MPAS_POOL_LAYOUT_SOURCE',
                                root / 'src/hexcore/mpasphys/pool_v842.py'))
    if not provider.is_file() or not layout.is_file():
        pytest.skip('f4 native device pool provider is not in this merged base')
    path = Path(__file__).with_name('thompson_v842_oracle') / 'native_binding_replay.py'
    spec = importlib.util.spec_from_file_location('thompson_native_binding_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.replay(Path(reference), native, provider, layout, Path(library),
                  tmp_path / 'native-binding.json')
