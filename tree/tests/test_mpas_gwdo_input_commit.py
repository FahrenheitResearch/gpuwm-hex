"""YSU GWDO input-only transfer ownership, pinned native declarations and raw words."""
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.ysu_gwdo_v842 import (
    _NATIVE_PROFILE_INPUTS, _discard_native_profile_inputs,
)
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.mpasphys.native_backend import NativePhysics


def test_omitted_profiles_are_native_intent_in():
    from hashlib import sha256
    from hexcore.mpasphys import ysu_gwdo_v842 as port
    root = os.environ.get('MPAS_REF')
    if not root:
        pytest.skip('MPAS_REF required for pinned native intent audit')
    base = Path(root) / 'src/core_atmosphere/physics'
    for name, expected in port.SOURCE_SHA256.items():
        assert sha256((base / name).read_bytes()).hexdigest() == expected
    source = (base / 'physics_wrf/module_bl_gwdo.F').read_text().lower()
    source = re.sub(r'!.*', '', source)
    source = re.sub(r'&\s*\n\s*', '', source)
    inputs = set()
    for line in source.splitlines():
        if 'intent(in)' in line and '::' in line:
            inputs.update(re.findall(r'\b\w+\b', line.split('::', 1)[1]))
    aliases = dict(zip(_NATIVE_PROFILE_INPUTS, ('u3d','v3d','t3d','qv3d','p3d','pi3d','z','p3di')))
    assert set(aliases.values()) <= inputs
    driver = (base / 'mpas_atmphys_driver_gwdo.F').read_text().lower()
    for profile, argument in aliases.items():
        assert re.search(r'\b' + argument + r'\s*=\s*' + profile + r'\b', driver)
    signature = port._KERNEL_BODY.split('void bl_ysu_gwdo_v842_f32(', 1)[1].split(') {', 1)[0]
    for name in ('u', 'v', 'temperature', 'qv', 'p_mass_dry_hyd', 'p_interface_dry_hyd', 'exner', 'zmid'):
        assert 'const float *' + name in signature
    assert not set(_NATIVE_PROFILE_INPUTS).intersection(port.LAYER_OUTPUTS)
    _discard_native_profile_inputs({})  # non-native mapping contract


@pytest.mark.gpu
@pytest.mark.parametrize('count,levels,padded', [(4, 2, False), (7, 3, True)])
def test_actual_native_input_retirement_preserves_words_and_writable_outputs(count, levels, padded):
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires explicit GPU grant and native library')
    cp = pytest.importorskip('cupy')
    words = np.array([0x80000000, 1, 0x7fc12345, 0x7f812345, 0xff800000, 0x3f800000], dtype=np.uint32)
    with NativePhysics(os.environ['MPAS_NATIVE_LIBRARY'], test_only=True) as native:
        owner = NativeDevicePoolViews(SimpleNamespace(native=native, layout=SimpleNamespace(
            dimensions={'nCellsSolve': count}, fields=[dict(pool='tend_physics', name='rublten', pool_name='rublten', dimensions=['nVertLevels','nCells'])])), cp)
        initial = {}
        for key in _NATIVE_PROFILE_INPUTS:
            shape = (count + int(padded), levels, 1)
            value = np.array(np.resize(words, shape).view(np.float32), order='F')
            native.put_variable(key, value, lower=(1,) * len(shape))
            initial[key] = value.tobytes(order='F')
        output = np.full((levels, count + int(padded)), -123, dtype=np.float32, order='F')
        native.put_field('tend_physics', 'rublten', output)
        # Establish the old generic copy-back result for the same IEEE words.
        for key in initial:
            owner.prepared(key)
        owner.commit()
        for key, expected in initial.items():
            assert native.get_variable(key)[0].tobytes(order='F') == expected
        # Actual getter, dispatch retirement, and generic commit across two calls.
        for _ in range(2):
            for key in initial:
                device = owner.prepared(key)
                assert np.isin(device.get().view(np.uint32), words).all()
            owner.field('tend_physics', 'rublten')[...] = cp.float32(7)
            _discard_native_profile_inputs(owner['prepared'])
            assert set(owner.buffers) == {('tend_physics', 'rublten', 1)}
            owner.commit()
            assert not owner.buffers
            for key, expected in initial.items():
                assert native.get_variable(key)[0].tobytes(order='F') == expected
            result = native.get_field('tend_physics', 'rublten')
            assert (result[:, :count] == 7).all()
            assert (result[:, count:] == -123).all()
