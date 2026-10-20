"""MYNN input-only transfer ownership, pinned native declarations and raw words."""
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.bl_mynn_v842 import (
    _MYNN_PREPARED, _MYNN_PREPARED_INPUTS, _discard_native_inputs,
    MYNN_ARGUMENTS, WRAPPER_TO_MMM,
)
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.mpasphys.native_backend import NativePhysics


def test_omitted_prepared_fields_are_native_intent_in():
    root = os.environ.get('MPAS_REF')
    if not root:
        pytest.skip('MPAS_REF required for pinned native intent audit')
    source = (Path(root) / 'src/core_atmosphere/physics/physics_wrf/module_bl_mynn.F').read_text().lower()
    source = source.split('subroutine mynn_bl_driver', 1)[1]
    source = source.split('!--- inout arguments:', 1)[0]
    source = re.sub(r'!.*', '', source)
    source = re.sub(r'&\s*\n\s*', '', source)
    inputs = set()
    for line in source.splitlines():
        if 'intent(in)' in line and '::' in line:
            inputs.update(re.findall(r'\b\w+\b', line.split('::', 1)[1]))
    assert set(_MYNN_PREPARED_INPUTS) == set(_MYNN_PREPARED) & inputs
    assert len(_MYNN_PREPARED_INPUTS) == 18
    outputs = {WRAPPER_TO_MMM.get(n, n) for n, s in MYNN_ARGUMENTS.items()
               if s['dimensions'] and s['intent'] in ('out', 'inout')}
    assert not outputs.intersection(_MYNN_PREPARED_INPUTS)


@pytest.mark.gpu
@pytest.mark.parametrize('count,levels,padded', [(4, 1, False), (7, 3, True)])
def test_actual_native_input_retirement_preserves_words_and_writable_outputs(count, levels, padded):
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires explicit GPU grant and native library')
    cp = pytest.importorskip('cupy')
    words = np.array([0x80000000, 1, 0x7fc12345, 0x7f812345, 0xff800000, 0x3f800000], dtype=np.uint32)
    with NativePhysics(os.environ['MPAS_NATIVE_LIBRARY'], test_only=True) as native:
        owner = NativeDevicePoolViews(SimpleNamespace(native=native, layout=SimpleNamespace(
            dimensions={'nCellsSolve': count})), cp)
        initial = {}
        for name in _MYNN_PREPARED_INPUTS:
            key = _MYNN_PREPARED[name]
            shape = (count + int(padded), 1) if name == 'ps' else (count + int(padded), levels, 1)
            value = np.array(np.resize(words, shape).view(np.float32), order='F')
            native.put_variable(key, value, lower=(1,) * len(shape))
            initial[key] = value.tobytes(order='F')
        output = np.full((count + int(padded), levels, 1), -123, dtype=np.float32, order='F')
        native.put_variable('rublten_p', output)
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
            owner.prepared('rublten_p')[...] = cp.float32(7)
            _discard_native_inputs(owner)
            assert set(owner.buffers) == {('prepared', 'rublten_p')}
            owner.commit()
            assert not owner.buffers
            for key, expected in initial.items():
                assert native.get_variable(key)[0].tobytes(order='F') == expected
            result = native.get_variable('rublten_p')[0]
            assert (result[:count] == 7).all()
            assert (result[count:] == -123).all()
