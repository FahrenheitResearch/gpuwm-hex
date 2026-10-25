"""Grell–Freitas input-only transfer ownership, pinned native declarations and raw words."""
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.gf_mpas_v842 import (
    _PREPARED_FIELDS, _discard_native_profile_inputs,
)
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.mpasphys.native_backend import NativePhysics


def test_omitted_profiles_are_native_intent_in():
    from hashlib import sha256
    from hexcore.mpasphys import gf_mpas_v842 as port
    root = os.environ.get('MPAS_REF')
    if not root:
        pytest.skip('MPAS_REF required for pinned native intent audit')
    base = Path(root) / 'src/core_atmosphere/physics'
    assert sha256((base / 'physics_wrf/module_cu_gf.mpas.F').read_bytes()).hexdigest() == port.SOURCE_SHA256
    driver_path = base / 'mpas_atmphys_driver_convection.F'
    assert sha256(driver_path.read_bytes()).hexdigest() == 'c2a95c625d000f55d6434fea00225d03ab23f9539da6c4db95b57b0f17909edb'
    source = (base / 'physics_wrf/module_cu_gf.mpas.F').read_text().lower()
    source = re.sub(r'!.*', '', source)
    source = re.sub(r'&\s*\n\s*', '', source)
    inputs = set()
    for line in source.splitlines():
        if 'intent(in)' in line and '::' in line:
            inputs.update(re.findall(r'\b\w+\b', line.split('::', 1)[1]))
    assert set(_PREPARED_FIELDS) <= inputs
    driver = driver_path.read_text().lower()
    for argument, profile in _PREPARED_FIELDS.items():
        assert re.search(r'\b' + argument + r'\s*=\s*' + profile + r'\b', driver)
    assert not set(_PREPARED_FIELDS).intersection(port.LEVEL_OUTPUTS)
    import ast, inspect
    # Gather word preservation and independent storage are exercised by
    # test_gf_pack_inputs.py, including padded native views.
    slot = ast.parse(inspect.getsource(port.run_slot)).body[0]
    assert ast.unparse(slot.body[-2]) == '_discard_native_profile_inputs(prepared)'
    assert ast.unparse(slot.body[-3]) == "pools['cu_act_flag'][:n].fill(False)"
    sentinel = object()
    ordinary = {"u_p": sentinel}
    _discard_native_profile_inputs(ordinary)
    assert ordinary == {"u_p": sentinel}  # non-native mapping contract


@pytest.mark.gpu
@pytest.mark.parametrize('count,levels,padded', [(4, 2, False), (7, 3, True)])
def test_actual_native_input_retirement_preserves_words_and_writable_outputs(count, levels, padded):
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires explicit GPU grant and native library')
    cp = pytest.importorskip('cupy')
    words = np.array([0x80000000, 1, 0x7fc12345, 0x7f812345, 0xff800000, 0x3f800000], dtype=np.uint32)
    with NativePhysics(os.environ['MPAS_NATIVE_LIBRARY'], test_only=True) as native:
        owner = NativeDevicePoolViews(SimpleNamespace(native=native, layout=SimpleNamespace(
            dimensions={'nCellsSolve': count}, fields=[dict(pool='tend_physics', name='rthcuten', pool_name='rthcuten', dimensions=['nVertLevels','nCells'])])), cp)
        initial = {}
        for key in _PREPARED_FIELDS.values():
            shape = (count + int(padded), levels + int(key == 'pres2_hyd_p'), 1)
            value = np.array(np.resize(words, shape).view(np.float32), order='F')
            native.put_variable(key, value, lower=(1,) * len(shape))
            initial[key] = value.tobytes(order='F')
        output = np.full((levels, count + int(padded)), -123, dtype=np.float32, order='F')
        native.put_field('tend_physics', 'rthcuten', output)
        flags = np.ones((count + int(padded), 1), dtype=np.bool_, order='F')
        native.put_variable('cu_act_flag', flags, lower=(1, 1))
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
            owner.field('tend_physics', 'rthcuten')[...] = cp.float32(7)
            owner.prepared('cu_act_flag')[...].fill(False)
            _discard_native_profile_inputs(owner['prepared'])
            assert set(owner.buffers) == {('tend_physics', 'rthcuten', 1), ('prepared', 'cu_act_flag')}
            owner.commit()
            assert not owner.buffers
            for key, expected in initial.items():
                assert native.get_variable(key)[0].tobytes(order='F') == expected
            result = native.get_field('tend_physics', 'rthcuten')
            assert (result[:, :count] == 7).all()
            assert (result[:, count:] == -123).all()
            flags_after = native.get_variable('cu_act_flag')[0]
            assert not flags_after[:count].any()
            assert flags_after[count:].all()
