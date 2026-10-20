"""Surface MYNN input-only transfer ownership, pinned native declarations and raw words."""
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.sf_mynn_v842 import (
    _NATIVE_PROFILE_INPUTS, _discard_native_profile_inputs, _PROFILE,
)
from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
from hexcore.mpasphys.native_backend import NativePhysics


def test_omitted_profiles_are_native_intent_in():
    import json
    from hexcore.mpasphys import sf_mynn_v842 as port
    root = os.environ.get('MPAS_REF')
    if not root:
        pytest.skip('MPAS_REF required for pinned native intent audit')
    source = (Path(root) / 'src/core_atmosphere/physics/physics_wrf/sf_mynn_pre.F').read_text().lower()
    source = source.split('subroutine sf_mynn_pre_run', 1)[1].split('!--- inout arguments:', 1)[0]
    source = re.sub(r'!.*', '', source)
    source = re.sub(r'&\s*\n\s*', '', source)
    inputs = set()
    for line in source.splitlines():
        if 'intent(in)' in line and '::' in line:
            inputs.update(re.findall(r'\b\w+\b', line.split('::', 1)[1]))
    assert {('dz3d' if n == 'dz8w' else n) for n in _PROFILE} <= inputs
    table = json.loads((Path(port.__file__).parents[1] / 'data/profiles/mpas-physics-suites.v1.json').read_text())
    def entries(value):
        if isinstance(value, dict):
            if value.get('module') == 'sf_mynn_v842':
                yield value
            for child in value.values():
                yield from entries(child)
        elif isinstance(value, list):
            for child in value:
                yield from entries(child)
    descriptors = list(entries(table))
    assert descriptors
    for descriptor in descriptors:
        assert set(descriptor['profiles']) == set(_PROFILE)
        assert set(descriptor['profiles'].values()) == set(_NATIVE_PROFILE_INPUTS)
    # All eight arrays enter the only profile consumer through const pointers.
    kernel = port.DRIVER_SOURCE.split('void sf_mynn_pre_columns(', 1)[1].split(') {', 1)[0]
    for name in ('u', 'v', 't', 'qv', 'qc', 'p', 'rho', 'dz'):
        assert 'const float *' + name in kernel
    assert not set(_NATIVE_PROFILE_INPUTS).intersection(n + '_sea' for n in port._SEA_HANDOFF)


@pytest.mark.gpu
@pytest.mark.parametrize('count,levels,padded', [(4, 2, False), (7, 3, True)])
def test_actual_native_input_retirement_preserves_words_and_writable_outputs(count, levels, padded):
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires explicit GPU grant and native library')
    cp = pytest.importorskip('cupy')
    words = np.array([0x80000000, 1, 0x7fc12345, 0x7f812345, 0xff800000, 0x3f800000], dtype=np.uint32)
    with NativePhysics(os.environ['MPAS_NATIVE_LIBRARY'], test_only=True) as native:
        owner = NativeDevicePoolViews(SimpleNamespace(native=native, layout=SimpleNamespace(
            dimensions={'nCellsSolve': count})), cp)
        initial = {}
        for key in _NATIVE_PROFILE_INPUTS:
            shape = (count + int(padded), levels, 1)
            value = np.array(np.resize(words, shape).view(np.float32), order='F')
            native.put_variable(key, value, lower=(1,) * len(shape))
            initial[key] = value.tobytes(order='F')
        output = np.full((count + int(padded), 1), -123, dtype=np.float32, order='F')
        native.put_variable('hfx_sea', output)
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
            owner.prepared('hfx_sea')[...] = cp.float32(7)
            _discard_native_profile_inputs(owner)
            assert set(owner.buffers) == {('prepared', 'hfx_sea')}
            owner.commit()
            assert not owner.buffers
            for key, expected in initial.items():
                assert native.get_variable(key)[0].tobytes(order='F') == expected
            result = native.get_variable('hfx_sea')[0]
            assert (result[:count] == 7).all()
            assert (result[count:] == -123).all()


@pytest.mark.gpu
@pytest.mark.parametrize('count,levels,padded', [(4, 2, False), (7, 55, True)])
def test_surface_two_level_upload_matches_full_preprocessing(count, levels, padded):
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires explicit GPU grant and native library')
    cp = pytest.importorskip('cupy')
    from hexcore.mpasphys.device_pools_v842 import _ProfileViews
    from hexcore.mpasphys.sf_mynn_v842 import _SurfaceProfileViews, preprocess, FIELDS
    words = np.array([0x80000000, 1, 0x7fc12345, 0x7f812345, 0xff800000, 0x3f800000], dtype=np.uint32)
    with NativePhysics(os.environ['MPAS_NATIVE_LIBRARY'], test_only=True) as native:
        owner = NativeDevicePoolViews(SimpleNamespace(native=native, layout=SimpleNamespace(
            dimensions={'nCellsSolve': count})), cp)
        aliases = dict(zip(_PROFILE, _NATIVE_PROFILE_INPUTS))
        for turn in range(2):
            expected = {}
            for name in aliases.values():
                shape = (count + int(padded), levels, 1)
                value = np.array(np.resize(np.roll(words, turn), shape).view(np.float32), order='F')
                native.put_variable(name, value)
                expected[name] = value.tobytes(order='F')
            full = _ProfileViews(owner, aliases)
            limited = _SurfaceProfileViews(full)
            for name in aliases:
                assert limited[name].shape == (2, count)
                assert limited[name] is limited[name]
                assert limited[name].get().tobytes() == full[name][:2].get().tobytes()
            a = cp.zeros((len(FIELDS), count), dtype=cp.float32)
            b = a.copy()
            qa = preprocess(a, full, itimestep=0)
            qb = preprocess(b, limited, itimestep=0)
            assert a.get().tobytes() == b.get().tobytes()
            assert qa.get().tobytes() == qb.get().tobytes()
            owner.buffers.clear()  # discard reference path's input-only buffers
            owner.commit()
            for name, original in expected.items():
                assert native.get_variable(name)[0].tobytes(order='F') == original
