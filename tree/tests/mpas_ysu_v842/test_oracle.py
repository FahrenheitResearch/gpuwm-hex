"""Strict word tests of the complete scheme and native wrapper.

Set YSU_ORACLE_DIR to build_oracle.py's output and YSU_FIXTURE_DIR to the
run_columns CSV folder. Native binaries are prerequisites, never built by pytest.
"""
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from hexcore.mpasphys import ysu_v842 as ysu
from oracle import Oracle, fixtures, branch_fixtures, OUTPUTS
from replay_witness import replay as replay_raw
from replay_wrapper import replay as replay_wrapper


@pytest.fixture(scope="module")
def paths():
    work, columns = os.environ.get("YSU_ORACLE_DIR"), os.environ.get("YSU_FIXTURE_DIR")
    if not work or not columns:
        pytest.skip("strict standalone MPAS oracle and column fixtures not supplied")
    return Path(work), Path(columns)


def test_complete_scheme_words(paths):
    result = replay_raw(*paths)
    assert result["compared_calls"] >= 864
    assert result["differing_words"] == 0, result["mismatches"][:5]


def test_native_wrapper_words(paths):
    result = replay_wrapper(*paths)
    assert result["compared_calls"] == 2016
    assert result["differing_words"] == 0, result["mismatches"][:5]


def test_rounding_branch_fixtures_have_finite_outputs(paths):
    oracle = Oracle(paths[0])
    checked = 0
    for tag, arrays, flags in branch_fixtures(paths[1]):
        if not any(name in tag for name in ("rounded-thermal", "zero-entrainment")):
            continue
        out, status = oracle.run(arrays, flags, native=True)
        assert status == 0
        for name in OUTPUTS:
            if out[name] is not None:
                assert np.all(np.isfinite(out[name])), (tag, name)
        checked += 1
    assert checked == 48


def test_missing_shared_device_math_fails_before_compile():
    with pytest.raises(ValueError, match="shared MPAS glibc 2.43"):
        ysu.cuda_source("")


def test_source_authority():
    root = os.environ.get("MPAS_REF")
    if not root:
        pytest.skip("campaign's pinned MPAS source unavailable")
    physics = Path(root)/"src/core_atmosphere/physics"
    for name, expected in (("physics_mmm/bl_ysu.F90", ysu.SOURCE_SHA256),
                            ("physics_wrf/module_bl_ysu.F", ysu.WRAPPER_SHA256),
                            ("mpas_atmphys_driver_pbl.F", ysu.DRIVER_SHA256)):
        assert hashlib.sha256((physics/name).read_bytes()).hexdigest() == expected


def test_compiled_artifact_rejects_stale_source_and_ftz(tmp_path):
    source = os.environ.get('YSU_COMPILED_ARTIFACT')
    if not source:
        pytest.skip('strict compiled CUDA artifact not supplied')
    artifact = Path(source)
    header = ysu.shared_device_math_header()
    metadata = json.loads(artifact.with_suffix('.json').read_text())
    architecture = metadata['compiler_options'][-1].split('sm_', 1)[1]
    expected = artifact.read_bytes()
    assert ysu.validated_cuda_binary(artifact, header, architecture) == expected
    changed = tmp_path / 'ysu.cubin'
    changed.write_bytes(expected)
    manifest = changed.with_suffix('.json')
    for key, value in (
        ('source_sha256', '0' * 64),
        ('compiler_options', metadata['compiler_options'] + ['--ftz=true']),
        ('device_math_sha256', '0' * 64),
    ):
        manifest.write_text(json.dumps({**metadata, key: value}))
        with pytest.raises(ValueError, match=key):
            ysu.validated_cuda_binary(changed, header, architecture)
    manifest.write_text(json.dumps(metadata))
    changed.write_bytes(expected + b'corrupt')
    with pytest.raises(ValueError, match='binary SHA256'):
        ysu.validated_cuda_binary(changed, header, architecture)

def complete_forecast_fixture():
    summary = {'model_execution': {'model_door_invoked': True, 'model_door_exit': 0,
                                    'process_count': 1},
               'progress': {'completed': True, 'steps_completed': 72, 'steps_requested': 72,
                            'physics': {'selectors': {'pbl': 'bl_ysu'},
                                        'backends': {'pbl': 'hex-cuda', 'surface_layer': 'native-cpu'}}},
               'actual_backend_calls': {'pbl/hex-cuda': 72, 'surface_layer/native-cpu': 72},
               'comparison_exit': 0}
    comparison = {'passed': True, 'tier': 'BITWISE', 'comparison': {'fields': [
        {'status': 'compared', 'differing_words': 0}]}}
    return summary, comparison
