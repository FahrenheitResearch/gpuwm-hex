"""Complete-column and MPAS conversion comparison with pristine Fortran.

Prepare THOMPSON_ORACLE_DIR with thompson_v842_oracle/Makefile through
the build queue before testing. These host witnesses do not establish GPU or G2.
"""
import os
from pathlib import Path
import sys
import pytest


@pytest.fixture(scope='module')
def oracle():
    path=os.environ.get('THOMPSON_ORACLE_DIR')
    if not path:pytest.skip('pristine standalone Thompson oracle not prepared')
    directory=Path(path).resolve()
    for name in ('liboracle.so','host_port.so','conversion_oracle.so','diagnostics_oracle.so','lookup_decode.so'):
        assert (directory/name).is_file(),f'missing pristine oracle artifact {name}'
    assert os.environ.get('GFORTRAN_CONVERT_UNIT')=='big_endian:71','MPAS big-endian lookup records must reach unit 71'
    sys.path.insert(0,str(Path(__file__).parent))
    return directory


def test_whole_column_and_radar_words(oracle,tmp_path):
    from thompson_v842_oracle.replay import replay
    result=replay(oracle,tmp_path/'column.json')
    assert all(c['differing_words']==0 for c in result['cases']),result


def test_mpas_driver_conversion_words(oracle,tmp_path):
    from thompson_v842_oracle.conversion_replay import replay
    result=replay(oracle,tmp_path/'conversion.json')
    assert all(c['differing_words']==0 for c in result),result


def test_precip_and_humidity_words(oracle,tmp_path):
    from thompson_v842_oracle.diagnostic_replay import replay
    result=replay(oracle,tmp_path/'diagnostic.json')
    assert all(c['differing_words']==0 for c in result),result


def test_optional_radii_and_negative_vapor_words(oracle,tmp_path):
    from thompson_v842_oracle.cleanup_replay import replay
    result=replay(oracle,tmp_path/'cleanup.json')
    assert all(c['differing_words']==0 for c in result),result


def test_active_helper_words(oracle,tmp_path):
    from thompson_v842_oracle.helper_replay import replay
    result=replay(oracle,tmp_path/'helpers.json')
    assert all(c['words']==0 and c['reference_nan']==0 and c['actual_nan']==0 for c in result),result


def test_effectrad_limits_and_vertical_masks(oracle,tmp_path):
    from thompson_v842_oracle.effectrad_replay import replay
    result=replay(oracle,tmp_path/'effectrad.json')
    assert all(c['words']==0 and c['reference_nan']==0 and c['actual_nan']==0 for c in result.values()),result


def test_empty_layers_and_sedimentation_loop_completion(oracle,tmp_path):
    from thompson_v842_oracle.replay import replay
    result=replay(oracle,tmp_path/'profiles.json',profiles=True)
    assert all(c['differing_words']==0 for c in result['cases']),result


def test_ice_zero_sedimentation_substeps(oracle,tmp_path):
    from thompson_v842_oracle.replay import replay,sedimentation_limits
    result=replay(oracle,tmp_path/'sedimentation-limits.json',column_inputs=sedimentation_limits())
    assert all(c['differing_words']==0 for c in result['cases']),result


def test_radius_native_nan_operand_order(oracle,tmp_path):
    from thompson_v842_oracle.effectrad_replay import replay
    result=replay(oracle,tmp_path/'radius-nan.json',nonfinite=True)
    assert all(c['words']==0 for c in result.values()),result


@pytest.mark.parametrize('changed', ('source_sha256', 'options', 'cubin_sha256', 'production_sha256'))
def test_offline_cuda_rejects_altered_certificate_before_device_use(tmp_path, monkeypatch, changed):
    import hashlib
    import json
    from hexcore.mpasphys import thompson_v842 as port
    from thompson_v842_oracle import certified_cuda
    source = 'extern "C" __global__ void exact_kernel() {}'
    key = certified_cuda.source_key(source)
    image = b'certificate-validation-fixture'
    metadata = dict(source_sha256=key, options=list(port.COMPILE_OPTIONS),
                    architecture='unused', cubin_sha256=hashlib.sha256(image).hexdigest(),
                    production_sha256=hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest())
    metadata[changed] = [] if changed == 'options' else 'altered'
    (tmp_path / (key + '.json')).write_text(json.dumps(metadata))
    (tmp_path / (key + '.cubin')).write_bytes(image)
    with monkeypatch.context() as context:
        context.setattr(port, 'compile_module', port.compile_module)
        load = certified_cuda.install_loader(tmp_path)
        with pytest.raises(AssertionError):
            load(source)


def test_oracle_preparation_preserves_native_metadata(tmp_path):
    import hashlib
    import importlib.util
    import json
    from hexcore.mpasphys import thompson_v842 as port
    path = Path(__file__).with_name('thompson_v842_oracle') / 'prepare_qualified_oracle.py'
    spec = importlib.util.spec_from_file_location('thompson_qualified_preparation', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native = tmp_path / 'native'
    assets = tmp_path / 'assets'
    native.mkdir()
    assets.mkdir()
    metadata = b'{"scope": "retained native input provenance"}\n'
    (native / 'preparation.json').write_bytes(metadata)
    host = b'authentication fixture; no numerical execution'
    (assets / 'host_port.so').write_bytes(host)
    (assets / 'manifest.json').write_text(json.dumps(dict(
        production_sha256=hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(), files=[dict(
            path='host_port.so', bytes=len(host), sha256=hashlib.sha256(host).hexdigest())])))
    (assets / 'native-fixtures-manifest.json').write_text(json.dumps(dict(root=str(native), files=[dict(
        path='preparation.json', bytes=len(metadata), sha256=hashlib.sha256(metadata).hexdigest())])))
    destination = tmp_path / 'qualified'
    helper.prepare(Path(__file__).resolve().parents[1], assets, destination)
    assert (native / 'preparation.json').read_bytes() == metadata
    assert (destination / 'preparation.json').read_bytes() == metadata
    assert (destination / 'qualified-preparation.json').is_file()
    (native / 'preparation.json').write_bytes(metadata.replace(b'retained', b'modified'))
    with pytest.raises(AssertionError):
        helper.prepare(Path(__file__).resolve().parents[1], assets, tmp_path / 'altered')
    assert not (tmp_path / 'altered').exists()
