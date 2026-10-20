"""Replay published native forecast fields through the actual serial adapter.

MPAS_NATIVE_STREAM_CAMPAIGN_DIR supplies a published all-stream
forecast directory. This tests real data transport, not a hex forecast claim.
Generated NetCDF output is deleted after each file's receipt is appended.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from hexcore.output import HistoryField
from hexcore.restart_v842 import read_stream, write_stream
from hexcore.streams_v842 import _all_packages, parse_streams


STREAMS = ('output', 'restart', 'diagnostics', 'surface', 'diag_ugwp')


@pytest.fixture(scope='session')
def campaign_deck(tmp_path_factory):
    configured = os.environ.get('MPAS_NATIVE_STREAM_CAMPAIGN_DIR')
    if not configured:
        pytest.skip('published native campaign artifacts not supplied')
    root = Path(configured)
    # Publication carries the unchanged deck; field-list assets are named and
    # hashed in its MANIFEST and supplied by the published native executable.
    assets = Path(os.environ['MPAS_NATIVE_STREAM_DEFAULTS'])
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    deck = tmp_path_factory.mktemp('published-native-deck')
    raw = (root / 'streams.atmosphere').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['inputs']['streams.atmosphere']
    (deck / 'streams.atmosphere').write_bytes(raw)
    for node in ET.fromstring(raw).iter('file'):
        name = node.attrib['name']
        data = (assets / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['inputs'][name]
        (deck / name).write_bytes(data)
    return parse_streams(deck / 'streams.atmosphere')


@pytest.fixture(scope='session')
def campaign_packages(campaign_deck):
    root = Path(os.environ['MPAS_NATIVE_STREAM_CAMPAIGN_DIR'])
    published = json.loads((root / 'PUBLICATION.json').read_text())
    raw = (root / 'log.atmosphere.0000.out').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == published['files']['log.atmosphere.0000.out']
    # Native physics/mpas_atmphys_packages.F:98-194 prints the actual active
    # physics packages; no field presence is used to infer package activation.
    packages = {name for name, value in re.findall(r'^\s*(\w+)\s*=\s*([TF])\s*$',
                                                  raw.decode(), re.MULTILINE)
                if value == 'T' and name in _all_packages()}
    # Registry.xml:488-490 surface-layer activation, using the resolved scheme
    # printed by mpas_atmphys_control.F:126-166 rather than selecting a suite.
    scheme = re.search(r'^\s*config_sfclayer_scheme\s*=\s*(\w+)\s*$', raw.decode(), re.MULTILINE)
    assert scheme
    if scheme[1] in {'sf_monin_obukhov', 'sf_monin_obukhov_rev', 'sf_mynn'}:
        packages.add('sfclayer')
    # These published decks enable UGWP diagnostics and embed invariant fields.
    assert campaign_deck['diag_ugwp'].options['output_interval'] != 'none'
    assert 'invariant' not in campaign_deck
    packages.update({'no_invariant_stream', 'ugwp_diags_stream'})
    return packages


def _cases():
    configured = os.environ.get('MPAS_NATIVE_STREAM_CAMPAIGN_DIR')
    if not configured:
        return [(None, None)]
    root = Path(configured)
    result = []
    for name in STREAMS:
        files = sorted(root.glob(name + '.*.nc'))
        if not files:
            raise ValueError(f'published native campaign lacks {name} files')
        result.extend((name, path) for path in files)
    return result


@pytest.mark.parametrize('name,path', _cases())
def test_published_native_campaign_stream_words(tmp_path, campaign_deck, campaign_packages, name, path):
    if path is None:
        pytest.skip('published native campaign artifacts not supplied')
    publication = path.parent / 'PUBLICATION.json'
    published = json.loads(publication.read_text())
    source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert source_digest == published['files'][path.name]
    packages = campaign_packages
    stream = campaign_deck[name]
    output = tmp_path / 'reencoded.nc'
    try:
        with read_stream(path, scratch=tmp_path) as native:
            assert set(native.variables) == set(stream.fields(packages))
            valid = native.variables['xtime'].text[0] if 'xtime' in native.variables else '2000-01-01_00:00:00'
            initial = native.variables['initial_time'].text[0] if 'initial_time' in native.variables else valid
            mesh = SimpleNamespace(arrays={}, dimensions=native.dimensions, attrs=native.global_attrs)
            fields = {n: HistoryField(v.values, v.dimensions, v.attrs) for n, v in native.variables.items()}
            write_stream(output, stream, mesh, fields, valid, initial_time=initial, active_packages=packages)
            with read_stream(output, scratch=tmp_path) as actual:
                assert native.dimensions == actual.dimensions
                assert native.unlimited == actual.unlimited
                assert native.global_attrs == actual.global_attrs
                assert set(native.variables) == set(actual.variables)
                elements = 0
                for field, reference in native.variables.items():
                    candidate = actual.variables[field]
                    assert reference.dimensions == candidate.dimensions, field
                    assert reference.attrs == candidate.attrs, field
                    assert reference.values.dtype == candidate.values.dtype, field
                    assert reference.values.shape == candidate.values.shape, field
                    assert reference.values.tobytes() == candidate.values.tobytes(), field
                    elements += reference.values.size
                receipt = {'stream': name, 'file': path.name, 'source_sha256': source_digest,
                           'native_process_count': json.loads((path.parent / 'MANIFEST.json').read_text())['ranks'],
                           'transport_process_count': 1,
                           'campaign': path.parent.name, 'active_packages': sorted(packages),
                           'publication_sha256': hashlib.sha256(publication.read_bytes()).hexdigest(),
                           'fields': len(native.variables), 'primitive_elements': elements,
                           'different_elements': 0, 'dimensions_attributes_equal': True,
                           'whole_hex_forecast_proven': False}
                destination = os.environ.get('MPAS_NATIVE_STREAM_CAMPAIGN_RECEIPT')
                if destination:
                    with Path(destination).open('a') as handle:
                        handle.write(json.dumps(receipt, sort_keys=True) + '\n')
    finally:
        output.unlink(missing_ok=True)


def test_published_native_campaign_restart_authority_layout(tmp_path, campaign_deck, campaign_packages):
    import resource
    from hexcore.restart_v842 import read_restart
    configured = os.environ.get('MPAS_NATIVE_STREAM_CAMPAIGN_DIR')
    if not configured:
        pytest.skip('published native campaign artifacts not supplied')
    files = sorted(Path(configured).glob('restart.*.nc'))
    assert files
    path = files[len(files) // 2]
    packages = campaign_packages
    output = tmp_path / 'authority-reencoded.nc'
    # Exercise three simultaneous, full, authentic native restart views under
    # a limit tighter than the lander's 1024. Never weaken the word comparison.
    descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, descriptor_limit[0]), descriptor_limit[1]))
    try:
        with read_stream(path, scratch=tmp_path) as reference:
            valid = reference.variables['xtime'].text[0]
            initial = reference.variables['initial_time'].text[0]
            with read_restart(path, valid_time=valid, scratch=tmp_path,
                              active_packages=packages, authority_layout=True) as authority:
                mesh = SimpleNamespace(arrays={}, dimensions=authority.dimensions, attrs=authority.global_attrs)
                fields = {n: HistoryField(v.values, v.dimensions, v.attrs) for n, v in authority.variables.items()}
                write_stream(output, campaign_deck['restart'], mesh, fields, valid,
                             initial_time=initial, active_packages=packages)
                with read_stream(output, scratch=tmp_path) as actual:
                    assert reference.dimensions == actual.dimensions
                    assert reference.global_attrs == actual.global_attrs
                    assert set(reference.variables) == set(actual.variables)
                    for name, variable in reference.variables.items():
                        restored = actual.variables[name]
                        assert variable.dimensions == restored.dimensions, name
                        assert variable.attrs == restored.attrs, name
                        assert variable.values.dtype == restored.values.dtype, name
                        assert variable.values.shape == restored.values.shape, name
                        assert variable.values.tobytes() == restored.values.tobytes(), name
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, descriptor_limit)
        output.unlink(missing_ok=True)
