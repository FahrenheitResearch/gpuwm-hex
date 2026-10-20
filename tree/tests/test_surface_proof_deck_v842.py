"""Check the metadata needed to exercise surface services in real forecasts."""
import xml.etree.ElementTree as ET

import pytest

from hexcore.config import parse_fortran_namelist
from surface_proof_deck_v842 import prepare, surface_input_path


def source_deck(tmp_path, surface_count=1):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'namelist.atmosphere').write_text('''&nhyd_model
 config_start_time = '2001-01-01_00:00:00'
 config_dt = 120
 config_stop_time = '2001-01-02_00:00:00'
/
&physics
 config_physics_suite = 'mesoscale_reference'
 config_sst_update = false
/
''')
    root = ET.Element('streams')
    for name in ('output', 'restart', 'diagnostics'):
        stream = ET.SubElement(root, 'stream', name=name, type='input;output',
                               output_interval='3:00:00')
        ET.SubElement(stream, 'file', name='stream_list.atmosphere.' + name)
    for _ in range(surface_count):
        stream = ET.SubElement(root, 'stream', name='surface', type='input;output',
                               input_interval='none', output_interval='3:00:00')
        ET.SubElement(stream, 'file', name='stale_surface_list')
    (source / 'streams.atmosphere').write_text(ET.tostring(root, encoding='unicode'))
    return source


def build(source, destination):
    return prepare(source, destination, 'records.nc', duration='5_00:00:00',
                   interval='6:00:00', bucket_interval='6:00:00',
                   greeness_interval='24:00:00')


def test_prepared_surface_contract(tmp_path):
    source = source_deck(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / 'prepared'
    receipt = build(source, destination)
    groups = parse_fortran_namelist((destination / 'namelist.atmosphere').read_text())
    assert groups['nhyd_model']['config_start_time'] == '2001-01-01_00:00:00'
    assert groups['nhyd_model']['config_dt'] == 120
    assert groups['nhyd_model']['config_run_duration'] == '5_00:00:00'
    assert 'config_stop_time' not in groups['nhyd_model']
    assert groups['physics']['config_physics_suite'] == 'mesoscale_reference'
    assert all(groups['physics'][key] for key in (
        'config_sst_update', 'config_sstdiurn_update', 'config_deepsoiltemp_update',
        'config_frac_seaice', 'config_sfc_albedo'))
    root = ET.parse(destination / 'streams.atmosphere').getroot()
    surface = next(item for item in root if item.get('name') == 'surface')
    assert surface.attrib == dict(name='surface', type='input',
                                  filename_template='records.nc', input_interval='6:00:00',
                                  filename_interval='none')
    assert [(item.tag, item.get('name')) for item in surface] == [('var', 'sst'), ('var', 'xice')]
    for stream in root:
        if stream is not surface:
            assert stream.get('output_interval') == '6:00:00'
            assert stream[0].get('name') == 'stream_list.atmosphere.' + stream.get('name')
    assert not receipt['surface_file_exists']
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError, match='overwrite'):
        build(source, destination)


@pytest.mark.parametrize('count', [0, 2])
def test_surface_membership_is_unambiguous(tmp_path, count):
    with pytest.raises(ValueError, match='exactly one'):
        build(source_deck(tmp_path, count), tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_candidate_surface_binding_preserves_native_control_bytes(tmp_path):
    source = source_deck(tmp_path)
    build(source, tmp_path / 'prepared')
    streams = tmp_path / 'prepared/streams.atmosphere'
    original = streams.read_bytes()
    candidate = tmp_path / 'i5.nc'
    command = ['gpuwm-hex', 'run', '--stream-input', 'surface=' + str(candidate)]
    assert surface_input_path(command, streams) == candidate
    assert surface_input_path(['gpuwm-hex', 'run'], streams) == streams.parent / 'records.nc'
    assert streams.read_bytes() == original
    with pytest.raises(ValueError, match='unique'):
        surface_input_path(command + ['--stream-input', 'surface=' + str(candidate)], streams)
    with pytest.raises(ValueError, match='absolute'):
        surface_input_path(['--stream-input', 'surface=i5.nc'], streams)
    with pytest.raises(ValueError, match='NAME=FILE'):
        surface_input_path(['--stream-input'], streams)
