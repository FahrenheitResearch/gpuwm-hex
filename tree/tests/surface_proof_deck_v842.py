"""Prepare surface acceptance decks from a published native suite deck.

Only namelist and stream metadata are changed. This does not generate inputs,
execute a forecast, or admit options through the production hex door.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from hexcore.config import parse_fortran_namelist


def _scalar(value):
    if isinstance(value, bool):
        return '.true.' if value else '.false.'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def surface_input_path(command, streams):
    """mpas_atm_core.F:779-806; Registry.xml:1161-1171.

    Resolve the candidate's explicit surface binding without changing the
    native-reference streams or authenticating a different native manifest.
    """
    streams = Path(streams)
    surface = [row for row in ET.parse(streams).getroot() if row.get('name') == 'surface']
    if len(surface) != 1:
        raise ValueError('exactly one surface stream is required')
    chosen = streams.parent / surface[0].get('filename_template', '')
    overridden = False
    for index, argument in enumerate(command):
        if argument != '--stream-input':
            continue
        if index + 1 == len(command) or '=' not in command[index + 1]:
            raise ValueError('stream input override requires NAME=FILE')
        name, path = command[index + 1].split('=', 1)
        if name != 'surface':
            continue
        if overridden or not Path(path).is_absolute():
            raise ValueError('surface override must be unique and use an absolute pinned path')
        chosen, overridden = Path(path), True
    return chosen.resolve()


def prepare(source, destination, surface_file, *, duration, interval,
            bucket_interval, greeness_interval):
    """Registry.xml:2261-2286,2341-2350; mpas_atmphys_manager.F:549-612.

    Surface records have the native SST/xice membership and one file with
    multiple times. All enabled output streams retain their membership but
    use the requested output cadence. Preserve suite selection and dynamics.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('destination exists: preparing would overwrite a proof deck')
    paths = [source / 'namelist.atmosphere', source / 'streams.atmosphere']
    groups = {name: dict(values) for name, values in
              parse_fortran_namelist(paths[0].read_text()).items()}
    model, physics = groups['nhyd_model'], groups['physics']
    model.pop('config_stop_time', None)
    model['config_run_duration'] = duration
    physics.update(config_sst_update=True, config_sstdiurn_update=True,
                   config_deepsoiltemp_update=True, config_frac_seaice=True,
                   config_sfc_albedo=True, config_bucket_update=bucket_interval,
                   config_greeness_update=greeness_interval)
    root = ET.fromstring(paths[1].read_text())
    surfaces = [item for item in root if item.get('name') == 'surface']
    if len(surfaces) != 1:
        raise ValueError('exactly one surface stream is required for the SST alarm')
    for stream in root:
        if 'output' in stream.get('type', '').split(';'):
            stream.set('output_interval', interval)
    surface = surfaces[0]
    surface.attrib.clear()
    surface.attrib.update(name='surface', type='input', filename_template=str(surface_file),
                          input_interval=interval, filename_interval='none')
    # Registry.xml:1161-1171: native surface stream contains exactly SST/xice.
    surface[:] = [ET.Element('var', name=name) for name in ('sst', 'xice')]
    namelist = ''.join('&' + name + '\n' + ''.join(
        f' {key} = {_scalar(value)}\n' for key, value in values.items()) + '/\n'
        for name, values in groups.items())
    destination.mkdir(parents=True)
    (destination / 'namelist.atmosphere').write_text(namelist)
    ET.indent(root)
    (destination / 'streams.atmosphere').write_text(ET.tostring(root, encoding='unicode') + '\n')
    manifest = dict(
        tier='prepared deck metadata; forecast and input validation not run',
        source=str(source), source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in paths},
        prepared_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (destination / name for name in
                                   ('namelist.atmosphere', 'streams.atmosphere'))},
        suite=physics['config_physics_suite'], surface_file=str(surface_file),
        surface_file_exists=Path(surface_file).is_file(), duration=duration,
        update_interval=interval, output_interval=interval,
        bucket_interval=bucket_interval, greeness_interval=greeness_interval,
        required_surface_fields=['sst', 'xice'])
    (destination / 'PREPARED.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'destination', 'surface-file', 'duration', 'interval',
                 'bucket-interval', 'greeness-interval'):
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args())
    prepare(args.pop('source'), args.pop('destination'), args.pop('surface_file'), **args)
