"""Run pristine native initialization to expose Registry diagnostic columns.

MPAS source: src/core_atmosphere/physics/mpas_atmphys_interface.F:235-561.
Python schedules the executable and files; native MPAS performs all decoding
and diagnostic arithmetic. This is a zero-duration fixture, not forecast proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


FIELDS = (
    'rho_zz', 'theta_m', 'pressure_base', 'pressure_p', 'rtheta_base',
    'rtheta_p', 'uReconstructZonal', 'uReconstructMeridional', 'exner',
    'w', 'surface_pressure',
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(*, init: Path, native: Path, namelist: Path, work: Path,
            receipt: Path) -> dict:
    init, native, namelist = init.resolve(), native.resolve(), namelist.resolve()
    work = work.absolute()
    if not init.is_file():
        raise FileNotFoundError(init)
    text = namelist.read_text()
    if not re.search(r"(?im)^\s*config_microp_scheme\s*=\s*'mp_kessler'", text):
        raise ValueError('this fixture requires an explicit mp_kessler namelist')
    text, changes = re.subn(r"(?im)^(\s*config_run_duration\s*=\s*)'[^']*'",
                           r"\g<1>'0_00:00:00'", text)
    if changes != 1:
        raise ValueError('namelist must declare exactly one config_run_duration')
    # Validate the deck first; fresh output prevents a failed run reusing old data.
    work.mkdir(parents=True, exist_ok=False)
    (work / 'namelist.atmosphere').write_text(text)
    for path in init.parent.iterdir():
        if path.is_file() and (path.suffix == '.nc' or '.graph.info' in path.name):
            (work / path.name).symlink_to(path)
    tables = {}
    for path in native.iterdir():
        if path.is_file() and (path.suffix in ('.TBL', '.DBL') or path.name in (
                'RRTMG_LW_DATA', 'RRTMG_SW_DATA', 'CCN_ACTIVATE_DATA')):
            (work / path.name).symlink_to(path)
            tables[path.name] = digest(path)
    streams = ET.Element('streams')
    ET.SubElement(streams, 'immutable_stream', name='input', type='input',
                  filename_template=init.name, input_interval='initial_only')
    output = ET.SubElement(streams, 'stream', name='output', type='output',
                           filename_template='columns.nc', output_interval='initial_only')
    for name in FIELDS:
        ET.SubElement(output, 'var', name=name)
    ET.SubElement(output, 'var_array', name='scalars')
    diagnostics = ET.SubElement(streams, 'stream', name='diagnostics', type='output',
                                filename_template='diagnostics.nc', output_interval='initial_only')
    ET.SubElement(diagnostics, 'var', name='surface_pressure')
    ET.indent(streams)
    ET.ElementTree(streams).write(work / 'streams.atmosphere', encoding='unicode')
    executable = native / 'atmosphere_model'
    with (work / 'process.log').open('wb') as log:
        result = subprocess.run([str(executable)], cwd=work, stdout=log,
                                stderr=subprocess.STDOUT, timeout=90)
    record = {
        'schema': 'mpas-native-initial-columns-v1', 'forecast_proof': False,
        'duration': '0_00:00:00', 'returncode': result.returncode,
        'executable': str(executable), 'executable_sha256': digest(executable),
        'init': str(init), 'init_sha256': digest(init),
        'source_namelist': str(namelist), 'source_namelist_sha256': digest(namelist),
        'tables': tables,
        'files': {p.name: digest(p) for p in work.iterdir()
                  if p.is_file() and not p.is_symlink()},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    result.check_returncode()
    if not (work / 'columns.nc').is_file():
        raise RuntimeError('native initialization produced no column output')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('init', 'native', 'namelist', 'work', 'receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
