#!/usr/bin/env python3
"""Run one real-door visual forecast against an authenticated NSSL publication.

This records admission and execution, never infers numerical equality from an
exit status. Successful outputs still require the Rust comparison sheets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(a):
    checkout = Path(__file__).resolve().parents[3]
    status = lambda: subprocess.check_output(
        ['git', '-C', str(checkout), 'status', '--porcelain'], text=True)
    assert not status(), 'A clean committed checkout is required'
    tip = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha(a.publication/'PUBLICATION.json') == a.publication_sha256
    publication = json.loads((a.publication/'PUBLICATION.json').read_text())
    manifest = json.loads((a.publication/'MANIFEST.json').read_text())
    assert sha(a.publication/'MANIFEST.json') == publication['files']['MANIFEST.json']
    settings = manifest['resolved_configuration']['resolved_settings']
    assert manifest['ranks'] == 1 and manifest['recipe'] == 'gnu-strict'
    assert settings['config_microp_scheme'] == 'mp_nssl2m'
    assert settings['config_run_duration'] in ('1_00:00:00', '24:00:00')
    assert sha(a.init) == manifest['inputs']['init.nc']
    controls = ['namelist.atmosphere', 'streams.atmosphere']
    controls += [name for name in publication['files'] if name.startswith('stream_list/')]
    assert any(name.startswith('stream_list/') for name in controls)
    for name in controls:
        assert sha(a.publication/name) == publication['files'][name]
    assert not a.out.resolve().is_relative_to(checkout)
    assert not a.scratch.resolve().is_relative_to(a.out.resolve())
    a.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'hexcore.cli', 'forecast', '--mesh', a.mesh,
        '--grid', str(a.grid.resolve()), '--init', str(a.init.resolve()),
        '--physics-backend', 'mpasphys', '--physics-mode', 'exact',
        '--physics-namelist', str((a.publication/'namelist.atmosphere').resolve()),
        '--streams', str((a.publication/'streams.atmosphere').resolve()),
        '--stream-list-directory', str((a.publication/'stream_list').resolve()),
        '--mpas-source', str(a.source.resolve()),
        '--native-physics-library', str(a.library.resolve()),
        '--native-pool-reader', str(a.pool_reader.resolve()),
        '--physics', "config_microp_backend='hex-cuda'",
        '--hours', '24', '--history-every-minutes', '60', '--out', str(a.scratch.resolve())]
    for table in a.tables:
        command += ['--native-physics-tables', str(table.resolve())]
    dependencies = [a.init, a.grid, a.library, a.pool_reader]
    if a.resources is not None:
        command += ['--physics-resources', str(a.resources.resolve())]
        dependencies.append(a.resources)
    if a.orographic_statistics is not None:
        command += ['--orographic-statistics', str(a.orographic_statistics.resolve())]
        dependencies.append(a.orographic_statistics)
    for binding in a.stream_input:
        name, separator, filename = binding.partition('=')
        assert separator and name and filename, 'stream input requires NAME=PATH'
        path = Path(filename).resolve()
        command += ['--stream-input', name+'='+str(path)]
        dependencies.append(path)
    receipt = dict(schema='r16-nssl2m.visual-forecast-attempt/v1', source_commit=tip,
        host=os.uname().nodename, process_count=1, command=command,
        native_publication=str(a.publication.resolve()),
        native_publication_sha256=sha(a.publication/'PUBLICATION.json'),
        native_reference_host=manifest['host'], cross_host_identity_claim=False,
        inputs={str(p.resolve()): sha(p) for p in dependencies},
        physics_mode='exact', publication_controls={name: publication['files'][name] for name in controls},
        requested_backend_overlay={'microphysics': 'hex-cuda'},
        requested_other_slots='native-cpu; resolved configuration required if the door admits',
        forecast_acceptance=False, differing_words=None, comparison_complete=False)
    (a.out/'request.json').write_text(json.dumps(receipt, indent=2)+'\n')
    with (a.out/'forecast.log').open('w') as log:
        completed = subprocess.run(command, cwd=checkout/'tree', stdout=log, stderr=subprocess.STDOUT)
    # Preserve the authentic output/diagnostic stream filenames. RRFS publishes
    # history.* and diag.*; an output.* glob would omit the actual run products.
    streams = ET.parse(a.publication/'streams.atmosphere').getroot()
    emitted = {
        stream.attrib['name']: sorted(str(path) for path in a.scratch.glob(
            re.sub(r'\$[YMDhms]', '*', stream.attrib['filename_template'])))
        for stream in streams if 'output' in stream.attrib.get('type', '').split(';')
    }
    receipt.update(exit_code=completed.returncode, source_dirty_after=status(),
        emitted_stream_files=emitted,
        emitted_output_files=sorted({p for paths in emitted.values() for p in paths}),
        status='EXECUTION_FAILED' if completed.returncode else 'COMPLETED_REQUIRES_VISUAL_REVIEW')
    assert not receipt['source_dirty_after']
    (a.out/'attempt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('source_commit', 'exit_code', 'status', 'differing_words')}))
    return completed.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('publication', 'grid', 'init', 'library', 'pool-reader', 'source', 'out', 'scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--publication-sha256', required=True)
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--orographic-statistics', type=Path)
    parser.add_argument('--stream-input', action='append', default=[])
    parser.add_argument('--tables', type=Path, action='append', default=[])
    raise SystemExit(main(parser.parse_args()))
