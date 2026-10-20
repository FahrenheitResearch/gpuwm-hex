"""Seal all native Tiedtke calls; record I/O stays in f1's Rust transport."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(a):
    log = a.log.read_text()
    if ('Logging complete.' not in log or
            'Error messages =                     0' not in log or
            'Critical error messages =            0' not in log):
        raise ValueError('native forecast has no completion witness')
    if (a.bank/'manifest.json').exists():
        raise ValueError('refusing to replace a sealed native capture')
    inputs = json.loads(a.inputs.read_text())
    provenance = dict(schema='p6.native-forecast-inputs/v1', process_count=1,
        expected_calls=a.calls, inputs=inputs, executable_sha256=digest(a.executable),
        inventory_sha256=digest(a.inventory), native_log_sha256=digest(a.log),
        capture_selection='cu_tiedtke__driver_convection__L557; every step and entry',
        forecast_duration_hours=24)
    (a.bank/'INPUT-PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    tool = Path(__file__).parents[2]/'tools/column_oracle_replay.py'
    subprocess.run([sys.executable, str(tool), '--scheme', 'cu_tiedtke', '--seal', '--dump', str(a.bank),
        '--inventory', str(a.inventory), '--pins', str(a.pins), '--words', str(a.words),
        '--process-count', '1', '--recipe',
        'f1 frozen v20 strict GNU native executable; complete 24h original Tiedtke forecast; MPI1'], check=True)
    manifest = json.loads((a.bank/'manifest.json').read_text())
    captures = manifest['captures']
    expected = {'cu_act_flag', 'f_qc', 'f_qi', 'pratec', 'qfx', 'raincv',
        'rqccuten', 'rqicuten', 'rqvcuten', 'rthcuten', 'rucuten', 'rvcuten',
        'f_qc.present', 'f_qi.present', 'rqccuten.present', 'rqicuten.present',
        'rqvcuten.present', 'rthcuten.present', 'rucuten.present', 'rvcuten.present'}
    steps = set()
    for prefix, info in captures.items():
        if info['site'] != 'cu_tiedtke__driver_convection__L557':
            raise ValueError('unexpected native call site')
        steps.add(int(prefix.split('.s')[1].split('.')[0]))
        outputs = {name.split('.post.', 1)[1][:-4] for name in manifest['files']
                   if name.startswith(prefix+'.post.')}
        if outputs != expected:
            raise ValueError('incomplete native output set: '+prefix)
    if len(captures) != a.calls or steps != set(range(1, a.calls+1)):
        raise ValueError('native capture does not cover every forecast call')
    publication = dict(schema='p6.native-tap-publication/v1', captures=len(captures),
        process_count=1, manifest_sha256=digest(a.bank/'manifest.json'),
        input_provenance_sha256=digest(a.bank/'INPUT-PROVENANCE.json'),
        output_records=len(captures)*len(expected), dump_root_sha256=manifest['dump_root_sha256'])
    (a.bank/'PUBLICATION.json').write_text(json.dumps(publication, indent=2)+'\n')
    print(json.dumps(publication))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'inventory', 'pins', 'words', 'inputs', 'executable', 'log'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--calls', type=int, required=True)
    main(p.parse_args())
