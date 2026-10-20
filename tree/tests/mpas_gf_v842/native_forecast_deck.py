"""Seal an owned complete strict-native rerun for the real verify-mpas door.

Uses native manifest metadata and pinned files only; no weather data processing.
The owned publication is explicitly distinct from f0's original publications.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(template, run, receipt, suite):
    run = run.resolve()
    m = json.loads((run/'MANIFEST.json').read_text())
    log = run/'log.atmosphere.0000.out'
    assert 'Finished running the atmosphere core' in log.read_text()
    assert m['ranks'] == 1 and m['outputs']
    if not isinstance(m.get('runtime'), dict) or not {'start', 'dt'} <= m['runtime'].keys():
        raise ValueError('Owned native manifest must pin its actual runtime clock before sealing a deck')
    files = dict(m['outputs'])
    assert all((run/name).is_file() for name in files)
    for name in ('MANIFEST.json', 'namelist.atmosphere', 'streams.atmosphere', log.name):
        files[name] = sha(run/name)
    publication = run/'OWNED-PUBLICATION.json'
    publication.write_text(json.dumps(dict(schema='mpas.native-cache-publication/v1',
        source='Owned completed rerun of f0 published strict binary and original native deck',
        execution_host=socket.gethostname(), files=files), indent=2)+'\n')
    d = json.loads(template.read_text())
    d['id'] = 'p8-gf-owned-native-'+run.name
    d['pin_set'] = m['pin_set_hash']; d['suite'] = suite
    d['native'] = dict(path=str(publication), sha256=sha(publication))
    for name, file in [('namelist', 'namelist.atmosphere'), ('streams', 'streams.atmosphere')]:
        d['inputs'][name] = dict(path=str(run/file), sha256=sha(run/file))
    assert m['inputs'][Path(d['inputs']['init']['path']).name] == d['inputs']['init']['sha256']
    assert all(m['inputs'][name] == ref['sha256'] for name, ref in d['auxiliary_inputs'].items())
    receipt.write_text(json.dumps(d, indent=2)+'\n')
    return d


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--suite', required=True)
    a = p.parse_args(); prepare(a.template, a.run, a.receipt, a.suite)
