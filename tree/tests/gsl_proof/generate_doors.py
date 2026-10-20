"""Replay all configured real Hex GSL doors on the CPU build/run host.

Rust owns generation, file I/O and stored-word comparison. This is orchestration.
MPAS driver interface: mpas_init_atm_gwd_gsl.F:49-203.
"""
import hashlib
import json
from pathlib import Path
import socket
import shutil
import subprocess
import sys


def run(command, log):
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def transport(cfg):
    # Canonical inputs stay on the source host. Only disposable CPU copies go elsewhere.
    if socket.gethostname() == cfg.get('source_host'):
        return None
    source_host = cfg.get('source_host')
    if not source_host:
        return None
    base = Path(cfg['transient_inputs'])
    assert not base.exists(), 'fresh disposable input copies required'
    base.mkdir(parents=True)
    copied = {}
    def copy(source, directory=False):
        if source in copied:
            return copied[source]
        target = base / hashlib.sha256(source.encode()).hexdigest() / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            target.mkdir()
        try:
            subprocess.run(['rsync', '-a', source_host + ':' + source + ('/' if directory else ''),
                            str(target) + ('/' if directory else '')], check=True)
        except BaseException:
            shutil.rmtree(base)
            raise
        copied[source] = str(target)
        return str(target)
    for row in cfg['front_doors']:
        for key in ('grid', 'native'):
            row[key] = copy(row[key])
        row['native_publication']['path'] = copy(row['native_publication']['path'])
        row['geog'] = copy(row['geog'], directory=True)
    return base


def replay(cfg):
    out = Path(cfg['receipts'])
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    assert len(cfg['front_doors']) == 6
    for i, row in enumerate(cfg['front_doors']):
        publication = row['native_publication']
        published = Path(publication['path'])
        assert hashlib.file_digest(published.open('rb'), 'sha256').hexdigest() == publication['sha256']
        native = Path(row['native'])
        native_sha = json.loads(published.read_text())['files'][native.name]
        assert hashlib.file_digest(native.open('rb'), 'sha256').hexdigest() == native_sha
        generated = out / (str(i) + '.nc')
        run([sys.executable, '-m', 'hexcore.cli', 'gsl-oro', '--engine', cfg['engine'],
             '--grid', row['grid'], '--geog', row['geog'], '--out', str(generated),
             '--receipt', str(out / (str(i) + '-generation.json'))], out / (str(i) + '-generate.log'))
        receipt = out / (str(i) + '-compare.json')
        run([cfg['engine'], '--compare', row['native'], str(generated), '--receipt', str(receipt)],
            out / (str(i) + '-compare.log'))
        comparison = json.loads(receipt.read_text())
        assert comparison['words'] > 0 and comparison['unequal_words'] == 0
        products = [generated]
        if 'metadata_output' in row:
            stamped = out / (str(i) + '-metadata.nc')
            run([cfg['metadata_engine'], str(generated), str(stamped), row['valid_time']],
                out / (str(i) + '-metadata.log'))
            run([cfg['engine'], '--compare', str(generated), str(stamped)],
                out / (str(i) + '-metadata-science.log'))
            products.append(stamped)
        rows.append({'index': i, 'comparison': comparison, 'native_publication': publication,
                     'native_sha256': native_sha,
                     'products': [{'name': path.name, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
                                  for path in products]})
    (out / 'six-front-doors.json').write_text(json.dumps({
        'passed': True, 'decks': rows, 'run_host': socket.gethostname(), 'process_count': 1,
        'engine_sha256': hashlib.file_digest(Path(cfg['engine']).open('rb'), 'sha256').hexdigest(),
    }, indent=2) + '\n')


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    transient = None
    try:
        transient = transport(cfg)
        replay(cfg)
    finally:
        if transient is not None:
            shutil.rmtree(transient)


if __name__ == '__main__':
    main()
