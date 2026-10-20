"""Admit the published reviewed driver artifact without editing owner sources.

The installed owner artifact is copied byte for byte. The only added package
file is this lane's complete MYNN port. This is a diagnostic forecast until
the same driver correction is landed and the checkout passes its reproducer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(a):
    source = a.owner/'reviewed-foundation-installed'
    provenance = a.owner/'receipts/reviewed-foundation-wheel-provenance.json'
    proof = json.loads(provenance.read_text())
    source_proof = a.owner/'receipts/reviewed-foundation-source-provenance.json'
    if sha(source_proof) != proof['source_provenance_sha256']:
        raise ValueError('Owner source provenance changed')
    for name, value in proof['installed_python_files'].items():
        if sha(source/name) != value:
            raise ValueError('Published installed owner source changed: '+name)
    if sha(source/'bin/gpuwm-hex') != proof['console_sha256']:
        raise ValueError('Published console artifact changed')
    module = Path('hex/tree/src/hexcore/mpasphys/bl_mynn_v842.py').resolve()
    target = a.artifact/'hexcore/mpasphys/bl_mynn_v842.py'
    if (source/'hexcore/mpasphys/bl_mynn_v842.py').exists():
        raise ValueError('Owner already supplies MYNN; reconcile before replacing it')
    if a.reuse_artifact:
        if sha(target) != sha(module):
            raise ValueError('Reused artifact has a different MYNN module')
    else:
        shutil.copytree(source, a.artifact, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(module, target)
    for name, value in proof['installed_python_files'].items():
        if sha(a.artifact/name) != value:
            raise ValueError('Owned copy differs from published source: '+name)
    a.receipts.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(provenance, a.receipts/provenance.name)
    shutil.copyfile(source_proof, a.receipts/source_proof.name)
    deck = json.loads(a.deck.read_text())
    deck['native']['path'] = str(a.native/'PUBLICATION.json')
    if sha(a.native/'PUBLICATION.json') != deck['native']['sha256']:
        raise ValueError('Strict native publication differs from requested deck')
    # Retained native metadata is authenticated before queueing. Weather bytes
    # are authenticated by the Rust verifier on the comparison host.
    for key in ['namelist', 'streams']:
        if sha(Path(deck['inputs'][key]['path'])) != deck['inputs'][key]['sha256']:
            raise ValueError('Native input changed: '+key)
    deck['hex_command'][0] = str(a.artifact/'bin/gpuwm-hex')
    (a.receipts/'deck.json').write_text(json.dumps(deck, indent=2)+'\n')
    resources = Path(deck['hex_command'][deck['hex_command'].index('--physics-resources')+1])
    artifacts = {'scheme_module': target, 'native_provider': Path(deck['hex_command'][deck['hex_command'].index('--native-physics-library')+1]),
                 'native_pool_reader': Path(deck['hex_command'][deck['hex_command'].index('--native-pool-reader')+1]),
                 'resources': resources, 'console': a.artifact/'bin/gpuwm-hex'}
    result = dict(scope=__doc__, process_count=1, landing_proof=False,
                  own_commit=subprocess.check_output(['git', '-C', 'hex', 'rev-parse', 'HEAD'], text=True).strip(),
                  owner_provenance_sha256=sha(provenance),
                  owner_python_files_verified=len(proof['installed_python_files']),
                  artifacts={k: dict(path=str(p), sha256=sha(p)) for k, p in artifacts.items()},
                  native_publication_sha256=deck['native']['sha256'],
                  deck_sha256=sha(a.receipts/'deck.json'), excluded_words=[])
    (a.receipts/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['owner_python_files_verified', 'landing_proof', 'deck_sha256']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['owner', 'artifact', 'deck', 'native', 'receipts']:
        p.add_argument('--'+name, type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--reuse-artifact', action='store_true')
    run(p.parse_args())
