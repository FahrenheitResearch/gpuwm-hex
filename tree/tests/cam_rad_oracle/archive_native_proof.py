"""Losslessly retain owned native forecast bytes and restore one proof at a time.

Only files named by the unchanged native manifest are archived or retired.
An archive is accepted after a complete extraction and SHA256 comparison.
This transports bytes; it neither decodes fields nor establishes identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive(directory, retire=False):
    directory = directory.resolve()
    manifest = directory/'MANIFEST.json'
    publication = directory/'PUBLICATION.json'
    outputs = json.loads(manifest.read_text())['outputs']
    published = json.loads(publication.read_text())['files']
    if not outputs or any(published.get(name) != value for name, value in outputs.items()):
        raise ValueError('Native publication and complete output manifest differ')
    for name in outputs:
        if Path(name).name != name:
            raise ValueError('Native output is not a direct owned file')
    target = directory/'forecast-outputs.tar.zst'
    receipt = directory/'ARCHIVE.json'
    if target.exists():
        proof = json.loads(receipt.read_text())
        if (proof['manifest_sha256'] != digest(manifest)
                or proof['publication_sha256'] != digest(publication)
                or proof['archive_sha256'] != digest(target)
                or proof['files'] != outputs or not proof['restored_all_bytes_verified']):
            raise ValueError('Retained native archive differs')
    else:
        for name, wanted in outputs.items():
            if digest(directory/name) != wanted:
                raise ValueError('Native output differs before archiving: '+name)
        stage = directory/'.archive-verification'
        stage.mkdir(exist_ok=False)
        partial = directory/'forecast-outputs.tar.zst.partial'
        try:
            subprocess.run(['tar', '-I', 'zstd -T4 -6', '-cf', str(partial),
                            '-C', str(directory), *sorted(outputs)], check=True)
            subprocess.run(['tar', '-I', 'zstd', '-xf', str(partial),
                            '-C', str(stage)], check=True)
            if {name:digest(stage/name) for name in outputs} != outputs:
                raise ValueError('Complete native archive round-trip differs')
            partial.rename(target)
            proof = dict(schema='mpas-cam-native-byte-archive-v1',
                archive=target.name, archive_sha256=digest(target),
                archive_bytes=target.stat().st_size, archive_host=socket.gethostname(),
                manifest_sha256=digest(manifest), publication_sha256=digest(publication),
                files=outputs, restored_all_bytes_verified=True,
                native_forecast_identity=False)
            receipt.write_text(json.dumps(proof, indent=2)+'\n')
        finally:
            shutil.rmtree(stage)
    if retire:
        for name, wanted in outputs.items():
            path=directory/name
            if path.exists():
                if digest(path) != wanted:
                    raise ValueError('Changed native output cannot be retired: '+name)
                path.unlink()
    return proof


def restore(directory):
    directory=directory.resolve()
    proof=archive(directory)
    subprocess.run(['tar','-I','zstd','-xf',str(directory/proof['archive']),
                    '-C',str(directory)],check=True)
    for name, wanted in proof['files'].items():
        if digest(directory/name) != wanted:
            raise ValueError('Restored native proof byte differs: '+name)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--restore',action='store_true')
    parser.add_argument('--retire',action='store_true')
    args=parser.parse_args()
    if args.restore and args.retire:
        parser.error('restore and retire are mutually exclusive')
    result=restore(args.directory) if args.restore else archive(args.directory,args.retire)
    print(json.dumps({key:result[key] for key in ('archive_sha256','archive_bytes',
                      'restored_all_bytes_verified')},sort_keys=True),flush=True)
