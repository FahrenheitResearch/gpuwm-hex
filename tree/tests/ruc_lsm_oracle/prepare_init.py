"""Authenticate and extract opaque native initialization/driver records, no field decode."""
import argparse,hashlib,json,subprocess,tarfile,tempfile
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    pins=json.loads(a.pins.read_text())
    temporary=None
    if 'archive_parts' in pins:
        temporary=tempfile.NamedTemporaryFile(prefix='ruc-bank-',suffix='.tar.zst')
        for part in pins['archive_parts']:
            name=part['name']
            if Path(name).name!=name:raise ValueError('unsafe native archive part')
            source=a.pins.parent/name
            if sha(source)!=part['sha256'] or source.stat().st_size!=part['size']:raise ValueError('sealed native archive part changed')
            with source.open('rb') as f:
                while block:=f.read(1024*1024):temporary.write(block)
        temporary.flush();a.archive=Path(temporary.name)
    if sha(a.archive)!=pins['archive_sha256']:raise ValueError('sealed native archive changed')
    a.out.mkdir(parents=True,exist_ok=True)
    names=set();proc=subprocess.Popen(['zstd','-q','-d','-c',str(a.archive)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
        for m in archive:
            if Path(m.name).name!=m.name or not m.isfile() or m.name in names:raise ValueError('unsafe native archive member')
            names.add(m.name)
            dest=a.out/m.name
            if dest.is_symlink():raise ValueError('native extraction target is a symlink')
            with archive.extractfile(m) as source,dest.open('wb') as output:
                while block:=source.read(1024*1024):output.write(block)
    if proc.wait() or sha(a.out/'manifest.json')!=pins['manifest_sha256']:raise ValueError('native bank manifest differs')
    manifest=json.loads((a.out/'manifest.json').read_text())
    scope_files={prefix+'.scope.txt' for prefix in manifest['captures']}
    if names!=set(manifest['files'])|{'manifest.json'}|scope_files:raise ValueError('native bank inventory differs')
    for name,spec in manifest['files'].items():
        if sha(a.out/name)!=spec['sha256']:raise ValueError('native capture hash differs: '+name)
    if temporary is not None:temporary.close()
    print('Authenticated',len(manifest['files']),'native initialization/driver records')

if __name__=='__main__':main()
