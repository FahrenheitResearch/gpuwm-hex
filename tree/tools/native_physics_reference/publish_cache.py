#!/usr/bin/env python3
"""Atomically publish explicitly selected cache files and append a hashed READY entry.

Glob selections are explicit: input cases include grid/graph/partition files;
forecast selections include output streams, decks, logs, and native manifests.
Resolved symlink inputs are copied so scratch removal cannot break the cache.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def seal(path):
    """Remove write bits from a completed cache; retain executable/read bits."""
    for child in path.rglob('*'):
        if not child.is_symlink():
            child.chmod(child.stat().st_mode & ~0o222)
    path.chmod(path.stat().st_mode & ~0o222)

def publish(source,destination,patterns,ready,description):
    source=source.resolve(strict=True)
    if destination.exists():raise ValueError('refusing existing cache publication')
    files=sorted({p for pattern in patterns for p in source.glob(pattern) if p.is_file()})
    if not files:raise ValueError('no selected cache products')
    for p in files:
        relative=p.relative_to(source)
        if '..' in relative.parts or relative==Path('PUBLICATION.json'):
            raise ValueError('selection escapes source or replaces publication metadata')
    stage=destination.with_name(destination.name+'.partial')
    stage.mkdir(parents=True,exist_ok=False)
    try:
        for p in files:
            target=stage/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p.resolve(strict=True),target)
        manifest={'schema':'mpas.native-cache-publication/v1','description':description,
                  'source':str(source),'selection':patterns,
                  'files':{str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
        (stage/'PUBLICATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
        digest=sha(stage/'PUBLICATION.json')
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage);raise
    with ready.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(f'{description}: `{destination}`; PUBLICATION.json sha256 `{digest}`.\n')
    seal(destination)
    return {'path':str(destination),'publication_sha256':digest,'files':len(files)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--include',action='append',required=True)
    p.add_argument('--ready',type=Path,required=True);p.add_argument('--description',required=True)
    a=p.parse_args();print(json.dumps(publish(a.source,a.destination,a.include,a.ready,a.description)))
if __name__=='__main__':main()
