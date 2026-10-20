#!/usr/bin/env python3
"""Publish a completed build and its receipt; READY never announces partial files."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from publish_cache import seal

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def publish(build, destination, core, recipe):
    binary=build/(core+'_model')
    if not binary.is_file(): raise ValueError('missing binary')
    if core=='atmosphere':
        audit=json.loads((build/'flag-audit.json').read_text())
        if not audit['all_ok']: raise ValueError('flags failed audit')
    pin={'upstream': 'f34984b2c89353cf31c235b08d3b4acd96b37bb3',
         'mmm':'a4baf7f3243d1db0dbc5f63473f895bdbdc05c30',
         'ugwp':'c1c893edcf171af5639af60e3a3a528816f6cc2b',
         'mpas_data':'c57dbc7be629802c6e848770a9e44b9bc602be41',
         'core':core, 'recipe':recipe, 'binary_sha256':sha(binary),
         'toolchain_sha256':sha(build/'toolchain.txt'), 'build_diff_sha256':sha(build/'build-source.diff')}
    key=hashlib.sha256(json.dumps(pin,sort_keys=True).encode()).hexdigest()
    final=destination/'native'/key
    if final.exists(): raise ValueError('refusing existing publication')
    stage=final.with_name(key+'.partial'); stage.mkdir(parents=True)
    try:
        files=[binary,build/'toolchain.txt',build/'build-source.diff',build/'executable.sha256',build/'build.log']
        files+=list(build.glob('namelist.*'))+list(build.glob('streams.*'))+list(build.glob('stream_list.*'))
        if core=='atmosphere':
            files += [build/'flag-audit.json',build/'build_tables']
            files += list((build/'src/core_atmosphere/physics/physics_wrf/files').glob('*'))
        for f in files:
            if f.is_file(): shutil.copy2(f,stage/f.name)
        pin['files']={p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage/'MANIFEST.json').write_text(json.dumps(pin,indent=2)+'\n')
        stage.rename(final)
    except BaseException:
        shutil.rmtree(stage); raise
    with (destination/'READY.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(f'{core} {recipe}: `{final}`; {binary.name} sha256 `{pin["binary_sha256"]}`; run using MPICH recorded in toolchain.txt. Native dycore acceptance is pending.\n')
    seal(final)
    return final

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('build',type=Path); p.add_argument('destination',type=Path)
    p.add_argument('--core',choices=('atmosphere','init_atmosphere'),required=True)
    p.add_argument('--recipe',choices=('gnu-strict','gnu-o3'),required=True)
    a=p.parse_args();print(publish(a.build,a.destination,a.core,a.recipe))
if __name__=='__main__': main()
