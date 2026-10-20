#!/usr/bin/env python3
"""Publish identical MPI runtime bytes without changing native executables."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from publish_cache import publish,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['prefix','work','ref','ready']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prefix=a.prefix.resolve(strict=True)
    names=['bin/mpiexec.hydra','bin/hydra_pmi_proxy','bin/mpichversion']
    names += [str(f.relative_to(prefix)) for f in sorted((prefix/'lib').glob('*.so*')) if f.is_file()]
    files={name:sha(prefix/name) for name in names}
    version=subprocess.check_output([str(prefix/'bin/mpichversion')],text=True)
    metadata={'schema':'mpas.native-mpi-runtime/v1','source':str(prefix),
              'version':version,'files':files,'native_executable_changed':False}
    import hashlib
    key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    target=a.ref/'native'/key
    if target.exists():
        publication=json.loads((target/'PUBLICATION.json').read_text())
        if json.loads((target/'RUNTIME.json').read_text())!=metadata:
            raise ValueError('runtime publication belongs to other bytes')
        for name,digest in publication['files'].items():
            if sha(target/name)!=digest:raise ValueError('runtime publication changed')
        print(json.dumps({'directory':str(target),'publication_sha256':sha(target/'PUBLICATION.json'),'reused':True}));return
    stage=a.work/key;stage.mkdir(parents=True,exist_ok=False)
    for name in names:
        destination=stage/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2((prefix/name).resolve(strict=True),destination)
        if sha(destination)!=files[name]:raise ValueError('runtime copy differs from original')
    # Hydra discovers its proxy relative to the relocated launcher directory.
    (stage/'bin/mpirun').symlink_to('mpiexec.hydra')
    (stage/'bin/mpiexec').symlink_to('mpiexec.hydra')
    for directory in [stage,stage/'bin',stage/'lib']:(directory/'.keep').touch()
    (stage/'RUNTIME.json').write_text(json.dumps(metadata,indent=2)+'\n')
    env=dict(os.environ,LD_LIBRARY_PATH=str(stage/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    relocated=subprocess.check_output([str(stage/'bin/mpichversion')],env=env,text=True)
    if relocated!=version:raise ValueError('relocated runtime version differs')
    (stage/'RUN.txt').write_text('Set LD_LIBRARY_PATH to this directory/lib, then run bin/mpiexec -np N NATIVE_EXECUTABLE.\nRuntime files are byte-identical to RUNTIME.json source; native executable hashes stay unchanged.\n')
    result=publish(stage,target,['RUNTIME.json','RUN.txt','.keep','bin/*','lib/*'],a.ready,
                   'Byte-pinned MPICH runtime; use LD_LIBRARY_PATH=<directory>/lib and <directory>/bin/mpiexec -np N <native executable>')
    result['directory']=str(target);print(json.dumps(result),flush=True)
    shutil.rmtree(stage)
if __name__=='__main__':main()
