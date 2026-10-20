#!/usr/bin/env python3
"""Run one generated native physics deck, hashing every produced stream.

Call under the proof host's native CPU affinity. Uses actual native
atmosphere_model, with unchanged physics and no Python field processing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from native_completion import completed


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('build','deck','case','out','tables'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mpi',required=True);p.add_argument('--ranks',type=int,required=True)
    p.add_argument('--pin-set-hash',required=True)
    p.add_argument('--digest-only',action='store_true',help='retire disposable output only after raw-word and stored-clock receipts')
    p.add_argument('--comparator',type=Path)
    p.add_argument('--inactive-package',action='append',default=[])
    p.add_argument('--static-coordinate-directory',type=Path,help='published UGWP xtime companion with unchanged science-word receipt')
    a=p.parse_args();build=a.build.resolve();case=a.case.resolve();deck=a.deck.resolve()
    if a.ranks<1:raise ValueError('positive rank count required')
    files={}
    def add(src,name=None):
        src=src.resolve(strict=True);target=name or src.name
        if target in files and files[target]!=src:raise ValueError(f'file collision {target}')
        files[target]=src
    for name in ('namelist.atmosphere','streams.atmosphere'):add(deck/name)
    add(build/'atmosphere_model')
    for src in build.glob('stream_list.*'):add(src)
    for src in build.iterdir():
        if src.is_file() and (src.suffix in ('.TBL','.DBL','.tbl') or 'DATA' in src.name or src.suffix=='.nc'):
            add(src)
    for src in a.tables.glob('*.DBL'):add(src)
    streams=ET.parse(deck/'streams.atmosphere')
    for s in streams.getroot():
        name=s.get('name')
        if name=='input':
            filename=s.get('filename_template')
            add(case/filename,filename)
    graph=list(case.glob('*.graph.info'))
    if len(graph)!=1:raise ValueError('case needs exactly one graph')
    add(graph[0])
    if a.ranks>1:add(case/(graph[0].name+f'.part.{a.ranks}'))
    for src in case.glob('*.nc'):
        if src.name not in files:add(src)
    coordinate_proof=None
    if a.static_coordinate_directory:
        directory=a.static_coordinate_directory.resolve(strict=True)
        coordinate_proof=directory/'MANIFEST.json';proof=json.loads(coordinate_proof.read_text())
        if (proof.get('schema')!='mpas.native-static-coordinate-copy/v1' or proof.get('existing_fields')!=20
            or proof.get('differing_words')!=0 or proof.get('added_coordinate')!='xtime'
            or proof.get('source_publication_sha256')!=sha(case/'PUBLICATION.json')):
            raise ValueError('static coordinate companion lacks unchanged native science proof')
        candidates=list(directory.glob('*.ugwp_oro_data.nc'))
        if len(candidates)!=1:raise ValueError('exactly one native UGWP coordinate companion required')
        src=candidates[0]
        if (src.name not in files or sha(files[src.name])!=proof['source_file_sha256'] or sha(src)!=proof['result_sha256']):
            raise ValueError('static coordinate companion input bytes differ from receipt')
        files[src.name]=src
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'.keep').touch()
    for name,src in files.items():
        if name in ('namelist.atmosphere','streams.atmosphere'):shutil.copy2(src,a.out/name)
        else:(a.out/name).symlink_to(src)
    retire=None
    if a.digest_only:
        from retire_native_outputs import RetireOutputs
        if not a.comparator:raise ValueError('digest-only forecasts require the real Rust word reader')
        runtime=json.loads((deck.parent/'INDEX.json').read_text())['runtime']
        retire=RetireOutputs(a.out,runtime,streams.getroot(),a.comparator,a.inactive_package)
    with (a.out/'native.stdout.log').open('w') as f:
        argv=[a.mpi,'-np',str(a.ranks),str(build/'atmosphere_model')]
        if retire:
            process=subprocess.Popen(argv,cwd=a.out,stdout=f,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    retire.collect();time.sleep(2)
            except BaseException:
                process.terminate();process.wait();raise
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,argv)
        else:subprocess.run(argv,cwd=a.out,stdout=f,stderr=subprocess.STDOUT,check=True)
    log=a.out/'log.atmosphere.0000.out'
    if not log.exists() or not completed(log.read_text(),'atmosphere'):raise RuntimeError('native did not finish')
    if retire:
        retire.collect(finished=True)
        if set(retire.files)!=set(retire.expected):raise RuntimeError('native disposable output set is incomplete')
        outputs={name:entry['sha256'] for name,entry in retire.files.items()}
    else:outputs={p.name:sha(p) for p in sorted(a.out.glob('*.nc')) if not p.is_symlink()}
    if not outputs:raise RuntimeError('no native output files')
    manifest={'schema':'mpas.native-forecast-cache/v1','pin_set_hash':a.pin_set_hash,
              'deck':deck.name,'ranks':a.ranks,'binary_sha256':sha(build/'atmosphere_model'),
              'inputs':{name:sha(src) for name,src in files.items()},
              'outputs':outputs,
              'namelist_sha256':sha(a.out/'namelist.atmosphere'),'streams_sha256':sha(a.out/'streams.atmosphere')}
    if (deck.parent/'INDEX.json').is_file():
        index=json.loads((deck.parent/'INDEX.json').read_text())
        manifest['runtime']=index['runtime']
        manifest['deck_index_sha256']=sha(deck.parent/'INDEX.json')
    if retire:
        manifest.update(outputs_retired=True,retired_clock_receipt_sha256=sha(retire.receipt),output_digest_tool_sha256=retire.tool_sha256)
    if coordinate_proof:manifest['static_coordinate_manifest_sha256']=sha(coordinate_proof)
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'outputs':len(outputs),'manifest':str(a.out/'MANIFEST.json')}))
if __name__=='__main__':main()
