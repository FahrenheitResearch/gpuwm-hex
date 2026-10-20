#!/usr/bin/env python3
"""Seal the explicitly pinned acceptance inputs without changing their bytes."""
import argparse,hashlib,json,shutil
from pathlib import Path
from publish_cache import publish,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','work','ref','ready','out-config'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    sources={}
    def select(path,digest):
        source=Path(path).resolve(strict=True)
        if sha(source)!=digest:raise ValueError('acceptance input differs from its frozen hash: '+str(source))
        sources.setdefault(digest,source)
        return 'files/'+digest
    initial_name=select(config['initial'],config['initial_sha256'])
    selected={}
    for row in config['runs']:
        selected[row['name']]={name:select(item['path'],item['sha256']) for name,item in row['files'].items()}
    metadata={'schema':'mpas.native-acceptance-input-publication/v1','source_config_sha256':sha(args.config),
              'files':{digest:str(source) for digest,source in sources.items()},'initial':initial_name,
              'runs':selected,'native_ranks':config['ranks'],'native_binary_sha256':config['binary_sha256']}
    key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    target=args.ref/'cases'/'dycore-acceptance'/key
    if not target.exists():
        stage=args.work/key;(stage/'files').mkdir(parents=True,exist_ok=False)
        for digest,source in sources.items():
            shutil.copy2(source,stage/'files'/digest)
            if sha(stage/'files'/digest)!=digest:raise ValueError('acceptance input copy changed')
        for directory in (stage,stage/'files'):(directory/'.keep').touch()
        (stage/'MANIFEST.json').write_text(json.dumps(metadata,indent=2)+'\n')
        publish(stage,target,['MANIFEST.json','.keep','files/*'],args.ready,
                'Byte-pinned real-terrain dry dycore acceptance inputs/decks/partition, native process count '+str(config['ranks']))
        shutil.rmtree(stage)
    publication=json.loads((target/'PUBLICATION.json').read_text())
    if json.loads((target/'MANIFEST.json').read_text())!=metadata:raise ValueError('acceptance publication metadata changed')
    for name,digest in publication['files'].items():
        if sha(target/name)!=digest:raise ValueError('acceptance publication file changed')
    config['initial']=str(target/initial_name)
    for row in config['runs']:
        for name,item in row['files'].items():item['path']=str(target/selected[row['name']][name])
    config['input_publication']=str(target)
    config['input_publication_sha256']=sha(target/'PUBLICATION.json')
    args.out_config.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'directory':str(target),'publication_sha256':sha(target/'PUBLICATION.json'),
                      'files':len(sources),'config_sha256':sha(args.out_config),'original_config_sha256':sha(args.config),
                      'input_bytes_changed':False}))
if __name__=='__main__':main()
