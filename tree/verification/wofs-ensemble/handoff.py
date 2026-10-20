"""Stage only pictures and compact receipts for the requested desktop handoff."""
import os
from pathlib import Path
import gzip,hashlib,json,shutil,sys
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))

def stage(phase):
    source=W/'products'/phase
    if not (source/'complete.json').is_file() or not (source/'map-artifacts.json').is_file():
        raise ValueError('products and picture hashes must close before staging')
    target=W/'handoff'/phase
    target.mkdir(parents=True,exist_ok=True)
    copied=[]
    def copy(path,relative):
        compress=path.suffix in ('.json','.log') and path.stat().st_size>128*1024
        if compress: relative=Path(str(relative)+'.gz')
        dest=target/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with path.open('rb') as src, dest.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as dst:
                    shutil.copyfileobj(src,dst)
        else: shutil.copy2(path,dest)
        with dest.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        copied.append(dict(path=str(relative),bytes=dest.stat().st_size,sha256=sha))
    for row in json.loads((source/'map-artifacts.json').read_text()):
        path=Path(row['path'])
        if path.suffix!='.png' or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('picture artifact escapes this product run')
        if path.stat().st_size!=row['bytes']: raise ValueError('picture size changed')
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=row['sha256']: raise ValueError('picture hash changed')
        copy(path,Path('maps')/path.relative_to(source))
    copy(source/'CAPTIONS.md',Path('maps/CAPTIONS.md'))
    for path in source.iterdir():
        if path.name in ('score-chart.png','runtime-chart.png') or path.suffix in ('.csv','.json'):
            copy(path,Path('scores')/path.name)
    root=W/'runs'/phase
    copy(root/'ensemble-receipt.json',Path('evidence/ensemble-receipt.json'))
    receipt=json.loads((root/'ensemble-receipt.json').read_text())
    native=source/'aggregate/d01/ensemble-manifest.json'
    if native.is_file():
        copy(native,Path('evidence/ensemble-products.json'))
        roster=source/'reducer-roster.json'
        roster.write_text(json.dumps([dict(reducer_member_id=index,member=job['id'],
            segments=job['segments']) for index,job in enumerate(
                job for job in receipt['jobs'] if job['returncode']==0)],indent=2)+'\n')
        copy(roster,Path('evidence/reducer-roster.json'))
    for job in receipt['jobs']:
        if job['returncode']!=0: continue
        for name in ('forecast-receipt.json','cuda-v841-forecast-receipt.json','fullstep-lts-receipt.json'):
            copy(root/job['id']/'history'/name,Path('evidence')/job['id']/name)
        copy(Path(job['log']),Path('evidence')/job['id']/'forecast.log')
    (target/'ARTIFACTS.json').write_text(json.dumps(copied,indent=2)+'\n')
    return copied

if __name__=='__main__': stage(sys.argv[1])
