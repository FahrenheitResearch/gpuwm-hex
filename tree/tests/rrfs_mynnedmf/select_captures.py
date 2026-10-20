#!/usr/bin/env python3
"""Select complete wrapper inputs, entry storage and every output from r1 seals."""
import argparse, hashlib, json
from pathlib import Path
from authenticate_inventory import selected_records
p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--out',required=True);a=p.parse_args()
publication=Path(a.publication);pub=json.loads(publication.read_text());root=Path(pub['path'])
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
selections=[];members=[]
for name, seal in pub['members'].items():
 if not name.startswith('mynnedmf_driver__') or not name.endswith('.manifest.json'):continue
 raw=(root/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=seal['sha256']:raise RuntimeError('unauthenticated r1 manifest')
 mf=json.loads(raw);capture=name.removesuffix('.manifest.json');archive=json.loads((root/(capture+'.archive.json')).read_text())
 files=selected_records(mf,capture)
 item={'capture':capture,'source_manifest_sha256':seal['sha256'],'source_dump_root_sha256':archive['dump_root_sha256'],'publication_sha256':hashlib.sha256(publication.read_bytes()).hexdigest(),'complete_archive_sha256':pub['members']['complete.tar.zst']['sha256'],'files':files}
 dest=out/(capture+'.json');dest.write_text(json.dumps(item,indent=2)+'\n');members.extend('./'+n+'\n' for n in sorted(files));selections.append(str(dest))
(out/'members.txt').write_text(''.join(members));(out/'cases.json').write_text(json.dumps({'publication':pub,'selections':selections},indent=2)+'\n')
print(json.dumps({'cases':len(selections),'records':len(members),'out':str(out)}))
