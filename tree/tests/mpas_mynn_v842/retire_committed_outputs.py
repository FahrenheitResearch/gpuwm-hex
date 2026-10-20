"""Retire only owned output authenticated by a complete stored-word grade."""
from pathlib import Path
import argparse,json,hashlib,shutil
p=argparse.ArgumentParser(description=__doc__)
for n in ['output','grade','receipt']:p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();r=json.loads(a.grade.read_text());assert r['words']>0 and r['streams']
retired=[];admitted=[]
for name,row in r['candidate_hashes'].items():
 f=a.output/name
 if (not f.resolve().is_relative_to(a.output.resolve()) or f.is_symlink()
     or not f.is_file()):raise ValueError('Refuse unknown output '+name)
 with f.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
 admitted.append((f,row))
for f,row in admitted:
 f.unlink();retired.append(dict(file=str(f),sha256=row['sha256'],bytes=row['bytes']))
 a.receipt.write_text(json.dumps(dict(scope=__doc__,files=retired),indent=2)+'\n')
