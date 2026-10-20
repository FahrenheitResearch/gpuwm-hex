"""Pristine two-pass list reader's partially assigned record witness."""
from pathlib import Path
import subprocess,json,struct,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
rows=[]
for label,text in {'comment-first':'!comment\n1 2 first\n3 4 last\n','multiline':'1\n2 station\n3 4 last\n','null-first':',,first\n1 2 last\n','slash-first':'/\n1 2 last\n','null-second':'1 2 first\n,,last\n'}.items():
 f=a.directory/(label+'.txt');f.write_text(text)
 for repeat in range(2):
  out=a.directory/(label+str(repeat)+'.raw');r=subprocess.run([str(a.native.resolve()),'read',str(f.resolve()),str(out.resolve())],capture_output=True)
  row=dict(fixture=label,repeat=repeat,exit=r.returncode,stderr=r.stderr.decode(errors='replace'))
  if out.exists():
   raw=out.read_bytes();n=struct.unpack('<I',raw[:4])[0];row.update(nSoundings=n,sha256=hashlib.sha256(raw).hexdigest(),names_hex=[raw[4+i*512:4+(i+1)*512].hex() for i in range(n)],coordinate_words=list(struct.unpack('<'+'I'*(2*n),raw[4+n*512:])))
  rows.append(row)
a.receipt.write_text(json.dumps(dict(tier='strict standalone undefined native list-storage witness',rows=rows),indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='names_hex'} for r in rows],indent=2))
