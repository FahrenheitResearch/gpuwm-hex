#!/usr/bin/env python3
"""Schedule Rust comparison of every native NetCDF field and output stream."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def compare(reference,other,comparator,pattern):
    names=[{f.name for f in p.glob(pattern) if f.is_file() and not f.is_symlink()} for p in (reference,other)]
    errors=[]
    if not names[0]:errors.append('reference has no output files')
    if names[0]!=names[1]:errors.append({'missing':sorted(names[0]-names[1]),'extra':sorted(names[1]-names[0])})
    files={}; ok=not errors
    for name in sorted(names[0]&names[1]):
        result=subprocess.run([str(comparator),str(reference/name),str(other/name)],capture_output=True,text=True)
        if result.returncode not in (0,1):raise RuntimeError(result.stderr)
        entry=json.loads(result.stdout)
        entry['sha256']=[digest(p/name) for p in (reference,other)]
        entry['file_byte_identical']=entry['sha256'][0]==entry['sha256'][1]
        files[name]=entry;ok &= entry['all_identical']
    return {'schema':'mpas.native-full-stream-identity/v1','all_identical':ok,
            'all_floating_words_identical':not errors and bool(files) and all(
                r.get('all_floating_words_identical',r['all_identical']) for r in files.values()),
            'all_file_bytes_identical':not errors and bool(files) and all(r['file_byte_identical'] for r in files.values()),
            'reference':str(reference),'other':str(other),'errors':errors,'files':files}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('other',type=Path)
    p.add_argument('--comparator',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pattern',default='*.nc')
    a=p.parse_args();r=compare(a.reference,a.other,a.comparator,a.pattern)
    a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('all_identical','all_file_bytes_identical','errors')}))
    return 0 if r['all_identical'] else 1
if __name__=='__main__':raise SystemExit(main())
