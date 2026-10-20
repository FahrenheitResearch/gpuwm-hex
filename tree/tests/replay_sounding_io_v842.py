"""Rust/pristine GNU sounding parsing, formatting and full-file byte replay."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess
import numpy as np


def run(native,rust,directory):
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    def compare(name,a,b):
        if a!=b:
            first=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y) if a[:min(len(a),len(b))]!=b[:min(len(a),len(b))] else min(len(a),len(b))
            raise AssertionError((name,first,a[max(0,first-15):first+25],b[max(0,first-15):first+25]))
        rows.append(dict(test=name,bytes=len(a),differing_bytes=0,sha256=hashlib.sha256(a).hexdigest()))
    def call(*args,input=None,cwd=None):
        return subprocess.run([str(a) for a in args],input=input,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    values=np.array([0,-0.,1.125,1.375,-1.125,-1.375,9999.999,-999.999,1e-40,-1e-40,np.nan,np.inf,-np.inf],dtype='<f4')
    rng=np.random.default_rng(9872)
    random_words=rng.integers(0,2**32,size=8192,dtype=np.uint32)
    # Include negative NaNs and native format threshold neighbors.
    data=values.tobytes()+random_words.tobytes()+np.arange(-16384,16384,dtype=np.float32).tobytes()
    raw=directory/'format.raw';raw.write_bytes(data)
    a=call(native,'format',raw);b=call(rust,'format',input=data)
    assert a.returncode==b.returncode==0,(a.stderr,b.stderr)
    compare('Fw.2 random/ties/subnormals/signed-zero/nonfinite',a.stdout,b.stdout)
    fixtures={
        'standard':"45 -120 alpha\n-90 180 south\n0 0 equator\n",
        'quoted-repeat':"45,-1.2D2,'A B'\n2*0,\"quoted\"\n1+1,-1-1,'a''b'\n",
        'invalid-stops-count':"1 2 station\ninvalid record\n3 4 later\n",
        'empty':'',
        'unicode':"1 2 'é☁'\n",
        'byte-name':b"1 2 '\xfeA'\n",
        'embedded-nul':b"1 2 'A\x00B'\n",
        'leading-nul':b"1 2 '\x00A'\n",
        'gnu-byte-eof':b"1 2 '\xffA'\n3 4 later\n",
        'unclosed-name':"1 2 'unterminated\n",
        'truncated-byte-name':b"1 2 "+b'a'*507+b'\xc3\xa9\n',
        'comment-first':'!comment\n1 2 first\n3 4 last\n',
        'multiline':'1\n2 station\n3 4 last\n',
        'null-first':',,first\n1 2 last\n',
        'slash-first':'/\n1 2 last\n',
        'null-second':'1 2 first\n,,last\n',
        'first-extra':"1 2 station 23 unused\n",
        'trailing-comment':"1 2 station !ignored\n",
    }
    for name,text in fixtures.items():
        loc=directory/(name+'.txt');loc.write_bytes(text.encode() if isinstance(text,str) else text);out=directory/(name+'.raw')
        a=call(native,'read',loc,out);b=call(rust,'read',loc)
        if a.returncode:
            assert b.returncode,(name,a.stderr,b.stdout)
            rows.append(dict(test='list-'+name,native_failure=True,rust_failure=True));continue
        assert b.returncode==0,(name,b.stderr)
        n=struct.unpack('<I',a_raw:=out.read_bytes()[:4])[0]
        native_bytes=out.read_bytes()
        names=[native_bytes[4+512*i:4+512*(i+1)].rstrip(b' ') for i in range(n)]
        mlen=struct.unpack('<I',b.stdout[:4])[0];meta=json.loads(b.stdout[4:4+mlen])
        compare('list-'+name+'-names',json.dumps([v.hex() for v in names]).encode(),json.dumps(meta['names_hex']).encode())
        compare('list-'+name+'-RKIND',native_bytes[4+512*n:],b.stdout[4+mlen:])
    # Entire native writer including replacement and minute-only filenames.
    loc=directory/'standard.txt';ns=3;nz=55
    native_dir=directory/'native';rust_dir=directory/'rust';native_dir.mkdir(exist_ok=True);rust_dir.mkdir(exist_ok=True)
    numeric=np.asfortranarray(rng.normal(0,50,(6,nz,ns)).astype('<f4'));numeric[0]+=1000;numeric[1]+=10000
    numeric[:,0,0]=values[:6]
    owned=np.array([1,0,1],dtype='<i4')
    raw=directory/'profiles.raw';raw.write_bytes(struct.pack('<I',nz)+owned.tobytes()+numeric.tobytes(order='F'))
    meta=dict(exists=True,opened=True,names=['alpha','south','equator'],levels=nz)
    j=json.dumps(meta).encode();coordinates=np.array([45,-90,0,-120,180,0],dtype='<f4').tobytes()
    packet=struct.pack('<I',len(j))+j+coordinates+owned.tobytes()+numeric.tobytes(order='F')
    for time in ('2000-02-29_23:59:03','2000-02-29_23:59:59','2023-01-01_00:00:00'):
        a=call(native,'write',loc,raw,time,cwd=native_dir);b=call(rust,'write',time,rust_dir,input=packet)
        assert a.returncode==b.returncode==0,(a.stderr,b.stderr)
        for file in native_dir.glob('*.snd'):compare('file-'+time+'-'+file.name,file.read_bytes(),(rust_dir/file.name).read_bytes())
        assert sorted(p.name for p in native_dir.glob('*.snd'))==sorted(p.name for p in rust_dir.glob('*.snd'))
    # Fortran CHARACTER widths count bytes, including UTF8 and raw bytes.
    for name in ('unicode','byte-name'):
        loc=directory/(name+'.txt');packet_read=call(rust,'read',loc).stdout
        prefix=struct.unpack('<I',packet_read[:4])[0];meta=json.loads(packet_read[4:4+prefix]);meta['levels']=nz
        j=json.dumps(meta).encode();payload=struct.pack('<I',len(j))+j+packet_read[4+prefix:]+np.array([1],dtype='<i4').tobytes()+numeric[:,:,:1].tobytes(order='F')
        raw=directory/(name+'-profile.raw');raw.write_bytes(struct.pack('<I',nz)+np.array([1],dtype='<i4').tobytes()+numeric[:,:,:1].tobytes(order='F'))
        nd=directory/(name+'-native');rd=directory/(name+'-rust');nd.mkdir(exist_ok=True);rd.mkdir(exist_ok=True)
        a=call(native,'write',loc,raw,'2000-02-29_23:59:59',cwd=nd);b=call(rust,'write','2000-02-29_23:59:59',rd,input=payload)
        assert a.returncode==b.returncode==0,(name,a.stderr,b.stderr)
        for file in nd.glob('*.snd'):compare('native-byte-header-'+name,file.read_bytes(),(rd/file.name).read_bytes())
    # Concrete native internal WRITE failures and their file creation order.
    for label,name in [('ten-byte',b'1234567890'),('stid-overflow',b'12345678901'),('filename-overflow',b'a'*496),('null-name',None),('embedded-nul',b'A\x00B'),('leading-nul',b'\x00A'),('quoted-leading-nul',b"'\x00A'")]:
        loc=directory/(label+'.txt');loc.write_bytes(b'0 0 '+(b'/' if name is None else name)+b'\n')
        response=call(rust,'read',loc);assert response.returncode==0,(label,response.stderr)
        nmeta=struct.unpack('<I',response.stdout[:4])[0];meta=json.loads(response.stdout[4:4+nmeta]);meta['levels']=nz
        j=json.dumps(meta).encode();payload=struct.pack('<I',len(j))+j+response.stdout[4+nmeta:]+np.array([1],dtype='<i4').tobytes()+numeric[:,:,:1].tobytes(order='F')
        raw=directory/(label+'-profile.raw');raw.write_bytes(struct.pack('<I',nz)+np.array([1],dtype='<i4').tobytes()+numeric[:,:,:1].tobytes(order='F'))
        nd=directory/(label+'-native');rd=directory/(label+'-rust');nd.mkdir(exist_ok=True);rd.mkdir(exist_ok=True)
        a=call(native,'write',loc,raw,'2000-02-29_23:59:59',cwd=nd);b=call(rust,'write','2000-02-29_23:59:59',rd,input=payload)
        assert bool(a.returncode)==bool(b.returncode),(label,a.stderr,b.stderr)
        if a.returncode:rows.append(dict(test='writer-'+label,native_failure=True,rust_failure=True))
        native_files={f.name:f.read_bytes() for f in nd.iterdir()};rust_files={f.name:f.read_bytes() for f in rd.iterdir()}
        assert native_files==rust_files,(label,native_files.keys(),rust_files.keys())
        for name,data in native_files.items():compare('writer-'+label+'-file',data,rust_files[name])
    receipt=dict(tier='strict standalone GNU versus Rust',whole_forecast_proof=False,checks=rows,
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rust_sha256=hashlib.sha256(rust.read_bytes()).hexdigest())
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('rust',type=Path);p.add_argument('scratch',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();receipt=run(a.native.resolve(),a.rust.resolve(),a.scratch.resolve());a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2));print(f"{len(receipt['checks'])} exact byte comparisons passed")
