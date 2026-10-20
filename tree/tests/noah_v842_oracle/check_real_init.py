"""Replay real initial surface/soil state, packed by the Rust input path."""
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import os
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--prefix',type=Path,default=Path('n1out/real-init'))
p.add_argument('--output',type=Path,default=Path('n1out/real-init-columns'))
p.add_argument('--input-file',type=Path,default=Path('scratch/real-init/input.nc'))
p.add_argument('--metadata',type=Path,default=Path('n1out/real-init-decode-metadata.json'))
p.add_argument('--receipt',type=Path,default=Path('n1out/real-init-host.json'))
p.add_argument('--publication',type=Path)
p.add_argument('--withdrawn',action='store_true',help='label invalid input debugging evidence explicitly')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
base=Path(__file__).resolve().parent
metadata=json.loads(a.metadata.read_text())
expected=('tslb','smois','sh2o','snoalb','snow','snowh','ivgtyp','isltyp')
assert tuple(v['name'] for v in metadata['variables'])==expected
x=np.fromfile(str(a.prefix)+'.f32',dtype='<f4').reshape(-1,15)
ix=np.fromfile(str(a.prefix)+'.i32',dtype='<i4').reshape(-1,4)
assert len(x)==len(ix) and len(x)%4==0
native=C.CDLL(str(base/'libnoah_oracle.so'));host=C.CDLL(str(base/'libnoah_host.so'))
old=Path.cwd();os.chdir(base);native.read_tables();os.chdir(old)
tb=np.load('n1out/init-synthetic.npz')['table_bytes']
a1=x.copy();i1=ix.copy();b=x.copy();ib=ix.copy()
fn=native.replay_init;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int];fn(a1.ctypes.data,i1.ctypes.data,len(x))
fn=host.replay_init;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int];fn(b.ctypes.data,ib.ctypes.data,tb.ctypes.data,len(x))
d=a1.view(np.uint32)!=b.view(np.uint32)
result={'scheme':'init','input_file_sha256':hashlib.sha256(a.input_file.read_bytes()).hexdigest(),'input_decoder':'rw_netcdf dump --raw + pack_real_init.rs','physical_cells':len(x)//4,'columns':len(x),'flag_variants':'restart and input_sfc_albedo: all four combinations','differing_words':int(d.sum()),'integer_differing_words':int((i1!=ib).sum()),'scope':'real initialization states; no driver forcing or forecast proof','publication_status':'withdrawn debugging input' if a.withdrawn else 'published reference input' if a.publication else 'publication not supplied','acceptance_evidence':bool(a.publication) and not a.withdrawn}
if a.publication:
    result['publication_sha256']=hashlib.sha256(a.publication.read_bytes()).hexdigest()
    publication=json.loads(a.publication.read_text())
    assert result['input_file_sha256'] in publication['files'].values(), 'input is absent from published file hashes'
np.savez(a.output/'init-synthetic.npz',input=x,input_int=ix,native=a1,native_int=i1,table_bytes=tb)
a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if d.any() or (i1!=ib).any():raise SystemExit(1)
