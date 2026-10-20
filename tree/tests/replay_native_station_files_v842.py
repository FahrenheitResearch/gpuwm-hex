"""Pristine numerical slice/writer versus the actual native sounding files."""
import argparse,ctypes,hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('writer',type=Path);p.add_argument('patch',type=Path);p.add_argument('files',type=Path);p.add_argument('scratch',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
m=json.loads((a.patch/'patch.json').read_text());h={r['field']:np.fromfile(a.patch/(r['field']+'.raw'),dtype=r['dtype']).reshape(r['shape'],order='F') for r in m['fields']}
ns,nz,nc=(m[n] for n in ('nCellsSolve','nVertLevels','nCells'));cells=np.arange(1,ns+1,dtype=np.int32);owned=np.ones(ns,dtype=np.int32);out=np.zeros((6,nz,ns),dtype=np.float32,order='F')
arrays=[cells,owned,*[h[n] for n in ('theta_m','qv','exner','pressure_base','pressure_p','uReconstructZonal','uReconstructMeridional','zgrid')],out]
ctypes.CDLL(str(a.library.resolve())).replay_sounding(ctypes.c_int(ns),ctypes.c_int(nz),ctypes.c_int(nc+1),*[ctypes.c_void_p(v.ctypes.data) for v in arrays])
a.scratch.mkdir(parents=True,exist_ok=True);raw=a.scratch/'profiles.raw';raw.write_bytes(struct.pack('<I',nz)+owned.tobytes()+out.tobytes(order='F'))
files=sorted(a.files.glob('col*.snd'));assert len(files)==ns
stamp=files[0].name.split('.')[1];time=f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}_{stamp[8:10]}:{stamp[10:12]}:00'
subprocess.run([str(a.writer.resolve()),'write',str((a.files/'sounding_locations.txt').resolve()),str(raw.resolve()),time],cwd=a.scratch,check=True,capture_output=True)
rows=[]
for f in files:
 actual=f.read_bytes();replay=(a.scratch/f.name).read_bytes();assert actual==replay,(f.name,len(actual),len(replay))
 rows.append(dict(file=f.name,bytes=len(actual),differing_bytes=0,sha256=hashlib.sha256(actual).hexdigest()))
a.receipt.write_text(json.dumps(dict(tier='pristine strict standalone versus actual strict-native executable files',whole_forecast_proof=False,rows=rows),indent=2)+'\n');print(f'{len(rows)} actual native station files matched')
