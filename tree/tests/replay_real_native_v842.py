"""Compare pristine strict numerical slice with real captured native PV words."""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('patch',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
m=json.loads((a.patch/'patch.json').read_text());h={r['field']:np.fromfile(a.patch/(r['field']+'.raw'),dtype=r['dtype']).reshape(r['shape'],order='F') for r in m['fields']}
nc,nz,me,nv,ne,ns=(m[n] for n in ('nCells','nVertLevels','maxEdges','nVertices','nEdges','nCellsSolve'))
args=[nc,nz,me,nv,ne,*[h[n] for n in ('nEdgesOnCell','cellsOnCell','edgesOnCell','cellsOnEdge','verticesOnCell','cellsOnVertex','kiteAreasOnVertex','areaCell','dvEdge','edgeNormalVectors','cellTangentPlane','localVerticalUnitVectors','zgrid')],ns,nz-1,*[h[n] for n in ('uReconstructX','uReconstructY','uReconstructZ','w','vorticity','rho','theta')]]
out=np.zeros((nz,nc+1),dtype=np.float32,order='F');lib=ctypes.CDLL(str(a.reference.resolve()));lib.replay_mesh(*[ctypes.c_void_p(v.ctypes.data) if isinstance(v,np.ndarray) else ctypes.c_int(v) for v in (*args,out)])
w=h['ertel_pv'][:-1,:ns];g=out[:-1,:ns];bad=w.view(np.uint32)!=g.view(np.uint32);ix=np.argwhere(bad)
r=dict(words=w.size,different_words=int(bad.sum()),samples=[dict(level=int(k+1),column=int(c+1),native=float(w[k,c]),standalone=float(g[k,c]),native_word=int(w.view(np.uint32)[k,c]),standalone_word=int(g.view(np.uint32)[k,c])) for k,c in ix[:24]],whole_forecast_proof=False)
a.receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
