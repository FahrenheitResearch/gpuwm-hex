"""Observe the native unassigned top velCellp; debug initialization is not proof."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from test_mpasdiag_v842 import F, mesh_fixture, native_library


def run(library):
    m,s,d,h,f,w=mesh_fixture(np,14)
    nc,nz,me,ne,nv=(m[n] for n in ("nCells","nVertLevels","maxEdges","nEdges","nVertices"))
    out=F(np.full((nz,nc+1),-999.0))
    args=[h[name] for name in ("nEdgesOnCell","cellsOnCell","edgesOnCell","cellsOnEdge","verticesOnCell",
        "cellsOnVertex","kiteAreasOnVertex","areaCell","dvEdge","edgeNormalVectors","cellTangentPlane",
        "localVerticalUnitVectors","zgrid")]
    native_library(library)("replay_mesh",nc,nz,me,nv,ne,*args,nc,nz,
        f["uReconstructX"],f["uReconstructY"],f["uReconstructZ"],w,f["vorticity"],f["rho"],f["theta"],out)
    return out[:,:nc].copy()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("strict",type=Path)
    p.add_argument("debug_initialized",type=Path)
    p.add_argument("receipt",type=Path)
    a=p.parse_args()
    strict=run(a.strict);debug=run(a.debug_initialized)
    receipt={"defect":"MPAS-PV-TOP-UNDEFINED", "source":"mpas_pv_diagnostics.F:1032",
        "strict_recipe":"-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations",
        "debug_only_extra_flag":"-finit-real=snan",
        "strict_top_words":[hex(int(v)) for v in strict[-1].view(np.uint32)],
        "debug_top_words":[hex(int(v)) for v in debug[-1].view(np.uint32)],
        "debug_top_nan_count":int(np.count_nonzero(np.isnan(debug[-1]))),
        "defined_levels_different_words":int(np.count_nonzero(strict[:-1].view(np.uint32)!=debug[:-1].view(np.uint32))),
        "strict_library_sha256":hashlib.sha256(a.strict.read_bytes()).hexdigest(),
        "debug_library_sha256":hashlib.sha256(a.debug_initialized.read_bytes()).hexdigest(),
        "whole_forecast_proof":False}
    a.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    assert receipt["defined_levels_different_words"]==0
    assert receipt["debug_top_nan_count"]==4
