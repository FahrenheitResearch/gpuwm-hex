"""Report native-vs-native build spread without converting it into a tolerance."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("tree","init","strict","optimized","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.tree/"src"))
    from hexcore.driver import load_mpas_vertical_grid,load_mpas_initial_state
    from hexcore.mesh import Mesh
    from compare_dry_identity import compare
    mesh=Mesh.from_netcdf(a.init)
    grid=load_mpas_vertical_grid(a.init,mesh,config_coef_3rd_order=0.25)
    def load(root):
        paths=sorted(root.glob("tracer.*.nc"))
        if len(paths)<2: raise ValueError("missing native step one")
        return load_mpas_initial_state(paths[1],mesh,grid.vertical_grid,scalar_names=("qv",),
              reference_path=paths[1],terrain_metrics=grid.terrain_metrics,return_saved_diagnostics=True)
    strict,optimized=load(a.strict),load(a.optimized)
    fields=compare(strict[0],strict[2],optimized[0],optimized[2])
    result={"schema":"gpuwm-hex.native-build-spread/v1","strict":str(a.strict),
            "optimized":str(a.optimized),"fields":fields}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Native strict vs optimized, step one")
    for name,row in fields.items():
        print(f'{name:27} {row.get("different_words",0)}/{row["words"]} max ULP={row.get("max_ulp",0)} max abs={row.get("max_abs",0):.8g}')


if __name__=="__main__": main()
