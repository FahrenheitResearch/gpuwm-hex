"""Compare corresponding physical members between two packing/timing arms."""
import argparse
import json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from hexcore.ensemble import FIELDS, checked_array, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("left",type=Path); p.add_argument("right",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    left=json.loads((a.left / "ensemble-receipt.json").read_text())
    right=json.loads((a.right / "ensemble-receipt.json").read_text())
    index={(r["member"],r["forecast_hour"]):r for r in right["outputs"]}
    rows=[]
    for item in left["outputs"]:
        other=index[(item["member"],item["forecast_hour"])]
        record={"member":item["member"],"forecast_hour":item["forecast_hour"],"variables":{}}
        with Dataset(item["path"]) as x,Dataset(other["path"]) as y:
            for name in (*FIELDS,"skintemp","tslb","smois"):
                maximum=0.0; equal=True
                var=x[name]; axis=var.dimensions.index("nEdges" if "nEdges" in var.dimensions else "nCells")
                for start in range(0,var.shape[axis],8192):
                    key=[slice(None)]*len(var.dimensions); key[axis]=slice(start,start+8192)
                    xv,yv=checked_array(var,tuple(key)),checked_array(y[name],tuple(key))
                    equal=equal and bool(np.array_equal(xv,yv))
                    maximum=max(maximum,float(np.max(np.abs(xv.astype(np.float64)-yv))))
                record["variables"][name]={"identical":equal,"max_abs_difference":maximum}
        rows.append(record)
    result={"left_receipt":str(a.left / "ensemble-receipt.json"),"right_receipt":str(a.right / "ensemble-receipt.json"),
            "states":rows,"states_compared":len(rows),
            "all_physical_arrays_identical":all(v["identical"] for r in rows for v in r["variables"].values())}
    write_json(a.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!="states"}))


if __name__=="__main__":
    main()
