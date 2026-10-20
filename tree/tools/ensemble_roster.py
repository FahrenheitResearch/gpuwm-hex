"""Build a manifest from distinct prepared MPAS member files, including EnKF."""
import argparse
from pathlib import Path

from hexcore.ensemble import SCHEMA, EnsembleRefusal, validate_inputs, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--pattern", default="member-*/init.nc")
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--source", required=True)
    p.add_argument("--start-time", required=True)
    p.add_argument("--resolution-km", choices=(60,30), type=int, default=60)
    p.add_argument("--mesh", required=True)
    for key in ("grid", "static", "mesh-rows", "native-template", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    paths = sorted(a.prepared.resolve().glob(a.pattern))[:a.members]
    if len(paths) != a.members or a.members < 2:
        raise EnsembleRefusal(f"need {a.members} distinct prepared members, found {len(paths)}")
    if a.out.exists():
        raise EnsembleRefusal(f"roster exists: {a.out}")
    data = {"schema": SCHEMA, "resolution_km": a.resolution_km, "mesh":a.mesh,
            **{k:str(getattr(a,k).resolve()) for k in ("grid", "static", "mesh_rows", "native_template")},
            "start_time":a.start_time, "forecast_flags": [],
            "members":[{"id":path.parent.name,"init":str(path),"source":a.source + "; " + path.parent.name}
                       for path in paths]}
    validate_inputs(data)
    write_json(a.out, data)


if __name__ == "__main__":
    main()
