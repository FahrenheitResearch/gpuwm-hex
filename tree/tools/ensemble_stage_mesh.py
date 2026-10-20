"""Stage one stock MPAS mesh under a caller-owned shared-prepared root.

CPU only. Reuse a completed receipt; flock prevents duplicate preparations.
Archive paths are checked before extraction. Geometry admissions remain in
the normal hex forecast bind, rather than a made-up qualification receipt.
"""
import argparse
from dataclasses import replace
import fcntl
from pathlib import Path
import tarfile

from hexcore import mesh_rows
from hexcore.ensemble import EnsembleRefusal, digest, write_json
from hexcore.ensemble_prepare import download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-km", choices=(60,30), type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve(); root.mkdir(parents=True, exist_ok=True)
    stock = {60:"x1.163842",30:"x1.655362"}[args.resolution_km]
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ((root / "mesh-rows.json").exists() and
                all((root / f"{stock}.{role}.nc").is_file() for role in ("grid","static"))):
            print(root / "mesh-rows.json")
            return
        receipt = {"stock":stock,"resolution_km":args.resolution_km,"downloads":[],"deleted":[],
                   "licence":"MPAS-Atmosphere mesh distribution; MPAS code BSD-3-Clause; data source recorded"}
        for suffix in ("", "_static"):
            role = "grid" if not suffix else "static"
            if (root / f"{stock}.{role}.nc").is_file():
                receipt.setdefault("reused",[]).append(str(root / f"{stock}.{role}.nc"))
                continue
            name = stock + suffix + ".tar.gz"
            archive = root / name
            receipt["downloads"].append(download("https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes/" + name, archive))
            with tarfile.open(archive) as tar:
                for entry in tar.getmembers():
                    destination = (root / entry.name).resolve()
                    if (not destination.is_relative_to(root) or entry.issym() or entry.islnk() or
                            not (entry.isfile() or entry.isdir())):
                        raise EnsembleRefusal(f"unsafe mesh archive entry {entry.name}")
                tar.extractall(root, filter="data")
            receipt["deleted"].append({"path":str(archive),"bytes":archive.stat().st_size})
            write_json(root / "mesh-receipt.json", receipt)
            archive.unlink()
        spec = root / "mesh-receipt.json"
        row = replace(mesh_rows.describe_generated(
            name=f"ensemble-global{args.resolution_km}-native", grid=root / f"{stock}.grid.nc",
            static=root / f"{stock}.static.nc", spec_path=spec, generator_receipt=spec,
            static_receipt=spec, point_deg=(0,0), fine_dx_m=args.resolution_km*1000,
            core_radius_km=0, background_km=args.resolution_km, dt_seconds=args.resolution_km*6,
            n_levels=55, admission={"scope":"stock archive; normal forecast bind measures mesh geometry"}),
            kind=mesh_rows.ARCHIVED_GLOBAL_KIND)
        mesh_rows.write_rows(root / "mesh-rows.json", [row])
        print(root / "mesh-rows.json")


if __name__ == "__main__":
    main()
