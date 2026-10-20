"""Remove recorded ensemble simulation files, retaining logs and receipts.

This command consumes a finished run. Never point it at another lane's run.
It checks exact containment, never follows links, and records each size before
unlinking. Directory traversal is limited to member history/scratch folders
which this ensemble door exclusively creates.
"""
import argparse
import json
from pathlib import Path

from hexcore.ensemble import SCHEMA, EnsembleRefusal, write_json


def dispose(root):
    root = Path(root).resolve()
    receipt = json.loads((root / "ensemble-receipt.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("status") not in ("failed", "forecast_and_native_export_passed"):
        raise EnsembleRefusal("disposal needs a finished ensemble receipt; deleting active member state would break its forecast or consumer")
    targets = set(root.glob("mpas.*.nc"))
    for job in receipt["jobs"]:
        name = job["id"]
        if not name or Path(name).name != name or name in (".",".."):
            raise EnsembleRefusal("invalid member identity in disposal receipt")
        for kind in ("history", "scratch"):
            directory = root / name / kind
            if directory.is_symlink():
                raise EnsembleRefusal("member directory is a link; refusing traversal outside its owned run")
            if directory.exists():
                targets.update(p for p in directory.rglob("*")
                               if p.suffix in (".nc", ".npz", ".npy", ".bin", ".raw", ".so"))
    # Recentered init files were created by this run, unlike the input manifest.
    if (root / "recentered").is_dir() and not (root / "recentered").is_symlink():
        targets.update((root / "recentered").glob("*.nc"))
    deleted = {"schema":SCHEMA,"source_receipt":str(root / "ensemble-receipt.json"),"files":[]}
    log = root / "disposal-receipt.json"
    for path in sorted(targets):
        if not path.absolute().is_relative_to(root):
            raise EnsembleRefusal(f"disposal path escapes run: {path}")
        if not path.is_symlink() and not path.resolve().is_relative_to(root):
            raise EnsembleRefusal(f"disposal target escapes run: {path}")
        if path.is_dir():
            continue
        size = path.lstat().st_size
        entry = {"path":str(path),"bytes":size,"removed":False}
        deleted["files"].append(entry)
        write_json(log, deleted)
        path.unlink()
        entry["removed"] = True
    deleted["total_bytes"] = sum(row["bytes"] for row in deleted["files"])
    write_json(log, deleted)
    print(json.dumps({"files":len(deleted["files"]),"bytes":deleted["total_bytes"],"receipt":str(log)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    dispose(parser.parse_args().run)
