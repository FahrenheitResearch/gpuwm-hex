"""Prepare native real-column captures toggling every pinned MYNN option.

This prepares private namelists and directories only. Run the original native
capture executable on CPU; it writes field records without Python decoding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from hexcore.mpasphys.bl_mynn_v842 import MynnOptions
from prepare_real_native import prepare


def matrix(args):
    baseline = vars(MynnOptions()) | {"edmf_dd": 1, "edmf_output": 1, "tkebudget": 1}
    rows = []
    for name, current in baseline.items():
        value = {"closure": 3.0, "cloudpdf": 0, "mixlength": 0, "stfunc": 2}.get(name)
        if value is None:
            value = not current if isinstance(current, bool) else 1 - current
        option = "config_mynn_" + name
        work = args.work / name
        prepare(work, args.native, args.input, args.template, args.registry,
                args.mesh, args.mesh_root, args.dt, args.duration)
        path = work / "namelist.atmosphere"
        source = path.read_text()
        for field, selected in baseline.items():
            if field == name:
                selected = value
            key = "config_mynn_" + field
            rendered = (".true." if selected else ".false.") if isinstance(selected, bool) else str(selected)
            source = re.sub(r"^\s*" + key + r"\s*=.*$", "", source, flags=re.M)
            marker = 'config_physics_suite = "none"'
            assert source.count(marker) == 1
            source = source.replace(marker, marker + f"\n    {key} = {rendered}")
        path.write_text(source)
        # The original preparer's receipt was for its initial baseline; seal
        # the final deck hash and explicit option values after mutation.
        receipt = json.loads((work / "probe.json").read_text())
        receipt["namelist_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        receipt["option_baseline"] = baseline
        receipt["option_toggle"] = {option: value}
        (work / "probe.json").write_text(json.dumps(receipt, indent=2) + "\n")
        rows.append({"option": option, "baseline": current, "selected": value,
                     "work": str(work.resolve()), "dump": str((args.work / (name + "-dump")).resolve()),
                     "namelist_sha256": receipt["namelist_sha256"]})
    (args.work / "matrix.json").write_text(json.dumps({"scope": "Native column capture option matrix, no forecast acceptance", "decks": rows}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "native", "input", "template", "registry", "mesh-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--dt", type=int, required=True)
    parser.add_argument("--duration", required=True)
    matrix(parser.parse_args())
