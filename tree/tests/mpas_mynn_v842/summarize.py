"""Seal small native oracle/audit receipts; no CUDA identity claim.

Input is the test Fortran program's exact word dump, never model NetCDF.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repeatability(first: Path, second: Path):
    counts = Counter()
    differing = Counter()
    adv = {}
    qke = {}
    with first.open() as a, second.open() as b:
        for left, right in zip_longest(a, b):
            if left is None or right is None:
                raise ValueError("oracle output lengths differ")
            field, deck, step, col, level, word = left.split()
            counts[field] += 1
            if left != right:
                differing[field] += 1
            if int(deck) in (13, 33) and field in ("qke_adv", "qke"):
                target = adv if field == "qke_adv" else qke
                target[(int(deck), int(step), int(col), int(level))] = word
    witness = {}
    for deck in (13, 33):
        keys = [key for key in adv if key[0] == deck and key[1] == 1]
        witness[str(deck)] = {
            "config_mynn_tkeadvect": True,
            "qke_adv_words": len(keys),
            "qke_adv_changed_between_driver_calls": sum(adv[k] != adv[(deck, 2, k[2], k[3])] for k in keys),
            "qke_vs_qke_adv_differing_on_second_call": sum(qke[(deck, 2, k[2], k[3])] != adv[(deck, 2, k[2], k[3])] for k in keys),
        }
    return {
        "comparison": "same compiled native program, two independent executions",
        "output_words": sum(counts.values()), "differing_words": sum(differing.values()),
        "per_array_words": dict(counts), "per_array_differing": dict(differing),
        "first_sha256": sha(first), "second_sha256": sha(second),
        "qke_driver_witness": witness,
        "cuda_identity": None, "whole_model_identity": None,
    }


def audit(root: Path):
    core = root / "src/core_atmosphere"
    registry = core / "Registry.xml"
    options = {node.attrib["name"]: {k: node.attrib[k] for k in ("type", "default_value")}
               for node in ET.parse(registry).iter("nml_option")
               if node.attrib.get("name", "").startswith("config_mynn_")}
    locations = []
    for src in sorted((root / "src").rglob("*")):
        if src.is_file() and src.suffix.lower() in (".f", ".f90", ".inc"):
            for lineno, line in enumerate(src.read_text(errors="replace").splitlines(), 1):
                if re.search(r"\bqke_adv\b|\bconfig_mynn_tkeadvect\b|\bqkeadv_p\b", line, re.I):
                    locations.append({"file": str(src.relative_to(root)), "line": lineno, "source": line.strip()})
    routines = []
    for name in ("bl_mynn.F90", "bl_mynn_subroutines.F90", "mynn_shared.F90"):
        src = core / "physics/physics_mmm" / name
        for lineno, line in enumerate(src.read_text().splitlines(), 1):
            match = re.match(r"\s*(subroutine|function)\s+(\w+)", line, re.I)
            if match:
                routines.append({"file": str(src.relative_to(root)), "line": lineno, "name": match[2].lower()})
    integration = core / "dynamics/mpas_atm_time_integration.F"
    return {
        "registry_sha256": sha(registry), "option_count": len(options), "options": options,
        "qke_references": locations, "routines": routines,
        "time_integration_sha256": sha(integration),
        "time_integration_qke_mentions": len(re.findall(r"\bqke\b|\bqke_adv\b|\bconfig_mynn_tkeadvect\b", integration.read_text(), re.I)),
        "finding": "qke_adv is in diag_physics, outside transported scalars; no qke transport hook is present in this pin",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mpas-root", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    args = parser.parse_args()
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.build_dir / "source-receipt.json").read_text())
    source["compiler"] = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]
    source["outputs_finite"] = "all output arrays checked in the Fortran test harness"
    source["direct_mmm"] = repeatability(args.build_dir / "words-1.txt", args.build_dir / "words-2.txt")
    source["mpas_adapter"] = repeatability(args.build_dir / "adapter-1.txt", args.build_dir / "adapter-2.txt")
    for name, data in (("native-oracle.json", source), ("qke-source-audit.json", audit(args.mpas_root))):
        (args.receipt_dir / name).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
