"""Measure pristine strict unknown-overlap behavior, with both radiations active.

Native-only defect diagnostics, not CUDA forecast acceptance. Each process
count has its own repeated reference. Rust reads every field before retirement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    a.work.mkdir(parents=True, exist_ok=False)
    a.receipts.mkdir(parents=True, exist_ok=True)
    source = Path(plan["source_deck"])
    source_text = (source / "namelist.atmosphere").read_text()
    text, count = re.subn(r"(config_run_duration\s*=\s*)'[^']+'",
                         r"\1'0_01:00:00'", source_text)
    assert count == 1
    source_index = json.loads((source.parent / "INDEX.json").read_text())
    source_index["runtime"]["duration"] = "0_01:00:00"
    (a.work / "INDEX.json").write_text(json.dumps(source_index, indent=2) + "\n")
    rows = []
    for option in ("none", "unknown", "maximum_random"):
        deck = a.work / option
        deck.mkdir()
        configured, count = re.subn(r"(config_radt_cld_overlap\s*=\s*)'[^']+'",
                                   lambda m: m[1] + repr(option), text)
        if count == 0:
            configured, count = re.subn(r"(?m)^(&physics\s*)$",
                lambda m: m[0] + "\n config_radt_cld_overlap = " + repr(option), text)
        assert count == 1
        (deck / "namelist.atmosphere").write_text(configured)
        shutil.copy2(source / "streams.atmosphere", deck / "streams.atmosphere")
        for ranks in (1, 2):
            for repeat in ((1,) if option == "maximum_random" else (1, 2)):
                name = f"{option}-rank{ranks}-repeat{repeat}"
                directory = a.work / name
                argv = [sys.executable, "hex/tree/tools/native_physics_reference/run_forecast.py"]
                for key in ("build", "tables", "mpi", "pin-set-hash"):
                    argv += ["--" + key, plan[key]]
                argv += ["--deck", str(deck), "--case", plan["case"],
                         "--out", str(directory), "--ranks", str(ranks),
                         "--digest-only", "--comparator", plan["comparator"],
                         "--inactive-package", "jedi_da"]
                with (a.receipts / (name + ".log")).open("w") as log:
                    result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
                row = dict(name=name, option=option, process_count=ranks,
                           repeat=repeat, host=socket.gethostname(), native_exit=result.returncode,
                           namelist_sha256=sha(deck / "namelist.atmosphere"),
                           streams_sha256=sha(deck / "streams.atmosphere"))
                if result.returncode == 0:
                    with (a.receipts / (name + "-verify.log")).open("w") as log:
                        checked = subprocess.run([sys.executable,
                            "hex/tree/tools/native_physics_reference/verify_products.py",
                            str(directory), "--comparator", plan["comparator"],
                            "--inactive-package", "jedi_da", "--receipt",
                            str(a.receipts / (name + "-clocks.json"))],
                            stdout=log, stderr=subprocess.STDOUT)
                    row["clock_exit"] = checked.returncode
                    for file in ("MANIFEST.json", "CLOCKS.json", "log.atmosphere.0000.out"):
                        original = directory / file
                        assert original.stat().st_size < 10_000_000
                        shutil.copy2(original, a.receipts / (name + "-" + file))
                    row["manifest_sha256"] = sha(directory / "MANIFEST.json")
                    row["field_digests_sha256"] = sha(directory / "CLOCKS.json")
                row["complete"] = result.returncode == 0 and row.get("clock_exit") == 0
                rows.append(row)
                (a.receipts / "matrix.json").write_text(json.dumps(dict(
                    purpose="native unknown-overlap defect measurement, both RRTMG spectra active",
                    plan_sha256=sha(a.plan), source_namelist_sha256=sha(source / "namelist.atmosphere"),
                    source_streams_sha256=sha(source / "streams.atmosphere"),
                    rows=rows, complete=len(rows) == 10 and all(r["complete"] for r in rows),
                    cuda_forecast_identity_claim=False), indent=2) + "\n")
                print(json.dumps(row), flush=True)
    return 0 if all(row["complete"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
