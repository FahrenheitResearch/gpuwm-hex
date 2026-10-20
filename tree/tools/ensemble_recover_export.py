"""Recover export after a completed ensemble forecast without rerunning it."""
import argparse
import json
from pathlib import Path
import time

from hexcore.ensemble import (EnsembleRefusal, digest, export_ensemble,
                              read_manifest, validate_inputs, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--io-workers", type=int, default=8)
    a = p.parse_args(); root = a.run.resolve()
    receipt = json.loads((root / "ensemble-receipt.json").read_text())
    if len(receipt["jobs"]) != receipt["members"] or any(j["returncode"] for j in receipt["jobs"]):
        raise EnsembleRefusal("export recovery needs every forecast completed successfully; failed members must be rerun")
    if digest(receipt["manifest"]) != receipt["manifest_sha256"]:
        raise EnsembleRefusal("manifest changed since forecast; cannot attribute recovered states")
    data = read_manifest(receipt["manifest"],receipt["members"],receipt["resolution_km"])
    jobs = {j["id"]:j for j in receipt["jobs"]}
    for row in data["members"]:
        job = jobs[row["id"]]
        row["init"] = job["command"][job["command"].index("--init")+1]
        if digest(row["init"]) != job["init_sha256"]:
            raise EnsembleRefusal(f"{row['id']}: init changed since forecast")
    validate_inputs(data)
    backup = root / "before-export-recovery.json"
    if not backup.exists():
        write_json(backup,receipt)
    receipt["recovered_export"] = True
    receipt["previous_error"] = receipt.pop("error",None)
    receipt["status"] = "recovering_export"
    started = time.monotonic()
    try:
        export_ensemble(data,root,receipt,receipt["cycle_hours"],a.io_workers)
        receipt["status"] = "forecast_and_native_export_passed"
        receipt["measured_phase_sum_seconds"] = receipt["forecast_wall_seconds"] + time.monotonic()-started
        receipt["total_wall_seconds"] = None
        receipt["timing_note"] = "forecast and recovered export measured separately; no continuous end-to-end time claimed"
    except BaseException as error:
        receipt["status"] = "failed"; receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(root / "ensemble-receipt.json",receipt)


if __name__ == "__main__":
    main()
