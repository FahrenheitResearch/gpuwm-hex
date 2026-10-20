#!/usr/bin/env python3
"""Use the existing C3 collector, adding matched composite plus health timing."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    lane = Path(os.environ["L"])
    runtime = Path(os.environ["STEP_BATCH_RUNTIME"])
    path = runtime / "tools/bench/run_hex_case.py"
    spec = importlib.util.spec_from_file_location("run_hex_case", path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    receipt = output / "step-timing.json"
    native_call = subprocess.call
    def call(command, *values, **kwargs):
        if command[:4] != [sys.executable, "-m", "hexcore", "forecast"]:
            raise ValueError("collector expected the real forecast door")
        command = [sys.executable, str(lane / "src/tree/tools/check_step_gate_batch.py"),
                   "--receipt", str(receipt), "--", *command[3:]]
        return native_call(command, *values, **kwargs)
    collector.subprocess.call = call
    rc = collector.main()
    if rc == 0:
        metrics_path = output / "hex-metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["host_step_median_seconds"] = json.loads(receipt.read_text())["host_step_median_seconds"]
        metrics["host_step_basis"] = "same observer of composite plus health, both arms"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
