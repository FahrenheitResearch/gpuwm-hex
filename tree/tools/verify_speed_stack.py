"""Run the declared CPU battery with explicit source and native dependencies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("L"):
        raise ValueError("export L for the owned lane; verification cannot place receipts without an explicit owner")
    lane = Path(os.environ["L"]).resolve()
    output = args.output.resolve()
    if not output.is_relative_to(lane):
        raise ValueError("verification receipts must stay inside the owned lane")
    if not 1 <= args.workers <= 16:
        raise ValueError("verification needs 1..16 shared CPU workers")
    files = [line.strip() for line in (root / "tools/battery/cpu_files.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    for name in files:
        path = (root / name).resolve()
        if not path.is_relative_to(root / "tests") or not path.is_file():
            raise ValueError(f"declared test file is absent or outside tests: {name}")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "-n", str(args.workers),
               "-m", "not gpu and not bigcard and not assets", *files,
               "--basetemp", str(output / "pytest")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    keys = ("PYTHONPATH", "GPUWM_HEX_RW_MPAS_HOSTPREP", "GPUWM_HEX_RW_MPAS_LBC",
            "GPUWM_PROJECTED_INTERMEDIATE_BRIDGE", "GPUWM_HEX_CONTRACT_HOSTPREP",
            "GPUWM_CPU_PREPROCESS_BRIDGE")
    document = {"argv": command, "cwd": str(root),
                "environment": {key: env.get(key) for key in keys},
                "test_files": len(files), "GPUWM_NO_LOCAL_GPU": "1"}
    (output / "command.json").write_text(json.dumps(document, indent=2) + "\n")
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    document["returncode"] = result.returncode
    (output / "command.json").write_text(json.dumps(document, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
