"""Run focused device checks only inside a single explicit card grant."""
from pathlib import Path
import json
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    lane = root.parents[1]
    physical = os.environ.get("GPU_MUTEX_CARDS", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical.isdecimal() or not visible or "," in visible:
        raise ValueError("device checks need an explicit single-card mutex grant; unqueued GPU use is refused")
    if visible.isdecimal() and visible != physical:
        raise ValueError("physical and visible card indices disagree; the checks could use another card")
    companion = lane / "runtime-src"
    sources = [str(root / "src"), str(root / "tests")]
    if (companion / "gpuwm/kernel_pack.py").is_file():
        sources.append(str(companion))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sources),
               GPUWM_CARD=physical, GPUWM_NO_LOCAL_GPU="0", GPUWM_HEX_NO_LOCAL_GPU="0",
               CUPY_CACHE_DIR=str(lane / "cache/gpu-tests/cupy"),
               CUDA_CACHE_PATH=str(lane / "cache/gpu-tests/cuda"))
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_cuda_subcycle_graphs.py",
               "tests/test_hex_aot_gpu.py",
               "tests/test_step_gate_batch.py", "tests/test_late_health_pipeline.py",
               "--basetemp", str(lane / "pytest-gpu")]
    output = lane / "receipts/gpu-focused"
    output.mkdir(parents=True, exist_ok=True)
    record = {"argv": command, "cwd": str(root), "physical_card": physical,
              "cuda_visible_devices": visible}
    with (output / "pytest.log").open("w") as log:
        done = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    record["returncode"] = done.returncode
    (output / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
