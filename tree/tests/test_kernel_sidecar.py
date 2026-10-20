import json
from pathlib import Path
import subprocess
import sys

from hexcore.cuda_backend.regional_admission import kernel_set_sha256


def test_build_hook_binds_the_actual_imported_package(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/write_kernel_sidecar.py"
    output = tmp_path / "kernel-set.json"
    result = subprocess.run([sys.executable, str(script), "--package", "hexcore", "--output", str(output)],
                            capture_output=True, text=True, check=True)
    document = json.loads(output.read_text())
    assert document == json.loads(result.stdout)
    assert document == {"schema": "gpuwm-hex.kernel-set/v1", "kernel_set_sha256": kernel_set_sha256()}
