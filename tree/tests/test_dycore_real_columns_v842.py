"""Real 55-level columns, synthetic module stencil; no forecast claim."""
from pathlib import Path
import json
import os
import pytest
from test_dycore_options_v842 import (
    read_inputs, device_d1, cpu_receipt_rows,
    test_complete_nonles_module_bits,
    test_native_cam_and_rayleigh_bits,
    test_rust_upper_matches_extracted_native_driver_bits,
)


@pytest.fixture(scope="module")
def native_d1():
    folder=os.environ.get("HEXCORE_NATIVE_D1_REAL_FIXTURE")
    if not folder:
        pytest.skip("set HEXCORE_NATIVE_D1_REAL_FIXTURE to the Rust-carried real fixture")
    folder=Path(folder)
    metadata=json.loads((folder.parent/"columns"/"real-columns.json").read_text())
    data=read_inputs(folder)
    assert data["rho"].shape[0] == metadata["levels"]
    return folder,data
