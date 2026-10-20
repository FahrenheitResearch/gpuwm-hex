"""Resident-reference ABI replay against the actual strict GNU oracle."""
import os
from pathlib import Path
import numpy as np
import pytest

from hexcore.transport_reference_v842 import prepare_nonles_angles


def test_reference_carrier_precision_is_checked_before_execution():
    with pytest.raises(TypeError):
        prepare_nonles_angles(np.zeros(4, dtype=np.float64))
    with pytest.raises(TypeError):
        prepare_nonles_angles(np.zeros((2, 2), dtype=np.float32))


def test_reference_angle_rust_abi_matches_native_words():
    root = os.environ.get("D1_NATIVE_ANGLE_FIXTURE")
    binary = os.environ.get("D1_NATIVE_ANGLE_ENGINE")
    if not root or not binary:
        pytest.skip("strict native reference-angle fixture not selected")
    path = Path(root)
    # Native binary data is read solely for a replay assertion; production
    # netCDF decoding and transcendental processing are owned by Rust.
    angles = np.fromfile(path / "angles.bin", dtype="<f4")
    cos, sin, provenance = prepare_nonles_angles(angles, engine=Path(binary))
    for name, value in (("native-cos.bin", cos), ("native-sin.bin", sin)):
        assert value.tobytes() == (path / name).read_bytes()
    assert "GPUWM_BRIDGE_SOURCE_REV=" in provenance["abi"]
    assert len(provenance["binary_sha256"]) == 64
