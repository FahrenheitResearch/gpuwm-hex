"""Rust/GNU native reference carriers; mpas_atm_dissipation_models.F:818-860.

The initialized angleEdge carrier is decoded by the existing Rust mesh reader.
This launcher transfers its storage to Rust's shared glibc 2.43 math routines.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import subprocess
import numpy as np

from .engines import EngineRefusal, EngineSpec, resolve

ANGLES = EngineSpec(
    name="rw_mpas_nonles_angles", flag="non-LES reference-angle bridge",
    env_names=("GPUWM_HEX_RW_MPAS_NONLES_ANGLES",),
    gpuwm_env="GPUWM_RW_MPAS_NONLES_ANGLES", subject="GNU RKIND reference angles",
    what_breaks="vertical perturbation momentum diffusion requires native cos(angleEdge) and sin(angleEdge)",
    cargo_package="rw-mpas")
MARKER = b"rw_mpas_nonles_angles --protocol mpas-nonles-angles-v1"


def prepare_nonles_angles(angle_edge, *, engine: Path | None = None):
    """mpas_atm_dissipation_models.F:859: prepare separate GNU float intrinsics."""
    angle = np.asarray(angle_edge)
    if angle.ndim != 1 or angle.dtype != np.float32:
        raise TypeError("native angleEdge carrier must be one-dimensional float32")
    binary = resolve(ANGLES, explicit=engine)
    abi = subprocess.run([str(binary), "--abi"], capture_output=True, check=True).stdout
    if MARKER not in abi:
        raise EngineRefusal("reference-angle bridge lacks the native RKIND ABI marker")
    with binary.open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    carrier = np.ascontiguousarray(angle, dtype="<f4")
    request = struct.pack("<8sQ", b"NONLES01", carrier.size) + memoryview(carrier).cast("B").tobytes()
    result = subprocess.run([str(binary), "--protocol", "mpas-nonles-angles-v1"],
                            input=request, capture_output=True, check=True)
    if len(result.stdout) != carrier.nbytes * 2:
        raise EngineRefusal("reference-angle bridge returned incomplete or excess carriers")
    cos = np.empty(carrier.shape, dtype="<f4")
    sin = np.empty(carrier.shape, dtype="<f4")
    memoryview(cos).cast("B")[:] = result.stdout[:carrier.nbytes]
    memoryview(sin).cast("B")[:] = result.stdout[carrier.nbytes:]
    return cos, sin, {"binary_sha256": binary_sha256, "abi": abi.decode().strip()}
