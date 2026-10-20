"""Strict CUDA source assembly and authentication for precompiled artifacts."""
from hexcore import numerics as _hex_numerics
import hashlib
import json
import os
from pathlib import Path

OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
           "--prec-div=true", "--prec-sqrt=true")


def cuda_source():
    from gpuwm.core import kernels
    from ._kernels import CUDA_SOURCE
    root = Path(kernels.__file__).parent
    return "\n".join((root / name).read_text() for name in
                     ("glibc_flt32.cuh", "flt32_expf_fma.cuh")) + "\n" + CUDA_SOURCE


def authenticate(path, source, architecture):
    """Prevent stale source, changed flags or corrupted CUDA artifact loading."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    fmt = manifest.get("format", "ptx")
    if fmt not in ("ptx", "cubin"):
        raise ValueError(f"diagnostics CUDA artifact format mismatch: {path}")
    expected = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    options=[*_hex_numerics.options(OPTIONS), "--gpu-architecture=" + architecture])
    expected[fmt + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"diagnostics CUDA artifact {key} mismatch: {path}")
    return path


def artifact_architecture(path, capability):
    """Select the exact device architecture without JIT of unsupported PTX."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    fmt = manifest.get("format", "ptx")
    if fmt not in ("ptx", "cubin"):
        raise ValueError(f"diagnostics CUDA artifact format mismatch: {path}")
    return ("sm_" if fmt == "cubin" else "compute_") + capability


def artifact_receipt():
    path = os.environ.get("HEX_MPASDIAG_CUDA_ARTIFACT")
    if path:
        path = Path(path)
        return json.loads(path.with_suffix(path.suffix + ".json").read_text())
    return None
