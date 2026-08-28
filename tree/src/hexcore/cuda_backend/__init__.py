"""CUDA backend for the MPAS Python port."""

from .containers import (
    DeviceAtmosphere,
    DeviceMesh,
    DevicePrognosticState,
    DeviceReferenceState,
    DeviceSavedDiagnostics,
    DeviceTerrainMetrics,
    DeviceVerticalGrid,
    TransferStats,
    require_resident_array,
)
from .runtime import (
    CudaCapability,
    CudaRefusal,
    KernelCache,
    KernelTiming,
    launch_timed,
    require_cuda,
)
from .compile_contract import (
    COMPILE_MANIFEST_SCHEMA,
    CompileContractError,
    canonical_sha256,
    compile_platform_binding,
    resolve_compile_platform_fingerprint,
    source_sha256,
    validate_compile_platform_fingerprint,
)
from .recovery import DeviceRecoveredState, RECOVERY_CUDA_SOURCE, recover_state


__all__ = [
    "CudaCapability",
    "CudaRefusal",
    "COMPILE_MANIFEST_SCHEMA",
    "CompileContractError",
    "DeviceAtmosphere",
    "DeviceMesh",
    "DevicePrognosticState",
    "DeviceReferenceState",
    "DeviceRecoveredState",
    "DeviceSavedDiagnostics",
    "DeviceTerrainMetrics",
    "DeviceVerticalGrid",
    "KernelCache",
    "KernelTiming",
    "RECOVERY_CUDA_SOURCE",
    "TransferStats",
    "canonical_sha256",
    "compile_platform_binding",
    "launch_timed",
    "require_cuda",
    "require_resident_array",
    "resolve_compile_platform_fingerprint",
    "recover_state",
    "source_sha256",
    "validate_compile_platform_fingerprint",
]
