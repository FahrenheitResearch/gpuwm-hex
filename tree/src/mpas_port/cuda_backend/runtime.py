"""CUDA runtime, capability, compilation-cache, and launch authority.

CuPy's process-global default cache can serialize unrelated NVRTC jobs on
Windows.  This module gives every process an isolated cache directory unless
``MPAS_PORT_CUDA_CACHE_DIR`` or ``CUPY_CACHE_DIR`` is explicitly declared.
The in-process :class:`KernelCache` then owns compiled RawModule functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

from .compile_contract import (
    CompileContractError,
    CompileManifest,
    NvrtcCompileCapture,
    compile_platform_binding,
    validate_module_key,
)


class CudaRefusal(RuntimeError):
    """A named CUDA capability or runtime contract was not satisfied."""


def _cupy() -> Any:
    try:
        import cupy as cp
    except Exception as error:  # pragma: no cover - exercised on non-CUDA hosts
        raise CudaRefusal(f"CuPy CUDA backend is unavailable: {error}") from error
    return cp


def _cache_directory(device_id: int, compute: tuple[int, int], raw: str | Path | None) -> Path:
    declared = raw
    if declared is None:
        declared = os.environ.get("MPAS_PORT_CUDA_CACHE_DIR")
    if declared is None:
        declared = os.environ.get("CUPY_CACHE_DIR")
    if declared is None:
        declared = (
            Path(tempfile.gettempdir())
            / "mpas-port-cupy-cache"
            / f"pid-{os.getpid()}-device-{device_id}-sm_{compute[0]}{compute[1]}"
        )
    result = Path(declared).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    os.environ["CUPY_CACHE_DIR"] = str(result)
    return result


@dataclass(frozen=True, slots=True)
class CudaCapability:
    device_id: int
    name: str
    compute_major: int
    compute_minor: int
    total_memory_bytes: int
    multiprocessor_count: int
    runtime_version: int
    driver_version: int
    nvrtc_version: tuple[int, int]
    cupy_version: str
    cache_directory: str

    @property
    def compute(self) -> tuple[int, int]:
        return (self.compute_major, self.compute_minor)

    @property
    def sm(self) -> str:
        return f"sm_{self.compute_major}{self.compute_minor}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "compute_capability": f"{self.compute_major}.{self.compute_minor}",
            "sm": self.sm,
            "total_memory_bytes": self.total_memory_bytes,
            "multiprocessor_count": self.multiprocessor_count,
            "runtime_version": self.runtime_version,
            "driver_version": self.driver_version,
            "nvrtc_version": list(self.nvrtc_version),
            "cupy_version": self.cupy_version,
            "cache_directory": self.cache_directory,
        }


def require_cuda(
    *,
    device_id: int = 0,
    min_compute: tuple[int, int] = (12, 0),
    required_compute: tuple[int, int] | None = None,
    min_runtime_version: int = 13_000,
    cache_dir: str | Path | None = None,
) -> CudaCapability:
    """Select a real CUDA device or refuse with the failing capability name."""

    cp = _cupy()
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:
        raise CudaRefusal(f"cuda.device_count unavailable: {error}") from error
    if device_id < 0 or device_id >= count:
        raise CudaRefusal(f"cuda.device_id={device_id} is outside [0,{count})")
    cp.cuda.Device(device_id).use()
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    compute = (int(properties["major"]), int(properties["minor"]))
    if compute < tuple(min_compute):
        raise CudaRefusal(
            f"cuda.compute_capability={compute[0]}.{compute[1]} is below "
            f"required {min_compute[0]}.{min_compute[1]}"
        )
    if required_compute is not None and compute != tuple(required_compute):
        raise CudaRefusal(
            f"cuda.compute_capability={compute[0]}.{compute[1]} != required "
            f"{required_compute[0]}.{required_compute[1]}"
        )
    runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    if runtime_version < min_runtime_version:
        raise CudaRefusal(
            f"cuda.runtime_version={runtime_version} < required {min_runtime_version}"
        )
    from cupy.cuda import nvrtc

    supported = tuple(int(value) for value in nvrtc.getSupportedArchs())
    architecture = compute[0] * 10 + compute[1]
    if architecture not in supported:
        raise CudaRefusal(
            f"cuda.nvrtc_arch=compute_{architecture} is not supported; available={supported}"
        )
    cache = _cache_directory(device_id, compute, cache_dir)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return CudaCapability(
        device_id=device_id,
        name=str(name),
        compute_major=compute[0],
        compute_minor=compute[1],
        total_memory_bytes=int(properties["totalGlobalMem"]),
        multiprocessor_count=int(properties["multiProcessorCount"]),
        runtime_version=runtime_version,
        driver_version=int(cp.cuda.runtime.driverGetVersion()),
        nvrtc_version=tuple(int(value) for value in nvrtc.getVersion()),
        cupy_version=str(cp.__version__),
        cache_directory=str(cache),
    )


@dataclass(frozen=True, slots=True)
class KernelTiming:
    compile_seconds: float
    first_launch_ms: float
    first_wall_ms: float
    mean_launch_ms: float
    min_launch_ms: float
    max_launch_ms: float
    repeats: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "compile_seconds": self.compile_seconds,
            "first_launch_ms": self.first_launch_ms,
            "first_wall_ms": self.first_wall_ms,
            "mean_launch_ms": self.mean_launch_ms,
            "min_launch_ms": self.min_launch_ms,
            "max_launch_ms": self.max_launch_ms,
            "repeats": self.repeats,
        }


_KERNEL_COMPILE_SECONDS: dict[int, float] = {}


class KernelCache:
    """In-process RawModule cache with compiler-bound compile inventory.

    ``module_key`` is mandatory at every resolution site.  It names the CUDA
    translation unit in receipts while the internal cache key binds its source,
    requested options, target SM and gpuwm compile-platform fingerprint.
    """

    def __init__(
        self,
        *,
        capability: CudaCapability | None = None,
        cache_dir: str | Path | None = None,
        base_options: Sequence[str] = ("--std=c++17", "--fmad=false"),
    ) -> None:
        self.capability = (
            require_cuda(cache_dir=cache_dir) if capability is None else capability
        )
        selected_cache = _cache_directory(
            self.capability.device_id,
            self.capability.compute,
            cache_dir or self.capability.cache_directory,
        )
        self.cache_directory = selected_cache
        self.base_options = tuple(str(option) for option in base_options)
        try:
            self._compile_manifest = CompileManifest(compile_platform_binding())
        except CompileContractError as error:
            raise CudaRefusal(
                f"cuda.compile_platform_fingerprint refused: {error}"
            ) from error
        self._modules: dict[str, Any] = {}
        self._kernels: dict[str, Any] = {}
        self._compile_seconds: dict[str, float] = {}

    def _module_key(self, source: str, options: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.capability.sm.encode("ascii"))
        digest.update(b"\0compile-platform\0")
        digest.update(self._compile_manifest.fingerprint_sha256.encode("ascii"))
        digest.update(source.encode("utf-8"))
        for option in (*self.base_options, *tuple(options)):
            digest.update(b"\0")
            digest.update(str(option).encode("utf-8"))
        return digest.hexdigest()

    def raw_kernel(
        self,
        name: str,
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> Any:
        try:
            stable_module_key = validate_module_key(module_key)
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.module_key refused: {error}") from error
        cp = _cupy()
        extra_options = tuple(str(value) for value in options)
        compiled_key = self._module_key(source, extra_options)
        requested_options = (
            *self.base_options,
            *extra_options,
        )
        try:
            self._compile_manifest.declare_module(
                stable_module_key,
                source=source,
                requested_options=requested_options,
                module_cache_key=compiled_key,
            )
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.compile_manifest refused: {error}") from error
        kernel_key = f"{compiled_key}:{name}"
        cached = self._kernels.get(kernel_key)
        if cached is not None:
            try:
                self._compile_manifest.record_kernel(stable_module_key, name)
            except CompileContractError as error:
                raise CudaRefusal(
                    f"cuda.compile_manifest refused: {error}"
                ) from error
            return cached
        module = self._modules.get(compiled_key)
        if module is None:
            os.environ["CUPY_CACHE_DIR"] = str(self.cache_directory)
            module = cp.RawModule(
                code=source,
                options=requested_options,
                backend="nvrtc",
                enable_cooperative_groups=False,
            )
        started = time.perf_counter()
        if compiled_key in self._modules:
            kernel = module.get_function(name)
        else:
            with NvrtcCompileCapture() as capture:
                kernel = module.get_function(name)
            try:
                self._compile_manifest.record_effective_compile(
                    stable_module_key,
                    capture.evidence(
                        hashlib.sha256(source.encode("utf-8")).hexdigest()
                    ),
                )
            except CompileContractError as error:
                raise CudaRefusal(
                    f"cuda.compile_manifest refused: {error}"
                ) from error
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - started
        self._modules[compiled_key] = module
        self._kernels[kernel_key] = kernel
        self._compile_seconds[kernel_key] = elapsed
        _KERNEL_COMPILE_SECONDS[id(kernel)] = elapsed
        try:
            self._compile_manifest.record_kernel(stable_module_key, name)
        except CompileContractError as error:
            raise CudaRefusal(f"cuda.compile_manifest refused: {error}") from error
        return kernel

    def raw_kernels(
        self,
        names: Sequence[str],
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Resolve several extern-C functions from one compiled RawModule."""

        selected = tuple(str(name) for name in names)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("raw_kernels names must be non-empty and unique")
        return {
            name: self.raw_kernel(
                name, source, module_key=module_key, options=options
            )
            for name in selected
        }

    def compile_manifest(self) -> dict[str, Any]:
        """JSON-ready snapshot for a CUDA step or long-run receipt."""

        return self._compile_manifest.snapshot()

    def compile_seconds(self, kernel: Any) -> float:
        return float(_KERNEL_COMPILE_SECONDS.get(id(kernel), 0.0))

    def clear_memory_cache(self) -> None:
        self._kernels.clear()
        self._modules.clear()
        self._compile_seconds.clear()


def _check_launch_error(cp: Any) -> None:
    checker = getattr(cp.cuda.runtime, "peekAtLastError", None)
    if checker is None:
        checker = getattr(cp.cuda.runtime, "getLastError", None)
    # CuPy 14's CUDA 13 Windows runtime binding exports neither helper.
    # RawKernel launch and the following event synchronization still surface
    # launch/configuration/execution failures as CUDARuntimeError.
    if checker is None:
        return
    code = int(checker())
    if code != 0:
        try:
            message = cp.cuda.runtime.getErrorString(code)
        except Exception:
            message = f"CUDA error {code}"
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA kernel launch failed: {message}")


def _one_launch(
    cp: Any,
    kernel: Any,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: tuple[Any, ...],
    stream: Any,
) -> tuple[float, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    wall = time.perf_counter()
    start.record(stream)
    kernel(grid, block, args, stream=stream)
    _check_launch_error(cp)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)), (time.perf_counter() - wall) * 1_000.0


def launch_timed(
    kernel: Any,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: tuple[Any, ...],
    *,
    warmup: int = 1,
    repeats: int = 1,
    stream: Any | None = None,
) -> KernelTiming:
    """Launch, check the CUDA error state, synchronize, and event-time work."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    cp = _cupy()
    selected = cp.cuda.get_current_stream() if stream is None else stream
    first_ms, first_wall_ms = _one_launch(
        cp, kernel, tuple(grid), tuple(block), tuple(args), selected
    )
    for _ in range(warmup):
        kernel(tuple(grid), tuple(block), tuple(args), stream=selected)
        _check_launch_error(cp)
    selected.synchronize()
    samples = [
        _one_launch(cp, kernel, tuple(grid), tuple(block), tuple(args), selected)[0]
        for _ in range(repeats)
    ]
    return KernelTiming(
        compile_seconds=float(_KERNEL_COMPILE_SECONDS.get(id(kernel), 0.0)),
        first_launch_ms=first_ms,
        first_wall_ms=first_wall_ms,
        mean_launch_ms=float(sum(samples) / len(samples)),
        min_launch_ms=float(min(samples)),
        max_launch_ms=float(max(samples)),
        repeats=repeats,
    )


__all__ = [
    "CudaCapability",
    "CudaRefusal",
    "KernelCache",
    "KernelTiming",
    "launch_timed",
    "require_cuda",
]
