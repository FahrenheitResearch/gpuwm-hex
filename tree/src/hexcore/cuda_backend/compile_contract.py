"""Fail-closed identity and manifest for MPAS NVRTC translation units.

The numerical gates certify compiled kernels, not CUDA source in isolation.
This module binds every translation unit to gpuwm's measured NVRTC platform
fingerprint and records the source, requested options, kernels resolved from
the module, and any effective compile evidence CuPy actually exposes.

CuPy's ``RawModule`` route appends ``-ftz=true`` after caller options on the
measured stack.  That is deliberately *not* inferred here.  When NVRTC really
fires, :class:`NvrtcCompileCapture` observes the options at its entry point.
A disk-cache hit, a changed private CuPy API, or an unexposed compiled image is
reported as unavailable rather than filled with a plausible value.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import re
import threading
from typing import Any, Iterable, Mapping


COMPILE_MANIFEST_SCHEMA = "mpas-port.cuda-compile-manifest/v1"
STATUS_RESOLVED = "resolved"
STATUS_UNAVAILABLE = "unavailable"
UNAVAILABLE = "unavailable"

_REQUIRED_FINGERPRINT_FIELDS = (
    "nvrtc_build",
    "nvrtc_build_id",
    "nvrtc_library_sha256",
    "cuda_driver_version",
    "device_compute_capability",
    "cupy_version",
    "numpy_version",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODULE_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_CAPTURE_LOCK = threading.RLock()


class CompileContractError(RuntimeError):
    """A compiler identity or translation-unit claim cannot be proved."""


def canonical_sha256(document: Mapping[str, Any]) -> str:
    """SHA-256 of a mapping's canonical JSON representation."""

    encoded = json.dumps(
        dict(document), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_sha256(source: str) -> str:
    """SHA-256 of one UTF-8 CUDA translation unit."""

    if not isinstance(source, str):
        raise TypeError("CUDA source must be a string")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_module_key(module_key: str) -> str:
    """Return a stable manifest key or refuse an ambiguous one."""

    if not isinstance(module_key, str) or not _MODULE_KEY_RE.fullmatch(module_key):
        raise CompileContractError(
            "CUDA module_key must be a non-empty stable identifier containing "
            "only letters, digits, '.', ':', '_' or '-'"
        )
    return module_key


def validate_compile_platform_fingerprint(
    measured: Mapping[str, Any],
) -> dict[str, str]:
    """Validate gpuwm's fingerprint at certification resolution.

    The four-part NVRTC build (for example ``13.0.48``), build id and loaded
    library digest are load-bearing.  Falling back to ``nvrtc.getVersion()``
    would conflate compiler builds already measured to generate different
    kernel answers.
    """

    if not isinstance(measured, Mapping):
        raise CompileContractError(
            "gpuwm compile_platform_fingerprint did not return a mapping"
        )
    missing = [name for name in _REQUIRED_FINGERPRINT_FIELDS if name not in measured]
    if missing:
        raise CompileContractError(
            f"gpuwm compile_platform_fingerprint is missing fields {missing}"
        )
    normalized: dict[str, str] = {}
    for key, value in measured.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CompileContractError(
                "gpuwm compile_platform_fingerprint keys and values must be strings"
            )
        if not value or value == UNAVAILABLE:
            raise CompileContractError(
                f"gpuwm compile_platform_fingerprint.{key} is unavailable"
            )
        normalized[key] = value
    if re.fullmatch(r"\d+\.\d+\.\d+", normalized["nvrtc_build"]) is None:
        raise CompileContractError(
            "gpuwm compile_platform_fingerprint.nvrtc_build does not identify "
            "the compiler at build resolution"
        )
    if not normalized["nvrtc_build_id"].startswith("CL-"):
        raise CompileContractError(
            "gpuwm compile_platform_fingerprint.nvrtc_build_id is invalid"
        )
    if _SHA256_RE.fullmatch(normalized["nvrtc_library_sha256"]) is None:
        raise CompileContractError(
            "gpuwm compile_platform_fingerprint.nvrtc_library_sha256 is invalid"
        )
    return dict(sorted(normalized.items()))


def resolve_compile_platform_fingerprint() -> dict[str, str]:
    """Measure and validate gpuwm's compiler fingerprint, or refuse.

    gpuwm owns the fingerprint contract.  MPAS binds it rather than carrying a
    second implementation which could silently drift to different semantics.
    """

    try:
        from gpuwm.certify.compile_platform import compile_platform_fingerprint
    except Exception as error:
        raise CompileContractError(
            "gpuwm.certify.compile_platform.compile_platform_fingerprint is "
            f"unavailable: {type(error).__name__}: {error}"
        ) from error
    try:
        measured = compile_platform_fingerprint()
    except Exception as error:
        raise CompileContractError(
            "gpuwm compile_platform_fingerprint measurement failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    return validate_compile_platform_fingerprint(measured)


def compile_platform_binding() -> dict[str, Any]:
    """JSON-ready full fingerprint and its canonical SHA-256 binding."""

    fingerprint = resolve_compile_platform_fingerprint()
    return {
        "fingerprint": fingerprint,
        "sha256": canonical_sha256(fingerprint),
    }


def _object_record(blob: Any) -> dict[str, Any]:
    if isinstance(blob, str):
        payload = blob.encode("utf-8")
        kind = "ptx"
    elif isinstance(blob, (bytes, bytearray, memoryview)):
        payload = bytes(blob)
        kind = "cubin" if payload[:4] == b"\x7fELF" else "ptx"
    else:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": "NVRTC returned no string or byte image to the capture",
        }
    return {
        "status": STATUS_RESOLVED,
        "kind": kind,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class NvrtcObservation:
    """One invocation observed at CuPy's NVRTC entry point."""

    source_sha256: str
    effective_flags: tuple[str, ...]
    include_path_count: int
    compiled_image: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "effective_flags": list(self.effective_flags),
            "include_path_count": self.include_path_count,
            "include_paths_omitted": (
                "-I entries describe this machine's toolkit layout and are "
                "counted but not copied into the arithmetic contract"
            ),
            "compiled_image": dict(self.compiled_image),
        }


class NvrtcCompileCapture(AbstractContextManager["NvrtcCompileCapture"]):
    """Observe effective NVRTC options if CuPy compiles inside the context.

    The wrapped name is private CuPy API, so absence is an ordinary explicit
    ``unavailable`` result.  A process-wide re-entrant lock prevents two MPAS
    caches from nesting incompatible wrappers around that global function.
    """

    def __init__(self) -> None:
        self.observations: list[NvrtcObservation] = []
        self.unavailable_reason: str | None = None
        self._compiler: Any | None = None
        self._original: Any | None = None
        self._locked = False

    def __enter__(self) -> "NvrtcCompileCapture":
        _CAPTURE_LOCK.acquire()
        self._locked = True
        try:
            from cupy.cuda import compiler
            original = getattr(compiler, "_compile_using_nvrtc_no_warning")
        except Exception as error:
            self.unavailable_reason = (
                "CuPy exposes no usable NVRTC entry capture: "
                f"{type(error).__name__}: {error}"
            )
            return self
        self._compiler = compiler
        self._original = original
        capture = self

        def wrapper(source: Any, options: Iterable[str] = (), *args: Any,
                    **kwargs: Any) -> Any:
            result = original(source, options, *args, **kwargs)
            blob = result[0] if isinstance(result, tuple) else result
            source_text = (
                source if isinstance(source, str)
                else bytes(source).decode("utf-8", errors="replace")
            )
            normalized_options = tuple(str(option) for option in options)
            flags = tuple(
                option for option in normalized_options
                if not option.startswith("-I")
            )
            capture.observations.append(NvrtcObservation(
                source_sha256=source_sha256(source_text),
                effective_flags=flags,
                include_path_count=len(normalized_options) - len(flags),
                compiled_image=_object_record(blob),
            ))
            return result

        compiler._compile_using_nvrtc_no_warning = wrapper
        return self

    def __exit__(self, *exc: Any) -> bool:
        try:
            if self._compiler is not None and self._original is not None:
                self._compiler._compile_using_nvrtc_no_warning = self._original
        finally:
            if self._locked:
                _CAPTURE_LOCK.release()
                self._locked = False
        return False

    def evidence(self, expected_source_sha256: str) -> dict[str, Any]:
        matched = [
            observation.as_dict() for observation in self.observations
            if observation.source_sha256 == expected_source_sha256
        ]
        if matched:
            return {
                "status": STATUS_RESOLVED,
                "method": (
                    "wrapped cupy.cuda.compiler."
                    "_compile_using_nvrtc_no_warning at the NVRTC entry point"
                ),
                "observations": matched,
            }
        reason = self.unavailable_reason or (
            "NVRTC did not fire while the RawModule function was resolved; "
            "the compiled image may have been a CuPy disk-cache hit"
        )
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": (
                reason + "; effective terminal -ftz=true is not claimed "
                "without a real NVRTC-entry capture"
            ),
        }


class CompileManifest:
    """Per-cache inventory of translation units bound to one compiler."""

    def __init__(self, binding: Mapping[str, Any]) -> None:
        if not isinstance(binding, Mapping) or "fingerprint" not in binding:
            raise CompileContractError(
                "compile platform binding is missing its fingerprint document"
            )
        fingerprint = validate_compile_platform_fingerprint(binding["fingerprint"])
        digest = str(binding.get("sha256", ""))
        expected = canonical_sha256(fingerprint)
        if digest != expected:
            raise CompileContractError(
                "compile platform fingerprint SHA-256 does not match its document"
            )
        self._binding = {"fingerprint": fingerprint, "sha256": digest}
        self._modules: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def fingerprint_sha256(self) -> str:
        return str(self._binding["sha256"])

    def declare_module(
        self,
        module_key: str,
        *,
        source: str,
        requested_options: Iterable[str],
        module_cache_key: str,
    ) -> None:
        key = validate_module_key(module_key)
        options = tuple(str(option) for option in requested_options)
        entry = {
            "source_sha256": source_sha256(source),
            "requested_options": list(options),
            "compile_platform_fingerprint_sha256": self.fingerprint_sha256,
            "module_cache_key": str(module_cache_key),
            "effective_compile": {
                "status": STATUS_UNAVAILABLE,
                "reason": (
                    "the RawModule has not produced observable NVRTC evidence; "
                    "its effective terminal -ftz=true is not claimed without a "
                    "real NVRTC-entry capture"
                ),
            },
            "resolved_kernels": [],
        }
        with self._lock:
            previous = self._modules.get(key)
            if previous is not None:
                identity_fields = (
                    "source_sha256", "requested_options",
                    "compile_platform_fingerprint_sha256", "module_cache_key",
                )
                if any(previous[name] != entry[name] for name in identity_fields):
                    raise CompileContractError(
                        f"CUDA module_key {key!r} was reused for a different "
                        "translation unit, options, or compiler"
                    )
                return
            self._modules[key] = entry

    def record_effective_compile(
        self, module_key: str, evidence: Mapping[str, Any]
    ) -> None:
        key = validate_module_key(module_key)
        with self._lock:
            if key not in self._modules:
                raise CompileContractError(
                    f"CUDA module_key {key!r} was not declared"
                )
            current = self._modules[key]["effective_compile"]
            if (current.get("status") == STATUS_RESOLVED
                    and evidence.get("status") != STATUS_RESOLVED):
                return
            self._modules[key]["effective_compile"] = json.loads(
                json.dumps(dict(evidence), sort_keys=True)
            )

    def record_kernel(self, module_key: str, kernel_name: str) -> None:
        key = validate_module_key(module_key)
        if not isinstance(kernel_name, str) or not kernel_name:
            raise CompileContractError("CUDA kernel name must be a non-empty string")
        with self._lock:
            if key not in self._modules:
                raise CompileContractError(
                    f"CUDA module_key {key!r} was not declared"
                )
            names = self._modules[key]["resolved_kernels"]
            if kernel_name not in names:
                names.append(kernel_name)
                names.sort()

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated, deterministic, JSON-ready manifest."""

        with self._lock:
            modules = json.loads(json.dumps(self._modules, sort_keys=True))
        return {
            "schema": COMPILE_MANIFEST_SCHEMA,
            "compile_platform": {
                "fingerprint": dict(self._binding["fingerprint"]),
                "sha256": self._binding["sha256"],
            },
            "modules": modules,
        }


__all__ = [
    "COMPILE_MANIFEST_SCHEMA",
    "CompileContractError",
    "CompileManifest",
    "NvrtcCompileCapture",
    "STATUS_RESOLVED",
    "STATUS_UNAVAILABLE",
    "canonical_sha256",
    "compile_platform_binding",
    "resolve_compile_platform_fingerprint",
    "source_sha256",
    "validate_compile_platform_fingerprint",
    "validate_module_key",
]
