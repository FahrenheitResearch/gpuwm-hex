#!/usr/bin/env python3
"""Isolated bootstrap for the frozen v8.4.1 FTZ measurement child.

The parent launches this file with ``-I -S -B``.  Consequently neither
``PYTHONPATH``, user-site startup, ``.pth`` files, ``sitecustomize`` nor
``usercustomize`` can run before this byte-pinned code.  Required package
directories are added only after startup, and the requested GPUWM tree is
exposed as two namespace packages rather than as a general-purpose import
root.
"""

from __future__ import annotations

import hashlib
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from pathlib import Path
import sys
from types import ModuleType
from typing import Mapping, Sequence


class BootstrapError(RuntimeError):
    """The child did not enter through the required isolated boundary."""


def _read_pinned_source(path: Path, expected_sha256: str, *, label: str) -> bytes:
    before = path.stat(follow_symlinks=False)
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise BootstrapError(f"{label} changed while reading its executable bytes")
    measured = hashlib.sha256(payload).hexdigest()
    if measured != expected_sha256:
        raise BootstrapError(
            f"{label} SHA-256 changed: {measured} != {expected_sha256}"
        )
    return payload


def _real_file(value: str, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise BootstrapError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_file():
        raise BootstrapError(f"{label} is not a real file: {selected}")
    return selected


def _real_directory(value: str, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise BootstrapError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_dir():
        raise BootstrapError(f"{label} is not a real directory: {selected}")
    return selected


def _assert_isolated_startup() -> None:
    required_flags = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_site": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
        "safe_path": True,
    }
    mismatches = {
        name: (getattr(sys.flags, name, None), expected)
        for name, expected in required_flags.items()
        if getattr(sys.flags, name, None) != expected
    }
    if mismatches:
        raise BootstrapError(f"Python isolated-startup flags are false: {mismatches}")
    loaded_hooks = sorted(
        name
        for name in ("site", "sitecustomize", "usercustomize")
        if name in sys.modules
    )
    if loaded_hooks:
        raise BootstrapError(
            f"Python startup customization ran before the bootstrap: {loaded_hooks}"
        )


def _namespace(name: str, directory: Path) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = None
    module.__loader__ = None
    module.__package__ = name
    module.__path__ = [str(directory)]  # type: ignore[attr-defined]
    spec = ModuleSpec(name=name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    module.__spec__ = spec
    return module


def _install_gpuwm_namespace(root: Path) -> None:
    package = root / "gpuwm"
    certify = package / "certify"
    compile_platform = certify / "compile_platform.py"
    if not package.is_dir() or not certify.is_dir() or not compile_platform.is_file():
        raise BootstrapError("GPUWM root lacks gpuwm/certify/compile_platform.py")
    if any(name in sys.modules for name in ("gpuwm", "gpuwm.certify")):
        raise BootstrapError(
            "GPUWM modules existed before the pinned namespace install"
        )
    sys.modules["gpuwm"] = _namespace("gpuwm", package)
    sys.modules["gpuwm.certify"] = _namespace("gpuwm.certify", certify)


class _SourceOnlyLoader(SourceFileLoader):
    """Compile the inventoried source bytes without consulting any .pyc."""

    def __init__(
        self,
        fullname: str,
        path: str,
        *,
        pinned_source: bytes | None = None,
    ) -> None:
        super().__init__(fullname, path)
        self.pinned_source = pinned_source

    def get_code(self, fullname: str):  # type: ignore[no-untyped-def]
        source_path = self.get_filename(fullname)
        source = (
            self.get_data(source_path)
            if self.pinned_source is None
            else self.pinned_source
        )
        return self.source_to_code(source, source_path)


class _SourceOnlyFinder:
    def __init__(
        self,
        roots: Sequence[Path],
        *,
        pinned_sources: Mapping[Path, bytes],
    ) -> None:
        self.roots = tuple(root.resolve() for root in roots)
        self.pinned_sources = {
            path.resolve(): payload for path, payload in pinned_sources.items()
        }

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[no-untyped-def]
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader):
            return spec
        if spec.origin is None:
            return spec
        origin = Path(spec.origin).resolve()
        if any(origin == root or root in origin.parents for root in self.roots):
            spec.loader = _SourceOnlyLoader(
                fullname,
                str(origin),
                pinned_source=self.pinned_sources.get(origin),
            )
            spec.cached = None
        return spec


def _parse_control(
    argv: Sequence[str],
) -> tuple[dict[str, str], list[Path], dict[str, str], list[str]]:
    values: dict[str, str] = {}
    package_roots: list[Path] = []
    gpuwm_closure: dict[str, str] = {}
    arguments = list(argv)
    if arguments.count("--") != 1:
        raise BootstrapError("bootstrap argv must contain one exact '--' separator")
    separator = arguments.index("--")
    control = arguments[:separator]
    child = arguments[separator + 1 :]
    if not child:
        raise BootstrapError("frozen child argv is empty")
    index = 0
    single = {"--frozen-tool", "--frozen-tool-sha256", "--gpuwm-root"}
    while index < len(control):
        option = control[index]
        if option not in {
            *single,
            "--package-root",
            "--gpuwm-closure",
        } or index + 1 >= len(control):
            raise BootstrapError(f"invalid bootstrap control argv near {option!r}")
        value = control[index + 1]
        if option == "--package-root":
            package_roots.append(
                _real_directory(value, label="post-startup package root")
            )
        elif option == "--gpuwm-closure":
            relative, separator, digest = value.partition("=")
            candidate = Path(relative)
            if (
                not separator
                or candidate.is_absolute()
                or ".." in candidate.parts
                or not relative
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or relative in gpuwm_closure
            ):
                raise BootstrapError("invalid or duplicate GPUWM closure pin")
            gpuwm_closure[relative] = digest
        else:
            if option in values:
                raise BootstrapError(f"duplicate bootstrap option {option}")
            values[option] = value
        index += 2
    if set(values) != single:
        raise BootstrapError("bootstrap control argv is incomplete")
    if not package_roots:
        raise BootstrapError("bootstrap has no post-startup package root")
    if not gpuwm_closure:
        raise BootstrapError("bootstrap has no transitive GPUWM closure pins")
    return values, package_roots, gpuwm_closure, child


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolated_startup()
    values, package_roots, gpuwm_closure, child_argv = _parse_control(
        sys.argv[1:] if argv is None else argv
    )
    frozen_tool = _real_file(values["--frozen-tool"], label="frozen measured tool")
    frozen_source = _read_pinned_source(
        frozen_tool,
        values["--frozen-tool-sha256"],
        label="frozen measured tool",
    )
    gpuwm_root = _real_directory(values["--gpuwm-root"], label="GPUWM root")
    pinned_gpuwm_sources: dict[Path, bytes] = {}
    for relative, digest in sorted(gpuwm_closure.items()):
        source = _real_file(
            str(gpuwm_root / relative), label=f"GPUWM closure source {relative}"
        )
        pinned_gpuwm_sources[source] = _read_pinned_source(
            source, digest, label=f"GPUWM closure source {relative}"
        )

    # ``-S`` left only the interpreter/stdlib roots.  Add runtime packages now;
    # Python does not process their .pth or startup-hook files at this stage.
    for package_root in package_roots:
        value = str(package_root)
        if value not in sys.path:
            sys.path.append(value)
    _install_gpuwm_namespace(gpuwm_root)
    mpas_source = frozen_tool.parents[1] / "src"
    if not mpas_source.is_dir() or mpas_source.is_symlink():
        raise BootstrapError("frozen measured tool has no real MPAS src/ sibling")
    sys.meta_path.insert(
        0,
        _SourceOnlyFinder(
            (mpas_source, gpuwm_root / "gpuwm"),
            pinned_sources=pinned_gpuwm_sources,
        ),
    )

    sys.argv = [str(frozen_tool), *child_argv]
    code = compile(frozen_source, str(frozen_tool), "exec", dont_inherit=True)
    scope = {
        "__name__": "__main__",
        "__file__": str(frozen_tool),
        "__cached__": None,
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
    }
    exec(code, scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
