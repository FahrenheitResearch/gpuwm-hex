"""Tier gates for the gpuwm-hex test battery.

Three things in this tree cannot run everywhere, and each one used to fail in
a way that told the reader nothing:

* **CUDA tests** compiled and ran kernels on whatever card was present, even
  on a box whose owner had asked that no GPU work happen there.
* **The x4.163842 full-physics tier** needs about 26.4 GiB of free device
  memory.  On a smaller card it does not print "this card is too small"; it
  dies inside a CuPy allocation with an out-of-memory traceback several
  frames below anything a reader recognises.
* **The byte-pinned authority tier** (mesh, static, init, native history)
  needs about 6.9 GiB of files that ship with no fetch path and live on
  the reference node.  Without them four tests raise ``FileNotFoundError`` naming
  a path inside this checkout that has never existed on this machine, which
  reads as a broken checkout rather than a missing asset.

So each tier gets a gate that names the concrete breakage, and each gate
fails closed.  The marker discipline is gpuwm's, deliberately: a convention
that depends on every author remembering a decorator fails open, and gpuwm
measured what that costs -- 34 CUDA gates once compiled and ran on a machine
that had been declared off limits, because an unmarked test is not excluded
by ``-m "not gpu"``.  Detection here is by AST, and an explicit marker is
additive rather than the only route in.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
from typing import Iterable

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "run_cuda_v841_full_physics_x4.py"

#: Device memory the x4.163842 full-physics tier holds resident, measured.
X4_FULL_PHYSICS_BYTES = int(26.4 * 1024**3)

#: Honoured under both spellings.  A box configured for the engine sets
#: GPUWM_NO_LOCAL_GPU; the port must not quietly ignore that and light up the
#: same card from a different distribution.
_GPU_BAN_VARIABLES = ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")

NO_LOCAL_GPU = any(
    os.environ.get(name, "") not in ("", "0") for name in _GPU_BAN_VARIABLES
)

if NO_LOCAL_GPU:
    # Planted where every route converges.  The marker automation cannot see
    # transitive device contact -- a helper three imports deep that allocates
    # -- and the CUDA runtime can.  An import-level ban was considered and
    # rejected: importing cupy is not the crime, opening the device is.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "gpu: needs a CUDA device and cupy; auto-applied to any test whose "
        "module or function imports cupy",
        "bigcard: needs about 26.4 GiB of free device memory (the x4.163842 "
        "full-physics footprint)",
        "assets: needs the byte-pinned mesh/static/init/native-authority "
        "files, which ship with no fetch path",
        "slow: minutes rather than seconds",
    ):
        config.addinivalue_line("markers", line)


# ---------------------------------------------------------------------------
# tier 1: the GPU marker, detected rather than declared
# ---------------------------------------------------------------------------
def _cupy_scope(source: str) -> tuple[bool, frozenset[str]]:
    """Return ``(whole_module, {function names})`` that reach cupy.

    Per-function rather than per-file because one file in gpuwm carried a
    single cupy import at line 1548 while its other two hundred tests
    deliberately stubbed cupy out; marking the whole module would have
    excluded them all from every CPU run.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a broken test file fails louder
        return False, frozenset()

    def _touches(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                if any(alias.name.split(".")[0] == "cupy" for alias in child.names):
                    return True
            elif isinstance(child, ast.ImportFrom):
                if (child.module or "").split(".")[0] == "cupy":
                    return True
            elif isinstance(child, ast.Call):
                function = child.func
                name = getattr(function, "attr", None) or getattr(function, "id", None)
                if name == "importorskip" and child.args:
                    first = child.args[0]
                    if isinstance(first, ast.Constant) and first.value == "cupy":
                        return True
        return False

    module_level: list[ast.AST] = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    whole_module = any(_touches(node) for node in module_level)

    functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _touches(node):
            functions.add(node.name)
    return whole_module, frozenset(functions)


# ---------------------------------------------------------------------------
# tier 2: the byte-pinned authority assets
# ---------------------------------------------------------------------------
_AUTHORITY_LITERAL = re.compile(
    r"^AUTHORITY_PINS[^=]*=\s*(?P<body>\{.*?^\})", re.DOTALL | re.MULTILINE
)


def _authority_relative_paths() -> tuple[str, ...]:
    """The pinned authority paths, read without importing the runner.

    The runner is 163 KB and pulls numpy and netCDF4 at import; paying that
    on every collection to learn nine filenames is not a trade worth making,
    and a conftest that imports the module under test is its own hazard.
    """

    try:
        source = RUNNER_PATH.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - only outside a checkout
        return ()
    match = _AUTHORITY_LITERAL.search(source)
    if match is None:  # pragma: no cover - the literal moved; fail open loudly
        return ()
    try:
        pins = ast.literal_eval(match.group("body"))
    except (ValueError, SyntaxError):  # pragma: no cover
        return ()
    return tuple(
        str(pin["relative_path"])
        for pin in pins.values()
        if isinstance(pin, dict) and "relative_path" in pin
    )


def _missing_authorities() -> tuple[str, ...]:
    return tuple(
        relative
        for relative in _authority_relative_paths()
        if not (ROOT / relative).is_file()
    )


def _authority_touching_functions(source: str) -> frozenset[str]:
    """Test functions that open a byte-pinned authority file.

    Detected on ``default_authority_paths``, the one accessor that turns a
    pin into a path on disk.  Reading ``AUTHORITY_PINS`` itself is not
    detected and must not be: several tests assert on the pin metadata --
    byte counts, digest shapes -- and those are exactly the tests that should
    keep running on a box with no assets, because they are what proves the
    pins are well formed before anyone spends 6.9 GiB fetching files.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return frozenset()
    functions: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "default_authority_paths"
            ) or (isinstance(child, ast.Name) and child.id == "default_authority_paths"):
                functions.add(node.name)
                break
    return frozenset(functions)


# ---------------------------------------------------------------------------
# tier 3: device capacity
# ---------------------------------------------------------------------------
def free_device_bytes() -> int | None:
    """Free memory on device 0, or ``None`` when there is no usable device.

    Answers ``None`` under the ban WITHOUT importing cupy, because importing
    it and asking for a device count is itself device contact and used to
    happen on every pytest invocation on this box.
    """

    if NO_LOCAL_GPU:
        return None
    try:
        import cupy
    except Exception:
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:  # pragma: no cover - driver present but unusable
        return None


def _bigcard_reason() -> str | None:
    if NO_LOCAL_GPU:
        return (
            "GPU work is banned on this box by "
            f"{'/'.join(_GPU_BAN_VARIABLES)}; the x4.163842 full-physics tier "
            "belongs on the dedicated device"
        )
    free = free_device_bytes()
    if free is None:
        return (
            "no CUDA device with a working cupy is reachable, and the "
            "x4.163842 full-physics tier is 26.4 GiB of resident device "
            "state -- there is nothing here to hold it"
        )
    if free < X4_FULL_PHYSICS_BYTES:
        return (
            "this card has "
            f"{free / 1024**3:.1f} GiB free and the x4.163842 full-physics "
            "tier holds about 26.4 GiB resident.  It would not fail a check, "
            "it would die inside a CuPy allocation part-way through a run.  "
            "Run this tier on a >=32 GiB device"
        )
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    missing = _missing_authorities()
    assets_skip = pytest.mark.skip(
        reason=(
            "byte-pinned authority assets are absent from this checkout "
            f"({len(missing)} of {len(_authority_relative_paths())} missing, "
            f"first: {missing[0] if missing else '-'}).  They are about "
            "6.9 GiB of mesh, static, init and native-history files that "
            "gpuwm-hex ships with no fetch path; they live on the dedicated "
            "node.  This is a missing asset, not a broken checkout"
        )
    )
    gpu_ban_skip = pytest.mark.skip(
        reason=(
            f"{'/'.join(_GPU_BAN_VARIABLES)} is set: GPU work belongs on the "
            "dedicated device"
        )
    )
    bigcard_reason = _bigcard_reason()
    bigcard_skip = (
        pytest.mark.skip(reason=bigcard_reason) if bigcard_reason else None
    )

    sources: dict[Path, str] = {}
    for item in items:
        path = Path(str(getattr(item, "fspath", "")))
        if path not in sources:
            try:
                sources[path] = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover
                sources[path] = ""
        source = sources[path]

        whole_module, cupy_functions = _cupy_scope(source)
        if whole_module or item.name.split("[")[0] in cupy_functions:
            item.add_marker(pytest.mark.gpu)

        if item.name.split("[")[0] in _authority_touching_functions(source):
            item.add_marker(pytest.mark.assets)

        if NO_LOCAL_GPU and item.get_closest_marker("gpu") is not None:
            item.add_marker(gpu_ban_skip)
        if missing and item.get_closest_marker("assets") is not None:
            item.add_marker(assets_skip)
        if bigcard_skip is not None and item.get_closest_marker("bigcard") is not None:
            item.add_marker(bigcard_skip)
