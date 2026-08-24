"""Every manifest of this repository's own files names a file that is here.

Two manifests declare which of this tree's modules a proof executed:
``PORT_SOURCE_FILES`` (recorded per run, hashes go in the receipt) and
``EXECUTION_SOURCE_PINS`` (byte-pinned, a mismatch refuses).  Both are hand-
maintained tuples of repository-relative paths.

THE BREAKAGE THIS PREVENTS.  ``PORT_SOURCE_FILES`` named
``tools/run_cuda_v841_partitioned_x4.py``, which has never existed at any
commit in this repository.  ``port_source_manifest`` recorded it as ``null``
and carried on, so a receipt stating "these are the twenty-five modules whose
bytes can move a number in this proof" was in fact stating twenty-four plus a
hole -- and the hole hashed identically whatever it stood for.  A reader
reconciling two receipts cannot see that, and a genuinely deleted module would
be swallowed the same silent way.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str) -> object:
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def partstream() -> object:
    return _load(ROOT / "tools" / "v841_partstream_common.py", "_test_partstream_common")


def _missing(names) -> list[str]:
    return sorted(name for name in names if not (ROOT / name).is_file())


def test_port_source_files_all_exist(partstream) -> None:
    assert _missing(partstream.PORT_SOURCE_FILES) == []


def test_port_source_files_has_no_repeated_row(partstream) -> None:
    declared = list(partstream.PORT_SOURCE_FILES)
    assert len(set(declared)) == len(declared)


def test_execution_source_pins_all_exist() -> None:
    runner = _load(
        ROOT / "tools" / "run_cuda_v841_full_physics_x4.py",
        "_test_manifest_run_cuda_v841_full_physics_x4",
    )
    assert _missing(runner.EXECUTION_SOURCE_PINS) == []


def test_manifest_hashes_every_declared_row(partstream, tmp_path: Path) -> None:
    for relative in partstream.PORT_SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative.encode("utf-8"))

    manifest = partstream.port_source_manifest(tmp_path)
    assert sorted(manifest["files"]) == sorted(partstream.PORT_SOURCE_FILES)
    assert all(entry["sha256"] for entry in manifest["files"].values())


def test_manifest_refuses_a_declared_row_that_is_not_there(
    partstream, tmp_path: Path
) -> None:
    absent = partstream.PORT_SOURCE_FILES[-1]
    for relative in partstream.PORT_SOURCE_FILES:
        if relative == absent:
            continue
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative.encode("utf-8"))

    with pytest.raises(FileNotFoundError) as refusal:
        partstream.port_source_manifest(tmp_path)
    assert absent in str(refusal.value)
