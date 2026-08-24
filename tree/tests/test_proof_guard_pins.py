"""The #228 proof-guard redesign: manifest-anchored checkout guard and
masked-content authority digests.

Leg 1: ``verify_arwen_checkout_git`` verifies the sixteen-file
``ARWEN_SOURCE_MANIFEST`` -- the seam's executed source -- and records the
checkout's HEAD/tree/dirty-state as receipt provenance instead of gating on
them.  A commit that moves nothing the seam executes must pass; a manifest
file that moved must refuse by name.

Leg 2: the three regenerated native authorities are pinned by masked-content
digest (the random 10-char ``file_id`` global attribute MPAS stamps into every
output is masked, value bytes only, located via the netCDF header), so a
bit-exact rerun satisfies the pin while a single flipped data byte refuses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "run_cuda_v841_full_physics_x4.py"

#: Names the gpuwm object store this box holds.  Set it and the store MUST
#: hold ``ARWEN_COMMIT``; there is no falling back to a skip.
#:
#: There is no built-in default and there must not be one.  A default is a
#: guess about somebody's filesystem: it is right on exactly the machine it
#: was written on, and on every other machine it is a path that does not
#: exist, which this guard cannot distinguish from a store that lost the
#: commit.  Unset means "this box has no store", which is a skip that names
#: what went unverified.
GPUWM_OBJECT_STORE_ENV = "GPUWM_OBJECT_STORE"


def _load_runner() -> object:
    name = "_test_proof_guard_pins_runner"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _git(repo: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _object_store_candidates() -> tuple[Path, ...]:
    """Every gpuwm object store this box might hold, best first."""

    declared = (os.environ.get(GPUWM_OBJECT_STORE_ENV) or "").strip()
    if declared:
        return (Path(declared),)
    return ()


def _holds_commit(store: Path, commit: str) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(store), "cat-file", "-t", commit],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "commit"


def _resolve_object_source(commit: str) -> Path:
    """The store holding ``ARWEN_COMMIT``, or a refusal / a named skip.

    THE DISTINCTION IS THE POINT.  A gpuwm checkout that EXISTS but no longer
    holds the proven commit is the failure this guard was written for: the
    commit lived only as the tip of a single work branch, and deleting
    that branch makes the port's only correctness anchor unreachable.  On a box
    that has the store, that is a REFUSAL, not a skip.

    A box with no gpuwm store at all -- a CI runner, a fresh clone of
    gpuwm-hex on someone else's machine -- cannot prove or disprove anything,
    so it skips.  The skip says which anchor went unverified rather than
    reading as a pass.
    """

    present = [store for store in _object_store_candidates()
               if (store / ".git").exists()]
    for store in present:
        if _holds_commit(store, commit):
            return store

    candidates = _object_store_candidates()
    looked = (
        "\n".join(f"    {store}" for store in candidates)
        if candidates
        else f"    nowhere -- ${GPUWM_OBJECT_STORE_ENV} is unset"
    )
    if present:
        raise AssertionError(
            f"the gpuwm object store exists but does NOT hold {commit}, the "
            "commit ARWEN_SOURCE_MANIFEST was taken from.\n"
            "  the breakage this refuses: that commit is the port's only "
            "correctness anchor. It is reachable only through refs that point "
            "at it, and it has lived as the tip of a lane branch. Once it is "
            "unreachable the sixteen manifest pins can never be verified "
            "again, and every proof built on them becomes unfalsifiable.\n"
            f"  stores present on this box:\n{looked}\n"
            "  what to do: restore a ref to it in gpuwm -- the annotated tag "
            "pin/mpas-port-arwen-seam exists for exactly this -- or point "
            f"${GPUWM_OBJECT_STORE_ENV} at a checkout that still has the "
            "object.")

    pytest.skip(
        f"NOT PROVEN on this box: no gpuwm object store holding {commit} is "
        "reachable, so the sixteen ARWEN_SOURCE_MANIFEST pins went "
        f"unverified. Looked at:\n{looked}\n"
        f"Set ${GPUWM_OBJECT_STORE_ENV} to a gpuwm checkout to turn this into "
        "a real check; a box that HAS the store and has lost the commit "
        "refuses instead of skipping.")


@pytest.fixture()
def proven_checkout(tmp_path: Path) -> tuple[object, Path]:
    """A real git checkout carrying exactly the proven manifest bytes.

    Seeded from the real gpuwm object store at ``ARWEN_COMMIT`` (blob bytes,
    no filters), committed in a throwaway repository -- never on a real
    branch.  Every seeded file is validated against the manifest pin before
    the tests rely on it.
    """

    runner = _load_runner()
    source = _resolve_object_source(runner.ARWEN_COMMIT)
    manifest = dict(runner.arwen_source_manifest())
    assert len(manifest) == 16
    repo = tmp_path / "arwen-checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "proof-guard@test")
    _git(repo, "config", "user.name", "proof-guard test")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "commit.gpgsign", "false")
    for relative, expected in manifest.items():
        blob = _git(
            source, "cat-file", "blob", f"{runner.ARWEN_COMMIT}:{relative}", binary=True
        )
        assert hashlib.sha256(blob).hexdigest() == expected, (
            f"gpuwm object store blob for {relative} does not hash to the "
            "manifest pin; the fixture instrument is invalid"
        )
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed: the proven manifest bytes")
    return runner, repo


def test_manifest_identical_checkout_at_a_different_commit_passes(
    proven_checkout: tuple[object, Path],
) -> None:
    runner, repo = proven_checkout
    scratch = repo / "docs" / "scratch-note.md"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text("a docs-only descendant; nothing the seam executes moved\n")
    _git(repo, "add", "docs/scratch-note.md")
    _git(repo, "commit", "-q", "-m", "docs: a commit that moves no executed source")
    head = _git(repo, "rev-parse", "HEAD")
    assert head != runner.ARWEN_COMMIT

    record = runner.verify_arwen_checkout_git(repo)

    assert record["head"] == head
    assert record["clean"] is True
    assert record["dirty_paths"] == []
    assert record["manifest"]["files"] == dict(runner.arwen_source_manifest())


def test_committed_manifest_edit_refuses_naming_the_file(
    proven_checkout: tuple[object, Path],
) -> None:
    runner, repo = proven_checkout
    target = repo / "gpuwm" / "core" / "physics.py"
    original = target.read_bytes()
    target.write_bytes(original + b"\n# one moved byte in the executed seam\n")
    _git(repo, "add", "gpuwm/core/physics.py")
    _git(repo, "commit", "-q", "-m", "drift: the executed source moved")
    expected = dict(runner.arwen_source_manifest())["gpuwm/core/physics.py"]
    found = hashlib.sha256(target.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(repo)

    text = str(error.value)
    assert "gpuwm/core/physics.py" in text
    assert "does not match the proven manifest" in text
    assert expected in text
    assert found in text
    assert "re-prove before running" in text


def test_uncommitted_manifest_edit_refuses_naming_the_file(
    proven_checkout: tuple[object, Path],
) -> None:
    runner, repo = proven_checkout
    target = repo / "gpuwm" / "core" / "gf.py"
    target.write_bytes(target.read_bytes() + b"\n# uncommitted drift\n")

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(repo)

    text = str(error.value)
    assert "gpuwm/core/gf.py" in text
    assert "does not match the proven manifest" in text


def test_missing_manifest_file_refuses_by_name(
    proven_checkout: tuple[object, Path],
) -> None:
    runner, repo = proven_checkout
    (repo / "docs" / "mpas-seam.md").unlink()

    with pytest.raises((RuntimeError, FileNotFoundError)) as error:
        runner.verify_arwen_checkout_git(repo)

    text = str(error.value)
    assert "docs/mpas-seam.md" in text
    assert "re-prove before running" in text


def test_dirty_manifest_file_refuses_even_when_bytes_match_the_manifest(
    proven_checkout: tuple[object, Path],
) -> None:
    runner, repo = proven_checkout
    target = repo / "gpuwm" / "core" / "physics.py"
    proven = target.read_bytes()
    target.write_bytes(proven + b"\n# drift\n")
    _git(repo, "add", "gpuwm/core/physics.py")
    _git(repo, "commit", "-q", "-m", "drift: committed, then reverted only on disk")
    target.write_bytes(proven)

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(repo)

    text = str(error.value)
    assert "gpuwm/core/physics.py" in text
    assert "dirty" in text


def test_dirty_unrelated_file_is_recorded_loudly_and_execution_proceeds(
    proven_checkout: tuple[object, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner, repo = proven_checkout
    (repo / "scratch.log").write_text("an unrelated untracked file\n")

    record = runner.verify_arwen_checkout_git(repo)

    assert record["clean"] is False
    assert "scratch.log" in record["dirty_paths"]
    assert record["manifest"]["files"] == dict(runner.arwen_source_manifest())
    loud = capsys.readouterr().out
    assert "scratch.log" in loud


def test_head_and_tree_are_provenance_not_gates() -> None:
    runner = _load_runner()
    import inspect

    source = inspect.getsource(runner.verify_arwen_checkout_git)
    assert "Arwen checkout HEAD changed" not in source
    assert "Arwen checkout tree changed" not in source
    assert "6b896c3dd5ef2fb94507210af49766f78f831d57" not in source


# --- leg 2: masked-content authority digests -------------------------------


REGENERATED_MASKED_DIGESTS = {
    "native_f000": "38575bfcbbe581c25ceffeec25d22061b6f22cea2308f639e8fcce093d58da17",
    "native_f030": "1cf267557cf394f0209fbce6a69350e386221d4e791c2ceefc72200d3a45da47",
    "native_f001": "2b867a3352d7580280c01120b5db7fb4e6979be528317770417dff04b9f58b4c",
}
RETIRED_WHOLE_FILE_DIGESTS = (
    "3c2917677726eac4b1514052e298f22418ad2b1ba63541016f42aaa632bcaf29",
    "e75ccb83b654a382a96b8ccb79b9232353f1119d19ee26efbe1b46158ca7ea16",
    "f6b1ec4aa0aac5c556147efac6806c094fd2e1945ee8c1a6038d2ea604604f01",
)

NETCDF3_FORMATS = ("NETCDF3_64BIT_DATA", "NETCDF3_64BIT_OFFSET", "NETCDF3_CLASSIC")


def _write_synthetic_history(path: Path, fmt: str, file_id: str) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w", format=fmt) as dataset:
        dataset.setncattr("on_a_sphere", "YES")
        dataset.setncattr("sphere_radius", 6371229.0)
        dataset.setncattr("file_id", file_id)
        dataset.setncattr("model_name", "mpas")
        dataset.createDimension("nCells", 8)
        dataset.createDimension("nVertLevels", 3)
        variable = dataset.createVariable("theta", "f8", ("nCells", "nVertLevels"))
        variable[:] = np.arange(24, dtype=np.float64).reshape(8, 3) + 250.0


@pytest.mark.parametrize("fmt", NETCDF3_FORMATS)
def test_file_id_value_span_is_located_via_the_netcdf_header(
    tmp_path: Path, fmt: str
) -> None:
    runner = _load_runner()
    path = tmp_path / f"history-{fmt}.nc"
    _write_synthetic_history(path, fmt, "abcdefghij")

    offset, length = runner.netcdf_file_id_value_span(path)

    data = path.read_bytes()
    assert length == 10
    assert data[offset : offset + length] == b"abcdefghij"


def test_rewritten_file_id_alone_preserves_the_masked_digest(tmp_path: Path) -> None:
    runner = _load_runner()
    original = tmp_path / "history-a.nc"
    _write_synthetic_history(original, "NETCDF3_64BIT_DATA", "aaaaaaaaaa")
    offset, length = runner.netcdf_file_id_value_span(original)

    rerun = tmp_path / "history-b.nc"
    data = bytearray(original.read_bytes())
    data[offset : offset + length] = b"zzzzzzzzzz"
    rerun.write_bytes(bytes(data))

    first = runner.netcdf_masked_digests(original)
    second = runner.netcdf_masked_digests(rerun)
    assert first["masked_sha256"] == second["masked_sha256"]
    assert first["sha256"] != second["sha256"]
    assert first["file_id"] == "aaaaaaaaaa"
    assert second["file_id"] == "zzzzzzzzzz"


def test_single_flipped_data_byte_changes_the_masked_digest(tmp_path: Path) -> None:
    runner = _load_runner()
    original = tmp_path / "history-a.nc"
    _write_synthetic_history(original, "NETCDF3_64BIT_DATA", "aaaaaaaaaa")

    corrupted = tmp_path / "history-c.nc"
    data = bytearray(original.read_bytes())
    data[-1] ^= 0x01
    corrupted.write_bytes(bytes(data))

    assert (
        runner.netcdf_masked_digests(original)["masked_sha256"]
        != runner.netcdf_masked_digests(corrupted)["masked_sha256"]
    )


def test_file_record_verifies_masked_digest_and_records_both(tmp_path: Path) -> None:
    runner = _load_runner()
    original = tmp_path / "history-a.nc"
    _write_synthetic_history(original, "NETCDF3_64BIT_DATA", "aaaaaaaaaa")
    offset, length = runner.netcdf_file_id_value_span(original)
    digests = runner.netcdf_masked_digests(original)
    pin = {"bytes": original.stat().st_size, "masked_sha256": digests["masked_sha256"]}

    rerun = tmp_path / "history-b.nc"
    data = bytearray(original.read_bytes())
    data[offset : offset + length] = b"zzzzzzzzzz"
    rerun.write_bytes(bytes(data))

    record = runner._file_record("native_f000", rerun, pin)
    assert record["masked_sha256"] == digests["masked_sha256"]
    assert record["sha256"] == hashlib.sha256(bytes(data)).hexdigest()
    assert record["file_id"] == "zzzzzzzzzz"
    assert record["bytes"] == original.stat().st_size

    corrupted = tmp_path / "history-c.nc"
    flipped = bytearray(original.read_bytes())
    flipped[-1] ^= 0x01
    corrupted.write_bytes(bytes(flipped))
    with pytest.raises(RuntimeError) as error:
        runner._file_record("native_f000", corrupted, pin)
    text = str(error.value)
    assert "native_f000" in text
    assert "masked content digest" in text


def test_native_authority_pins_are_the_regenerated_masked_digests() -> None:
    runner = _load_runner()
    for role, masked in REGENERATED_MASKED_DIGESTS.items():
        pin = runner.AUTHORITY_PINS[role]
        assert pin["masked_sha256"] == masked
        assert "sha256" not in pin, (
            f"{role} still carries a whole-file sha256 pin; a bit-exact rerun "
            "could never satisfy it"
        )
        assert pin["bytes"] == 1_584_808_024
    for role in (
        "grid",
        "static",
        "init",
        "native_validation_receipt",
        "native_launch_receipt",
        "native_closure",
    ):
        pin = runner.AUTHORITY_PINS[role]
        assert "sha256" in pin and "masked_sha256" not in pin


def test_retired_whole_file_digests_are_gone_from_the_execution_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for digest in RETIRED_WHOLE_FILE_DIGESTS:
        assert digest not in source
