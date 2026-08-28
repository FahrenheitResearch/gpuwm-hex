"""Ledger #379: an installed wheel must reach its own shipped modules.

Two gates in this program digest the port's modules by name, and both used
to resolve those names by assuming a source checkout under the file doing
the asking.  From a wheel that assumption lands on ``<venv>/lib/python3.13``,
which holds no ``src/``, so the regional digest refused and took 0.2.0's
headline limited-area lane with it on every machine.  Measured before the
fix on Linux (``evidence/xmachine-20260827`` section 6b) and on Windows
(``evidence/wheel-reach-20260827`` section 2).

EVERY TEST HERE IS WRITTEN BOTH WAYS.  A resolution fix is exactly the kind
of change that can be "proven" by a test that would pass with the gate
removed, so each admission below has a refusal beside it that exercises the
same code path with one byte moved.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import shutil
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hexcore import shipped_sources  # noqa: E402
from hexcore.cuda_backend import regional_admission  # noqa: E402


# ---------------------------------------------------------------------------
# the resolver itself
# ---------------------------------------------------------------------------
def test_a_declared_name_resolves_to_the_module_this_interpreter_imported():
    """The whole fix in one assertion: the file, not a guessed tree."""

    resolved = shipped_sources.resolve("src/hexcore/cuda_fp32.py")
    from hexcore import cuda_fp32

    assert resolved == Path(cuda_fp32.__file__).resolve()
    assert resolved.is_file()


def test_every_declared_regional_source_resolves_to_a_real_file():
    for name in regional_admission.REGIONAL_KERNEL_SOURCES:
        path = shipped_sources.resolve(name)
        assert path.is_file(), f"{name} -> {path}"


def test_a_name_outside_the_package_is_refused_rather_than_guessed():
    """The teeth of the prefix rule.

    A name the package cannot own has no package-relative meaning, and
    silently joining it onto the package directory would produce a path that
    exists nowhere and a refusal blaming the wrong thing.
    """

    with pytest.raises(shipped_sources.DeclaredSourceError, match="declared source"):
        shipped_sources.resolve("tools/run_cuda_v841_forecast.py")


def test_an_explicit_root_still_means_that_tree(tmp_path):
    """The A/B route: a caller digesting somebody else's checkout gets it."""

    planted = tmp_path / "src" / "hexcore" / "cuda_fp32.py"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"# not this tree\n")
    assert shipped_sources.resolve("src/hexcore/cuda_fp32.py", tmp_path) == planted


# ---------------------------------------------------------------------------
# the digest is the same number it was, and that matters more than it reads
# ---------------------------------------------------------------------------
def test_the_fix_did_not_move_the_minted_digest():
    """A resolution change with the same bytes under it must be a no-op.

    If this goes red the fix moved a pinned constant, every class in
    ``ADMITTED_CLASSES`` has silently lapsed, and the card time to re-mint
    them is owed.  It is the first thing to look at.
    """

    assert (
        regional_admission.kernel_set_sha256()
        == regional_admission.MINTED_KERNEL_SET_SHA256
    )


def test_the_checkout_route_and_the_package_route_agree_in_a_checkout():
    """Two resolutions, one answer, because they name the same files here."""

    assert regional_admission.kernel_set_sha256(ROOT) == (
        regional_admission.kernel_set_sha256()
    )


# ---------------------------------------------------------------------------
# the shape that was broken: a package with no checkout above it
# ---------------------------------------------------------------------------
def _plant_installed_package(destination: Path) -> Path:
    """A directory laid out the way an installed wheel lays site-packages out.

    ``<destination>/site-packages/hexcore/...`` with the fourteen declared
    regional sources in it and NO ``src/`` anywhere above -- which is the
    layout that produced the refusal this ledger item exists for.  Only the
    digested files are copied: the point is the resolution, and a full copy
    of the package would hide a resolver that quietly fell back to a
    directory it should not have been reading.
    """

    package = destination / "site-packages" / "hexcore"
    for name in regional_admission.REGIONAL_KERNEL_SOURCES:
        relative = name[len(shipped_sources.DECLARED_PREFIX):]
        target = package.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shipped_sources.resolve(name), target)
    return package


def test_the_digest_is_earned_from_a_package_with_no_checkout_above_it(
    tmp_path, monkeypatch
):
    """ADMITS.  The half of the proof that says the fix works."""

    package = _plant_installed_package(tmp_path)
    assert not (tmp_path / "src").exists()
    monkeypatch.setattr(shipped_sources, "PACKAGE_ROOT", package)
    assert (
        regional_admission.kernel_set_sha256()
        == regional_admission.MINTED_KERNEL_SET_SHA256
    )


def test_a_moved_byte_in_an_installed_package_lapses_every_class(
    tmp_path, monkeypatch
):
    """REFUSES.  The half of the proof that says the gate still has teeth.

    This is the direction that decides whether the fix is a fix or a hole.
    A digest gate that reads the right files and cannot notice them changing
    is worse than one that refuses everything, because it issues a receipt.
    One byte appended to one shipped module inside the installed package must
    lapse every admitted class BY NAME.
    """

    package = _plant_installed_package(tmp_path)
    monkeypatch.setattr(shipped_sources, "PACKAGE_ROOT", package)
    clean = regional_admission.kernel_set_sha256()

    altered = package / "cuda_fp32.py"
    altered.write_bytes(altered.read_bytes() + b"\n# one byte too many\n")

    moved = regional_admission.kernel_set_sha256()
    assert moved != clean
    assert moved != regional_admission.MINTED_KERNEL_SET_SHA256

    with pytest.raises(RuntimeError, match="those bytes have moved"):
        regional_admission.require_regional_anchor(
            "r4.75.14050",
            bdy_mask_sha256=(
                "a8e66046452db881bb4a9da08952610207ee5aa2e0a58d48b1d2348b48f84088"
            ),
            n_cells=14_050,
        )


def test_an_installed_package_missing_a_shipped_module_refuses_by_path(
    tmp_path, monkeypatch
):
    """The old refusal's job, kept, and now naming a path a user can look at."""

    package = _plant_installed_package(tmp_path)
    monkeypatch.setattr(shipped_sources, "PACKAGE_ROOT", package)
    (package / "cuda_acoustic_v841.py").unlink()

    with pytest.raises(
        regional_admission.RegionalAdmissionRefusal,
        match="cannot be digested",
    ) as caught:
        regional_admission.kernel_set_sha256()
    message = str(caught.value)
    assert "src/hexcore/cuda_acoustic_v841.py" in message
    assert str(package) in message
    assert "installed hexcore package" in message


def test_the_digest_does_not_read_a_checkout_that_happens_to_be_nearby(
    tmp_path, monkeypatch
):
    """The instrument's own control: prove the package root is what is read.

    A resolver that fell back to a tree walk would pass every test above and
    still be the defect.  So plant a COMPLETE, correct ``src/`` tree three
    directories above the package, corrupt the package's own copy, and
    require the refusal anyway.  If the digest comes out minted here, the
    resolution is reading the wrong files again.
    """

    package = _plant_installed_package(tmp_path)
    shadow = tmp_path / "site-packages" / "src" / "hexcore"
    for name in regional_admission.REGIONAL_KERNEL_SOURCES:
        relative = name[len(shipped_sources.DECLARED_PREFIX):]
        target = shadow.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shipped_sources.resolve(name), target)

    monkeypatch.setattr(shipped_sources, "PACKAGE_ROOT", package)
    altered = package / "cuda_transport.py"
    altered.write_bytes(altered.read_bytes() + b"\n# corrupted\n")

    assert (
        regional_admission.kernel_set_sha256()
        != regional_admission.MINTED_KERNEL_SET_SHA256
    ), "the digest read the shadow checkout instead of the executing package"


def test_the_framing_still_pins_a_name_to_its_payload(tmp_path, monkeypatch):
    """Renaming one module's bytes into another's must not reproduce the digest.

    The reason the digest is framed name-then-payload at all.  Kept as a test
    of the CURRENT code because the resolution change touched the loop the
    framing lives in.
    """

    package = _plant_installed_package(tmp_path)
    monkeypatch.setattr(shipped_sources, "PACKAGE_ROOT", package)
    first = package / "cuda_horizontal.py"
    second = package / "cuda_horizontal_v841.py"
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    assert first_bytes != second_bytes
    first.write_bytes(second_bytes)
    second.write_bytes(first_bytes)
    assert (
        regional_admission.kernel_set_sha256()
        != regional_admission.MINTED_KERNEL_SET_SHA256
    )


# ---------------------------------------------------------------------------
# the frozen execution set, the second site with the same defect
# ---------------------------------------------------------------------------
def test_the_frozen_execution_set_reads_the_executing_package():
    """``tools/`` verified the checkout's copies while site-packages ran.

    In a checkout the two are one directory, so this asserts the identity
    that makes the change a no-op for every historical proof, and the
    resolution that makes it correct in the hybrid shape a wheel user is
    told to assemble.
    """

    driver = ROOT / "tools" / "run_cuda_v841_full_physics_x4.py"
    source = driver.read_text(encoding="utf-8")
    assert "shipped_sources.resolve(relative)" in source, (
        "the frozen execution set is resolving against a checkout again"
    )
    assert "path = ROOT / relative" not in source

    names = [
        line.split('"')[1]
        for line in source.splitlines()
        if line.strip().startswith('"src/hexcore/')
    ]
    assert len(names) >= 20, names
    for name in names:
        assert shipped_sources.resolve(name).is_file(), name


# ---------------------------------------------------------------------------
# the extras table cannot contradict the runtime floor
# ---------------------------------------------------------------------------
def _declaration() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_gpu_alias_installs_a_cupy_the_runtime_floor_admits():
    """The durable form of ledger #379's second defect.

    ``pip install "gpuwm-hex[gpu]"`` -- the obvious spelling -- used to
    resolve to ``gpu-cu12``, and every GPU door in this port refuses a CUDA
    runtime below 13000.  So the obvious command installed a CuPy that
    imports, allocates, runs cuBLAS, and is then refused by name at forecast
    launch.  This test is what stops the two drifting apart again: it reads
    the floor off ``require_cuda`` rather than restating it, so moving the
    floor to 12000 makes the alias wrong here and the red says which.
    """

    doctor = importlib.import_module("hexcore.doctor")
    floor = doctor.cuda_runtime_floor()
    assert floor is not None
    expected_extra = doctor._GPU_EXTRA_BY_MAJOR[floor // 1000]

    extras = _declaration()["project"]["optional-dependencies"]
    assert extras["gpu"] == [f"gpuwm-hex[{expected_extra}]"], (
        f"the gpu alias must install the CuPy the runtime floor {floor} "
        f"admits, which is {expected_extra}"
    )
    wheel = extras[expected_extra][0]
    assert wheel.startswith(f"cupy-cuda{floor // 1000}x"), wheel


def test_both_per_major_extras_stay_resolvable():
    """A published name that vanishes turns a wrong answer into a hard error.

    ``gpu-cu12`` is in 0.1.0's published metadata and in written
    instructions.  It is no longer what the alias points at and doctor now
    reports an installed cu12 wheel as a gap, but the name resolves.
    """

    extras = _declaration()["project"]["optional-dependencies"]
    assert "gpu-cu12" in extras and "gpu-cu13" in extras


# ---------------------------------------------------------------------------
# doctor tells the truth about a CuPy the runtime will refuse
# ---------------------------------------------------------------------------
def _doctor():
    return importlib.import_module("hexcore.doctor")


def test_the_floor_doctor_reports_is_the_one_require_cuda_enforces():
    """Read, never restated -- so the number in the remedy is the number raised."""

    from inspect import signature

    from hexcore.cuda_backend.runtime import require_cuda

    declared = signature(require_cuda).parameters["min_runtime_version"].default
    assert _doctor().cuda_runtime_floor() == int(declared)


def test_a_wrong_major_cupy_is_reported_as_a_gap_naming_the_refusal(monkeypatch):
    """REFUSES.  The measured case: cupy-cuda12x on a box the doors want cu13.

    Before this, doctor printed ``verified`` on exactly this CuPy, because
    it imported.  The user then met ``CudaRefusal: cuda.runtime_version=12090
    < required 13000`` at forecast launch, from a card that had already been
    reserved.
    """

    doctor = _doctor()
    monkeypatch.setattr(doctor, "_import_probe", lambda module: (True, "imported, 14.2.0"))
    monkeypatch.setattr(
        doctor,
        "_cupy_runtime_probe",
        lambda: {"cupy_version": "14.2.0", "runtime_version": 12090},
    )
    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: ["cupy-cuda12x"])
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)

    findings = doctor.check_gpu_runtime()
    assert [f.status for f in findings] == [doctor.MISSING]
    finding = findings[0]
    assert "12090" in finding.detail
    assert "13000" in finding.detail
    assert "cuda.runtime_version=12090 < required 13000" in finding.remedy

    compact = doctor.render(findings, explain=False)
    assert "pip uninstall -y cupy-cuda12x" in compact, compact
    assert "gpu-cu12" not in compact, compact


def test_a_right_major_cupy_is_verified_and_says_what_it_measured(monkeypatch):
    """ADMITS.  The same instrument, the other way, or it measures nothing."""

    doctor = _doctor()
    monkeypatch.setattr(doctor, "_import_probe", lambda module: (True, "imported, 14.0.1"))
    monkeypatch.setattr(
        doctor,
        "_cupy_runtime_probe",
        lambda: {"cupy_version": "14.0.1", "runtime_version": 13000},
    )
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)

    findings = doctor.check_gpu_runtime()
    assert [f.status for f in findings] == [doctor.VERIFIED]
    assert "13000" in findings[0].detail


def test_the_compact_remedy_for_a_missing_cupy_names_the_admitted_extra(monkeypatch):
    """The one line a user actually sees must be the one that works.

    ``render(explain=False)`` filters comment lines and keeps the first
    command.  That filter is why the old remedy's careful "match your
    driver's CUDA major" comment never reached anybody and the cu12 command
    did.
    """

    doctor = _doctor()
    monkeypatch.setattr(
        doctor, "_import_probe", lambda module: (False, "ModuleNotFoundError: cupy")
    )
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)

    compact = doctor.render(doctor.check_gpu_runtime(), explain=False)
    assert 'pip install "gpuwm-hex[gpu-cu13]"' in compact, compact
    assert "gpu-cu12" not in compact, compact


def test_a_driver_below_the_floor_is_told_no_pip_command_helps(monkeypatch):
    """A remedy that cannot work is worse than a remedy that is absent.

    On a CUDA-12 driver neither extra opens the CUDA lane: cupy-cuda13x
    needs a CUDA-13 driver, and cupy-cuda12x is refused by the floor.  The
    compact view therefore offers no install command at all, and the
    headline -- the line the compact view always prints -- carries the
    reason.
    """

    doctor = _doctor()
    monkeypatch.setattr(
        doctor, "_import_probe", lambda module: (False, "ModuleNotFoundError: cupy")
    )
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 12)

    findings = doctor.check_gpu_runtime()
    assert [f.status for f in findings] == [doctor.MISSING]
    assert "serves CUDA 12" in findings[0].detail
    compact = doctor.render(findings, explain=False)
    assert "pip install" not in compact, compact


def test_the_driver_probe_is_suppressed_by_the_no_local_gpu_declaration(monkeypatch):
    """A box that has declared no GPU work happens on it is not touched."""

    doctor = _doctor()
    monkeypatch.setenv("GPUWM_HEX_NO_LOCAL_GPU", "1")
    assert doctor._driver_cuda_major() is None
    monkeypatch.setenv("GPUWM_HEX_NO_LOCAL_GPU", "0")
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    # Not asserting a value here: whether a driver answers is a fact about
    # the box, and this test is about the suppression, not the card.
    doctor._driver_cuda_major()


def test_an_unreadable_runtime_version_is_not_reported_as_verified(monkeypatch):
    """CuPy that imports and will not say what it carries is PRESENT, not OK.

    ``verified`` in this module means the deep check ran and passed.  A
    probe that could not read the runtime version did not run.
    """

    doctor = _doctor()
    monkeypatch.setattr(doctor, "_import_probe", lambda module: (True, "imported, 14.0.1"))
    monkeypatch.setattr(doctor, "_cupy_runtime_probe", lambda: {"error": "boom"})
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)
    assert [f.status for f in doctor.check_gpu_runtime()] == [doctor.PRESENT]


def test_the_digest_of_a_wheel_shaped_package_is_measured_not_asserted(tmp_path):
    """A guard against this file becoming a declaration.

    Every ``_plant_installed_package`` test above rests on the copy being
    faithful.  If ``shutil.copyfile`` ever stopped being byte-exact, the
    admitting tests would go red rather than silently pass, but the refusing
    ones would pass for the wrong reason.  So measure the copy.
    """

    package = _plant_installed_package(tmp_path)
    for name in regional_admission.REGIONAL_KERNEL_SOURCES:
        relative = name[len(shipped_sources.DECLARED_PREFIX):]
        copied = package.joinpath(*relative.split("/"))
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == (
            hashlib.sha256(shipped_sources.resolve(name).read_bytes()).hexdigest()
        ), name
