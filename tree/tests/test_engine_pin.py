"""The engine range, the drift check, and the two places that must name it.

THE DEFECT ALL OF THIS ANSWERS, measured on a real install on 2026-08-27
(``evidence/userwalk-20260827/``): ``pip install gpuwm-hex`` resolved gpuwm
2.5.7, whose bytes this port's sixteen-file seam manifest refuses; the
forecast lane stopped with two SHA-256 digests and no version number,
``gpuwm-hex doctor`` reported the estate healthy and exited 0, and no
document in the distribution named a version that works.

Three surfaces, one fact.  The declaration bounds what pip may resolve, the
report names the drift at install time, and the door names it at launch.
Every test below is written so that it fails when the surface stops doing
that -- the instrument is validated in both directions, with real bytes and
real SHA-256, and never against a stub of itself.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore import doctor
from hexcore import engine_pin
from hexcore import forecast_door as door


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------
def _row(version: str, *, on_pypi=True, moved=(), buildable=True):
    return engine_pin.PublishedEngine(
        version=version,
        on_pypi=on_pypi,
        moved=tuple(moved),
        offline_build_road=buildable,
    )


def test_the_floor_is_the_lowest_engine_that_is_all_three_things(monkeypatch):
    """Installable, byte-compatible, buildable from.  Any one missing is out.

    Each clause has a measured victim.  ``2.5.5`` satisfies the manifest and
    is a git tag with NO PyPI release, so a floor there named a version pip
    cannot resolve; and its published tree is missing a file of the vendored
    ``cc`` crate, so the manual's own printed ``cargo build --offline`` dies
    there while the same command finishes at 2.5.6.
    """

    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.4", moved=("gpuwm/config.py",)),
        _row("2.5.5", on_pypi=False, buildable=False),
        _row("2.5.6"),
        _row("2.5.7", moved=("gpuwm/core/microphysics.py",)),
    ))
    assert engine_pin.gpuwm_floor() == "2.5.6"

    # Teeth, one clause at a time: drop each disqualifier and the floor moves
    # down to 2.5.5, which proves the clause was doing the work.
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.5"),
        _row("2.5.6"),
    ))
    assert engine_pin.gpuwm_floor() == "2.5.5"


def test_the_ceiling_is_exclusive_at_the_first_engine_that_does_not_work(
    monkeypatch,
):
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.6"),
        _row("2.5.7", moved=("gpuwm/io/restart.py",)),
    ))
    assert engine_pin.gpuwm_ceiling() == "2.5.7"
    assert engine_pin.gpuwm_requirement() == "gpuwm>=2.5.6,<2.5.7"


def test_an_unmeasured_engine_is_never_admitted(monkeypatch):
    """The property that stops the NEXT engine cut re-opening this.

    When every measured engine above the floor works, the ceiling is the
    next patch after the highest one measured -- not "unbounded".  An engine
    nobody has compared against the manifest is exactly the engine the
    original defect resolved onto.
    """

    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.6"),
        _row("2.5.7"),
    ))
    assert engine_pin.gpuwm_ceiling() == "2.5.8"
    assert engine_pin.gpuwm_requirement() == "gpuwm>=2.5.6,<2.5.8"


def test_no_usable_engine_is_a_refusal_rather_than_a_silent_range(monkeypatch):
    monkeypatch.setattr(engine_pin, "PUBLISHED_ENGINES", (
        _row("2.5.7", moved=("gpuwm/io/restart.py",)),
    ))
    with pytest.raises(engine_pin.EnginePinError) as raised:
        engine_pin.gpuwm_requirement()
    assert "measure_engine_verdicts.py" in str(raised.value)


def test_the_shipped_table_says_what_the_measurement_said():
    """The rows this distribution actually ships, against the measurement.

    Not a re-derivation -- the network instrument is
    ``evidence/standalone-20260827/measure_engine_verdicts.py`` and the JSON
    behind the shipped table is
    ``evidence/repin-261-20260901/engine-verdicts.json``.  This is the claim
    the receipts make, kept true here so that editing a row without
    re-measuring is caught.

    THIS TEST'S OWN FIGURES MOVED ON 2026-08-28, and the gate was the stale
    side rather than the table.  It used to assert floor 2.5.6, ceiling
    2.5.7, and that 2.5.5 satisfied the manifest.  All three were correct
    against the 0.1.x pin and all three are now wrong, because the port
    re-pinned its manifest to gpuwm 2.5.8 and ``moved`` is measured AGAINST
    that manifest: re-pinning moves the whole column, including rows for
    engines cut long before.  Loosening the gate would have been the wrong
    repair; re-measuring and restating it is the right one.

    AND THEY MOVED AGAIN ON 2026-08-31, the same way for the same reason:
    the engine published 2.6.0 (the repaired steepest-gradient meter and
    P3), three manifest files moved at that cut -- ``gpuwm/config.py``,
    ``gpuwm/core/rrtmg_legacy.py``, ``gpuwm/io/restart.py`` -- and the port
    re-pinned to the published 2.6.0 bytes, so 2.5.8 joined the rows that
    fail its own former manifest.

    AND A THIRD TIME ON 2026-09-01: the engine published 2.6.1 (the restart
    schema v2 carriers payload and the P3 eight-species seam route), three
    manifest files moved at that cut -- ``gpuwm/core/mpas_column_batch.py``,
    ``gpuwm/config.py``, ``docs/mpas-seam.md`` -- and the port re-pinned to
    the published 2.6.1 bytes, so 2.6.0 dropped to "fails the manifest"
    exactly as 2.5.8 did the time before.
    """

    assert engine_pin.gpuwm_requirement() == "gpuwm>=2.6.1,<2.6.2"
    assert engine_pin.gpuwm_floor() == "2.6.1"
    assert engine_pin.gpuwm_ceiling() == "2.6.2"

    # EXACTLY ONE published engine is usable.  Asserted as a count, not as a
    # membership test, because "2.6.1 works" would still pass if a stale row
    # below it were left claiming to work too -- and that stale row is what
    # a resolver would actually pick when 2.6.1 is unavailable.
    usable = [row.version for row in engine_pin.PUBLISHED_ENGINES if row.usable]
    assert usable == ["2.6.1"], (
        "the re-pinned manifest is satisfied by exactly one published "
        f"engine.  Usable: {usable}"
    )
    assert engine_pin.engine("2.6.1").moved == ()

    # The engine that was the floor the day before now fails, and so does
    # every other pre-2.6.1 cut.  Named individually so that a row silently
    # reverting to "matches" is a failure here rather than a wider range.
    assert not engine_pin.engine("2.5.7").usable
    assert not engine_pin.engine("2.5.8").usable
    assert not engine_pin.engine("2.6.0").usable
    assert engine_pin.engine("2.6.0").moved == (
        "docs/mpas-seam.md",
        "gpuwm/config.py",
        "gpuwm/core/mpas_column_batch.py",
    )
    for row in engine_pin.PUBLISHED_ENGINES:
        if row.version != "2.6.1":
            assert row.moved, (
                f"{row.version} claims to satisfy the manifest.  Only 2.6.1 "
                "does; a pre-2.6.1 row with an empty moved list means the "
                "table was patched by hand instead of re-measured"
            )

    # 2.5.5 is disqualified three separate ways now, and all three are
    # asserted so that removing any one from the table is a failure rather
    # than a silent floor move.  The byte clause is new; the other two are
    # the 2026-08-27 findings and still stand.
    tag_only = engine_pin.engine("2.5.5")
    assert not tag_only.satisfies_manifest
    assert not tag_only.on_pypi
    assert not tag_only.offline_build_road


def test_the_table_is_the_instruments_output_not_a_transcription(receipts):
    """The shipped rows must equal the shipped JSON, field for field.

    THE BREAKAGE THIS PREVENTS is the one this module's docstring counts:
    three separate hand-transcriptions of these figures, three different
    wrong answers.  The defence is that the table is rendered from the
    instrument's JSON, so this test re-renders it and demands equality --
    which fails on a hand-patched row even when the patch looks plausible.

    It takes the ``receipts`` fixture because it reads a receipt, and #378
    closed exactly this shape thirty times over: a tree that holds
    ``evidence/`` out -- the assembled public tree, and the sdist -- has no
    JSON to re-render, and a direct read raises ``FileNotFoundError`` there
    instead of skipping with the reason.  This file is in CI's tier-1 list
    (``tools/battery/cpu_files.txt``), and ``ci.yml`` runs tier 1 on push, so
    the direct read fired a RED CI on the public repository on the release
    commit itself.  Measured on a full assembly of the public tree,
    2026-08-28: 1 failed / 799 passed, and this was the one.  The fixture
    keeps the gate real where the receipts exist -- a tree that carries them
    and is missing THIS one still fails.
    """

    import json

    verdicts = json.loads(
        (receipts / "repin-261-20260901" / "engine-verdicts.json").read_text(
            encoding="utf-8"
        )
    )
    measured = {
        row["tag"].lstrip("v"): (
            bool(row["on_pypi"]),
            tuple(row["moved"]),
            bool(row["offline_build_road"]),
        )
        for row in verdicts["tags"]
    }
    shipped = {
        row.version: (row.on_pypi, row.moved, row.offline_build_road)
        for row in engine_pin.PUBLISHED_ENGINES
    }
    assert shipped == measured, (
        "PUBLISHED_ENGINES disagrees with the JSON it is supposed to be "
        "rendered from.  Re-splice it: python "
        "evidence/repin-258-20260828/render_engine_pin_table.py "
        "evidence/repin-261-20260901/engine-verdicts.json --splice "
        "src/hexcore/engine_pin.py"
    )


# ---------------------------------------------------------------------------
# the byte check, on real bytes
# ---------------------------------------------------------------------------
def _seam_tree(tmp_path: Path, version: str) -> tuple[Path, dict[str, str]]:
    """A fake gpuwm checkout and the manifest that pins its exact bytes.

    Real files, real SHA-256, real reads.  The manifest is computed FROM the
    bytes written, so the clean case is genuinely clean and every drift below
    is a byte that actually moved rather than a mocked verdict.
    """

    root = tmp_path / f"gpuwm-{version}"
    payloads = {
        "gpuwm/core/mpas_column_batch.py": b"def run_mpas_column_batch(): ...\n",
        "gpuwm/core/microphysics.py": b"def apply(): ...\n",
        "gpuwm/io/restart.py": b"RESTART = 1\n",
        "docs/mpas-seam.md": b"# the seam\n",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "gpuwm"\nversion = "{version}"\n', encoding="utf-8"
    )
    manifest = {
        relative: sha256(payload).hexdigest()
        for relative, payload in payloads.items()
    }
    return root, manifest


def test_a_matching_tree_reports_no_drift(tmp_path, monkeypatch):
    root, manifest = _seam_tree(tmp_path, "2.5.6")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)

    inspection = engine_pin.inspect_seam(root)
    assert not inspection.drifted
    assert inspection.absent == ()
    assert inspection.checked == len(manifest)
    assert door.seam_pin_problem(root) is None


def test_a_moved_byte_is_found_and_a_missing_file_is_a_different_finding(
    tmp_path, monkeypatch
):
    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)

    (root / "gpuwm/core/microphysics.py").write_bytes(b"def apply(): pass\n")
    (root / "docs/mpas-seam.md").unlink()

    inspection = engine_pin.inspect_seam(root)
    assert inspection.moved == ("gpuwm/core/microphysics.py",)
    assert inspection.absent == ("docs/mpas-seam.md",)
    assert inspection.drifted


def test_the_checkout_version_is_read_from_the_tree_not_from_the_install(
    tmp_path, monkeypatch
):
    """A refusal that names a version must name the version ON DISK.

    ``gpuwm/__init__.py`` reads its own version back out of INSTALLED
    metadata, so importing it from a checkout answers with whatever pip has
    rather than what the user pointed at -- the exact confusion this refusal
    exists to end.
    """

    root, _ = _seam_tree(tmp_path, "2.5.7")
    assert engine_pin.checkout_version(root) == "2.5.7"
    assert engine_pin.checkout_version(tmp_path / "not-a-checkout") is None


# ---------------------------------------------------------------------------
# the door's refusal names a version
# ---------------------------------------------------------------------------
def test_the_door_refusal_names_the_version_found_and_the_version_wanted(
    tmp_path, monkeypatch
):
    """The refusal law: name the breakage AND the remedy.

    What a user met before this existed was two digests, an instruction to
    "re-prove before running" addressed to a developer of this port, and the
    door's own wrapper saying the message should not be trusted.  A manifest
    holds digests; a user cannot invert one into a version.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    (root / "gpuwm/core/microphysics.py").write_bytes(b"moved\n")

    problem = door.seam_pin_problem(root)
    assert problem is not None
    assert "gpuwm 2.5.7" in problem, "the version FOUND is not named"
    assert f"gpuwm {engine_pin.wanted_version()}" in problem, (
        "the version WANTED is not named"
    )
    assert "gpuwm/core/microphysics.py" in problem, "the moved file is not named"
    assert engine_pin.gpuwm_requirement() in problem, "the remedy is not printed"
    assert f"--branch v{engine_pin.wanted_version()}" in problem, (
        "the forecast lane needs a SOURCE CHECKOUT and the refusal must say "
        "which tag to clone; naming only the pip line leaves that lane shut"
    )


def test_the_door_refuses_before_the_run_rather_than_relaying_the_driver(
    tmp_path, monkeypatch
):
    """A named refusal, raised by the door, is what reaches the user.

    ``resolve_request`` is where every check a user can act on happens.  The
    driver's own sixteen-file check is NOT retired by this -- it is the wall,
    it also reads the checkout's git state, and it lives inside the frozen
    proof harness -- but the sentence the user reads comes from here.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    (root / "gpuwm/io/restart.py").write_bytes(b"RESTART = 2\n")

    problem = door.seam_pin_problem(root)
    assert problem is not None
    assert isinstance(door._refuse(problem), door.ForecastDoorRefusal), (
        "the door's refusal type is what run_forecast relays verbatim; an "
        "exception that is not one of this program's named refusals gets "
        "wrapped in 'that is not one of this program's named refusals'"
    )


# ---------------------------------------------------------------------------
# doctor sees it at install time
# ---------------------------------------------------------------------------
def _doctor_seam_finding(findings):
    return next(f for f in findings if f.subject.startswith("gpuwm seam bytes"))


def test_doctor_passes_when_the_installed_engine_matches(tmp_path, monkeypatch):
    root, manifest = _seam_tree(tmp_path, "2.5.6")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: "2.5.6"
    )

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.VERIFIED
    assert doctor.blocking_gaps(findings) == []


def test_doctor_refuses_and_names_the_version_when_the_bytes_have_moved(
    tmp_path, monkeypatch
):
    """The check the walk found missing, and the reason it is BLOCKING.

    Fifteen of the sixteen pinned paths sit in site-packages, so this drift
    was visible at install time, on any box, with no card and no checkout.
    Nothing looked, and the report said ``Every check passed`` and exited 0
    while the forecast lane was dead.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: "2.5.7"
    )
    (root / "gpuwm/core/microphysics.py").write_bytes(b"moved\n")

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.MISSING
    assert seam.required, (
        "an engine whose seam bytes this port refuses is a WRONG install, "
        "not a missing optional asset: pip resolved something the package's "
        "own declared range excludes.  A non-required finding leaves the "
        "report saying every required check passed, which is the measured "
        "defect"
    )
    assert "2.5.7" in seam.detail and engine_pin.wanted_version() in seam.detail
    assert engine_pin.gpuwm_requirement() in seam.remedy
    assert [f.subject for f in doctor.blocking_gaps(findings)] == [seam.subject]


def test_doctor_says_so_rather_than_crashing_when_gpuwm_cannot_be_located(
    monkeypatch,
):
    monkeypatch.setattr(engine_pin, "installed_root", lambda: None)
    finding = doctor._check_seam_bytes("2.5.7")
    assert finding.status == doctor.INFO
    assert doctor.blocking_gaps([finding]) == []


# ---------------------------------------------------------------------------
# the checkout finding must not contradict the line above it
# ---------------------------------------------------------------------------
def _doctor_checkout_finding(findings):
    return next(f for f in findings if f.subject.startswith("gpuwm git checkout"))


def test_the_checkout_finding_stops_claiming_the_pin_is_unsatisfiable(
    tmp_path, monkeypatch
):
    """THE DEFECT THIS PREVENTS, measured 2026-08-28 on a real 2.5.8 install.

    ``gpuwm-hex doctor`` printed these two lines one under the other::

        OK      gpuwm seam bytes: 16 of 16 pinned files are in this install
                and all 16 match
        INFO    gpuwm source checkout: ... An installed gpuwm satisfies pip
                and does not satisfy the pin, so that lane needs a source
                checkout at the pinned commit.

    The second was a CONSTANT string, written when the manifest pinned a
    document no wheel carried, and it outlived that.  It is now derived from
    the same inspection the first line prints, so an install that carries
    every pinned path can never be told it does not.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.8")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.5.8")

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    checkout = _doctor_checkout_finding(findings)

    assert seam.status == doctor.VERIFIED
    assert checkout.status == doctor.INFO
    assert checkout.evidence["absent"] == []
    assert "does not satisfy the pin" not in checkout.detail
    assert "no wheel places" not in checkout.detail
    assert f"all {len(manifest)} pinned paths resolve" in checkout.detail, (
        "when every pinned path is in the install, the finding under the OK "
        "line has to say so; the two lines are read together"
    )
    assert "HEAD, tree and dirty paths" in checkout.detail, (
        "a refusal or a requirement stands only if it names the concrete "
        "breakage it prevents.  What survives at 2.5.8 is receipt "
        "provenance, and that is what this line must say"
    )


def test_the_checkout_finding_still_names_a_path_an_older_engine_lacks(
    tmp_path, monkeypatch
):
    """The other direction, so the instrument is not a constant either way.

    Engines through 2.5.7 ship no ``docs/mpas-seam.md`` in the wheel.  There
    the finding must go back to naming the absent path, because there an
    install genuinely cannot satisfy the pin on its own.
    """

    root, manifest = _seam_tree(tmp_path, "2.5.7")
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: root)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.5.7")
    (root / "docs/mpas-seam.md").unlink()

    checkout = _doctor_checkout_finding(doctor.check_physics_seam())
    assert checkout.evidence["absent"] == ["docs/mpas-seam.md"]
    assert "docs/mpas-seam.md" in checkout.detail
    assert "cannot satisfy the seam pin on its own" in checkout.detail


def test_the_remedy_stops_giving_a_reason_that_is_false_at_the_pinned_engine():
    """``engine_pin.remedy()`` is printed by doctor and by the door.

    It told every reader that one of the sixteen pinned paths is a document
    no wheel places in site-packages.  At the only engine this port admits
    that is false, and a remedy carrying a dead reason is a refusal that
    does not name its breakage.
    """

    text = engine_pin.remedy()
    assert "no wheel" not in text
    assert "--gpuwm-checkout" in text
    assert "provenance" in text
    assert "HEAD, tree and dirty" in text
