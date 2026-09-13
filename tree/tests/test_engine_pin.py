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
    ``tools/measure_engine_verdicts.py`` and the JSON behind the shipped
    table is ``verification/engine-verdicts/repin-273-20260913.json``.  This
    is the claim the measurement makes, kept true here so that editing a row
    without re-measuring is caught.

    AND A SEVENTH TIME ON 2026-09-13, the widest move yet: the engine
    published 2.7.0, 2.7.1, 2.7.2 and 2.7.3, ELEVEN manifest files moved
    between the published 2.6.5 wheel and the published 2.7.3 wheel, and
    the port re-pinned to the 2.7.3 bytes, so 2.6.4 and 2.6.5 -- the whole
    admitted range the day before -- dropped to "fails the manifest" by
    those eleven.  2.7.0, 2.7.1 and 2.7.2 carry the same sixteen bytes as
    each other and fail by eight (the Grell-Freitas adapter, its kernel and
    the Noah-MP libm unit moved at 2.7.0 and stayed; the rest moved again
    at 2.7.3).  Measured first the other way, against the manifest 0.2.3
    shipped, all four 2.7 engines failed it -- the three earlier by seven
    and 2.7.3 by eleven -- which is the ``<2.6.6`` ceiling doing its work
    for the ten days between the 2.7.0 publish and this re-pin.  Exactly
    one engine is usable now, asserted below as the whole list.

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

    AND A FOURTH TIME ON 2026-09-02: the engine published 2.6.3 (the
    aerosol-aware Thompson seam row and a config-relative file key), the
    same three manifest files moved at that cut, and the port re-pinned to
    the published 2.6.3 bytes, so 2.6.1 dropped to "fails the manifest".
    2.6.2 is the 2.5.5 shape again -- a tag whose bytes match and whose
    publish job died, so nothing of it is on PyPI -- and is asserted below
    as satisfying the manifest and still not usable.

    AND A FIFTH TIME THE SAME DAY: the engine published 2.6.4 three hours
    after 2.6.3 (a third cumulus scheme and two fixes), SIX manifest files
    moved between the two published wheels -- ``gpuwm/core/physics.py``,
    ``gpuwm/core/gf.py``, ``gpuwm/core/kernels/gf.cu``,
    ``gpuwm/core/kernels/__init__.py``, ``gpuwm/config.py``,
    ``gpuwm/io/restart.py`` -- and the port re-pinned to the published 2.6.4
    bytes, so 2.6.3 dropped to "fails the manifest" by exactly those six.
    2.6.2 fails by the same six now, so its tag-only clause no longer
    decides anything; it is asserted below as failing on bytes AND absent
    from PyPI, so neither clause can be edited back alone.

    AND A SIXTH TIME THE SAME DAY, WITHOUT A PIN MOVE: the engine published
    2.6.5 ninety minutes after 2.6.4 (a soil-archive layout fix off the
    physics path), all sixteen manifest files are byte-identical between
    the two published wheels (``evidence/repin-265-20260902/
    wheel-pair-264-265.txt``), so the manifest stayed where it was, 2.6.4
    stayed the floor, 2.6.5 joined the usable rows and the ceiling moved
    one patch up to 2.6.6.  That is the only widening the derivation can
    produce -- a measured engine that passed -- and it is asserted below as
    exactly two usable rows, so a third appearing without a measurement is
    a failure here.
    """

    assert engine_pin.gpuwm_requirement() == "gpuwm>=2.7.3,<2.7.4"
    assert engine_pin.gpuwm_floor() == "2.7.3"
    assert engine_pin.gpuwm_ceiling() == "2.7.4"

    # EXACTLY ONE published engine is usable.  Asserted as the whole list,
    # not as a membership test, because "2.7.3 works" would still pass if a
    # stale row below it were left claiming to work too -- and that stale
    # row is what a resolver would actually pick when the admitted engine
    # is unavailable.
    usable = [row.version for row in engine_pin.PUBLISHED_ENGINES if row.usable]
    assert usable == ["2.7.3"], (
        "the re-pinned manifest is satisfied by exactly one published "
        f"engine.  Usable: {usable}"
    )
    assert engine_pin.engine("2.7.3").moved == ()
    assert engine_pin.engine("2.7.3").on_pypi
    assert engine_pin.engine("2.7.3").offline_build_road

    # The two engines that were the whole range the day before now fail,
    # and so does every other pre-2.7.3 cut.  Named individually so that a
    # row silently reverting to "matches" is a failure here rather than a
    # wider range.
    assert not engine_pin.engine("2.5.7").usable
    assert not engine_pin.engine("2.5.8").usable
    assert not engine_pin.engine("2.6.0").usable
    assert not engine_pin.engine("2.6.1").usable
    assert not engine_pin.engine("2.6.3").usable
    assert not engine_pin.engine("2.6.4").usable
    assert not engine_pin.engine("2.6.5").usable
    moved_at_273 = (
        "gpuwm/config.py",
        "gpuwm/core/gf.py",
        "gpuwm/core/kernels/__init__.py",
        "gpuwm/core/kernels/gf.cu",
        "gpuwm/core/kernels/noahmp_leaves.cu",
        "gpuwm/core/microphysics.py",
        "gpuwm/core/noahmp_glacier_gpu.py",
        "gpuwm/core/noahmp_kernel_sources.py",
        "gpuwm/core/physics.py",
        "gpuwm/core/rrtmg_legacy.py",
        "gpuwm/io/restart.py",
    )
    assert engine_pin.engine("2.6.5").moved == moved_at_273
    # 2.6.4 and 2.6.5 carried the same sixteen bytes, so they fail the same
    # way; a row that differs between them was edited, not measured.
    assert engine_pin.engine("2.6.4").moved == moved_at_273
    # The three earlier 2.7 engines carry the same sixteen bytes as each
    # other and fail by the eight that moved again at 2.7.3 -- the GF
    # adapter, its kernel and the Noah-MP libm unit moved at 2.7.0 and did
    # not move again.
    moved_at_273_since_270 = (
        "gpuwm/config.py",
        "gpuwm/core/kernels/__init__.py",
        "gpuwm/core/microphysics.py",
        "gpuwm/core/noahmp_glacier_gpu.py",
        "gpuwm/core/noahmp_kernel_sources.py",
        "gpuwm/core/physics.py",
        "gpuwm/core/rrtmg_legacy.py",
        "gpuwm/io/restart.py",
    )
    for version in ("2.7.0", "2.7.1", "2.7.2"):
        row = engine_pin.engine(version)
        assert row.moved == moved_at_273_since_270, version
        assert row.on_pypi and row.offline_build_road and not row.usable
    # 2.6.2: the tag whose publish job died.  It fails this manifest by the
    # same eleven files 2.6.5 does, so it is disqualified twice; both
    # clauses named so that neither can be quietly edited back on its own.
    tagged_262 = engine_pin.engine("2.6.2")
    assert not tagged_262.satisfies_manifest
    assert tagged_262.moved == moved_at_273
    assert tagged_262.offline_build_road
    assert not tagged_262.on_pypi
    assert not tagged_262.usable
    for row in engine_pin.PUBLISHED_ENGINES:
        if row.version != "2.7.3":
            assert row.moved, (
                f"{row.version} claims to satisfy the manifest.  Only 2.7.3 "
                "does; any other row with an empty moved list means the "
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


#: The measurement the shipped table is rendered from.  Under ``verification/``
#: rather than ``evidence/`` because it is the table's SOURCE, not a record of
#: a run: the sdist grafts ``verification/`` and holds ``evidence/`` out, so
#: this gate runs on every published surface instead of skipping there.
VERDICTS_OF_RECORD = (
    ROOT / "verification" / "engine-verdicts" / "repin-273-20260913.json"
)


def test_the_table_is_the_instruments_output_not_a_transcription():
    """The shipped rows must equal the shipped JSON, field for field, and
    the generated block must be the instrument's own rendering of it.

    THE BREAKAGE THIS PREVENTS is the one this module's docstring counts:
    three separate hand-transcriptions of these figures, three different
    wrong answers.  The defence is that the table is rendered from the
    instrument's JSON, so this test re-renders it -- with the instrument's
    own renderer, imported from ``tools/`` -- and demands equality, which
    fails on a hand-patched row even when the patch looks plausible.

    Until 2026-09-13 this read a receipt under ``evidence/`` and took the
    ``receipts`` fixture, so on the assembled public tree and in the sdist
    it SKIPPED: the one gate that keeps the table honest was disarmed on
    exactly the surfaces a user installs from.  The JSON now ships beside
    the contracts and the gate runs everywhere.
    """

    import importlib.util
    import json

    assert VERDICTS_OF_RECORD.is_file(), (
        f"{VERDICTS_OF_RECORD.relative_to(ROOT).as_posix()} is missing; the "
        "table has no measurement to be rendered from"
    )
    verdicts = json.loads(VERDICTS_OF_RECORD.read_text(encoding="utf-8"))
    assert verdicts["pin"]["version"] == "2.7.3"
    assert verdicts["instrument"] == "tools/measure_engine_verdicts.py"
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
        "rendered from.  Re-splice it: python tools/measure_engine_verdicts.py "
        "--offline --engine-git <arwen clone> --wheel-cache <cache> "
        "--pin-to 2.7.3 --splice"
    )

    # The manifest the JSON was measured against is the one the package
    # ships: a JSON re-measured against some other bytes would render the
    # same shape of table for the wrong pin.
    assert verdicts["manifest"] == engine_pin.seam_manifest()

    spec = importlib.util.spec_from_file_location(
        "_measure_engine_verdicts", ROOT / "tools" / "measure_engine_verdicts.py"
    )
    assert spec is not None and spec.loader is not None
    instrument = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instrument)
    source = (ROOT / "src" / "hexcore" / "engine_pin.py").read_text(encoding="utf-8")
    start = source.index(instrument.BEGIN_MARK)
    end = source.index(instrument.END_MARK) + len(instrument.END_MARK)
    assert source[start:end] == instrument.render_rows(verdicts["tags"]), (
        "the generated block in engine_pin.py is not the instrument's own "
        "rendering of the JSON of record; it was edited by hand"
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


# ---------------------------------------------------------------------------
# admitted engines: a measured row for a build that is not on PyPI
# ---------------------------------------------------------------------------
#: The document the admitted table is rendered from; one file for every
#: admitted engine, keyed by version, written by the instrument's --admit.
ADMITTED_OF_RECORD = ROOT / "verification" / "engine-verdicts" / "admitted-engines.json"


def _instrument():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_measure_engine_verdicts", ROOT / "tools" / "measure_engine_verdicts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_admitted_table_is_the_instruments_output_not_a_transcription():
    """The second generated block is held to the same standard as the first.

    A row here admits an engine the published table cannot see, so a
    hand-typed digest would admit whatever bytes the typist believed in.
    The block must be the instrument's own rendering of the document it
    wrote, field for field, and every shipped row must equal its JSON row.
    """

    import json

    assert ADMITTED_OF_RECORD.is_file()
    document = json.loads(ADMITTED_OF_RECORD.read_text(encoding="utf-8"))
    instrument = _instrument()
    assert document["schema"] == instrument.ADMITTED_SCHEMA
    assert document["instrument"] == "tools/measure_engine_verdicts.py"

    source = (ROOT / "src" / "hexcore" / "engine_pin.py").read_text(encoding="utf-8")
    start = source.index(instrument.ADMITTED_BEGIN_MARK)
    end = source.index(instrument.ADMITTED_END_MARK) + len(instrument.ADMITTED_END_MARK)
    assert source[start:end] == instrument.render_admitted_rows(document["engines"]), (
        "the ADMITTED_ENGINES block is not the instrument's own rendering of "
        "admitted-engines.json; re-splice it with --splice-admitted"
    )

    shipped = {row.version: row for row in engine_pin.ADMITTED_ENGINES}
    assert sorted(shipped) == sorted(row["version"] for row in document["engines"])
    for measured in document["engines"]:
        row = shipped[measured["version"]]
        assert row.base == measured["base"]
        assert row.build_commit == measured["build_commit"]
        assert row.manifest == measured["digests"]
        assert row.glacier_composed_tu_sha256 == measured["glacier_composed_tu_sha256"]
        assert list(row.moved_from_base) == measured["moved_from_base"]
        assert [list(pair) for pair in row.artifacts] == [
            [artifact["name"], artifact["sha256"]] for artifact in measured["artifacts"]
        ]
        # The row was measured against the manifest this package ships:
        # what it says moved is what differs from the shipped digests now.
        manifest = engine_pin.seam_manifest()
        assert set(row.manifest) == set(manifest)
        assert row.moved_from_base == tuple(
            sorted(path for path in manifest if row.manifest[path] != manifest[path])
        )


def test_an_admitted_row_rides_the_specifier_and_is_not_a_published_engine():
    """What a row may be.  A local version on a base inside the derived
    range, because the specifier is what pip enforces and the row only
    decides bytes; distinct from the pin (a row that moved nothing would be
    the pin under another name); and absent from the published table,
    which measures PyPI and has no business admitting it."""

    floor, ceiling = engine_pin.gpuwm_floor(), engine_pin.gpuwm_ceiling()
    assert engine_pin.ADMITTED_ENGINES, "the release admits one private engine"
    for row in engine_pin.ADMITTED_ENGINES:
        assert "+" in row.version and row.version.split("+", 1)[0] == row.base
        assert engine_pin._key(floor) <= engine_pin._key(row.base) < engine_pin._key(ceiling)
        assert row.moved_from_base
        assert engine_pin.engine(row.version) is None
        assert engine_pin.admitted_engine(row.version) is row
        identity = engine_pin.identity_for(row.version)
        assert identity.admitted and identity.build_commit == row.build_commit
        assert identity.manifest == row.manifest

    # The one row this release ships, as measured: five of sixteen paths
    # differ from the public 2.7.3 bytes and the composed glacier unit does
    # not, so the Noah-MP compile site is the pin's even though the kernel
    # loader beside it moved.
    row = engine_pin.admitted_engine("2.7.3+soma.1")
    assert row is not None and row.base == "2.7.3"
    assert row.moved_from_base == (
        "gpuwm/config.py",
        "gpuwm/core/kernels/__init__.py",
        "gpuwm/core/microphysics.py",
        "gpuwm/core/physics.py",
        "gpuwm/io/restart.py",
    )
    assert row.glacier_composed_tu_sha256 == engine_pin.pinned_engine().glacier_composed_tu_sha256
    assert engine_pin.pinned_engine().version == "2.7.3"
    assert not engine_pin.pinned_engine().admitted


def _admitted_tree(tmp_path: Path, monkeypatch, version: str = "2.7.3+local.1"):
    """A fake public checkout, a fake private build of it whose bytes differ
    on one path, and the admitted row measured FROM that build's bytes.

    Real files and real digests again: the row's manifest is computed from
    what was written, so acceptance below is a byte match and every refusal
    is a byte or a version string that actually differs.
    """

    root, manifest = _seam_tree(tmp_path, "2.7.3")
    private = tmp_path / "gpuwm-private"
    for relative in manifest:
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    (private / "gpuwm/core/microphysics.py").write_bytes(b"def apply(): ...  # private\n")
    (private / "pyproject.toml").write_text(
        f'[project]\nname = "gpuwm"\nversion = "{version}"\n', encoding="utf-8"
    )
    digests = {
        relative: sha256((private / relative).read_bytes()).hexdigest()
        for relative in manifest
    }
    row = engine_pin.AdmittedEngine(
        version=version,
        base="2.7.3",
        build_commit="9" * 40,
        digests=tuple(sorted(digests.items())),
        glacier_composed_tu_sha256="7" * 64,
        moved_from_base=("gpuwm/core/microphysics.py",),
        artifacts=(("gpuwm-fake.whl", "0" * 64),),
    )
    monkeypatch.setattr(engine_pin, "seam_manifest", lambda: manifest)
    monkeypatch.setattr(engine_pin, "ADMITTED_ENGINES", (row,))
    return root, private, row


def test_an_admitted_engines_bytes_under_its_own_version_are_accepted(
    tmp_path, monkeypatch
):
    root, private, row = _admitted_tree(tmp_path, monkeypatch)

    inspection = engine_pin.inspect_seam(private)
    assert inspection.engine == row.version
    assert inspection.declared == row.version
    assert inspection.admitted and inspection.accepted
    assert not inspection.drifted and inspection.moved == () and inspection.absent == ()
    assert inspection.checked == len(row.manifest)
    assert door.seam_pin_problem(private) is None

    # The public tree beside it is untouched by the row's existence.
    public = engine_pin.inspect_seam(root)
    assert public.engine == engine_pin.wanted_version()
    assert not public.admitted and public.accepted
    assert door.seam_pin_problem(root) is None


def test_the_same_bytes_under_another_version_string_are_refused_by_name(
    tmp_path, monkeypatch
):
    """A re-versioned copy of a measured build is neither that build nor the
    pin.  The refusal names the version the bytes belong to and the version
    the tree claims, and the remedy is not a pip line."""

    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    (private / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\nversion = "2.7.3"\n', encoding="utf-8"
    )

    inspection = engine_pin.inspect_seam(private)
    assert inspection.engine == row.version and inspection.declared == "2.7.3"
    assert inspection.misdeclared and not inspection.accepted
    assert inspection.moved == (), "the bytes ARE the row's; only the name is wrong"

    problem = door.seam_pin_problem(private)
    assert problem is not None
    assert f"gpuwm {row.version}" in problem and "gpuwm 2.7.3" in problem
    assert "version string and its bytes together" in problem


def test_an_admitted_version_over_other_bytes_is_refused_against_the_pin(
    tmp_path, monkeypatch
):
    """The other half of the key.  A tree that calls itself the admitted
    version and carries bytes the row did not measure is an unmeasured
    engine, and is refused the way every unmeasured engine is: against the
    public pin, naming what moved."""

    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    (private / "gpuwm/io/restart.py").write_bytes(b"RESTART = 3  # nobody measured this\n")

    inspection = engine_pin.inspect_seam(private)
    assert inspection.engine is None and not inspection.admitted
    assert inspection.declared == row.version
    assert inspection.moved == ("gpuwm/core/microphysics.py", "gpuwm/io/restart.py"), (
        "measured against the PUBLIC pin, so the row's own private byte "
        "counts as moved too: the row admits one engine, not a family"
    )
    problem = door.seam_pin_problem(private)
    assert problem is not None
    assert f"gpuwm {row.version}" in problem
    assert "gpuwm/io/restart.py" in problem
    assert engine_pin.gpuwm_requirement() in problem


def test_doctor_names_the_admitted_engine_it_verified(tmp_path, monkeypatch):
    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: private)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: row.version)

    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.VERIFIED
    assert row.version in seam.detail and "admitted engine" in seam.detail
    assert seam.evidence["engine"] == row.version and seam.evidence["admitted"]
    assert doctor.blocking_gaps(findings) == []


def test_doctor_refuses_an_install_whose_metadata_disowns_its_bytes(
    tmp_path, monkeypatch
):
    """An INSTALL root carries no pyproject.toml; what it declares is pip's
    metadata.  Bytes of an admitted row under metadata that says the public
    version is the misdeclared case seen from site-packages."""

    _, private, row = _admitted_tree(tmp_path, monkeypatch)
    (private / "pyproject.toml").unlink()
    monkeypatch.setattr(engine_pin, "installed_root", lambda: private)
    monkeypatch.setattr(engine_pin, "installed_version", lambda: "2.7.3")
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.7.3")

    assert engine_pin.declared_version(private) == "2.7.3"
    findings = doctor.check_physics_seam()
    seam = _doctor_seam_finding(findings)
    assert seam.status == doctor.MISSING and seam.required
    assert row.version in seam.detail and "2.7.3" in seam.detail
    assert [f.subject for f in doctor.blocking_gaps(findings)] == [seam.subject]


def test_the_adapter_admits_the_row_and_provenances_the_run_by_it(
    tmp_path, monkeypatch
):
    """The wall itself.  The adapter's checkout guard must accept the
    admitted tree, hand back THAT row's identity for the receipts, and
    refuse the two impostors with the spelling every earlier refusal used."""

    from hexcore import cuda_arwen_physics_v841 as adapter

    root, private, row = _admitted_tree(tmp_path, monkeypatch)
    surface = sha256()
    for relative in ("gpuwm/core/mpas_column_batch.py", "docs/mpas-seam.md"):
        surface.update((private / relative).read_bytes())
    monkeypatch.setattr(adapter, "MPAS_SEAM_CONTRACT_SURFACE_SHA256", surface.hexdigest())

    identity = adapter._verify_checkout_root(private)
    assert identity.admitted and identity.version == row.version
    assert identity.build_commit == row.build_commit
    assert identity.manifest == row.manifest
    assert identity.glacier_composed_tu_sha256 == row.glacier_composed_tu_sha256

    (private / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\nversion = "2.7.3"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version string and its bytes together"):
        adapter._verify_checkout_root(private)

    (private / "pyproject.toml").write_text(
        f'[project]\nname = "gpuwm"\nversion = "{row.version}"\n', encoding="utf-8"
    )
    (private / "gpuwm/io/restart.py").write_bytes(b"RESTART = 3\n")
    with pytest.raises(
        ValueError,
        match="pinned Arwen source hash mismatch for gpuwm/core/microphysics.py",
    ):
        adapter._verify_checkout_root(private)

    # And the public tree still resolves to the public pin's identity, with
    # the surface digest the public tree actually has.
    surface = sha256()
    for relative in ("gpuwm/core/mpas_column_batch.py", "docs/mpas-seam.md"):
        surface.update((root / relative).read_bytes())
    monkeypatch.setattr(adapter, "MPAS_SEAM_CONTRACT_SURFACE_SHA256", surface.hexdigest())
    public = adapter._verify_checkout_root(root)
    assert not public.admitted and public.version == engine_pin.wanted_version()
    assert public.build_commit == adapter.ARWEN_BUILD_COMMIT
