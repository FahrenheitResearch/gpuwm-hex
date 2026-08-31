"""Writers that land on a TRACKED file write LF, whatever the platform.

THE BREAKAGE THIS PREVENTS, and why the blobs cannot show it.
``.gitattributes`` here declares ``* text=auto eol=lf``, so git NORMALIZES
every text file on the way into the object database.  That is the right
setting -- ``tests/test_checkout_reproducibility.py`` records what it cost
to learn it -- but it has one consequence that is easy to miss: a CRLF
write to a tracked file leaves the blob perfectly clean while the WORKING
TREE keeps the carriage returns.  Nothing that reads the object database
can see it, and ``git status`` stays clean because git normalizes before
it compares.

Measured in this checkout on 2026-08-29: twelve tracked non-binary files
carry 1,261 CR bytes on disk and ZERO in their blobs -- six ``.txt``
transcripts under ``evidence/assembly-rehearsal-20260828/``, four under
``evidence/repin-258-20260828/``, and ``PLAIN-LANGUAGE.md`` and
``RECEIPT.md`` under ``evidence/cpr-bias-20260828/``.  Those came from
shell redirection on Windows rather than from Python, and they are
harmless as transcripts; they are recorded here because they show the
class is real and that no blob-reading check can find it.

Where it stops being harmless is any digest taken FROM DISK.  This port
pins ``.nc``, ``.npy`` and ``.npz`` payloads by SHA-256 through its proof
guards, and a receipt that hashes its own inputs on a Windows machine
disagrees with the same commit on a Linux one, with both blobs identical.

``Path.write_text`` is the Python half of that: it opens in TEXT mode, so
on Windows it translates every "\\n" to "\\r\\n" on the way out.  That is
the right default for scratch and run directories, which is where nearly
all of this repository's writes go.  Four call sites, in three files, had
a receiver that resolved to a path this repository TRACKS, and those four
now pass ``newline="\\n"``.  This module pins them red-on-revert.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: `(script that writes, the tracked file it writes)`.  Each entry was
#: reached by folding the call's receiver through module-scope
#: assignments -- `HERE = Path(__file__).resolve().parent` in all three --
#: and checking the result against `git ls-files`.  Sites resolving to a
#: run directory, an argparse destination or an absolute path on a rented
#: node are deliberately absent: `evidence/cpr-bias-20260828/tools/
#: patch_class.py` and its siblings rewrite a source file inside a rented
#: node's snapshot tree, which is neither this checkout nor reachable from
#: Windows.
_WRITERS = (
    ("tree/evidence/memory-row-refit-20260826/refit.py",
     "tree/evidence/memory-row-refit-20260826/REFIT.json"),
    ("tree/evidence/memory-shape-20260827/confirm_366_on_card.py",
     "tree/evidence/memory-shape-20260827/ON-CARD-366.json"),
    ("tree/evidence/memory-shape-20260827/validate.py",
     "tree/evidence/memory-shape-20260827/VALIDATION.json"),
)

#: Text-mode writes left alone, per script, keyed by the receiver as
#: `ast.unparse` spells it, because the destination is not tracked.
#: `confirm_366_on_card.py` writes an `owner` file into a mutex directory
#: it creates and deletes in the same run.
_UNTRACKED_TEXT_WRITES = {
    "tree/evidence/memory-shape-20260827/confirm_366_on_card.py":
        {"MUTEX / 'owner'"},
}

#: An unpacked sdist is not a repository: no `.gitattributes`, no `tree/`
#: beside it, and `git ls-files` has nothing to answer.  Skipping there
#: names what went unverified instead of reading as a pass, the same way
#: `test_checkout_reproducibility.py` does.
_IN_REPOSITORY = ((REPO_ROOT / ".gitattributes").is_file()
                  and (REPO_ROOT / "tree").is_dir())

pytestmark = pytest.mark.skipif(
    not _IN_REPOSITORY,
    reason=("not a repository checkout, so the writers' destinations cannot "
            "be checked against git ls-files from here"))


def _shipped_or_skip(relative: str) -> Path:
    """The path, or a by-name skip in a tree that holds `tree/evidence/` out.

    The three writers and their destinations live under `tree/evidence/`,
    which the public assembly ships only `EVIDENCE.md` of.  A tree that does
    not carry the file cannot have it checked from here, and #378 closed
    exactly this shape thirty times over: a direct read fires a red CI on
    the public release commit while the private tree, which carries the
    evidence, keeps the gate real.  Measured on the 0.2.1 assembly's own
    battery before this guard existed: 9 failed, every one in this module.
    """

    path = REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not in this tree; this gate runs where "
                    "tree/evidence/ ships, which the public assembly does not")
    return path


@pytest.mark.parametrize("script,destination", _WRITERS)
def test_the_writer_uses_no_translating_write(script, destination):
    """No text-mode write without an explicit `newline` survives here.

    Scoped to the three files whose writes land on a tracked path, not to
    the repository: a rule over every `write_text` call would be a style
    gate with no breakage behind it, since the overwhelming majority write
    run directories where the platform's line ending is correct.
    """

    source = _shipped_or_skip(script)
    allowed = _UNTRACKED_TEXT_WRITES.get(script, set())
    tree = ast.parse(source.read_bytes().decode("utf-8"))
    translating = [
        (node.lineno, ast.unparse(node.func.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
        and not any(kw.arg == "newline" for kw in node.keywords)
        and ast.unparse(node.func.value) not in allowed]
    assert not translating, (
        f"{script} writes {destination} and still has a text-mode write at "
        f"(line, receiver) {translating}.  On Windows that translates every "
        "newline on the way out, and because `* text=auto eol=lf` normalizes "
        "the blob, the damage would be invisible to every check that reads "
        "the object database.  Pass newline='\\n', or write_bytes -- or, if "
        "that receiver is genuinely untracked, name it in "
        "_UNTRACKED_TEXT_WRITES.")


@pytest.mark.parametrize("script,destination", _WRITERS)
def test_the_destination_is_tracked_and_still_lf(script, destination):
    """The list stays honest: tracked, present, and LF on disk right now.

    On disk rather than in the blob, because the blob is the one place
    this defect can never appear: `* text=auto eol=lf` strips the CR
    before the object exists.  If a destination stops being tracked the
    entry belongs elsewhere; if one is already CRLF, the writer above it
    has been run on Windows and this pin arrived after the fact.
    """

    target = _shipped_or_skip(destination)
    listed = subprocess.check_output(
        ["git", "ls-files", "--error-unmatch", "--", destination],
        cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True)
    assert listed.strip() == destination
    raw = target.read_bytes()
    assert b"\r" not in raw, (
        f"{destination} carries {raw.count(chr(13).encode())} CR bytes on "
        f"disk while its blob is clean; the writer in {script} has run on "
        "Windows")


@pytest.mark.parametrize("script,destination", _WRITERS)
def test_the_fixed_spelling_survives_what_the_old_one_did_not(
        script, destination, tmp_path):
    """The two spellings, over the destination's own bytes, side by side.

    `newline="\\n"` has to reproduce the file exactly.  The spelling it
    replaced has to be caught doing the damage on this platform rather
    than asserted to -- without that half, a green result proves only that
    nothing was ever at stake.  On a POSIX runner `os.linesep` is "\\n"
    and there is nothing to demonstrate, so that half is skipped there and
    the round trip stands alone; Windows is where the failure lives.
    """

    real = _shipped_or_skip(destination).read_bytes()
    text = real.decode("utf-8")
    copy = tmp_path / Path(destination).name

    copy.write_text(text, encoding="utf-8", newline="\n")
    assert copy.read_bytes() == real

    if os.linesep == "\r\n" and "\n" in text:
        copy.write_text(text, encoding="utf-8")
        assert b"\r" in copy.read_bytes(), (
            "the text-mode writer did not translate, so this platform is "
            "not the one that produced the defect and this half of the test "
            "is proving nothing")
        assert copy.read_bytes() != real
