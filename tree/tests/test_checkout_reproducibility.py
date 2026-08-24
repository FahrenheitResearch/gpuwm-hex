"""The checkout is LF on every platform, so the built artefact is one artefact.

THE BREAKAGE THESE PREVENT, measured 2026-08-21.  This repository is developed
with ``core.autocrlf=false``, so its working tree is LF.  A fresh ``git clone``
on Windows inherits Git-for-Windows' default ``core.autocrlf=true``, converts
the whole tree to CRLF on checkout, and ``python -m build`` packs those CRLF
bytes into the wheel and the sdist.

Measured: the wheel built from such a clone carried a ``LICENSE`` of 11,063
bytes where this tree's is 10,861 -- the same text, 202 CRLF bytes of
difference, a different SHA-256.  ``NOTICE`` moved the same way.  Those two
files are the entire licence posture of the distribution, they land in
``dist-info/licenses/`` where a downstream consumer reads them, and a file
whose hash depends on who built it cannot be pinned by anyone.

``.gitattributes`` is what stops it.  Without it the distribution's bytes are a
property of the builder's machine rather than of the source.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTES = REPO_ROOT / ".gitattributes"

#: These tests are about the REPOSITORY, not about the distribution, and an
#: unpacked sdist is not a repository: it has no ``.gitattributes``, no
#: ``tree/`` directory, and ``parents[2]`` points at whatever happens to sit
#: above the unpacked root.  Run there, every assertion below fails on a
#: missing file and reads as a licence defect in a distribution whose licence
#: bytes are in fact correct -- the CRLF question was already settled at build
#: time, which is the only place it can be settled.
#:
#: So the guard is presence of the repository layout, and the skip names what
#: went unverified rather than reading as a pass.
_IN_REPOSITORY = ATTRIBUTES.is_file() and (REPO_ROOT / "tree").is_dir()

pytestmark = pytest.mark.skipif(
    not _IN_REPOSITORY,
    reason=(
        "not a repository checkout (no .gitattributes and no tree/ beside it), "
        "so the line-ending posture of the SOURCE cannot be checked from here. "
        "These bytes were fixed when this distribution was built; run this "
        "test in the repository to verify the source that produced it."
    ),
)

#: Files whose bytes the licence posture depends on.  They are checked in both
#: locations because the sdist takes them from ``tree/`` and the repository
#: presents them at the root.
LICENCE_FILES = (
    Path("LICENSE"),
    Path("NOTICE"),
    Path("tree/LICENSE"),
    Path("tree/NOTICE"),
)


def test_the_repository_pins_line_endings_for_every_text_file() -> None:
    assert ATTRIBUTES.is_file(), (
        f"{ATTRIBUTES} is missing.\n"
        "  the breakage this refuses: without it, a clone made with "
        "core.autocrlf=true (the Git-for-Windows default) checks the tree out "
        "as CRLF, and `python -m build` packs those bytes into the wheel. The "
        "LICENSE inside dist-info/licenses/ then hashes differently depending "
        "on who built the wheel -- 11,063 bytes instead of 10,861, measured. "
        "The distribution's licence bytes must be a property of the source, "
        "not of the builder's machine."
    )

    rules = ATTRIBUTES.read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in rules, (
        ".gitattributes exists but does not force LF for every text file. "
        "The wildcard rule `* text=auto eol=lf` is the one that survives a "
        "cloner's core.autocrlf setting; a per-extension list does not, "
        "because the file that gets added next is the one nobody listed."
    )


@pytest.mark.parametrize("relative", LICENCE_FILES, ids=lambda p: str(p))
def test_the_licence_files_are_lf_in_this_checkout(relative: Path) -> None:
    path = REPO_ROOT / relative
    assert path.is_file(), f"{relative} is missing from the checkout"

    raw = path.read_bytes()
    assert b"\r\n" not in raw, (
        f"{relative} carries CRLF in this checkout ({len(raw)} bytes).\n"
        "  the breakage: these bytes go into the wheel verbatim, so the "
        "licence file a consumer reads out of dist-info/licenses/ would hash "
        "differently from the one in the repository. Re-clone with "
        "`git -c core.autocrlf=false clone`, or `git add --renormalize .` if "
        "the CRLF reached the index."
    )


def test_the_two_licence_locations_are_the_same_bytes() -> None:
    """The root pair and the packaged pair must not be allowed to drift.

    ``tree/LICENSE`` and ``tree/NOTICE`` are what the sdist and the wheel
    carry; the root pair is what a reader of the repository sees.  If they ever
    differ, one of the two audiences is reading a licence the artefact does not
    ship.
    """

    for name in ("LICENSE", "NOTICE"):
        root = (REPO_ROOT / name).read_bytes()
        packaged = (REPO_ROOT / "tree" / name).read_bytes()
        assert root == packaged, (
            f"{name} differs between the repository root ({len(root)} bytes) "
            f"and tree/ ({len(packaged)} bytes). The repository would then "
            "state one licence and the distribution ship another."
        )
