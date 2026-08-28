"""No tracked text file may carry an unresolved merge-conflict marker.

The concrete breakage this prevents: the 2026-08-25 L1 merge shipped
``>>>>>>> the regional admission work` inside CHANGELOG.md because nothing
in the battery reads prose files for marker residue, and the next merge
then conflicted against the stale marker itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_TEXT_SUFFIXES = {".py", ".md", ".rs", ".toml", ".json", ".yml", ".yaml", ".cfg", ".ini", ".txt"}
_MARKER_PREFIXES = ("<<<<<<< ", ">>>>>>> ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_tracked_text_files_carry_no_conflict_markers() -> None:
    root = _repo_root()
    try:
        listing = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: git ls-files unavailable, marker scan has no file list")

    offenders: list[str] = []
    for rel in listing.splitlines():
        path = root / rel
        if path.suffix.lower() not in _TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.startswith(_MARKER_PREFIXES):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:60]}")

    assert not offenders, (
        "unresolved merge-conflict markers are present in tracked files; "
        "a merge was concluded without cleaning every marker:\n" + "\n".join(offenders)
    )
