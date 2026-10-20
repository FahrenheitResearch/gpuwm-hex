"""Receipt-ledger aliases must not bypass pin or withdrawal checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("d1_publication",
    Path(__file__).parent / "dycore_options_native/check_publication.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def publication(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "ref"
    alias.symlink_to(actual, target_is_directory=True)
    field = actual / "input.nc"
    field.write_bytes(b"immutable input fixture")
    manifest = actual / "PUBLICATION.json"
    manifest.write_text(json.dumps({"files": {
        field.name: hashlib.sha256(field.read_bytes()).hexdigest()}}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ready = tmp_path / "READY.md"
    ready.write_text(f"Accepted: `{alias}`; sha256 `{digest}`.\n")
    return actual, alias, digest, ready


def test_accepts_canonical_storage_alias(publication):
    actual, alias, digest, ready = publication
    for root in (actual, alias):
        result = module.accepted_publication(root / "input.nc", ready)
        assert result["publication_sha256"] == digest
        assert result["input_path"] == str(actual / "input.nc")


def test_alias_cannot_admit_another_directory(publication):
    actual, _, digest, ready = publication
    ready.write_text(f"Accepted: `{actual.parent}`; sha256 `{digest}`.\n")
    with pytest.raises(ValueError, match="not accepted"):
        module.accepted_publication(actual / "input.nc", ready)


def test_withdrawal_wins_through_alias(publication):
    actual, alias, digest, ready = publication
    with ready.open("a") as stream:
        stream.write(f"WITHDRAWN: `{alias}`; publication{digest[:8]}.\n")
    with pytest.raises(ValueError, match="withdrew"):
        module.accepted_publication(actual / "input.nc", ready)
