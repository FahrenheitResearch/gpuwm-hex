"""Verify f0's accepted input ledger before preparing proof fixtures.

This is receipt orchestration only; NetCDF fields are decoded by the engine Rust tools.
The concrete failure prevented is reuse of f0's withdrawn publication7d5bc404,
documented in ref/READY.md (native Noah aborts on geography category21).
"""
from pathlib import Path
import hashlib
import json
import re
import sys


def _mentions_directory(line, directory):
    """Match immutable ledger paths through storage symlinks."""
    return any(Path(path).resolve() == directory
               for path in re.findall(r"`(/[^`]+)`", line))


def accepted_publication(input_path, ready_path):
    input_path = Path(input_path).resolve()
    publication = input_path.parent / "PUBLICATION.json"
    data = publication.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    metadata = json.loads(data)
    lines = Path(ready_path).read_text().splitlines()
    if not any(_mentions_directory(line, input_path.parent) and digest in line
               for line in lines):
        raise ValueError("the input publication is not accepted in f0 READY.md")
    if any(line.startswith("WITHDRAWN") and (
            digest[:8] in line or _mentions_directory(line, input_path.parent))
           for line in lines):
        raise ValueError("f0 withdrew this publication; use its accepted replacement")
    expected = metadata["files"][input_path.name]
    return {"publication_sha256": digest, "input_sha256": expected,
            "publication_path": str(publication), "input_path": str(input_path)}


if __name__ == "__main__":
    input_path, ready_path, receipt_path, *columns = sys.argv[1:]
    record = accepted_publication(input_path, ready_path)
    if columns:
        decoded = json.loads(Path(columns[0]).read_text())
        if decoded["input_sha256"] != record["input_sha256"]:
            raise ValueError("the Rust-decoded input differs from f0's published hash")
        record["column_sha256"] = decoded["column_sha256"]
        record["decoder_source_revision"] = decoded["source_revision"]
    Path(receipt_path).write_text(json.dumps(record, indent=2) + "\n")
