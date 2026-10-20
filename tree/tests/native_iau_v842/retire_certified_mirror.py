"""Retire owned mirrors only after matching complete Rust-certified file hashes."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("run", "certificate", "receipt"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
work = Path(__file__).resolve().parents[4]
run = args.run.resolve()
assert run.is_relative_to(work), "mirror must belong to this lane"
assert args.receipt.resolve().is_relative_to(work) and not args.receipt.exists()
certificate = json.loads(args.certificate.read_text())
assert certificate["schema"] == "mpas.iau-compared-output-retirement/v1"
certified = {row["file"]: row for row in certificate["files"]}
actual = {path.name: path for path in run.glob("*.nc") if not path.is_symlink()}
assert set(actual) <= set(certified), "unexpected raw output is preserved"
rows = []
for name, path in sorted(actual.items()):
    row = certified[name]
    assert "variables" in row["inventory"]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    assert path.stat().st_size == row["bytes"] and digest.hexdigest() == row["sha256"], name
    rows.append(dict(file=name, bytes=row["bytes"], sha256=row["sha256"]))
receipt = dict(schema="mpas.iau-certified-mirror-retirement/v1", run=str(run),
    certificate=str(args.certificate),
    certificate_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
    files=rows, bytes=sum(row["bytes"] for row in rows))
args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
for path in actual.values():
    path.unlink()
print(json.dumps(dict(files=len(rows), bytes=receipt["bytes"])))
