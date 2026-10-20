"""Persist the deck's CPU references through the existing exact prep codec.

No pickle, numerical conversion or field implementation lives here. Arrays and
dataclasses use prep_cache's bit-preserving codec. Input bytes and producer
identity are checked again before a device process consumes the references.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path

from hexcore import prep_cache

SCHEMA = "gpuwm-hex.contract-host/v2"


def source_digest():
    """Bind CPU reference producers across separate build and GPU hosts."""
    package = Path(prep_cache.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _key(value):
    if isinstance(value, str):
        return {"string": value}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, tuple):
        return {"tuple": [_key(item) for item in value]}
    raise TypeError("unsupported reference key; prevents a lossy host-deck snapshot")


def _unkey(value):
    if len(value) != 1:
        raise ValueError("host reference key malformed; prevents rebinding cached references")
    if "string" in value:
        return value["string"]
    if "bytes" in value:
        return bytes.fromhex(value["bytes"])
    if "datetime" in value:
        return datetime.fromisoformat(value["datetime"])
    if "tuple" in value:
        return tuple(_unkey(item) for item in value["tuple"])
    raise ValueError("host reference key unknown; prevents rebinding cached references")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(bundle, selected):
    return {"schema": SCHEMA, "producer_sources": source_digest(),
            "grid": _sha(bundle.grid_path), "init": _sha(bundle.init_path),
            "lbc": [[Path(path).name, _sha(path)] for path in bundle.lbc_paths],
            "start": bundle.start_time.isoformat(), "decks": [deck.name for deck in selected]}


def save(bundle, selected, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = binding(bundle, selected)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    payload = [{"key": _key(name), "value": value} for name, value in bundle._references.items()]
    arrays = []
    structure = prep_cache._encode_result(payload, arrays)
    digests = [value.hex() for value in prep_cache._array_digests(arrays)]
    prep_cache._store(root / "cache", key, "contract-host-references", structure, arrays, digests)
    record = {"schema": SCHEMA, "binding": identity, "key": key,
              "host_producer": prep_cache.producer_digest(),
              "references": len(payload), "arrays": len(arrays),
              "array_bytes": sum(array.nbytes for array in arrays), "array_sha256": digests}
    path = root / "host-deck.json"
    tmp = root / "host-deck.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return record


def load(bundle, selected, root):
    root = Path(root)
    record = json.loads((root / "host-deck.json").read_text())
    expected = binding(bundle, selected)
    if record.get("schema") != SCHEMA or record.get("binding") != expected:
        raise ValueError("host deck input or producer mismatch; prevents comparing device fields against another analysis")
    key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if record.get("key") != key:
        raise ValueError("host deck key mismatch; prevents selecting another CPU reference payload")
    held = prep_cache._load(root / "cache", key)
    if held is None:
        raise ValueError("host deck arrays missing or corrupt; prevents unverified CPU reference use")
    structure, arrays = held
    decoded = prep_cache._decode_result(structure, arrays)
    references = {_unkey(row["key"]): row["value"] for row in decoded}
    if len(references) != len(decoded) or len(references) != record["references"]:
        raise ValueError("host deck reference inventory changed; prevents dropping a CPU comparison")
    bundle._references.update(references)
    return record
