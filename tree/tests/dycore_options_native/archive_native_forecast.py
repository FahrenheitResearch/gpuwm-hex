"""Losslessly retain lane-owned forecast bytes with a full extraction witness.

Scientific files are never decoded here. GNU tar/zstd transport original bytes;
all native inventory hashes must match before any owned output is retired.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(manifest, owner):
    data = json.loads(manifest.read_text())
    if data.get("schema") != "mpas-native-run-v1" or data.get("complete") is not True:
        raise ValueError("only authenticated complete native forecasts can be archived")
    root = Path(data["output_root"]).resolve()
    if not root.is_relative_to((owner.resolve() / "scratch").resolve()):
        raise ValueError("output retirement is restricted to the lane's owned scratch")
    files = {row["path"]: row["sha256"] for row in data["outputs"]}
    if len(files) != len(data["outputs"]) or any(Path(name).name != name for name in files):
        raise ValueError("archive inventory must contain distinct local file names")
    return root, files


def verify(root, files):
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise ValueError(f"native bytes changed: {root / name}")


def archive(manifest, owner, archive_root, receipt):
    root, files = inventory(manifest, owner)
    digest = sha(manifest)
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / (digest + ".tar.zst")
    if target.exists() or receipt.exists():
        raise ValueError("an archive or receipt already exists")
    verify(root, files)
    names = archive_root / (digest + ".files")
    names.write_text("\n".join(sorted(files)) + "\n")
    partial = target.with_suffix(".partial")
    subprocess.run(["tar", "-I", "zstd -T4 -3", "-cf", str(partial),
                    "-C", str(root), "-T", str(names)], check=True)
    stage = archive_root / (digest + "-restored")
    stage.mkdir(exist_ok=False)
    try:
        subprocess.run(["tar", "-I", "zstd", "-xf", str(partial), "-C", str(stage)], check=True)
        verify(stage, files)
        verify(root, files)
        if sha(manifest) != digest:
            raise ValueError("native manifest changed during archive")
        partial.rename(target)
        proof = dict(schema="d1-native-byte-archive-v1", native_manifest=str(manifest),
                     native_manifest_sha256=digest, owner_root=str(owner.resolve()),
                     output_root=str(root), archive=str(target), archive_sha256=sha(target),
                     archive_bytes=target.stat().st_size, files=files,
                     restored_all_bytes_verified=True)
        receipt.write_text(json.dumps(proof, indent=2) + "\n")
        for name in files:
            (root / name).unlink()
        print(json.dumps(dict(receipt=str(receipt), files=len(files),
                              archive_bytes=proof["archive_bytes"])), flush=True)
        return proof
    finally:
        shutil.rmtree(stage)


def restore(receipt, manifest):
    proof = json.loads(receipt.read_text())
    if proof.get("schema") != "d1-native-byte-archive-v1" or not proof.get("restored_all_bytes_verified"):
        raise ValueError("archive lacks the full-byte extraction witness")
    root, files = inventory(manifest, Path(proof["owner_root"]))
    if (sha(manifest) != proof["native_manifest_sha256"] or files != proof["files"]
            or root != Path(proof["output_root"]).resolve()):
        raise ValueError("archive differs from this native inventory")
    target = Path(proof["archive"])
    if sha(target) != proof["archive_sha256"]:
        raise ValueError("archive bytes changed")
    # Run directories are disposable too. Recreate only the authenticated
    # lane-owned scratch root, after validating the original archive bytes.
    root.mkdir(parents=True, exist_ok=True)
    present = [name for name in files if (root / name).exists()]
    verify(root, {name: files[name] for name in present})
    missing = sorted(set(files) - set(present))
    if missing:
        subprocess.run(["tar", "-I", "zstd", "-xf", str(target), "-C", str(root), *missing], check=True)
    verify(root, files)
    return root, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    archive(args.manifest.resolve(), args.owner_root.resolve(),
            args.archive_root.resolve(), args.receipt.resolve())


if __name__ == "__main__":
    main()
