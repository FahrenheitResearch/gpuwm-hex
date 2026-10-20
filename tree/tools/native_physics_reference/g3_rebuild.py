#!/usr/bin/env python3
"""G3: the native reference rebuilds from this tree's recipe and reproduces the pinned references.

THE BREAKAGE THIS PREVENTS (release gate g; the project rule that "a claim whose
reference cannot be rebuilt does not exist"): every G1 and G2 proof compares hex with
native outputs that one strict binary, built once, wrote.  If the recipe in
tools/native_physics_reference no longer rebuilds a binary whose forecasts
match those outputs word for word, every bitwise claim rests on an artifact
nobody can reproduce, and nothing would notice until the binary was lost.

G3 rebuilds that reference from scratch from the pinned MPAS-Model source,
on whatever host runs it, and requires:

1. build:   build_reference.sh (the recipe G1/G2's binary came from) builds
            the atmosphere core; its compiler-flag audit passes.
2. tables:  the rebuilt build_tables writes the four Thompson DBL files, each
            byte-identical to the pinned table manifest.
3. run:     every deck row of the G3 index runs on the rebuilt binary with
            run_forecast.py, from exactly the pinned row's inputs.
4. compare: every output file of every row equals the pinned reference in
            every field word (the Rust comparator).  Whole-file bytes may
            differ: MPAS writes a random file_id attribute into each file.

Hex is then equal to the rebuilt reference wherever G1/G2 show hex equal to
the pinned one on the same tree; mpas_nightly.sh runs both.

    g3_rebuild.py build   --index I --work W      # compilers; any host
    g3_rebuild.py run     --index I --work W      # same host as build
    g3_rebuild.py compare --index I --work W --comparator C --receipt R
                                                  # host holding the references

$NATIVE_BUILD_WRAPPER, when set, prefixes the build (a shared host's build
queue).  No index entry is a default: meshes, dates, decks and paths are table
rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slotted(command: list[str]) -> list[str]:
    wrapper = os.environ.get("NATIVE_BUILD_WRAPPER", "").split()
    return wrapper + command


def read_index(path: Path) -> dict:
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != "mpas-g3-index-v1":
        raise SystemExit(f"{path}: not an mpas-g3-index-v1 table")
    if not index.get("decks"):
        raise SystemExit("G3 has no deck rows; an empty gate cannot pass")
    return index


def build_dir(index: dict, work: Path) -> Path:
    return work / "native" / f"v8.4.2-{index['recipe']}-atmosphere"


def build(index: dict, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    command = slotted(["bash", str(HERE / "build_reference.sh"), str(work), index["source"],
                       index["recipe"], "atmosphere", index["mpi_prefix"], index["pnetcdf_prefix"]])
    with (work / "build.stdout.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    built = build_dir(index, work)
    tables = work / "tables"
    tables.mkdir(exist_ok=False)
    # build_tables reads nothing and writes the four Thompson tables in its cwd.
    with (tables / "build_tables.log").open("w") as log:
        subprocess.run([str(built / "build_tables")], cwd=tables, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    pinned = json.loads(Path(index["pinned_tables_manifest"]).read_text(encoding="utf-8"))["files"]
    produced = {name: sha256(tables / name) for name in pinned}
    # Publish the rebuild exactly as the reference was published
    # (publish_build.py: the binary, build_tables, the namelist/stream
    # templates and the physics_wrf data files), so every run reads the same
    # input set as the reference run did.
    from publish_build import publish

    published = publish(built, work, "atmosphere", index["recipe"])
    ours = json.loads((published / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    theirs = json.loads((Path(index["pinned_publication"]) / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    host_records = {"toolchain.txt", "build.log", "flag-audit.json"}  # name the host and its paths
    publication_differs = sorted(name for name in set(ours) | set(theirs)
                                 if name not in host_records and ours.get(name) != theirs.get(name))
    record = {"binary_sha256": sha256(built / "atmosphere_model"),
              "pinned_binary_sha256": index.get("pinned_binary_sha256"),
              "binary_identical": sha256(built / "atmosphere_model") == index.get("pinned_binary_sha256"),
              "published": str(published), "publication_differs": publication_differs,
              "toolchain": (built / "toolchain.txt").read_text(encoding="utf-8"),
              "flag_audit_sha256": sha256(built / "flag-audit.json"),
              "build_tables_sha256": sha256(built / "build_tables"),
              "tables": produced, "tables_identical": produced == pinned}
    (work / "build-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if produced != pinned:
        differ = sorted(name for name in pinned if produced[name] != pinned[name])
        raise SystemExit(f"G3 tables: the rebuilt build_tables wrote different {differ}")
    print(f"G3 build: {published} ({record['binary_sha256'][:12]}, byte-identical to the pinned "
          f"binary: {record['binary_identical']}); tables identical; publication differs at "
          f"{publication_differs or 'nothing outside the host records'}")
    return record


def run(index: dict, work: Path) -> None:
    built = Path(json.loads((work / "build-record.json").read_text(encoding="utf-8"))["published"])
    for row in index["decks"]:
        out = work / "forecasts" / row["id"]
        if out.exists():
            raise SystemExit(f"{out} exists; a G3 run never reuses a forecast directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(HERE / "run_forecast.py"), "--build", str(built),
                   "--deck", row["deck"], "--case", row["case"], "--out", str(out),
                   "--tables", str(work / "tables"), "--mpi", index["mpiexec"],
                   "--ranks", str(row["ranks"]), "--pin-set-hash", index["pin_set_hash"]]
        for package in row.get("inactive_packages", []):
            command += ["--inactive-package", package]
        # The index names the MPI runtime the references ran with (its lib
        # directory on LD_LIBRARY_PATH); nothing about it is assumed here.
        environment = dict(os.environ, OMP_NUM_THREADS="1")
        for key, value in index.get("run_environment", {}).items():
            environment[key] = value + (os.pathsep + environment[key] if environment.get(key) else "")
        subprocess.run(command, check=True, env=environment)
        print(f"G3 run: {row['id']} -> {out}")


def compare(index: dict, work: Path, comparator: Path, receipt: Path) -> int:
    from compare_runs import compare as compare_runs

    rows = []
    passed = True
    for row in index["decks"]:
        reference = Path(row["reference"])
        other = work / "forecasts" / row["id"]
        manifests = [json.loads((p / "MANIFEST.json").read_text(encoding="utf-8")) for p in (reference, other)]
        contract = {key: [m.get(key) for m in manifests]
                    for key in ("deck", "ranks", "namelist_sha256", "streams_sha256", "pin_set_hash")}
        inputs_differ = sorted(key for key in set(manifests[0]["inputs"]) | set(manifests[1]["inputs"])
                               if key != "atmosphere_model"
                               and manifests[0]["inputs"].get(key) != manifests[1]["inputs"].get(key))
        result = compare_runs(reference, other, comparator, "*.nc")
        words = sum(f.get("words", 0) for f in result["files"].values())
        differing = sum(f.get("differing_words", 0) for f in result["files"].values())
        ok = (all(a == b for a, b in contract.values()) and not inputs_differ
              and result["all_identical"] and not result["errors"])
        rows.append({"id": row["id"], "reference": str(reference), "rebuilt_run": str(other),
                     "contract": contract, "inputs_differ": inputs_differ,
                     "files": len(result["files"]), "words": words, "differing_words": differing,
                     "all_identical": result["all_identical"],
                     "all_file_bytes_identical": result["all_file_bytes_identical"],
                     "errors": result["errors"], "passed": ok})
        passed &= ok
        print(f"G3 compare: {row['id']}: {len(result['files'])} files, {words} words, "
              f"{differing} differing -> {'PASS' if ok else 'FAIL'}")
    record = json.loads((work / "build-record.json").read_text(encoding="utf-8"))
    passed = passed and record["tables_identical"] and not record.get("publication_differs", ["unrecorded"])
    document = {"schema": "mpas-g3-receipt-v1", "passed": passed,
                "index_sha256": sha256(Path(index["_path"])), "build": record, "decks": rows,
                "note": "whole-file bytes differ only where MPAS writes its random file_id attribute; "
                        "field words are what G3 compares"}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"G3 {'PASSED' if document['passed'] else 'FAILED'}: {receipt}")
    return 0 if document["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("build", "run", "compare"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--comparator", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    index = read_index(args.index)
    index["_path"] = str(args.index.resolve())
    work = args.work.resolve()
    if args.step == "build":
        build(index, work)
    elif args.step == "run":
        run(index, work)
    else:
        if args.comparator is None or args.receipt is None:
            parser.error("compare needs --comparator and --receipt")
        return compare(index, work, args.comparator, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
