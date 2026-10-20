"""Repeat entire pinned native decks; Rust compares every stored output word.

This is a native stability witness, never evidence of a CUDA port forecast.
The caller supplies existing binaries, exact deck receipts and owned storage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path):
    return json.loads(Path(path).read_text())


def run(command, log):
    with log.open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    log.with_suffix(log.suffix + ".exit").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError(f"Native witness command failed ({result.returncode}): {log}")


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    index = load(args.decks)
    rows = index["decks"]
    if not rows or args.ranks < 1:
        raise ValueError("Nonempty exact deck list and fixed positive MPI rank count required")
    os.environ["OMP_NUM_THREADS"] = "1"
    result = dict(schema="gf-native-full-repeat-matrix-v1", process_count=args.ranks,
        run_host=os.uname().nodename, deck_index=pin(args.decks),
        native_executable=pin(args.build / "atmosphere_model"),
        runner=pin(args.runner), comparison_runner=pin(args.compare),
        comparator=pin(args.comparator), decks=[], forecast_port_proven=False,
        exclusions_proposed=[], mapped_native_defect_word_set=False,
        scope="Whole pinned native forecasts repeated twice at one fixed process count; "
              "no CUDA forecast, rank invariance or native fault-site reproduction claimed")
    for entry in rows:
        deck_path = Path(entry["path"]).resolve(strict=True)
        if pin(deck_path)["sha256"] != entry["sha256"]:
            raise ValueError("Deck receipt pin changed: " + str(deck_path))
        deck = load(deck_path)
        root = args.out / entry["id"]
        root.mkdir()
        inputs = root / "inputs"
        inputs.mkdir()
        for key, filename in (("namelist", "namelist.atmosphere"),
                              ("streams", "streams.atmosphere")):
            source = Path(deck["inputs"][key]["path"])
            if pin(source)["sha256"] != deck["inputs"][key]["sha256"]:
                raise ValueError("Published native input pin changed: " + str(source))
            shutil.copy2(source, inputs / filename)
        command = deck["hex_command"]
        case = Path(command[command.index("--grid") + 1]).parent
        native_runs = []
        for repeat in (1, 2):
            native_run = root / f"repeat{repeat}"
            run([sys.executable, str(args.runner), "--build", str(args.build),
                 "--case", str(case), "--deck", str(inputs), "--out", str(native_run),
                 "--tables", str(args.tables), "--mpi", str(args.mpi),
                 "--ranks", str(args.ranks), "--pin-set-hash", args.pin_set],
                root / f"repeat{repeat}.log")
            manifest = load(native_run / "MANIFEST.json")
            if (manifest["ranks"] != args.ranks
                or manifest["binary_sha256"] != result["native_executable"]["sha256"]
                or manifest["namelist_sha256"] != deck["inputs"]["namelist"]["sha256"]
                or manifest["streams_sha256"] != deck["inputs"]["streams"]["sha256"]):
                raise ValueError("Native executable, rank or exact deck changed")
            native_runs.append(native_run)
        comparison = root / "repeat-identity.json"
        run([sys.executable, str(args.compare), *map(str, native_runs),
             "--comparator", str(args.comparator), "--out", str(comparison)],
            root / "compare.log")
        identity = load(comparison)
        if (identity.get("all_identical") is not True
            or identity.get("all_floating_words_identical") is not True
            or identity.get("errors") or not identity["files"]
            or any(v["differing_words"] for v in identity["files"].values())):
            raise ValueError("Full native repeats differ: " + entry["id"])
        result["decks"].append(dict(id=entry["id"], source_deck=pin(deck_path),
            exact_input_pins=deck["inputs"], process_count=args.ranks,
            manifests=[pin(p / "MANIFEST.json") for p in native_runs],
            comparison=pin(comparison), output_files=len(identity["files"]),
            compared_words=sum(v["words"] for v in identity["files"].values()),
            differing_words=0, all_file_bytes_identical=identity["all_file_bytes_identical"]))
        (args.out / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result["decks"][-1].items()
                          if k in ("id", "output_files", "compared_words", "differing_words")}),
              flush=True)
    result["all_native_repeats_word_equal"] = True
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("decks", "out", "receipt", "build", "tables", "mpi", "runner",
                "compare", "comparator"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--pin-set", required=True)
    main(p.parse_args())
