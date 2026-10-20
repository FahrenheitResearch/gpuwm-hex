"""Seal an owned completed strict native forecast for the real f2 comparator.

Only metadata is processed here. Rust reads netCDF clocks and inventories.
This produces mpas-native-run-v1, without claiming an f0 publication.
"""
import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from hexcore.config import parse_fortran_namelist
from hexcore.verify_mpas import native_manifest, stream_filename_times


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(root, deck_path, comparator, destination):
    root = root.resolve()
    deck = json.loads(deck_path.read_text())
    cache_path = root / "MANIFEST.json"
    cache = json.loads(cache_path.read_text())
    if cache.get("schema") != "mpas.native-forecast-cache/v1" or cache.get("ranks") != 1:
        raise ValueError("requires a completed strict native forecast at one rank")
    if sha(root / "atmosphere_model") != cache["binary_sha256"]:
        raise ValueError("native executable changed after forecast")
    logs = sorted(root.glob("log.atmosphere.*.out"))
    if not logs or not any("Finished running the atmosphere core" in p.read_text() for p in logs):
        raise ValueError("native forecast completion witness absent")
    inputs = {key: ref["sha256"] for key, ref in deck["inputs"].items()}
    for key, ref in deck["inputs"].items():
        path = (deck_path.parent / ref["path"]).resolve()
        if sha(path) != ref["sha256"]:
            raise ValueError("changed primary input")
        if key == "init" and cache["inputs"].get(path.name) != ref["sha256"]:
            raise ValueError("native initialization differs")
    if (cache["namelist_sha256"] != inputs["namelist"]
            or cache["streams_sha256"] != inputs["streams"]
            or cache["pin_set_hash"] != deck["pin_set"]):
        raise ValueError("native namelist, streams or pin set differs")
    auxiliary = {name: ref["sha256"] for name, ref in deck["auxiliary_inputs"].items()}
    for name, ref in deck["auxiliary_inputs"].items():
        if sha((deck_path.parent / ref["path"]).resolve()) != ref["sha256"] or cache["inputs"].get(name) != ref["sha256"]:
            raise ValueError(f"native auxiliary input differs: {name}")
    namelist = parse_fortran_namelist((root / "namelist.atmosphere").read_text())
    options = {k: v for group in namelist.values() for k, v in group.items()}
    start = datetime.strptime(options["config_start_time"], "%Y-%m-%d_%H:%M:%S")
    dt = Decimal(str(options["config_dt"]))
    streams = ET.parse(root / "streams.atmosphere").getroot()
    patterns = []
    for stream in streams:
        if "output" not in stream.get("type", ""):
            continue
        pattern = re.escape(stream.get("filename_template", ""))
        for token, width in (("Y", 4), ("M", 2), ("D", 2), ("h", 2), ("m", 2), ("s", 2)):
            pattern = pattern.replace(re.escape("$" + token), rf"(?P<{token}>\d{{{width}}})")
        patterns.append((stream, re.compile(pattern)))
    outputs = []
    schedule = {}
    for name, expected_sha in cache["outputs"].items():
        path = root / name
        if path.is_symlink() or sha(path) != expected_sha:
            raise ValueError("native output changed")
        matched = [(s, p.fullmatch(name)) for s, p in patterns if p.fullmatch(name)]
        if len(matched) != 1:
            raise ValueError("output stream mapping is ambiguous")
        stream, match = matched[0]
        describe = subprocess.run([str(comparator), "--describe", str(path)],
                                  capture_output=True, text=True, check=True)
        metadata = json.loads(describe.stdout)
        times = metadata["times"]
        row = {"stream": stream.get("name"), "path": name, "sha256": expected_sha}
        if times is None:
            times = stream_filename_times(stream, match.groupdict(), metadata["records"])
            row["clock_witness"] = {"filename_template": stream.get("filename_template")}
        steps = []
        for stamp in times:
            elapsed = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S") - start
            seconds = Decimal(elapsed.days * 86400 + elapsed.seconds)
            if seconds < 0 or seconds % dt:
                raise ValueError("native output off timestep clock")
            steps.append(int(seconds / dt))
        row.update(times=times, steps=steps)
        outputs.append(row)
        schedule.setdefault(row["stream"], []).extend(steps)
    measured = {stream: sorted(steps) for stream, steps in schedule.items()}
    if measured != deck["expected_samples"]:
        raise ValueError(f"native output schedule differs: {measured}")
    result = dict(schema="mpas-native-run-v1", pin_set=deck["pin_set"],
        process_count=1, ranks=1, complete=True, output_root=str(root),
        inputs=inputs, auxiliary_inputs=auxiliary, outputs=outputs,
        producer_cache_sha256=sha(cache_path), comparator_sha256=sha(comparator),
        binary_sha256=cache["binary_sha256"],
        completion_logs={p.name: sha(p) for p in logs},
        scope="owned strict native forecast; metadata sealed through Rust; no Hex identity claim")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    ref = {"path": str(destination.resolve()), "sha256": sha(destination)}
    # The actual verifier authenticates the complete inventory, including files
    # omitted from a producer manifest. No handcrafted field exclusions.
    native_manifest(ref, deck_path.parent, deck, comparator)
    deck["native"] = ref
    deck_path.write_text(json.dumps(deck, indent=2) + "\n")
    print(json.dumps({"manifest": ref, "outputs": len(outputs), "schedule": measured}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "deck", "comparator", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    seal(args.root, args.deck, args.comparator, args.destination)
