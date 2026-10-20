"""Authenticate original native schemas for honest surface-product omissions.

Registry.xml:3029-3042; mpas_atmphys_control.F:155-167 disables sfclayer
with physics off. Only Rust decodes file headers; Python handles proof metadata.
No native field is synthesized and no numerical acceptance is granted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def main(stage, snapshot, out):
    stage, snapshot, out = stage.resolve(), snapshot.resolve(), out.resolve()
    if stage.name != "native-sheet-header-control-v1" or stage.parent.name != "proof-tools":
        raise ValueError("header control may retire only its explicitly owned transfer stage")
    authority = load(stage / "AUTHORITY.json")
    if sha(stage / "native-run.json") != authority["native_manifest_sha256"]:
        raise ValueError("changed original native authority")
    tool = stage / "rw_mpas_verify"
    if sha(tool) != authority["comparator_sha256"]:
        raise ValueError("changed qualified Rust comparator")
    results = []
    for name, expected in authority["files"].items():
        raw = stage / name
        if sha(raw) != expected:
            raise ValueError("changed native input bytes")
        run = subprocess.run([str(tool), "--describe", str(raw)], capture_output=True,
                             text=True, check=True)
        metadata = json.loads(run.stdout)
        if set(metadata["fields"]).intersection(("t2m", "u10", "v10")):
            raise ValueError("native surface field exists; it must be rendered")
        if sha(raw) != expected:
            raise ValueError("Rust header control changed native bytes")
        results.append(dict(path=name, sha256=expected, metadata=metadata))
    inputs = load(snapshot / "snapshot.json")
    measured = []
    for case in inputs["cases"]:
        path = snapshot / case["verify"]
        if sha(path) != inputs["files"][case["verify"]]:
            raise ValueError("changed original forecast receipt")
        receipt = load(path)
        rows = [r for r in receipt["comparison"]["fields"] if r["stream"] == "output"]
        times = sorted({r["time"] for r in rows})
        for time in times:
            fields = sorted(r["field"] for r in rows if r["time"] == time)
            if not fields or set(fields).intersection(("t2m", "u10", "v10")):
                raise ValueError("original native output schema has surface fields")
            measured.append(dict(index=case["index"], time=time, native_fields=fields,
                                 original_verifier_receipt_sha256=sha(path)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema="d1-native-surface-sheet-schema-controls-v1",
        host=socket.gethostname(), numerical_acceptance=False, new_images_claimed=False,
        absent_native_fields=["t2m", "u10", "v10"], native_header_results=results,
        original_forecast_schema_results=measured, all_input_bytes_unchanged=True,
        original_native_authority=authority, snapshot_sha256=sha(snapshot / "snapshot.json"),
        orchestration_sha256=sha(Path(__file__)),
        scope="Three original full-day native histories plus every measured one-step output schema; native feature absence only"),
        indent=2) + "\n")
    shutil.rmtree(stage)  # Only this job's disposable, byte-authenticated copies.
    print(json.dumps(dict(native_headers=len(results), measured_samples=len(measured),
                          numerical_acceptance=False, new_images_claimed=False)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.stage, args.snapshot, args.out)
