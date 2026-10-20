"""Qualify the actual Rust reference carrier against strict GNU intrinsics.

The scientific site is mpas_atm_dissipation_models.F:859-860. Rust owns native
netCDF decoding, RKIND conversion and shared GNU maths; Python records receipts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log, env=None):
    with log.open("w") as output:
        subprocess.run(list(map(str, command)), stdout=output,
                       stderr=subprocess.STDOUT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "native", "rust-checkout", "work-root", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--grid", type=Path, action="append", required=True)
    args = parser.parse_args()
    engine, native = args.engine.resolve(), args.native.resolve()
    rust = args.rust_checkout.resolve()
    revision = subprocess.check_output(["git", "-C", str(rust), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(rust), "status", "--porcelain", "-uno"], text=True):
        raise ValueError("Rust tracked sources are modified")
    abi = subprocess.check_output([str(engine), "--abi"], text=True).strip()
    if "GPUWM_BRIDGE_SOURCE_REV=" + revision not in abi:
        raise ValueError("actual Rust artifact does not name the current clean source")
    root = args.work_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    run([native, "--synthetic", root / "synthetic-input.bin"], root / "synthetic-input.log")
    cases = [("synthetic", "--file-proof", root / "synthetic-input.bin")]
    cases.extend((f"real-{i}", "--grid-proof", grid.resolve()) for i, grid in enumerate(args.grid))
    rows = []
    tests = Path(__file__).resolve().parents[1] / "test_nonles_reference_v842.py"
    for name, mode, source in cases:
        destination = root / name
        run([engine, mode, source, destination], root / (name + "-rust.log"))
        run([native, destination / "angles.bin", destination], root / (name + "-native.log"))
        storage = []
        for field in ("cos", "sin"):
            expected = destination / ("native-" + field + ".bin")
            actual = destination / (field + ".bin")
            if expected.stat().st_size % 4 or expected.read_bytes() != actual.read_bytes():
                raise ValueError(f"{name}: {field} differs from strict GNU")
            storage.append(dict(field=field, words=actual.stat().st_size // 4,
                                native_sha256=sha(expected), rust_sha256=sha(actual),
                                differing_words=0))
        env = dict(os.environ, D1_NATIVE_ANGLE_FIXTURE=str(destination),
                   D1_NATIVE_ANGLE_ENGINE=str(engine))
        replay_log = root / (name + "-launcher.log")
        run([sys.executable, "-m", "pytest", tests, "-q"], replay_log, env)
        rows.append(dict(name=name, source=str(source), source_sha256=sha(source),
                         angles_sha256=sha(destination / "angles.bin"), fields=storage,
                         launcher_log_sha256=sha(replay_log)))
    tree = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), Path(__file__).with_name("angle_reference.f90"),
                    tests, tree / "src/hexcore/transport_reference_v842.py",
                    rust / "tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_nonles_angles.rs",
                    rust / "tools/rustwx/crates/rw-libm/src/lib.rs"]
    receipt = dict(schema="d1-native-reference-angle-replay-v1", status="PASS",
                   tier="BITWISE", native_process_count=1, hex_process_count=1,
                   host=subprocess.check_output(["hostname"], text=True).strip(),
                   engine_sha256=sha(engine), native_sha256=sha(native), abi=abi,
                   rust_source_revision=revision, sources={str(p): sha(p) for p in source_files},
                   native_flags="-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fPIC -cpp",
                   glibc=subprocess.check_output(["getconf", "GNU_LIBC_VERSION"], text=True).strip(),
                   compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
                   compared_words=sum(f["words"] for r in rows for f in r["fields"]),
                   differing_words=0, exclusions=[], cases=rows,
                   scope="reference-angle preparation and actual launcher; forecast unmeasured")
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "compared_words", "differing_words")}))


if __name__ == "__main__":
    main()
