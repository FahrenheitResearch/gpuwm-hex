"""Build the hex scorer against the installed engine's existing Rust readers.

This creates a private Cargo project, never edits the engine workspace.
Run on a compute box, not on the desktop.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib
import shutil


def write_project(engine: Path, out: Path) -> Path:
    """Write the scorer's private Cargo project under ``out`` and return its manifest.

    tools/helper_licences.py reads the same project, so the licence notice
    covers exactly the crates this build links.
    """
    root = engine.resolve() / "tools/rustwx"
    out.mkdir(parents=True, exist_ok=True)
    dependencies = {
        "netcrust": root / "vendor/netcrust",
        "grib-core": root.parent / "grib1_bridge/vendor/grib-core",
        "rw-wrfbatch": root / "crates/rw-wrfbatch",
        "rw-obs": root / "crates/rw-obs",
        "rw-isobaric": root / "crates/rw-isobaric",
        "rustwx-core": root / "crates/rustwx-core",
        "rustwx-io": root / "crates/rustwx-io",
    }
    source = Path(__file__).resolve().parents[1] / "rust/obs-scorecard/main.rs"
    manifest = ('[package]\nname="hex-obs-scorecard"\nversion="0.1.0"\nedition="2021"\npublish=false\n'
                '[[bin]]\nname="hex_obs_scorecard"\npath=' + json.dumps(str(source)) + '\n'
                '[dependencies]\nserde_json="1"\nchrono="0.4"\n'
                # IGRA2 station zips (the engine's rw_igra2 keeps NCEI's .txt.zip); rw-obs already locks flate2.
                'flate2={version="1",default-features=false,features=["rust_backend"]}\n')
    for name, path in dependencies.items():
        if not (path / "Cargo.toml").is_file():
            raise FileNotFoundError(path / "Cargo.toml")
        manifest += name + '={path=' + json.dumps(str(path)) + '}\n'
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    for registry, rows in workspace.get("patch", {}).items():
        manifest += '\n[patch.' + json.dumps(registry) + ']\n'
        for name, row in rows.items():
            if set(row) != {"path"}:
                raise ValueError(f"unsupported engine patch metadata for {name}")
            manifest += name + '={path=' + json.dumps(str((root / row["path"]).resolve())) + '}\n'
    target = out / "Cargo.toml"
    target.write_text(manifest, encoding="utf-8")
    lock = out / "Cargo.lock"
    if not lock.exists():
        shutil.copyfile(root / "Cargo.lock", lock)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--install", action="store_true", help="stage the native binary in this hex package's libexec for wheel packaging")
    args = parser.parse_args()
    root = args.engine.resolve() / "tools/rustwx"
    source = Path(__file__).resolve().parents[1] / "rust/obs-scorecard/main.rs"
    target = write_project(args.engine, args.out)
    code = subprocess.call(["cargo", "test" if args.test else "build", "--release", "--offline",
                            "--jobs", "2", "--manifest-path", str(target)], cwd=root)
    if code == 0 and args.install:
        destination = source.parents[2] / "src/hexcore/libexec"
        destination.mkdir(parents=True, exist_ok=True)
        executable = "hex_obs_scorecard.exe" if __import__("os").name == "nt" else "hex_obs_scorecard"
        shutil.copyfile(args.out / "target/release" / executable, destination / executable)
        (destination / executable).chmod(0o755)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
