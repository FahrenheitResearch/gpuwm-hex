"""Run the sealed MYJ visual request from committed code, explicitly in exact mode.

This controller authenticates inputs and invokes the real CLI. It neither changes
the driver route nor treats incomplete forecast output as successful evidence.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def preserve_sheet_inputs(request: dict, output: Path) -> dict:
    """Preserve both native-named streams as authenticated byte copies."""
    target = Path(request["retained_visual_directory"])
    streams = ET.parse(request["inputs"]["streams"]["path"]).getroot()
    start = datetime.datetime.strptime(request["simulation_start"], "%Y-%m-%d_%H:%M:%S")
    pending = []
    for lead in request["visual_leads_hours"]:
        clock = start + datetime.timedelta(hours=lead)
        for name in ("output", "diagnostics"):
            row = next((r for r in streams if r.get("name") == name), None)
            if row is None or not row.get("filename_template"):
                raise RuntimeError(f"visual evidence requires the {name} stream template")
            filename = row.get("filename_template")
            for token, fmt in (("$Y", "%Y"), ("$M", "%m"), ("$D", "%d"),
                               ("$h", "%H"), ("$m", "%M"), ("$s", "%S")):
                filename = filename.replace(token, clock.strftime(fmt))
            source = output / filename
            if not source.is_file():
                raise RuntimeError(f"visual evidence is incomplete: missing {name} {source}")
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            destination = target / filename
            if destination.exists():
                with destination.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise RuntimeError(f"refusing to overwrite different retained evidence: {destination}")
            pending.append((source, destination, dict(stream=name, lead_hours=lead,
                           source=str(source), retained=str(destination), sha256=digest,
                           bytes=source.stat().st_size)))
    # Authenticate the complete pair set before copying or declaring it retained.
    for source, destination, row in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise RuntimeError(f"retained evidence failed authentication: {destination}")
    return dict(encoding_changed=False, complete_stream_pairs=True,
                retire_only_after_sheets_inspected=True, files=[row for _, _, row in pending])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    receipt = args.request.resolve().parent
    repo = Path(request["hex_checkout"]).resolve()

    def git(*words: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *words], text=True).strip()

    if git("rev-parse", "HEAD") != request["source_commit"] or git("status", "--porcelain"):
        raise RuntimeError("visual request requires its clean committed checkout")
    if Path(__file__).resolve() != repo / "tree/tests/rrfs_myj_visual_attempt.py":
        raise RuntimeError("visual controller must come from the supplied checkout")
    from hexcore.verify_mpas import command_physics_mode

    if command_physics_mode(request["arguments"]) != "exact":
        raise RuntimeError("native parity request must explicitly pass --physics-mode exact")
    if not request["arguments"] or request["arguments"][0] != "forecast":
        raise RuntimeError("visual request must invoke the forecast door")
    for field in ("retained_visual_directory", "visual_leads_hours", "simulation_start"):
        if not request.get(field):
            raise RuntimeError(f"visual request must declare {field} before launching")
    for row in request["inputs"].values():
        with Path(row["path"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise RuntimeError(f"visual input changed: {row['path']}")
    import hexcore.cli

    if Path(hexcore.cli.__file__).resolve() != repo / "tree/src/hexcore/cli.py":
        raise RuntimeError("CLI must come from the supplied checkout")
    state = dict(request, physics_mode="exact", runtime_cli=str(Path(hexcore.cli.__file__).resolve()),
                 runtime_cli_sha256=hashlib.sha256(Path(hexcore.cli.__file__).read_bytes()).hexdigest(),
                 controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 runtime_source_commit=git("rev-parse", "HEAD"), runtime_dirty=git("status", "--porcelain"),
                 checked_utc=datetime.datetime.now(datetime.UTC).isoformat())
    if args.check_only:
        state.update(state="authenticated-only", forecast_started=False, compared_words=None,
                     differing_words=None, visual_acceptance=False)
        (receipt / "admission.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"physics_mode": "exact", "inputs_authenticated": len(request["inputs"]),
                          "source_commit": request["source_commit"], "forecast_started": False}))
        return 0
    os.environ.update(request["environment"])
    state.update(state="executing", started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                 runner_pid=os.getpid())
    (receipt / "attempt.json").write_text(json.dumps(state, indent=2) + "\n")
    started = time.monotonic()
    with (receipt / "forecast.log").open("w") as log:
        rc = subprocess.run([sys.executable, "-m", "hexcore.cli", *request["arguments"]],
                            cwd=repo / "tree", stdout=log, stderr=subprocess.STDOUT).returncode
    output = Path(request["arguments"][request["arguments"].index("--out") + 1])
    state.update(state="finished", exit_status=rc, elapsed_seconds=round(time.monotonic() - started, 3),
                 finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                 runtime_source_commit_after=git("rev-parse", "HEAD"), runtime_dirty_after=git("status", "--porcelain"),
                 compared_words=None, differing_words=None, visual_acceptance=False,
                 output_nc_files=sorted(str(p) for p in output.rglob("*.nc")) if output.exists() else [],
                 log_sha256=hashlib.sha256((receipt / "forecast.log").read_bytes()).hexdigest())
    (receipt / "attempt.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["runtime_source_commit_after"] != request["source_commit"] or state["runtime_dirty_after"]:
        raise RuntimeError("forecast checkout changed during execution")
    if rc == 0:
        retained = preserve_sheet_inputs(request, output)
        (receipt / "retained-streams.json").write_text(json.dumps(retained, indent=2) + "\n")
    print(json.dumps({"exit_status": rc, "elapsed_seconds": state["elapsed_seconds"], "visual_acceptance": False}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
