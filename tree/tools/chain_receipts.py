"""Collect source replay, C1 waterfalls and C4 verdicts inside the owned lane."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def call(argv):
    return subprocess.check_output(list(map(str, argv)), text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--timelines", action="store_true")
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    receipts = root / "receipts"
    if args.replay:
        result = {}
        for name, source in (("hex", "hex-src"), ("site", "site")):
            original = call(["git", "-C", root / source, "rev-parse", "HEAD^{tree}"])
            replay = call(["git", "-C", root / (name + "-replay-final"), "rev-parse", "HEAD^{tree}"])
            result[name] = {"source_tree": original, "replay_tree": replay, "equal": original == replay}
        (receipts / "final-patch-replay.json").write_text(json.dumps(result, indent=2))
        if not all(row["equal"] for row in result.values()):
            raise ValueError("patch replay tree differs from the reviewed source")
        print(json.dumps(result))
    if args.timelines:
        paths = [(receipts / f"cpu-{arm}-{rev}-ledger.jsonl", f"cpu-{arm}", True)
                 for arm, rev in (("serial", "r2"), ("overlap", "r3"))]
        paths += [(receipts / f"H1-{arm}-{mode}-ledger.jsonl", f"H1-{arm}-{mode}", False)
                  for arm in ("serial", "overlap") for mode in ("COLD", "WARM")]
        for path, name, cpu in paths:
            if not path.is_file():
                continue
            events = [json.loads(line) for line in path.read_text().splitlines()]
            first = [event for event in events if event["stage"] == "worker.claim"]
            last = [event for event in events if event["stage"] == "finish" and event["attrs"].get("scope") == "proxy chain"]
            if not first or not last:
                continue
            original = path
            if cpu:
                # Earlier preparation-only collectors incorrectly labelled card 0.
                # Retain the original record and correct only the CPU card metadata.
                for event in events:
                    event["card"] = None
                path = receipts / (name + "-cpu-normalized.jsonl")
                path.write_text("".join(json.dumps(event) + "\n" for event in events))
                (receipts / (name + "-normalization.json")).write_text(json.dumps({
                    "source": str(original), "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                    "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "change": "CPU preparation-only card metadata set to null; clocks and stage boundaries unchanged"}, indent=2))
            command = [sys.executable, "-m", "gpuwm.cli", "timeline", "report", str(path), "--product", "hex",
                       "--proxy-start", first[0]["t_utc"], "--proxy-end", last[-1]["t_utc"],
                       "--output", str(receipts / (name + "-timeline.json")),
                       "--png", str(receipts / (name + "-waterfall.png"))]
            if not cpu:
                command += ["--cards", "1"]
            with (receipts / (name + "-timeline-command.json")).open("w") as stream:
                json.dump(command, stream)
            subprocess.run(command, check=True)
    if args.identity:
        for mode in ("COLD", "WARM"):
            before = root / "data" / f"H1-serial-{mode}" / "artifacts"
            after = root / "data" / f"H1-overlap-{mode}" / "artifacts"
            if not before.is_dir() or not after.is_dir():
                continue
            command = [sys.executable, str(root / "runtime/tools/bench/identity_gate.py"), str(before), str(after),
                       "--hex-file-id-nonce", "--scratch-dir", str(root / "tmp"),
                       "--json", str(receipts / f"H1-{mode}-identity.json")]
            result = subprocess.run(command)
            if result.returncode:
                raise ValueError(f"H1 {mode} C4 identity did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
