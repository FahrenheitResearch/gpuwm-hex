"""Prepare an unapplied verifier-owner mapping from original forecast receipts.

Only proof metadata is processed. MPAS authority: mpas_atm_time_integration.F:
6319-6410,6585-6598,6610-6629,6880-6906. This never runs a forecast, modifies
the published ratchet, or upgrades an original diagnostic to acceptance.
"""
import argparse
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from hexcore.verify_mpas import aggregate, identity, ratchet_failures, recorded_missing_failures


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def qualify(inputs, checkout, out):
    manifest = load(inputs / "snapshot.json")
    for name, expected in manifest["files"].items():
        require(sha(inputs / name) == expected, "changed snapshot: " + name)
    baseline_path = checkout / "receipts/ratchet.json"
    require(sha(baseline_path) == manifest["baseline_sha256"], "published ratchet changed")
    baseline = load(baseline_path)
    proposal = deepcopy(baseline)
    records = []
    controls = []
    additions = []
    for item in manifest["cases"]:
        original = load(inputs / item["verify"])
        executed = load(inputs / item["execution_deck"])
        sealed = load(inputs / item["sealed_deck"])
        native = load(inputs / item["native"])
        require(original["evidence_kind"] == "FORECAST" and original["model_execution"] is True,
                "receipt does not witness a forecast")
        require(original["failed_model_exit"] is None, "forecast failed")
        require(original["provenance"]["hex_source"]["dirty"] == "", "dirty original model")
        require(original["provenance"]["deck_sha256"] == sha(inputs / item["execution_deck"]),
                "execution deck differs from original verifier provenance")
        require(native["complete"] and native["process_count"] == 1,
                "native reference is not complete at one process")
        require(sealed["native"]["sha256"] == sha(inputs / item["native"]), "native pin changed")
        require(executed["inputs"] == sealed["inputs"] and
                {k: v["sha256"] for k, v in executed["inputs"].items()} ==
                {k: v["sha256"] for k, v in original["provenance"]["inputs"].items()},
                "original input authorities changed")
        require({k: v for k, v in executed.items() if k not in ("id", "hex_command")} ==
                {k: v for k, v in sealed.items() if k not in ("id", "hex_command")},
                "execution changed scientific deck metadata")
        command = list(executed["hex_command"])
        for flag in ("--native-physics-library", "--native-pool-reader"):
            at = command.index(flag) + 1
            require(sealed["hex_command"][at - 1] == flag, "provider argument moved")
            command[at] = sealed["hex_command"][at]
        require(command == sealed["hex_command"], "execution changed scientific command")
        require(executed["id"] in (sealed["id"], sealed["id"] + "-disposable-diagnostic"),
                "unexpected original deck identity")
        old_context = identity(executed)
        require(all(original[k] == v for k, v in old_context.items()), "receipt context differs")
        context = {**identity(sealed), "process_count": 1}
        require(sealed["physics"] == "off" and sealed["process_count"] == 1,
                "wrong physics or process scope")
        result = original["comparison"]
        require(all(r["status"] in ("compared", "field missing in hex") for r in result["fields"]),
                "structural failure cannot enroll")
        written = [r for r in result["fields"] if r["status"] == "compared"]
        require(written and not any(r["differing_words"] for r in written), "computed words differ")
        require(original["candidate_ratchet"] == aggregate(result["fields"],
                {**old_context, "process_count": 1}), "original candidate metadata changed")
        candidate = aggregate(result["fields"], context)
        missing = [r for r in candidate["entries"] if r["missing_samples"]]
        # Same-mesh physics-off G1 is the existing published inventory witness.
        # Match every family exactly and every integer word count per sample;
        # a shorter stream clock never authorizes a new field or larger shape.
        witnesses = [r for r in baseline["entries"] if r["mesh"] == context["mesh"]
                     and r["deck"].startswith("f0-G1-explicit-dry-none-")
                     and r["statuses"] == ["field missing in hex"]]
        known = {(r["stream"], r["field"]): r for r in witnesses}
        require({(r["stream"], r["field"]) for r in missing} == set(known),
                "missing inventory differs from published physics-off witness")
        for row in missing:
            witness = known[(row["stream"], row["field"])]
            require(row["process_count"] == witness["process_count"] == 1 and
                    row["statuses"] == witness["statuses"] and
                    row["max_ulp"] == witness["max_ulp"] == 0 and
                    row["differing_words"] == row["compared_words"] and
                    row["compared_words"] * witness["missing_samples"] ==
                    witness["compared_words"] * row["missing_samples"] and
                    row["missing_samples"] <= witness["missing_samples"] and
                    row["compared_words"] <= witness["compared_words"],
                    "missing family exceeds published shape or clock")
        require(recorded_missing_failures(result, context, baseline), "context already enrolled")
        additions.extend(deepcopy(missing))
        records.append(dict(index=item["index"], context=context,
            duration_seconds=sealed["duration_seconds"], original_execution_deck_id=executed["id"],
            original_diagnostic_context={**old_context, "process_count": 1},
            model_commit=original["provenance"]["hex_source"]["commit"],
            missing_family_pairs=len(missing), missing_samples=sum(r["missing_samples"] for r in missing),
            missing_native_words=sum(r["compared_words"] for r in missing),
            written_words=sum(r["compared_words"] for r in written), written_differing_words=0,
            inventory_growth=0, same_mesh_witness_deck=witnesses[0]["deck"],
            original_verifier_passed=original["passed"], excluded_words=0))
    proposal["entries"].extend(additions)
    require(proposal["entries"][:len(baseline["entries"])] == baseline["entries"],
            "original entries changed")
    require({k: v for k, v in proposal.items() if k != "entries"} ==
            {k: v for k, v in baseline.items() if k != "entries"}, "published authorities changed")
    for item, record in zip(manifest["cases"], records):
        result = load(inputs / item["verify"])["comparison"]
        context = record["context"]
        require(not recorded_missing_failures(result, context, proposal), "proposed context fails")
        require(not ratchet_failures(aggregate(result["fields"], context), proposal), "ratchet fails")

        def refuses(name, changed_result, changed_context):
            try:
                failures = recorded_missing_failures(changed_result, changed_context, proposal)
            except (ValueError, TypeError) as error:
                failures = [str(error)]
            require(failures, "negative control passed: " + name)
            controls.append(dict(index=item["index"], name=name,
                                 failure_count=len(failures), first_failure=failures[0]))

        for key in ("pin_set", "mesh", "date", "deck"):
            changed = dict(context)
            changed[key] += "/changed-context"
            refuses("changed-" + key, result, changed)
        refuses("changed-process-count", result, {**context, "process_count": 2})
        written = next(i for i, r in enumerate(result["fields"]) if r["status"] == "compared")
        missing = next(i for i, r in enumerate(result["fields"])
                       if r["status"] == "field missing in hex" and r["compared_words"] > 1)
        for name, index, update in (
            ("computed-word-difference", written, dict(differing_words=1, max_ulp=1)),
            ("missing-word-growth", missing, dict(
                differing_words=result["fields"][missing]["differing_words"] + 1,
                compared_words=result["fields"][missing]["compared_words"] + 1)),
            ("fewer-native-words", missing, dict(compared_words=result["fields"][missing]["compared_words"] - 1)),
            ("structural-error", missing, dict(status="shape differs")),
            ("new-missing-family", missing, dict(field="unpublished_native_field"))):
            changed = deepcopy(result)
            changed["fields"][index].update(update)
            refuses(name, changed, context)
        pair = result["fields"][missing]
        changed = deepcopy(result)
        changed["fields"] = [r for r in changed["fields"]
                             if (r["stream"], r["field"]) != (pair["stream"], pair["field"])]
        refuses("omitted-native-family", changed, context)
        changed = deepcopy(result)
        changed["fields"] = [r for r in changed["fields"] if r["status"] != "compared"]
        refuses("no-written-native-words", changed, context)
    out.mkdir(parents=True, exist_ok=False)
    text = json.dumps(proposal, indent=2) + "\n"
    (out / "proposal-ratchet.json").write_text(text)
    patch = "".join(difflib.unified_diff(baseline_path.read_text().splitlines(keepends=True),
        text.splitlines(keepends=True), fromfile="a/receipts/ratchet.json", tofile="b/receipts/ratchet.json"))
    (out / "owner-context-proposed.diff").write_text(patch)
    checked = subprocess.run(["git", "-C", str(checkout), "apply", "--check", "-"],
                             input=patch, text=True, capture_output=True)
    require(checked.returncode == 0, "proposal does not dry apply: " + checked.stderr)
    require(sha(baseline_path) == manifest["baseline_sha256"], "published ratchet modified")
    control = dict(schema="d1-unapplied-owner-context-proposal-v1",
        scope="Only measured sealed contexts proposed; original diagnostic receipts remain unchanged",
        cases=records, negative_controls=controls, negative_controls_all_refuse=True,
        original_entry_count=len(baseline["entries"]), all_original_entries_unchanged=True,
        proposed_entry_count=len(additions), computed_difference_allowance=0, inventory_growth=0,
        snapshot_sha256=sha(inputs / "snapshot.json"), orchestration_sha256=sha(Path(__file__)),
        verifier_source_sha256=sha(checkout / "tree/src/hexcore/verify_mpas.py"),
        baseline_sha256=sha(baseline_path), proposal_sha256=sha(out / "proposal-ratchet.json"),
        diff_sha256=sha(out / "owner-context-proposed.diff"), apply_check_returncode=checked.returncode,
        proposal_mapping_qualified=True, owner_approved=False, policy_applied=False,
        ratchet_modified=False, whole_model_acceptance=False, excluded_words=0)
    (out / "proposal.json").write_text(json.dumps(control, indent=2) + "\n")
    require(all(p.stat().st_size < 10_000_000 for p in out.iterdir()), "receipt exceeds delivery limit")
    print(json.dumps(dict(qualified=True, contexts=len(records), negative_controls=len(controls),
        written_words=sum(r["written_words"] for r in records), differing_written_words=0,
        diff_sha256=control["diff_sha256"], whole_model_acceptance=False)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.inputs.resolve(), args.checkout.resolve(), args.out.resolve())
