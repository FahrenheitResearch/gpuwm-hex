"""Authenticate actual completed diagnostic calls and optional stream chronology.

Controlled forcing and copied owner proposals never establish lane acceptance.
This checks execution metadata only; Rust comparison supplies field identity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("output", "artifact", "receipt"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument("--boundary", type=Path)
parser.add_argument("--chronology", type=Path)
args = parser.parse_args()
sources = [args.output / "native-physics-request.json",
           args.output / "native-physics-receipt.json",
           args.output / "physics-calls.jsonl"]
request, runtime = [json.loads(path.read_text()) for path in sources[:2]]
rows = [json.loads(line) for line in sources[2].read_text().splitlines()]
steps = runtime["steps_completed"]
assert runtime["completed"] and steps == runtime["steps_requested"] == request["steps"]
assert len(rows) == steps and [row["step"] for row in rows] == list(range(1, steps + 1))
calls = [(row["step"], call) for row in rows for call in row["calls"]
         if call.get("operation") == "atm_add_tend_anal_incr"]
assert Counter(step for step, _ in calls) == Counter(range(1, steps + 1))
assert all(call["backend"] == "hex-cuda" and call["itimestep"] == step
           for step, call in calls)
assert all(call.get("backend", "native-cpu") == "native-cpu"
           for row in rows for call in row["calls"]
           if call.get("operation") != "atm_add_tend_anal_incr")
assert all(backend == "native-cpu" for backend in runtime["physics"]["backends"].values())
boundary_rows = []
if args.boundary:
    sources.append(args.boundary)
    boundary = json.loads(args.boundary.read_text())
    boundary_rows = [row for phase in ("pre", "post") for row in boundary[phase].values()]
    assert all(row["differing_words"] == 0 for row in boundary_rows)
observed = dict(source_selection=False, input_alarm_reset=False, before_density=False)
if args.chronology:
    sources.append(args.chronology)
    chronology = json.loads(args.chronology.read_text())
    transport = chronology["transport"]
    selected = transport["request.json"]["source_seek_witness"]
    assert len(selected) == 1 and selected[0]["stream"] == "iau"
    assert selected[0]["whence"] == "nearest"
    actual = transport["manifest.json"]["source_records"]
    requested = transport["request.json"]["source_records"]
    assert all(actual[path] == record for path, record in requested.items())
    assert requested[selected[0]["source"]] == selected[0]["record"]
    observed["source_selection"] = True
    events = chronology["events"]
    resets = [event for event in events if event["operation"] == "native_alarm_reset"]
    assert len(resets) == 1 and resets[0]["before"] and not resets[0]["after"]
    observed["input_alarm_reset"] = True
    pending = None
    counts = {}
    adds = []
    finishes = []
    for event in events:
        operation = event["operation"]
        if operation == "iau_add":
            assert pending is None
            pending = event["step"]
            counts[pending] = 0
            adds.append(pending)
        elif operation == "density_tendency":
            assert pending is not None and event["step"] == pending
            assert event["density_call"] == counts[pending]
            counts[pending] += 1
        elif operation == "finish_step":
            assert event["step"] == pending and counts[pending] > 0
            assert event["density_calls"] == counts[pending]
            finishes.append(pending)
            pending = None
    assert pending is None and adds == finishes == list(range(1, steps + 1))
    observed["before_density"] = True
else:
    selected = []
    counts = {}
result = dict(schema="mpas.iau-complete-forecast-witness-diagnostic/v1",
              acceptance=False, scope="controlled forcing and copied owner proposals",
              process_count=1, steps_completed=steps, iau_calls=len(calls),
              dt_seconds=float(request["dt"]), start=request["start"], stop=request["stop"],
              other_slots_native_cpu=True, observed=observed,
              source_selection=selected, density_calls=counts,
              boundary_compared_words=sum(row["words"] for row in boundary_rows) if boundary_rows else None,
              boundary_differing_words=0 if boundary_rows else None,
              module_sha256=hashlib.sha256((args.artifact / "hexcore/iau_v842.py").read_bytes()).hexdigest(),
              inputs_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
args.receipt.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
