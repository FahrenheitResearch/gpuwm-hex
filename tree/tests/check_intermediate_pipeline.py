"""Box receipt checks for complete MET identity and measured fetch deadlines."""
import argparse
import json
from pathlib import Path


def marker_check(root):
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    marks = [event for event in events if event["stage"] == "prep.intermediate.hour" and event["phase"] == "mark"]
    fetches = {event["attrs"]["hour"]: event for event in events
               if event["stage"] == "fetch.intermediate.hour" and event["phase"] == "end"}
    rows = []
    for event in marks:
        attrs = event["attrs"]
        marker = Path(attrs["marker"])
        document = json.loads(marker.read_text())
        if document["hour"] != attrs["hour"] or document["sha256"] != attrs["sha256"]:
            raise AssertionError("file marker differs from its C1 completion event")
        following = fetches.get(attrs["hour"] + 1)
        if following:
            delta = following["t_mono"] - event["t_mono"]
            rows.append({"hour": attrs["hour"], "next_fetch_end_minus_mark_s": delta, "pass": delta > 0})
    expected = json.loads((root / "native/intermediate-receipt.json").read_text())["hours"]
    ordered = [event["attrs"]["hour"] for event in marks] == expected
    return {"ordered": ordered, "markers": len(marks), "observed_deadlines": len(rows),
            "strict_deadline": ordered and bool(rows) and all(row["pass"] for row in rows), "rows": rows}


def compare(before, after):
    a, b = (json.loads(path.read_text()) for path in (before, after))
    if a["target"] != b["target"] or a["cycle"] != b["cycle"]:
        raise AssertionError("timing configurations differ")
    rows = [{"hour": x["hour"], "equal": x["hour"] == y["hour"] and
             x["native_sha256"] == y["native_sha256"] and x["bytes"] == y["bytes"],
             "sha256": y["native_sha256"], "bytes": y["bytes"]}
            for x, y in zip(a["hours"], b["hours"])]
    return {"identity": "PASS" if len(rows) == len(a["hours"]) == len(b["hours"]) and
            all(row["equal"] for row in rows) else "FAIL", "predecessor_s": a["prep_wall_s"],
            "native_s": b["prep_wall_s"], "saved_s": a["prep_wall_s"] - b["prep_wall_s"], "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markers", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = marker_check(args.markers) if args.markers else compare(args.before, args.after)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result.get("strict_deadline", result.get("identity") == "PASS"):
        raise SystemExit(1)
