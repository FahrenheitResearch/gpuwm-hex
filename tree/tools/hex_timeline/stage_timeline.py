"""Add the delivered C1/C2 modules and CLI registration to an owned install.

This leaves the installed engine's numerical files unchanged. Run with the
lane's private venv interpreter, never the shared box interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


# Narrow C2 hook: coverage is calculated inside gpuwm timeline, never by a
# separate forecast-fraction calculator.
_COVERAGE_HOOK = '''

# hex-ledger named-coverage hook
_report_before_named_coverage = Timeline.report
def _report_with_named_coverage(self, **kwargs):
    report = _report_before_named_coverage(self, **kwargs)
    origin, endpoint = report.get("origin"), report.get("endpoint")
    if origin is None or endpoint is None:
        report.update(named_coverage_seconds=None, unnamed_gaps=[], max_unnamed_gap_seconds=None)
        return report
    spans = sorted((max(origin, r["start"]), min(endpoint, r["end"]))
                   for r in report["intervals"] if r["end"] > origin and r["start"] < endpoint)
    cursor = origin
    gaps = []
    for start, end in spans:
        if start > cursor:
            gaps.append({"start": cursor, "end": start, "seconds": start - cursor})
        cursor = max(cursor, end)
    if cursor < endpoint:
        gaps.append({"start": cursor, "end": endpoint, "seconds": endpoint - cursor})
    report.update(unnamed_gaps=gaps,
                  max_unnamed_gap_seconds=max((g["seconds"] for g in gaps), default=0.0),
                  named_coverage_seconds=endpoint - origin - sum(g["seconds"] for g in gaps))
    return report
Timeline.report = _report_with_named_coverage
'''


def stage(source: Path, lane: Path):
    import gpuwm

    package = Path(gpuwm.__file__).resolve().parent
    if not package.is_relative_to(lane.resolve()):
        raise ValueError("timeline staging must target the lane's owned install")
    from hexcore.cuda_arwen_physics_v841 import ARWEN_SOURCE_MANIFEST
    from hexcore import engine_pin

    before = engine_pin.inspect_seam(package.parent)
    if before.moved or before.absent:
        raise ValueError("installed numerical seam does not match the hex pin")
    timeline = source / "gpuwm" / "timeline"
    if not (timeline / "ledger.py").is_file() or not (timeline / "report.py").is_file():
        raise ValueError("both delivered C1 and C2 are required")
    shutil.copytree(timeline, package / "timeline", dirs_exist_ok=True)
    report_path = package / "timeline/report.py"
    report_text = report_path.read_text()
    if "# hex-ledger named-coverage hook" not in report_text:
        report_path.write_text(report_text + _COVERAGE_HOOK)
    cli = package / "cli.py"
    original = cli.read_text()
    text = original
    if 'timeline_register_cli = _lazy_register("gpuwm.timeline.report")' not in text:
        anchor = 'report_register_cli = _lazy_register("gpuwm.report_bundle")'
        if anchor not in text:
            raise ValueError("CLI registration anchor missing; refuse an ambiguous edit")
        text = text.replace(anchor, anchor + '\ntimeline_register_cli = _lazy_register("gpuwm.timeline.report")', 1)
        anchor = '    doctor_register_cli(sub)'
        if anchor not in text:
            raise ValueError("CLI parser anchor missing")
        text = text.replace(anchor, anchor + '\n    timeline_register_cli(sub)', 1)
        anchor = '        list(sys.argv[1:] if argv is None else argv))'
        if anchor not in text:
            raise ValueError("CLI dispatch anchor missing")
        text = text.replace(anchor, anchor + '\n    if (len(tokens) > 1 and tokens[0] == "timeline"\n'
                            '            and tokens[1] not in {"report", "project", "--help", "-h"}):\n'
                            '        tokens.insert(1, "report")', 1)
        cli.write_text(text)
    after = engine_pin.inspect_seam(package.parent)
    if before.digests != after.digests or after.moved or after.absent:
        raise ValueError("timeline staging changed a numerical seam file")
    return {"package": str(package), "engine": after.engine, "manifest_fields": len(ARWEN_SOURCE_MANIFEST),
            "numerical_seam_unchanged": True,
            "cli_before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "cli_after_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source": str(source)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lane", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.lane), indent=2))
