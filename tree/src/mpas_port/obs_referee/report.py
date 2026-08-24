"""Deterministic JSON/CSV/Markdown/HTML evidence rendering."""

from __future__ import annotations

import csv
from html import escape
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_bytes, sha256_file, write_json


def write_evidence_directory(
    output_directory: str | Path,
    *,
    run_receipt: Mapping[str, Any],
    case_metrics: Sequence[Mapping[str, Any]],
    scorecard: Mapping[str, Any],
) -> dict[str, str]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run-receipt.json", dict(run_receipt))
    write_json(output / "case-metrics.json", list(case_metrics))
    write_json(output / "scorecard.json", dict(scorecard))
    atomic_write_bytes(output / "case-metrics.csv", _metrics_csv(case_metrics).encode("utf-8"))
    markdown = _markdown_report(run_receipt, case_metrics, scorecard)
    atomic_write_bytes(output / "REPORT.md", markdown.encode("utf-8"))
    atomic_write_bytes(
        output / "REPORT.html",
        _html_report(markdown, scorecard).encode("utf-8"),
    )
    inventory = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "output-inventory.json"
    }
    write_json(
        output / "output-inventory.json",
        {
            "schema": "gpuwm-hex.obs-referee-output-inventory/v1",
            "files": inventory,
        },
    )
    inventory["output-inventory.json"] = sha256_file(output / "output-inventory.json")
    return inventory


def _metrics_csv(items: Sequence[Mapping[str, Any]]) -> str:
    buffer = StringIO(newline="")
    fieldnames = [
        "case_id",
        "case_role",
        "arm_id",
        "metric_id",
        "status",
        "value",
        "n_valid",
        "reason",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in sorted(
        items,
        key=lambda row: (
            str(row["case_id"]),
            str(row["arm_id"]),
            str(row["metric_id"]),
        ),
    ):
        value = item.get("value")
        writer.writerow(
            {
                "case_id": item["case_id"],
                "case_role": item["case_role"],
                "arm_id": item["arm_id"],
                "metric_id": item["metric_id"],
                "status": item["status"],
                "value": "" if value is None else format(float(value), ".17g"),
                "n_valid": item.get("n_valid", 0),
                "reason": item.get("reason") or "",
            }
        )
    return buffer.getvalue()


def _markdown_report(
    run_receipt: Mapping[str, Any],
    case_metrics: Sequence[Mapping[str, Any]],
    scorecard: Mapping[str, Any],
) -> str:
    verdict = str(scorecard["scientific_verdict"])
    lines = [
        "# gpuwm-hex observational referee",
        "",
        f"**Scientific status: {verdict}**",
        "",
        str(scorecard["scientific_reason"]),
        "",
        "> Default-on decision: **DO NOT ENABLE**. Synthetic tests prove only the",
        "> verification machinery. Real atmospheric evidence and explicit owner",
        "> acceptance remain mandatory.",
        "",
        "## Run identity",
        "",
        f"- Suite: `{run_receipt['suite_id']}`",
        f"- Manifest SHA-256: `{run_receipt['manifest_sha256']}`",
        f"- Base commit: `{run_receipt['base_commit']}`",
        f"- Evidence class: `{run_receipt['evidence_class']}`",
        f"- Run status: `{run_receipt['status']}`",
    ]
    reason = run_receipt.get("not_measured_reason")
    if reason:
        lines.append(f"- NOT MEASURED reason: {reason}")
    lines.extend(
        [
            "",
            "## Paired metric comparisons",
            "",
            "| Metric | Paired cases | Mean improvement | Confidence interval | Verdict |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in scorecard["metric_comparisons"]:
        interval = item["improvement_interval"]
        estimate = _number(interval.get("estimate"))
        if interval.get("lower") is None:
            ci = "NOT MEASURED"
        else:
            ci = f"[{_number(interval['lower'])}, {_number(interval['upper'])}]"
        lines.append(
            f"| `{item['metric_id']}` | {interval['n_cases']} | {estimate} | {ci} | "
            f"`{item['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim assessments",
            "",
            "| Claim | Status | Reason |",
            "|---|---|---|",
        ]
    )
    if scorecard["claim_assessments"]:
        for item in scorecard["claim_assessments"]:
            lines.append(
                f"| `{item['claim_id']}` | `{item['status']}` | "
                f"{_escape_markdown_cell(str(item['reason']))} |"
            )
    else:
        lines.append("| — | `NOT MEASURED` | No claim rules were declared. |")
    lines.extend(
        [
            "",
            "## Case metrics",
            "",
            "| Case | Role | Arm | Metric | Status | Value | n |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for item in sorted(
        case_metrics,
        key=lambda row: (
            str(row["case_id"]),
            str(row["arm_id"]),
            str(row["metric_id"]),
        ),
    ):
        lines.append(
            f"| `{item['case_id']}` | `{item['case_role']}` | `{item['arm_id']}` | "
            f"`{item['metric_id']}` | `{item['status']}` | {_number(item.get('value'))} | "
            f"{item.get('n_valid', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Input provenance",
            "",
            "| Logical input | Producer | Version | SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    inputs = run_receipt.get("input_artifacts", [])
    if inputs:
        for item in inputs:
            lines.append(
                f"| `{item['logical_id']}` | `{item['producer']}` | "
                f"`{item['producer_version']}` | `{item['sha256']}` |"
            )
    else:
        lines.append("| — | — | — | No input artifacts were measured. |")
    lines.extend(
        [
            "",
            "## Audit notes",
            "",
            "- All times and cases are manifest-owned.",
            "- Paired uncertainty resamples whole cases, never individual pixels.",
            "- Raw MRMS and raw METAR parsing are outside this package and remain a rustwx boundary.",
            "- Upper-level theta/GF causal questions remain unresolved without a secondary vertical-profile referee.",
            "",
        ]
    )
    return "\n".join(lines)


def _html_report(markdown: str, scorecard: Mapping[str, Any]) -> str:
    """A dependency-free audit view; REPORT.md remains the canonical prose."""

    escaped = escape(markdown)
    verdict = escape(str(scorecard["scientific_verdict"]))
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Observational referee — {verdict}</title>"
        "<style>"
        "body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.45 system-ui,sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1rem;border-radius:.4rem}"
        "</style></head><body>"
        f"<h1>Scientific status: {verdict}</h1><pre>{escaped}</pre></body></html>\n"
    )


def _number(value: Any) -> str:
    if value is None:
        return "NOT MEASURED"
    return format(float(value), ".8g")


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")
