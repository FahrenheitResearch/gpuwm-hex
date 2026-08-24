"""Command-line entry point for the issue-283 observational referee."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .acquire import run_producer
from .canonical import canonical_pretty_json_bytes, sha256_file, write_json
from .errors import RefereeError
from .manifest import load_manifest
from .runner import emit_not_measured, run_suite
from .treatment import (
    compare_output_trees,
    validate_disabled_receipt,
    validate_treatment_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuwm-hex-obs-referee",
        description=(
            "Manifest-driven, deterministic MRMS/ASOS referee. "
            "Raw observation parsing remains a rustwx boundary."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strictly validate a suite manifest")
    validate.add_argument("manifest", type=Path)

    run = subparsers.add_parser("run", help="run all available case/arm metrics")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)

    unmeasured = subparsers.add_parser(
        "not-measured",
        help="emit an explicit unrun scorecard with no fabricated values",
    )
    unmeasured.add_argument("manifest", type=Path)
    unmeasured.add_argument("--output", type=Path, required=True)
    unmeasured.add_argument("--reason", required=True)

    producer = subparsers.add_parser(
        "run-producer",
        help="invoke an explicitly configured rustwx/model canonical-bundle producer",
    )
    producer.add_argument("manifest", type=Path)
    producer.add_argument("--case", required=True)
    source_group = producer.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--observation")
    source_group.add_argument("--arm")
    producer.add_argument("--timeout-seconds", type=int, default=3600)

    treatment = subparsers.add_parser(
        "verify-treatment",
        help="validate a narrow enabled GF-subsidence treatment receipt",
    )
    treatment.add_argument("receipt", type=Path)
    treatment.add_argument("--name", required=True)
    treatment.add_argument("--mode", required=True)
    treatment.add_argument("--value", required=True, type=float)

    disabled = subparsers.add_parser(
        "verify-disabled-treatment",
        help="validate the zero-call/unchanged-digest disabled receipt",
    )
    disabled.add_argument("receipt", type=Path)

    identity = subparsers.add_parser(
        "compare-identity",
        help="require byte identity between default and hook-disabled output trees",
    )
    identity.add_argument("first", type=Path)
    identity.add_argument("second", type=Path)
    identity.add_argument("--include", action="append", default=[])
    identity.add_argument("--exclude", action="append", default=[])
    identity.add_argument("--json-output", type=Path)

    fingerprint = subparsers.add_parser("fingerprint", help="print SHA-256 of one file")
    fingerprint.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            _print_json(
                {
                    "status": "VALID",
                    "schema": manifest.raw["schema"],
                    "suite_id": manifest.suite_id,
                    "manifest_sha256": manifest.digest,
                    "base_commit": manifest.raw["base_commit"],
                    "cases": len(manifest.cases),
                    "arms": len(manifest.arms),
                    "metrics": len(manifest.metrics),
                }
            )
            return 0
        if args.command == "run":
            result = run_suite(args.manifest, args.output)
            _print_json(
                {
                    "status": result["run_receipt"]["status"],
                    "scientific_verdict": result["scorecard"]["scientific_verdict"],
                    "output": str(args.output),
                }
            )
            return 0
        if args.command == "not-measured":
            result = emit_not_measured(
                args.manifest,
                args.output,
                reason=args.reason,
            )
            _print_json(
                {
                    "status": result["run_receipt"]["status"],
                    "scientific_verdict": result["scorecard"]["scientific_verdict"],
                    "output": str(args.output),
                }
            )
            return 0
        if args.command == "run-producer":
            manifest = load_manifest(args.manifest)
            case = _find_case(manifest, args.case)
            if args.observation:
                source = case["observations"].get(args.observation)
                if source is None:
                    parser.error(
                        f"case {args.case!r} has no observation source {args.observation!r}"
                    )
                arm_id = None
            else:
                source = case["model_inputs"].get(args.arm)
                if source is None:
                    parser.error(f"case {args.case!r} has no model input {args.arm!r}")
                arm_id = args.arm
            output = run_producer(
                manifest,
                source,
                case_id=args.case,
                arm_id=arm_id,
                timeout_seconds=args.timeout_seconds,
            )
            _print_json(
                {
                    "status": "PRODUCED",
                    "artifact": str(output),
                    "sha256": sha256_file(output),
                }
            )
            return 0
        if args.command == "verify-treatment":
            receipt = validate_treatment_receipt(
                args.receipt,
                expected_name=args.name,
                expected_mode=args.mode,
                expected_value=args.value,
            )
            _print_json(
                {
                    "status": "VALID",
                    "treatment_name": receipt["treatment_name"],
                    "call_count": receipt["call_count"],
                    "columns_touched": receipt["columns_touched"],
                }
            )
            return 0
        if args.command == "verify-disabled-treatment":
            receipt = validate_disabled_receipt(args.receipt)
            _print_json(
                {
                    "status": "VALID_DISABLED",
                    "treatment_name": receipt["treatment_name"],
                    "call_count": receipt["call_count"],
                }
            )
            return 0
        if args.command == "compare-identity":
            result = compare_output_trees(
                args.first,
                args.second,
                include=tuple(args.include) or ("*.nc", "*.json", "*.npz"),
                exclude=tuple(args.exclude) or ("*treatment-receipt*.json",),
            )
            if args.json_output:
                write_json(args.json_output, result)
            _print_json(result)
            return 0
        if args.command == "fingerprint":
            print(sha256_file(args.path))
            return 0
    except RefereeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unhandled command {args.command!r}")
    return 2


def _find_case(manifest, case_id: str):
    for case in manifest.cases:
        if case["case_id"] == case_id:
            return case
    raise RefereeError(f"manifest has no case {case_id!r}")


def _print_json(value) -> None:
    sys.stdout.buffer.write(canonical_pretty_json_bytes(value))


if __name__ == "__main__":
    raise SystemExit(main())
