#!/usr/bin/env python3
"""Pre-import launcher for the v8.4.1 compiled endpoint gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "tools" / "compare_v841_compiled_endpoint.py"
DEFAULT_FIXTURE = (
    ROOT / "oracle" / "jw-x1.2562-v8.4.1-split3-endpoint-nonclaim"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def hash_regular_tree(root: Path, *, suffix: str | None = None) -> dict[str, str]:
    base = root.resolve(strict=True)
    result: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"gate evidence tree contains symlink {path}")
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        result[path.relative_to(base).as_posix()] = sha256_file(path)
    if not result:
        raise RuntimeError(f"gate evidence tree is empty: {base}")
    return result


def capture_launcher_integrity(fixture: Path) -> dict[str, Any]:
    return {
        "source_capsule_sha256": hash_regular_tree(
            ROOT / "src" / "hexcore", suffix=".py"
        ),
        "test_capsule_sha256": hash_regular_tree(ROOT / "tests", suffix=".py"),
        "fixture_tree_sha256": hash_regular_tree(fixture),
        "comparator_sha256": sha256_file(COMPARATOR),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
    }


def validate_output_target(output: Path, fixture: Path) -> dict[str, Any]:
    candidate = output.absolute()
    if candidate.is_symlink():
        raise RuntimeError("report output target must not be a symlink")
    resolved = candidate.resolve()
    protected_directories = (
        (ROOT / "src").resolve(strict=True),
        (ROOT / "tests").resolve(strict=True),
        fixture.resolve(strict=True),
    )
    protected_files = (
        COMPARATOR.resolve(strict=True),
        Path(__file__).resolve(strict=True),
    )
    if any(resolved.is_relative_to(root) for root in protected_directories):
        raise RuntimeError("report output target is inside a protected input tree")
    if resolved in protected_files:
        raise RuntimeError("report output target is a protected gate program")
    if resolved.exists():
        raise RuntimeError("report output target already exists")
    if not resolved.parent.is_dir() or resolved.parent.is_symlink():
        raise RuntimeError("report output parent must be an existing real directory")
    return {
        "resolved_output": str(resolved),
        "protected_directories": [str(path) for path in protected_directories],
        "protected_files": [str(path) for path in protected_files],
        "requires_fresh_target": True,
        "refuses_symlink_target": True,
        "exclusive_create": True,
        "write_occurs_after_protected_input_post_hash": True,
    }


def materialize_report(serialized: str, policy: dict[str, Any]) -> None:
    output = Path(policy["resolved_output"])
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


def run_gate(
    fixture: Path,
    *,
    verify_only: bool = False,
    output_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_fixture = fixture.resolve(strict=True)
    integrity_start = capture_launcher_integrity(resolved_fixture)
    child_argv = [
        sys.executable,
        str(COMPARATOR.resolve()),
        "--fixture",
        str(resolved_fixture),
    ]
    if verify_only:
        child_argv.append("--verify-only")
    started = perf_counter()
    child = subprocess.run(
        child_argv,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    elapsed = perf_counter() - started
    if child.returncode != 0:
        raise RuntimeError(
            f"compiled endpoint child failed with exit {child.returncode}: "
            f"{child.stdout}\n{child.stderr}"
        )
    report = json.loads(child.stdout)
    integrity_end = capture_launcher_integrity(resolved_fixture)
    if integrity_start != integrity_end:
        raise RuntimeError("launcher evidence bytes changed during endpoint gate")
    if not verify_only:
        child_bootstrap = report["runtime_provenance"]["preimport_bootstrap"]
        if child_bootstrap["source_capsule_sha256"] != integrity_start[
            "source_capsule_sha256"
        ]:
            raise RuntimeError("child source bootstrap differs from launcher snapshot")
        if child_bootstrap["comparator_sha256"] != integrity_start[
            "comparator_sha256"
        ]:
            raise RuntimeError(
                "child comparator bootstrap differs from launcher snapshot"
            )
        if child_bootstrap["preexisting_mpas_modules"]:
            raise RuntimeError("child imported hexcore before its source snapshot")
        if not child_bootstrap["matches_gate_start"]:
            raise RuntimeError("child pre-import bootstrap did not reach gate start")
        report["ruler"]["launcher_integrity_passed"] = True
    report["launcher_provenance"] = {
        "command_argv": [sys.executable, *sys.argv],
        "child_argv": child_argv,
        "cwd": str(ROOT.resolve()),
        "elapsed_seconds": elapsed,
        "child_returncode": child.returncode,
        "child_stderr": child.stderr,
        "integrity_start": integrity_start,
        "integrity_end": integrity_end,
        "integrity_start_end_equal": True,
        "preimport_launcher": True,
        "output_materialization": output_policy,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_policy = (
        None
        if args.output is None
        else validate_output_target(args.output, args.fixture)
    )
    report = run_gate(
        args.fixture,
        verify_only=args.verify_only,
        output_policy=output_policy,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_policy is not None:
        materialize_report(serialized, output_policy)
    print(serialized, end="")


if __name__ == "__main__":
    main()
