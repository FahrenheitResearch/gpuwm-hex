#!/usr/bin/env python3
"""Fit and report gpuwm-hex device-capacity receipts without overstating gates.

This module is deliberately standard-library only.  It consumes receipts from
isolated hardware runs, fits the affine process-footprint model at two or more
mesh widths, checks dual-run hashes, and reports projections separately from
actual capacity gates.

A 12 GiB PASS is intentionally hard to obtain.  Extrapolation, a CuPy memory-
pool limit, or a single successful run can never produce it.  The gate requires
at least two byte-identical successful runs at the target cell count on either:

* a physical device whose reported total memory is no more than 12 GiB; or
* an externally enforced whole-device limit no more than 12 GiB whose receipt
  explicitly records that non-pool allocations and CUDA local-memory backing
  store were included.

In either case the sampled process peak plus the declared final-gate headroom
(default 512 MiB) must fit within the physical or enforced capacity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

GIB = 1024**3
MIB = 1024**2
SCHEMA = "gpuwm-hex.capacity-samples.v1"
REPORT_SCHEMA = "gpuwm-hex.capacity-report.v1"
DEFAULT_BUDGETS_GIB = (12.0, 16.0, 24.0, 32.0)
DEFAULT_FINAL_GATE_HEADROOM_BYTES = 512 * MIB


class ReceiptError(ValueError):
    """A receipt is incomplete or would support a misleading claim."""


@dataclass(frozen=True, slots=True)
class Sample:
    variant: str
    label: str
    cells: int
    process_peak_bytes: int
    success: bool
    state_sha256: str | None = None
    seconds_per_step: float | None = None
    forecast_seconds_per_wall_second: float | None = None
    pool_live_peak_bytes: int | None = None
    pool_total_peak_bytes: int | None = None
    physical_device_total_bytes: int | None = None
    effective_device_limit_bytes: int | None = None
    whole_device_limit_enforced: bool = False
    limit_includes_non_pool: bool = False
    limit_includes_local_backing_store: bool = False
    isolated_card: bool = False
    source_commit: str | None = None
    command: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Sample":
        required = ("variant", "label", "cells", "process_peak_bytes", "success")
        missing = [name for name in required if name not in raw]
        if missing:
            raise ReceiptError(f"sample is missing required keys: {', '.join(missing)}")
        command_raw = raw.get("command", ())
        if isinstance(command_raw, str):
            command = (command_raw,)
        elif isinstance(command_raw, Sequence):
            command = tuple(str(item) for item in command_raw)
        else:
            raise ReceiptError("sample command must be a string or sequence")
        sample = cls(
            variant=str(raw["variant"]),
            label=str(raw["label"]),
            cells=int(raw["cells"]),
            process_peak_bytes=int(raw["process_peak_bytes"]),
            success=bool(raw["success"]),
            state_sha256=_optional_digest(raw.get("state_sha256")),
            seconds_per_step=_optional_positive_float(raw.get("seconds_per_step")),
            forecast_seconds_per_wall_second=_optional_positive_float(
                raw.get("forecast_seconds_per_wall_second")
            ),
            pool_live_peak_bytes=_optional_nonnegative_int(
                raw.get("pool_live_peak_bytes")
            ),
            pool_total_peak_bytes=_optional_nonnegative_int(
                raw.get("pool_total_peak_bytes")
            ),
            physical_device_total_bytes=_optional_positive_int(
                raw.get("physical_device_total_bytes")
            ),
            effective_device_limit_bytes=_optional_positive_int(
                raw.get("effective_device_limit_bytes")
            ),
            whole_device_limit_enforced=bool(raw.get("whole_device_limit_enforced", False)),
            limit_includes_non_pool=bool(raw.get("limit_includes_non_pool", False)),
            limit_includes_local_backing_store=bool(
                raw.get("limit_includes_local_backing_store", False)
            ),
            isolated_card=bool(raw.get("isolated_card", False)),
            source_commit=(
                None if raw.get("source_commit") in (None, "")
                else str(raw["source_commit"])
            ),
            command=command,
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if not self.variant.strip():
            raise ReceiptError("sample variant must be non-empty")
        if not self.label.strip():
            raise ReceiptError("sample label must be non-empty")
        if self.cells <= 0:
            raise ReceiptError(f"{self.label}: cells must be positive")
        if self.process_peak_bytes <= 0:
            raise ReceiptError(f"{self.label}: process_peak_bytes must be positive")
        if (
            self.pool_live_peak_bytes is not None
            and self.pool_total_peak_bytes is not None
            and self.pool_live_peak_bytes > self.pool_total_peak_bytes
        ):
            raise ReceiptError(
                f"{self.label}: pool live peak exceeds pool total peak"
            )
        if self.state_sha256 is not None and len(self.state_sha256) != 64:
            raise ReceiptError(f"{self.label}: state_sha256 must be 64 hex characters")
        if self.whole_device_limit_enforced and self.effective_device_limit_bytes is None:
            raise ReceiptError(
                f"{self.label}: an enforced limit requires effective_device_limit_bytes"
            )
        if (
            self.effective_device_limit_bytes is not None
            and self.process_peak_bytes > self.effective_device_limit_bytes
            and self.success
        ):
            raise ReceiptError(
                f"{self.label}: successful process peak exceeds the stated effective limit"
            )


@dataclass(frozen=True, slots=True)
class AffineModel:
    sample_count: int
    distinct_cell_counts: int
    fixed_bytes: float
    bytes_per_cell: float
    r_squared: float
    residual_max_abs_bytes: float

    def predict_bytes(self, cells: int) -> float:
        if cells < 0:
            raise ValueError("cells must be non-negative")
        return self.fixed_bytes + self.bytes_per_cell * cells

    def max_cells(self, budget_bytes: int, headroom_bytes: int) -> int | None:
        usable = budget_bytes - headroom_bytes - self.fixed_bytes
        if usable < 0.0 or self.bytes_per_cell <= 0.0:
            return None
        return max(0, math.floor(usable / self.bytes_per_cell))


@dataclass(frozen=True, slots=True)
class DeterminismVerdict:
    status: str
    cells: int
    successful_runs: int
    hashes_present: int
    distinct_hashes: int
    detail: str


@dataclass(frozen=True, slots=True)
class GateVerdict:
    status: str
    name: str
    detail: str
    qualifying_labels: tuple[str, ...] = ()


def _optional_nonnegative_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    result = int(value)
    if result < 0:
        raise ReceiptError("integer field must be non-negative")
    return result


def _optional_positive_int(value: Any) -> int | None:
    result = _optional_nonnegative_int(value)
    if result is not None and result <= 0:
        raise ReceiptError("integer field must be positive")
    return result


def _optional_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ReceiptError("floating field must be finite and positive")
    return result


def _optional_digest(value: Any) -> str | None:
    if value in (None, ""):
        return None
    result = str(value).lower()
    if any(character not in "0123456789abcdef" for character in result):
        raise ReceiptError("digest contains non-hexadecimal characters")
    return result


def fit_affine_model(samples: Iterable[Sample]) -> AffineModel:
    selected = tuple(sample for sample in samples if sample.success)
    if len(selected) < 2:
        raise ReceiptError("an affine fit requires at least two successful samples")
    distinct = len({sample.cells for sample in selected})
    if distinct < 2:
        raise ReceiptError("an affine fit requires at least two distinct cell counts")

    xs = [float(sample.cells) for sample in selected]
    ys = [float(sample.process_peak_bytes) for sample in selected]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        raise ReceiptError("cell-count variance is zero")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    fixed = y_mean - slope * x_mean
    if not math.isfinite(fixed) or not math.isfinite(slope):
        raise ReceiptError("affine fit produced a non-finite coefficient")
    if fixed < 0.0:
        raise ReceiptError(
            "affine fit produced a negative fixed term; inspect the receipts rather "
            "than clipping it to a plausible-looking value"
        )
    if slope <= 0.0:
        raise ReceiptError(
            "affine fit produced a non-positive bytes-per-cell term; inspect the receipts"
        )

    predicted = [fixed + slope * x for x in xs]
    residuals = [actual - estimate for actual, estimate in zip(ys, predicted)]
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((value - y_mean) ** 2 for value in ys)
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return AffineModel(
        sample_count=len(selected),
        distinct_cell_counts=distinct,
        fixed_bytes=fixed,
        bytes_per_cell=slope,
        r_squared=r_squared,
        residual_max_abs_bytes=max(abs(value) for value in residuals),
    )


def determinism_verdict(samples: Iterable[Sample], *, cells: int) -> DeterminismVerdict:
    selected = tuple(
        sample for sample in samples if sample.cells == cells and sample.success
    )
    hashes = tuple(sample.state_sha256 for sample in selected if sample.state_sha256)
    distinct_hashes = len(set(hashes))
    if len(selected) < 2:
        return DeterminismVerdict(
            "NOT MEASURED",
            cells,
            len(selected),
            len(hashes),
            distinct_hashes,
            "fewer than two successful runs exist at this cell count",
        )
    if len(hashes) != len(selected):
        return DeterminismVerdict(
            "NOT MEASURED",
            cells,
            len(selected),
            len(hashes),
            distinct_hashes,
            "one or more successful runs lacks an exact state hash",
        )
    if distinct_hashes != 1:
        return DeterminismVerdict(
            "FAIL",
            cells,
            len(selected),
            len(hashes),
            distinct_hashes,
            "successful repeated runs produced different state hashes",
        )
    return DeterminismVerdict(
        "PASS",
        cells,
        len(selected),
        len(hashes),
        distinct_hashes,
        "all successful repeated runs are byte-identical",
    )


def _qualifies_for_12gib(
    sample: Sample,
    *,
    ceiling_bytes: int,
    headroom_bytes: int,
) -> tuple[bool, str]:
    if not sample.success:
        return False, "run failed"
    if not sample.isolated_card:
        return False, "card was not recorded as isolated"
    if headroom_bytes < 0:
        raise ReceiptError("12 GiB gate headroom must be non-negative")

    capacity_limit: int | None = None
    qualification = ""
    physical = sample.physical_device_total_bytes
    if physical is not None and physical <= ceiling_bytes:
        capacity_limit = physical
        qualification = "physical device total is at most 12 GiB"
    else:
        limit = sample.effective_device_limit_bytes
        if (
            not sample.whole_device_limit_enforced
            or limit is None
            or limit > ceiling_bytes
        ):
            return False, "no qualifying whole-device limit at or below 12 GiB"
        if not sample.limit_includes_non_pool:
            return False, "limit does not prove non-pool allocations are constrained"
        if not sample.limit_includes_local_backing_store:
            return False, "limit does not prove CUDA local backing store is constrained"
        capacity_limit = limit
        qualification = (
            "validated whole-device limit includes pool, non-pool, and local backing store"
        )

    assert capacity_limit is not None
    if sample.process_peak_bytes + headroom_bytes > capacity_limit:
        return (
            False,
            "process peak does not leave the required headroom: "
            f"{sample.process_peak_bytes} + {headroom_bytes} > {capacity_limit}",
        )
    return True, qualification


def twelve_gib_gate(
    samples: Iterable[Sample],
    *,
    target_cells: int,
    ceiling_bytes: int = 12 * GIB,
    headroom_bytes: int = DEFAULT_FINAL_GATE_HEADROOM_BYTES,
) -> GateVerdict:
    selected = tuple(sample for sample in samples if sample.cells == target_cells)
    qualifying: list[Sample] = []
    rejections: list[str] = []
    for sample in selected:
        accepted, reason = _qualifies_for_12gib(
            sample,
            ceiling_bytes=ceiling_bytes,
            headroom_bytes=headroom_bytes,
        )
        if accepted:
            qualifying.append(sample)
        else:
            rejections.append(f"{sample.label}: {reason}")

    deterministic = determinism_verdict(qualifying, cells=target_cells)
    if len(qualifying) >= 2 and deterministic.status == "PASS":
        return GateVerdict(
            "PASS",
            "x4-under-12-GiB",
            "two or more isolated qualifying runs completed at the target cell count "
            "with one exact state hash and the declared headroom",
            tuple(sample.label for sample in qualifying),
        )
    if deterministic.status == "FAIL":
        return GateVerdict(
            "FAIL",
            "x4-under-12-GiB",
            "qualifying capacity runs are non-deterministic: " + deterministic.detail,
            tuple(sample.label for sample in qualifying),
        )
    detail = (
        "NOT MEASURED: final capacity cannot be inferred from an affine fit. "
        "Need two byte-identical isolated target-cell runs on a physical <=12 GiB "
        "device or a validated whole-device limit, with the declared headroom."
    )
    if rejections:
        detail += " Rejections: " + "; ".join(rejections)
    return GateVerdict(
        "NOT MEASURED",
        "x4-under-12-GiB",
        detail,
        tuple(sample.label for sample in qualifying),
    )


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else statistics.fmean(present)


def build_variant_report(
    variant: str,
    samples: Sequence[Sample],
    *,
    target_cells: int,
    budgets_gib: Sequence[float],
    headroom_bytes: int,
) -> dict[str, Any]:
    model = fit_affine_model(samples)
    determinism_by_cells = {
        str(cells): asdict(determinism_verdict(samples, cells=cells))
        for cells in sorted({sample.cells for sample in samples})
    }
    capacities: dict[str, int | None] = {}
    for budget_gib in budgets_gib:
        capacities[f"{budget_gib:g}"] = model.max_cells(
            int(round(budget_gib * GIB)), headroom_bytes
        )
    successful = tuple(sample for sample in samples if sample.success)
    return {
        "variant": variant,
        "source_commits": sorted(
            {sample.source_commit for sample in samples if sample.source_commit}
        ),
        "sample_count": len(samples),
        "successful_sample_count": len(successful),
        "model": {
            **asdict(model),
            "fixed_mib": model.fixed_bytes / MIB,
            "bytes_per_cell": model.bytes_per_cell,
            "target_cells": target_cells,
            "predicted_target_peak_bytes": model.predict_bytes(target_cells),
            "predicted_target_peak_gib": model.predict_bytes(target_cells) / GIB,
            "projection_status": "PROJECTION ONLY",
        },
        "predicted_max_cells_by_budget_gib": capacities,
        "performance": {
            "mean_seconds_per_step": _mean_or_none(
                sample.seconds_per_step for sample in successful
            ),
            "mean_forecast_seconds_per_wall_second": _mean_or_none(
                sample.forecast_seconds_per_wall_second for sample in successful
            ),
        },
        "determinism_by_cells": determinism_by_cells,
        "twelve_gib_gate": asdict(
            twelve_gib_gate(
                samples,
                target_cells=target_cells,
                headroom_bytes=headroom_bytes,
            )
        ),
        "samples": [asdict(sample) for sample in samples],
    }


def load_samples(path: Path) -> tuple[Sample, ...]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ReceiptError("input must be a JSON object")
    if raw.get("schema") != SCHEMA:
        raise ReceiptError(f"input schema must be {SCHEMA!r}")
    entries = raw.get("samples")
    if not isinstance(entries, list):
        raise ReceiptError("input samples must be a JSON list")
    samples = tuple(Sample.from_mapping(entry) for entry in entries)
    if not samples:
        raise ReceiptError("input contains no samples")
    labels = [sample.label for sample in samples]
    if len(labels) != len(set(labels)):
        raise ReceiptError("sample labels must be unique")
    return samples


def build_report(
    samples: Sequence[Sample],
    *,
    target_cells: int,
    budgets_gib: Sequence[float] = DEFAULT_BUDGETS_GIB,
    headroom_bytes: int = DEFAULT_FINAL_GATE_HEADROOM_BYTES,
) -> dict[str, Any]:
    if target_cells <= 0:
        raise ReceiptError("target_cells must be positive")
    if headroom_bytes < 0:
        raise ReceiptError("headroom_bytes must be non-negative")
    if not budgets_gib or any(
        not math.isfinite(value) or value <= 0.0 for value in budgets_gib
    ):
        raise ReceiptError("budgets_gib must contain positive finite values")

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.variant, []).append(sample)
    reports = [
        build_variant_report(
            variant,
            tuple(grouped[variant]),
            target_cells=target_cells,
            budgets_gib=budgets_gib,
            headroom_bytes=headroom_bytes,
        )
        for variant in sorted(grouped)
    ]
    return {
        "schema": REPORT_SCHEMA,
        "target_cells": target_cells,
        "headroom_bytes": headroom_bytes,
        "budgets_gib": list(budgets_gib),
        "variants": reports,
        "claim_boundary": (
            "Predicted peaks and capacities are projections. Only a gate object with "
            "status PASS is an actual capacity claim."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-cells", type=int, default=163_842)
    parser.add_argument("--headroom-mib", type=float, default=512.0)
    parser.add_argument(
        "--budgets-gib",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS_GIB),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        samples = load_samples(arguments.input)
        report = build_report(
            samples,
            target_cells=arguments.target_cells,
            budgets_gib=tuple(arguments.budgets_gib),
            headroom_bytes=int(round(arguments.headroom_mib * MIB)),
        )
    except (OSError, json.JSONDecodeError, ReceiptError, ValueError) as exc:
        raise SystemExit(f"capacity report refused: {exc}") from exc
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
