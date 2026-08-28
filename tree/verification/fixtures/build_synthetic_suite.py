#!/usr/bin/env python3
"""Build a deterministic four-case synthetic suite and run it.

This is a software proof only. Every produced report is labeled SYNTHETIC_ONLY.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

TREE = Path(__file__).resolve().parents[2]
SRC = TREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexcore.obs_referee.bundle import write_grid_bundle, write_station_bundle  # noqa: E402
from hexcore.obs_referee.canonical import write_json  # noqa: E402
from hexcore.obs_referee.manifest import BASE_COMMIT, SCHEMA  # noqa: E402
from hexcore.obs_referee.runner import run_suite  # noqa: E402
from hexcore.obs_referee.treatment import TREATMENT_RECEIPT_SCHEMA  # noqa: E402


CASE_SPECS = (
    ("synthetic-known-gfs-20260812", "known_divergence", 1786492800),
    ("synthetic-known-era5-20240521", "known_divergence", 1716249600),
    ("synthetic-independent-control", "independent_control", 1748736000),
    ("synthetic-weak-convection-control", "weak_convection_control", 1751328000),
)


def build(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    lat_1d = np.linspace(34.0, 35.2, 7)
    lon_1d = np.linspace(-99.6, -98.4, 7)
    lon, lat = np.meshgrid(lon_1d, lat_1d)

    cases = []
    for case_index, (case_id, role, cycle) in enumerate(CASE_SPECS):
        case_dir = data / case_id
        times = np.asarray([cycle, cycle + 3600, cycle + 7200], dtype=np.int64)
        obs_fields = _observation_fields(case_index, lat, lon)
        write_grid_bundle(
            case_dir / "mrms.npz",
            time_unix_s=times,
            latitude_deg=lat,
            longitude_deg=lon,
            fields={
                "precip_1h_mm": obs_fields["precip_1h_mm"],
                "reflectivity_dbz": obs_fields["reflectivity_dbz"],
            },
            producer="synthetic-fixture",
            producer_version="obs-referee-fixture-v1",
            metadata={"case_id": case_id, "source_kind": "synthetic-mrms"},
        )
        stations = []
        for time_index, valid_time in enumerate(times):
            for station_index, (y, x) in enumerate(((1, 1), (3, 3), (5, 5))):
                stations.append(
                    {
                        "station_id": f"S{station_index:02d}",
                        "time_unix_s": int(valid_time),
                        "latitude_deg": float(lat[y, x]),
                        "longitude_deg": float(lon[y, x]),
                        "fields": {
                            "temperature_k": float(
                                obs_fields["temperature_k"][time_index, y, x]
                            ),
                            "wind_speed_ms": float(
                                obs_fields["wind_speed_ms"][time_index, y, x]
                            ),
                        },
                    }
                )
        write_station_bundle(
            case_dir / "asos.jsonl",
            records=stations,
            producer="synthetic-fixture",
            producer_version="obs-referee-fixture-v1",
            metadata={"case_id": case_id, "source_kind": "synthetic-asos"},
        )

        current = _model_fields(obs_fields, arm="arwen-current", case_index=case_index)
        experiment = _model_fields(
            obs_fields, arm="gf-subsidence-experiment", case_index=case_index
        )
        native = _model_fields(
            obs_fields, arm="native-history-context", case_index=case_index
        )
        operational = _model_fields(
            obs_fields, arm="operational-baseline", case_index=case_index
        )
        model_inputs = {}
        for arm_id, fields in (
            ("arwen-current", current),
            ("gf-subsidence-experiment", experiment),
            ("native-history-context", native),
            ("operational-baseline", operational),
        ):
            artifact = case_dir / f"{arm_id}.npz"
            write_grid_bundle(
                artifact,
                time_unix_s=times,
                latitude_deg=lat,
                longitude_deg=lon,
                fields=fields,
                producer="synthetic-fixture",
                producer_version="obs-referee-fixture-v1",
                metadata={"case_id": case_id, "arm_id": arm_id},
            )
            model_inputs[arm_id] = {
                "adapter": "canonical-grid-v1",
                "path": str(artifact.relative_to(root)),
                "receipt": str(
                    Path(f"{artifact}.receipt.json").relative_to(root)
                ),
                "optional": False,
            }

        treatment_receipt = case_dir / "gf-treatment-receipt.json"
        write_json(
            treatment_receipt,
            {
                "schema": TREATMENT_RECEIPT_SCHEMA,
                "treatment_name": "gf-subsidence-scale",
                "mode": "multiply_tendency",
                "value": 0.75,
                "enabled": True,
                "scope": "gf_subsidence_only",
                "call_count": 12,
                "columns_touched": 588,
                "pre_tendency_sha256": "1" * 64,
                "post_tendency_sha256": f"{case_index + 2:x}" * 64,
                "producer_commit": "a" * 40,
                "metadata": {"synthetic": True, "case_id": case_id},
            },
        )
        cases.append(
            {
                "case_id": case_id,
                "role": role,
                "selection_status": "selected",
                "cycle_time": _iso(cycle),
                "window_start": _iso(cycle),
                "window_end": _iso(cycle + 7200),
                "description": "Deterministic synthetic software fixture.",
                "observations": {
                    "mrms": {
                        "adapter": "canonical-grid-v1",
                        "path": str((case_dir / "mrms.npz").relative_to(root)),
                        "receipt": str(
                            Path(f"{case_dir / 'mrms.npz'}.receipt.json").relative_to(root)
                        ),
                        "optional": False,
                    },
                    "asos": {
                        "adapter": "canonical-stations-v1",
                        "path": str((case_dir / "asos.jsonl").relative_to(root)),
                        "receipt": str(
                            Path(f"{case_dir / 'asos.jsonl'}.receipt.json").relative_to(root)
                        ),
                        "optional": False,
                    },
                },
                "model_inputs": model_inputs,
            }
        )

    manifest = {
        "schema": SCHEMA,
        "suite_id": "obs-referee-283-synthetic-proof",
        "mode": "synthetic",
        "base_commit": BASE_COMMIT,
        "description": "Four-case deterministic machinery proof; not atmospheric evidence.",
        "producer_allowlist": ["synthetic-fixture"],
        "path_variables": {},
        "cases": cases,
        "arms": [
            {
                "arm_id": "arwen-current",
                "role": "baseline",
                "eligible_for_promotion": False,
                "treatment": {"kind": "none"},
            },
            {
                "arm_id": "gf-subsidence-experiment",
                "role": "experiment",
                "eligible_for_promotion": False,
                "treatment": {
                    "kind": "external-receipt-v1",
                    "receipt": "data/{case_id}/gf-treatment-receipt.json",
                    "expected_name": "gf-subsidence-scale",
                    "expected_mode": "multiply_tendency",
                    "expected_value": 0.75,
                },
            },
            {
                "arm_id": "native-history-context",
                "role": "context",
                "eligible_for_promotion": False,
                "treatment": {"kind": "none"},
            },
            {
                "arm_id": "operational-baseline",
                "role": "context",
                "eligible_for_promotion": False,
                "treatment": {"kind": "none"},
            },
        ],
        "metrics": [
            {
                "metric_id": "mrms-precip-csi-1mm",
                "source": "mrms",
                "field": "precip_1h_mm",
                "family": "categorical",
                "statistic": "csi",
                "direction": "higher",
                "threshold": 1.0,
                "time_tolerance_seconds": 0,
                "space_tolerance_km": 0.0,
                "minimum_valid_samples": 20,
                "primary": True,
                "guardrail": False,
            },
            {
                "metric_id": "mrms-precip-fss-r1",
                "source": "mrms",
                "field": "precip_1h_mm",
                "family": "fss",
                "statistic": "fss",
                "direction": "higher",
                "threshold": 1.0,
                "neighborhood_radius_cells": 1,
                "time_tolerance_seconds": 0,
                "space_tolerance_km": 0.0,
                "minimum_valid_samples": 20,
                "primary": True,
                "guardrail": False,
            },
            {
                "metric_id": "mrms-reflectivity-csi-20dbz",
                "source": "mrms",
                "field": "reflectivity_dbz",
                "family": "categorical",
                "statistic": "csi",
                "direction": "higher",
                "threshold": 20.0,
                "time_tolerance_seconds": 0,
                "space_tolerance_km": 0.0,
                "minimum_valid_samples": 20,
                "primary": True,
                "guardrail": False,
            },
            {
                "metric_id": "asos-temperature-rmse",
                "source": "asos",
                "field": "temperature_k",
                "family": "continuous",
                "statistic": "rmse",
                "direction": "lower",
                "time_tolerance_seconds": 0,
                "space_tolerance_km": 0.1,
                "minimum_valid_samples": 6,
                "primary": False,
                "guardrail": True,
            },
        ],
        "bootstrap": {
            "replicates": 500,
            "confidence": 0.95,
            "minimum_cases": 3,
            "seed": 283,
        },
        "policy": {
            "candidate_arm": "gf-subsidence-experiment",
            "reference_arm": "arwen-current",
            "required_case_roles": [
                "known_divergence",
                "independent_control",
                "weak_convection_control",
            ],
            "owner_acceptance_required": True,
            "allow_automatic_default": False,
            "guardrail_tolerance": 0.0,
        },
        "claims": [
            {
                "claim_id": "gf-treatment-improves-observed-precip-placement",
                "rule": "arm_delta",
                "metric_id": "mrms-precip-fss-r1",
                "candidate_arm": "gf-subsidence-experiment",
                "reference_arm": "arwen-current",
            },
            {
                "claim_id": "upper-band-theta-drift-cause",
                "rule": "outside_primary_scope",
                "reason": (
                    "MRMS and ASOS do not observe model theta above level 45; "
                    "a radiosonde or analysis-profile referee is required."
                ),
            },
            {
                "claim_id": "condensate-surplus-cause",
                "rule": "outside_primary_scope",
                "reason": (
                    "MRMS reflectivity is an indirect hydrometeor proxy and cannot "
                    "identify cloud-water/rain-water partition causality by itself."
                ),
            },
        ],
        "metadata": {"synthetic": True},
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _observation_fields(case_index: int, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    ny, nx = lat.shape
    yy, xx = np.mgrid[:ny, :nx]
    precip = np.empty((3, ny, nx), dtype=np.float64)
    reflectivity = np.empty_like(precip)
    temperature = np.empty_like(precip)
    wind = np.empty_like(precip)
    for time_index in range(3):
        center_y = 2.0 + 0.5 * time_index + 0.1 * case_index
        center_x = 2.0 + 0.7 * time_index
        gaussian = np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / 3.0)
        precip[time_index] = np.maximum(0.0, 8.0 * gaussian - 0.35)
        reflectivity[time_index] = 10.0 + 38.0 * gaussian
        temperature[time_index] = (
            295.0 + 0.25 * xx - 0.15 * yy + 0.4 * time_index + 0.1 * case_index
        )
        wind[time_index] = 5.0 + 0.2 * xx + 0.3 * time_index
    return {
        "precip_1h_mm": precip,
        "reflectivity_dbz": reflectivity,
        "temperature_k": temperature,
        "wind_speed_ms": wind,
    }


def _model_fields(
    observation: dict[str, np.ndarray],
    *,
    arm: str,
    case_index: int,
) -> dict[str, np.ndarray]:
    if arm == "gf-subsidence-experiment":
        shift = 0
        precip_scale = 0.97
        temp_offset = 0.10
        dbz_offset = -0.3
    elif arm == "arwen-current":
        shift = 1
        precip_scale = 0.72
        temp_offset = 0.55
        dbz_offset = -4.0
    elif arm == "operational-baseline":
        shift = 0
        precip_scale = 0.90
        temp_offset = 0.30
        dbz_offset = -1.0
    else:
        shift = -1
        precip_scale = 1.18
        temp_offset = -0.65
        dbz_offset = 3.5
    precip = np.roll(observation["precip_1h_mm"], shift=shift, axis=2) * precip_scale
    reflectivity = np.roll(
        observation["reflectivity_dbz"], shift=shift, axis=2
    ) + dbz_offset
    temperature = observation["temperature_k"] + temp_offset + 0.02 * case_index
    wind = observation["wind_speed_ms"] + temp_offset * 0.2
    return {
        "precip_1h_mm": precip,
        "reflectivity_dbz": reflectivity,
        "temperature_k": temperature,
        "wind_speed_ms": wind,
    }


def _iso(unix_seconds: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest = build(args.root.resolve())
    print(manifest)
    if args.run:
        output = args.root.resolve() / "evidence"
        result = run_suite(manifest, output)
        print(result["scorecard"]["scientific_verdict"])
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
